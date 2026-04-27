# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Float8 weight-only quantization of MoE expert weights.

The test model (SimpleMoEExperts) uses the same structure as real-world
MoE implementations (Qwen3, DeepSeek, etc.) with 3D weight tensors:
  - gate_up_proj: (num_experts, 2*intermediate_dim, hidden_dim)
  - down_proj: (num_experts, hidden_dim, intermediate_dim)
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

# Determine available test device
_has_cuda_sm90 = (
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (9, 0)
)
_has_xpu = torch.xpu.is_available() if hasattr(torch, "xpu") else False

if not _has_cuda_sm90 and not _has_xpu:
    pytest.skip(
        "Requires CUDA SM90+ or XPU device", allow_module_level=True
    )

# Choose test device: prefer XPU if available (our target), fallback to CUDA
_test_device = "xpu" if _has_xpu else "cuda"

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.config import Float8WeightOnlyMoEConfig
from torchao.prototype.moe_training.tensor import Float8InferenceWeightOnlyWrapperTensor
from torchao.quantization.quant_api import quantize_


class SimpleMoEExperts(nn.Module):
    """
    Simplified MoE Experts module matching the structure found in
    Qwen3MoE/DeepSeekV3/Qwen3.5MoE transformers models.

    Uses torch._grouped_mm for efficient batched expert computation.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_dim: int,
        intermediate_dim: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        # gate_up_proj: (E, 2*intermediate_dim, hidden_dim) — combined gate and up projection
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_dim, hidden_dim)
        )
        # down_proj: (E, hidden_dim, intermediate_dim)
        self.down_proj = nn.Parameter(
            torch.empty(num_experts, hidden_dim, intermediate_dim)
        )

    def forward(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor
    ) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)

        # gate_up = grouped_mm(x, gate_up_proj^T)
        # gate_up_proj is (E, 2*I, H), transposed to (E, H, 2*I) for right operand
        gate_up = torch._grouped_mm(
            x.bfloat16(),
            self.gate_up_proj.bfloat16().transpose(-2, -1),
            offs=offsets,
        )

        gate, up = gate_up.chunk(2, dim=-1)
        h = F.silu(gate) * up

        # out = grouped_mm(h, down_proj^T)
        # down_proj is (E, H, I), transposed to (E, I, H) for right operand
        out = torch._grouped_mm(
            h,
            self.down_proj.bfloat16().transpose(-2, -1),
            offs=offsets,
        )
        return out.type_as(x)

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.gate_up_proj, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)


def _create_test_inputs(
    num_experts: int,
    hidden_dim: int,
    tokens_per_expert: int,
    device: str = _test_device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Create test inputs with evenly distributed tokens across experts."""
    total_tokens = num_experts * tokens_per_expert
    x = torch.randn(total_tokens, hidden_dim, dtype=dtype, device=device)
    num_tokens_per_expert = torch.full(
        (num_experts,), tokens_per_expert, dtype=torch.int32, device=device
    )
    return x, num_tokens_per_expert


class TestFloat8WeightOnlyMoE:
    """Tests for Float8 weight-only quantization of MoE experts."""

    @pytest.fixture
    def model_params(self):
        return {
            "num_experts": 8,
            "hidden_dim": 1024,
            "intermediate_dim": 2048,
        }

    @pytest.fixture
    def model(self, model_params):
        model = SimpleMoEExperts(**model_params).to(torch.bfloat16).to(_test_device)
        torch.manual_seed(42)
        model.init_weights()
        return model

    def test_fp8_weight_only_tensor_from_hp(self, model_params):
        """Test Float8InferenceWeightOnlyWrapperTensor.from_hp quantization."""
        config = Float8WeightOnlyMoEConfig()
        weight = torch.randn(
            model_params["num_experts"],
            model_params["intermediate_dim"],
            model_params["hidden_dim"],
            dtype=torch.bfloat16,
            device=_test_device,
        )

        wo_tensor = Float8InferenceWeightOnlyWrapperTensor.from_hp(weight, config)

        assert isinstance(wo_tensor, Float8InferenceWeightOnlyWrapperTensor)
        assert wo_tensor._data.qdata.dtype == config.weight_dtype
        assert wo_tensor._data.scale is not None
        assert wo_tensor.dtype == torch.bfloat16
        assert wo_tensor.shape == weight.shape

    def test_fp8_weight_only_tensor_dequantize(self, model_params):
        """Test that dequantize recovers a close approximation of the original."""
        config = Float8WeightOnlyMoEConfig()
        weight = torch.randn(
            model_params["num_experts"],
            model_params["intermediate_dim"],
            model_params["hidden_dim"],
            dtype=torch.bfloat16,
            device=_test_device,
        )

        wo_tensor = Float8InferenceWeightOnlyWrapperTensor.from_hp(weight, config)
        dequantized = wo_tensor.dequantize()

        assert dequantized.shape == weight.shape
        assert dequantized.dtype == weight.dtype

        sqnr = compute_error(weight, dequantized)
        assert sqnr.item() >= 20.0, f"Dequantize SQNR {sqnr.item()} too low"

    def test_fp8_weight_only_moe_forward(self, model, model_params):
        """Test forward pass: compare bf16 reference vs FP8 weight-only."""
        import copy

        ref_model = copy.deepcopy(model)

        # Apply weight-only quantization
        config = Float8WeightOnlyMoEConfig()

        def moe_filter_fn(mod, fqn):
            return True

        quantize_(model, config=config, filter_fn=moe_filter_fn)

        # Create inputs
        x, num_tokens_per_expert = _create_test_inputs(
            model_params["num_experts"],
            model_params["hidden_dim"],
            tokens_per_expert=32,
        )

        with torch.no_grad():
            ref_out = ref_model(x, num_tokens_per_expert)
            out = model(x, num_tokens_per_expert)

        assert out.shape == ref_out.shape
        sqnr = compute_error(ref_out, out)
        assert sqnr.item() >= 18.0, (
            f"Forward SQNR {sqnr.item()} too low, expected >= 18.0"
        )

    def test_fp8_weight_only_moe_quantize_api(self, model, model_params):
        """Test quantize_() API integration with module filter."""
        config = Float8WeightOnlyMoEConfig()

        def moe_filter_fn(mod, fqn):
            return True

        # Apply quantize_ to the model
        quantize_(model, config=config, filter_fn=moe_filter_fn)

        # Verify parameters were converted
        assert isinstance(model.gate_up_proj.data, Float8InferenceWeightOnlyWrapperTensor), (
            "gate_up_proj should be Float8InferenceWeightOnlyWrapperTensor"
        )
        assert isinstance(model.down_proj.data, Float8InferenceWeightOnlyWrapperTensor), (
            "down_proj should be Float8InferenceWeightOnlyWrapperTensor"
        )

        # Verify requires_grad is False (inference-only)
        assert not model.gate_up_proj.requires_grad, (
            "gate_up_proj should not require grad"
        )
        assert not model.down_proj.requires_grad, (
            "down_proj should not require grad"
        )

    def test_fp8_weight_only_moe_quantize_with_filter(self, model, model_params):
        """Test quantize_() with a module filter function."""
        config = Float8WeightOnlyMoEConfig()

        def filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
            return True

        quantize_(model, config=config, filter_fn=filter_fn)

        # Both should be converted since filter returns True for all
        assert isinstance(model.gate_up_proj.data, Float8InferenceWeightOnlyWrapperTensor)
        assert isinstance(model.down_proj.data, Float8InferenceWeightOnlyWrapperTensor)

    def test_tensor_flatten_unflatten(self, model_params):
        """Test __tensor_flatten__ and __tensor_unflatten__."""
        config = Float8WeightOnlyMoEConfig()
        weight = torch.randn(
            model_params["num_experts"],
            model_params["intermediate_dim"],
            model_params["hidden_dim"],
            dtype=torch.bfloat16,
            device=_test_device,
        )

        wo_tensor = Float8InferenceWeightOnlyWrapperTensor.from_hp(weight, config)

        # Flatten
        inner_names, metadata = wo_tensor.__tensor_flatten__()
        assert "_data" in inner_names

        # Unflatten
        inner_tensors = {
            "_data": wo_tensor._data,
        }
        restored = Float8InferenceWeightOnlyWrapperTensor.__tensor_unflatten__(
            inner_tensors, metadata, wo_tensor.shape, None
        )

        assert isinstance(restored, Float8InferenceWeightOnlyWrapperTensor)
        assert torch.equal(restored._data.qdata.float(), wo_tensor._data.qdata.float())
        assert torch.equal(restored._data.scale, wo_tensor._data.scale)
