"""
3-Layer Transformer MoE WOQ Accuracy Verification — Utilities.

Uses transformers' built-in `use_experts_implementation` mechanism:
- Set `config._experts_implementation = "grouped_mm"` for auto forward dispatch
- Pre-transpose standard weights to (E,K,N) + set `is_transposed = True`
- GptOss: already transposed, no changes needed
- `quantize_()` skips 2D bias params automatically (fixed in _swap_params)
"""

import dataclasses
import importlib
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/root/xw/transformers/src")


@dataclasses.dataclass
class ModelSpec:
    """Specification for one 3-layer transformer MoE model."""

    name: str
    config_module: str
    config_class: str
    model_module: str
    model_class: str
    config_kwargs: dict
    is_gptoss: bool = False
    use_cache_false: bool = False
    output_key: str = "logits"


_COMMON = dict(
    hidden_size=256,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    intermediate_size=512,
    vocab_size=1000,
    moe_intermediate_size=192,
    num_experts=4,
    num_experts_per_tok=2,
)

MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        name="deepseek_v2",
        config_module="transformers.models.deepseek_v2.configuration_deepseek_v2",
        config_class="DeepseekV2Config",
        model_module="transformers.models.deepseek_v2.modeling_deepseek_v2",
        model_class="DeepseekV2ForCausalLM",
        config_kwargs={**_COMMON, "first_k_dense_replace": 0, "n_shared_experts": 0},
    ),
    ModelSpec(
        name="deepseek_v3",
        config_module="transformers.models.deepseek_v3.configuration_deepseek_v3",
        config_class="DeepseekV3Config",
        model_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
        model_class="DeepseekV3ForCausalLM",
        config_kwargs={
            **_COMMON,
            "first_k_dense_replace": 0,
            "n_shared_experts": 0,
            "n_group": 1,
            "topk_group": 1,
        },
    ),
    ModelSpec(
        name="gpt_oss",
        config_module="transformers.models.gpt_oss.configuration_gpt_oss",
        config_class="GptOssConfig",
        model_module="transformers.models.gpt_oss.modeling_gpt_oss",
        model_class="GptOssForCausalLM",
        config_kwargs={**_COMMON, "num_local_experts": 4},
        is_gptoss=True,
    ),
    ModelSpec(
        name="qwen2_moe",
        config_module="transformers.models.qwen2_moe.configuration_qwen2_moe",
        config_class="Qwen2MoeConfig",
        model_module="transformers.models.qwen2_moe.modeling_qwen2_moe",
        model_class="Qwen2MoeForCausalLM",
        config_kwargs={**_COMMON, "decoder_sparse_step": 1},
    ),
    ModelSpec(
        name="qwen3_moe",
        config_module="transformers.models.qwen3_moe.configuration_qwen3_moe",
        config_class="Qwen3MoeConfig",
        model_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
        model_class="Qwen3MoeForCausalLM",
        config_kwargs={**_COMMON, "decoder_sparse_step": 1},
    ),
    ModelSpec(
        name="qwen3_5_moe",
        config_module="transformers.models.qwen3_5_moe.configuration_qwen3_5_moe",
        config_class="Qwen3_5MoeTextConfig",
        model_module="transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        model_class="Qwen3_5MoeForCausalLM",
        config_kwargs={**_COMMON, "decoder_sparse_step": 1},
        use_cache_false=True,
    ),
    ModelSpec(
        name="qwen3_next",
        config_module="transformers.models.qwen3_next.configuration_qwen3_next",
        config_class="Qwen3NextConfig",
        model_module="transformers.models.qwen3_next.modeling_qwen3_next",
        model_class="Qwen3NextForCausalLM",
        config_kwargs={**_COMMON, "decoder_sparse_step": 1},
        use_cache_false=True,
    ),
    ModelSpec(
        name="qwen3_vl_moe",
        config_module="transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe",
        config_class="Qwen3VLMoeTextConfig",
        model_module="transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe",
        model_class="Qwen3VLMoeTextModel",
        config_kwargs={**_COMMON, "decoder_sparse_step": 1},
        output_key="last_hidden_state",
    ),
    ModelSpec(
        name="mistral4",
        config_module="transformers.models.mistral4.configuration_mistral4",
        config_class="Mistral4Config",
        model_module="transformers.models.mistral4.modeling_mistral4",
        model_class="Mistral4ForCausalLM",
        config_kwargs={
            **_COMMON,
            "first_k_dense_replace": 0,
            "n_shared_experts": 0,
            "n_group": 1,
            "topk_group": 1,
        },
    ),
]


def get_test_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("No XPU or CUDA device available")


def _is_experts_module(mod: nn.Module) -> bool:
    cls = type(mod).__name__
    return "Experts" in cls or "NaiveMoe" in cls


def find_experts_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if _is_experts_module(m)]


def create_model(spec: ModelSpec, device: str, seed: int = 42) -> nn.Module:
    cfg_mod = importlib.import_module(spec.config_module)
    mdl_mod = importlib.import_module(spec.model_module)
    ConfigCls = getattr(cfg_mod, spec.config_class)
    ModelCls = getattr(mdl_mod, spec.model_class)

    config = ConfigCls(**spec.config_kwargs)
    torch.manual_seed(seed)
    model = ModelCls(config).to(torch.bfloat16).to(device)
    model.eval()
    return model


def run_model_forward(
    model: nn.Module, input_ids: torch.Tensor, spec: ModelSpec
) -> torch.Tensor:
    kwargs = {}
    if spec.use_cache_false:
        kwargs["use_cache"] = False
    with torch.no_grad():
        out = model(input_ids, **kwargs)
    return getattr(out, spec.output_key).clone()


def prepare_for_grouped_mm(model: nn.Module, spec: ModelSpec) -> None:
    """Prepare model for grouped_mm dispatch:

    1. Set config._experts_implementation = "grouped_mm"
    2. For standard models (is_transposed=False): transpose 3D weights
       to (E,K,N) layout and set is_transposed=True so _grouped_linear
       passes weights directly to _grouped_mm without .transpose() that
       would dequantize WOQ tensors.
    3. For GptOss (is_transposed=True already): no changes needed.
    """
    model.config._experts_implementation = "grouped_mm"

    for _name, mod in find_experts_modules(model):
        if not mod.is_transposed:
            for pname, param in list(mod.named_parameters(recurse=False)):
                if param.ndim == 3:
                    t = param.data.transpose(-2, -1).contiguous()
                    setattr(mod, pname, nn.Parameter(t, requires_grad=False))
            mod.is_transposed = True


# Accuracy metrics
def compute_sqnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    noise = ref.float() - test.float()
    sig_pwr = (ref.float() ** 2).mean()
    noi_pwr = (noise**2).mean()
    if noi_pwr == 0:
        return float("inf")
    return (10 * torch.log10(sig_pwr / noi_pwr)).item()


def compute_max_abs_error(ref: torch.Tensor, test: torch.Tensor) -> float:
    return (ref.float() - test.float()).abs().max().item()


def compute_mean_abs_error(ref: torch.Tensor, test: torch.Tensor) -> float:
    return (ref.float() - test.float()).abs().mean().item()


def compute_cosine_similarity(ref: torch.Tensor, test: torch.Tensor) -> float:
    return F.cosine_similarity(
        ref.float().flatten().unsqueeze(0),
        test.float().flatten().unsqueeze(0),
    ).item()
