"""
Per-model 3-Layer Transformer MoE WOQ Accuracy Test.

Methodology:
1. Create 3-layer transformer model → BF16 → device
2. Baseline: model forward → BF16 logits (unquantized)
3. Deep copy → prepare_for_grouped_mm (set config, transpose weights)
4. quantize_() with Float8WeightOnlyMoEConfig (skips 2D bias automatically)
5. Expected: model forward → WOQ logits (dispatched via grouped_mm_experts_forward)
6. Compare baseline vs expected (SQNR, cosine similarity, etc.)
7. Verify dispatch count matches expected grouped_mm invocations
"""

import copy
import json
import time
from typing import Any, Dict

import torch

from test_utils import (
    ModelSpec,
    compute_cosine_similarity,
    compute_max_abs_error,
    compute_mean_abs_error,
    compute_sqnr,
    create_model,
    find_experts_modules,
    get_test_device,
    prepare_for_grouped_mm,
    run_model_forward,
)


def _count_expected_dispatches(model, spec) -> int:
    """Count expected grouped_mm dispatches per forward pass.

    Each Experts module calls grouped_mm twice (gate_up_proj, down_proj)
    per decoder layer. With 3 layers, standard: 3*2=6.
    """
    experts = find_experts_modules(model)
    return len(experts) * 2


def run_test(spec: ModelSpec) -> Dict[str, Any]:
    from unittest.mock import patch

    from torchao.prototype.moe_training.config import Float8WeightOnlyMoEConfig
    from torchao.prototype.moe_training.fp8_grouped_mm import _Float8GroupedMM
    from torchao.prototype.moe_training.tensor import (
        Float8InferenceWeightOnlyWrapperTensor,
    )
    from torchao.quantization.quant_api import quantize_

    device = get_test_device()
    result: Dict[str, Any] = {
        "model": spec.name,
        "status": "FAIL",
        "device": device,
    }

    t0 = time.time()

    try:
        # 1. Create model and get baseline
        model = create_model(spec, device)
        input_ids = torch.randint(0, spec.config_kwargs["vocab_size"], (1, 16), device=device)
        baseline = run_model_forward(model, input_ids, spec)
        result["baseline_shape"] = list(baseline.shape)

        # 2. Prepare quantized model
        expected_dispatches = _count_expected_dispatches(model, spec)
        model_q = copy.deepcopy(model)
        del model
        torch.xpu.empty_cache()

        prepare_for_grouped_mm(model_q, spec)

        def filter_fn(m, fqn):
            cls = type(m).__name__
            return "Experts" in cls or "NaiveMoe" in cls

        quantize_(model_q, Float8WeightOnlyMoEConfig(), filter_fn=filter_fn)

        # Verify quantization: 3D → WOQ, 2D → BF16
        for name, mod in find_experts_modules(model_q):
            for pname, p in mod.named_parameters(recurse=False):
                is_woq = isinstance(p.data, Float8InferenceWeightOnlyWrapperTensor)
                if p.ndim == 3:
                    if is_woq:
                        print(f"{name}.{pname}: 3D param quantized")
                    assert is_woq, f"{name}.{pname}: 3D param not quantized"
                else:
                    assert not is_woq, f"{name}.{pname}: non-3D param should not be quantized"

        # 3. Run quantized model with dispatch counting
        dispatch_count = [0]
        orig_apply = _Float8GroupedMM.apply

        def tracked_apply(*args, **kwargs):
            dispatch_count[0] += 1
            return orig_apply(*args, **kwargs)

        with torch.no_grad():
            with patch.object(_Float8GroupedMM, "apply", side_effect=tracked_apply):
                expected = run_model_forward(model_q, input_ids, spec)

        # 4. Compute accuracy metrics
        result["sqnr_db"] = compute_sqnr(baseline, expected)
        result["cosine_similarity"] = compute_cosine_similarity(baseline, expected)
        result["max_abs_error"] = compute_max_abs_error(baseline, expected)
        result["mean_abs_error"] = compute_mean_abs_error(baseline, expected)
        result["dispatch_count"] = dispatch_count[0]
        result["expected_dispatches"] = expected_dispatches
        result["dispatch_match"] = dispatch_count[0] == expected_dispatches

        # 5. Determine pass/fail
        passed = (
            result["sqnr_db"] > 15.0
            and result["cosine_similarity"] > 0.99
            and result["dispatch_match"]
        )
        result["status"] = "PASS" if passed else "FAIL"

        del model_q
        torch.xpu.empty_cache()

    except Exception as e:
        import traceback

        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        torch.xpu.empty_cache()

    result["elapsed_seconds"] = round(time.time() - t0, 2)
    return result
