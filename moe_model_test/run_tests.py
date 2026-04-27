"""
Run all 9 MoE model WOQ accuracy tests.

Usage:  python run_tests.py [model_name ...]
  No args → run all 9.  Pass names to filter.
"""

import json
import os
import sys
import time
from pathlib import Path

# Adjust path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/root/xw/transformers/src")

from test_moe_accuracy import run_test
from test_utils import MODEL_SPECS

LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)


def generate_report(results, report_path):
    lines = [
        "# MoE WOQ Accuracy Verification — 3-Layer Transformer Models",
        "",
        "## Methodology",
        "",
        "- **Baseline**: BF16 model forward pass (unquantized)",
        "- **Expected**: WOQ-quantized model via `Float8WeightOnlyMoEConfig`",
        "- **Dispatch**: `config._experts_implementation = \"grouped_mm\"` →",
        "  `grouped_mm_experts_forward` → `_grouped_linear` → `torch._grouped_mm`",
        "  → `Float8InferenceWeightOnlyWrapperTensor.__torch_function__` →",
        "  `_Float8GroupedMM` → dequant + `torch._grouped_mm` (XPU fallback)",
        "- **Weight layout**: Standard models pre-transposed to (E,K,N) with",
        "  `is_transposed=True`; GptOss already in (E,K,N) layout natively",
        "- **Bias handling**: `_swap_params` auto-skips non-3D params",
        "- **Config**: 3 decoder layers, hidden=256, moe_intermediate=192,",
        "  4 experts, top-2 routing",
        "",
        "## Results",
        "",
        "| Model | Status | SQNR (dB) | Cosine Sim | Dispatch | Time |",
        "|-------|--------|-----------|------------|----------|------|",
    ]

    pass_count = 0
    for r in results:
        st = r.get("status", "FAIL")
        if st == "PASS":
            pass_count += 1
        sqnr = f"{r.get('sqnr_db', 0):.2f}" if "sqnr_db" in r else "N/A"
        cos = f"{r.get('cosine_similarity', 0):.6f}" if "cosine_similarity" in r else "N/A"
        dc = r.get("dispatch_count", "?")
        ed = r.get("expected_dispatches", "?")
        disp = f"{dc}/{ed}"
        elapsed = f"{r.get('elapsed_seconds', 0):.1f}s"
        error = ""
        if "error" in r:
            error = f" ({r['error'][:60]})"
        lines.append(f"| {r['model']} | {st} | {sqnr} | {cos} | {disp} | {elapsed} |{error}")

    lines += [
        "",
        f"**Total: {pass_count}/{len(results)} PASSED**",
        "",
        "## Pass Criteria",
        "",
        "- SQNR > 15.0 dB",
        "- Cosine Similarity > 0.99",
        "- Dispatch count matches expected (2 per layer × 3 layers = 6)",
        "",
    ]

    # Add per-model details
    lines += ["## Per-Model Details", ""]
    for r in results:
        lines.append(f"### {r['model']}")
        lines.append("")
        if "error" in r:
            lines.append(f"**Error**: {r['error']}")
            if "traceback" in r:
                lines.append(f"```\n{r['traceback']}\n```")
        else:
            for k in [
                "sqnr_db",
                "cosine_similarity",
                "max_abs_error",
                "mean_abs_error",
                "dispatch_count",
                "expected_dispatches",
                "baseline_shape",
                "elapsed_seconds",
            ]:
                if k in r:
                    lines.append(f"- **{k}**: {r[k]}")
        lines.append("")

    report = "\n".join(lines)
    Path(report_path).write_text(report)
    return report


def main():
    filter_names = set(sys.argv[1:]) if len(sys.argv) > 1 else None
    specs = MODEL_SPECS
    if filter_names:
        specs = [s for s in specs if s.name in filter_names]
        if not specs:
            print(f"No models matched: {filter_names}")
            sys.exit(1)

    print(f"Running {len(specs)} model(s): {[s.name for s in specs]}")
    results = []
    for spec in specs:
        print(f"\n{'='*60}")
        print(f"Testing: {spec.name}")
        print(f"{'='*60}")
        r = run_test(spec)
        results.append(r)

        # Save per-model log
        log_path = LOGS_DIR / f"{spec.name}.json"
        with open(log_path, "w") as f:
            json.dump(r, f, indent=2, default=str)

        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"{icon} {spec.name}: {r['status']}")
        if "sqnr_db" in r:
            print(f"   SQNR={r['sqnr_db']:.2f} dB  Cosine={r['cosine_similarity']:.6f}  Dispatch={r['dispatch_count']}/{r['expected_dispatches']}")
        if "error" in r:
            print(f"   Error: {r['error']}")

    # Generate summary report
    report_path = Path(__file__).parent / "summary_report.md"
    report = generate_report(results, report_path)
    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"SUMMARY: {passed}/{len(results)} PASSED")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
