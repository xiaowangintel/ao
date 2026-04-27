# MoE WOQ Accuracy Verification — 3-Layer Transformer Models

## Methodology

- **Baseline**: BF16 model forward pass (unquantized)
- **Expected**: WOQ-quantized model via `Float8WeightOnlyMoEConfig`
- **Dispatch**: `config._experts_implementation = "grouped_mm"` →
  `grouped_mm_experts_forward` → `_grouped_linear` → `torch._grouped_mm`
  → `Float8InferenceWeightOnlyWrapperTensor.__torch_function__` →
  `_Float8GroupedMM` → dequant + `torch._grouped_mm` (XPU fallback)
- **Weight layout**: Standard models pre-transposed to (E,K,N) with
  `is_transposed=True`; GptOss already in (E,K,N) layout natively
- **Bias handling**: `_swap_params` auto-skips non-3D params
- **Config**: 3 decoder layers, hidden=256, moe_intermediate=192,
  4 experts, top-2 routing

## Results

| Model | Status | SQNR (dB) | Cosine Sim | Dispatch | Time |
|-------|--------|-----------|------------|----------|------|
| deepseek_v2 | PASS | 45.50 | 0.999986 | 6/6 | 2.1s |
| deepseek_v3 | PASS | 28.83 | 0.999346 | 6/6 | 0.6s |
| gpt_oss | PASS | 25.59 | 0.998619 | 6/6 | 0.2s |
| qwen2_moe | PASS | 36.43 | 0.999886 | 6/6 | 0.1s |
| qwen3_moe | PASS | 40.50 | 0.999955 | 6/6 | 0.0s |
| qwen3_5_moe | PASS | 19.50 | 0.994385 | 6/6 | 0.2s |
| qwen3_next | PASS | 18.98 | 0.993668 | 6/6 | 0.1s |
| qwen3_vl_moe | PASS | 33.41 | 0.999772 | 6/6 | 0.0s |
| mistral4 | PASS | 34.48 | 0.999822 | 6/6 | 0.3s |

**Total: 9/9 PASSED**

## Pass Criteria

- SQNR > 15.0 dB
- Cosine Similarity > 0.99
- Dispatch count matches expected (2 per layer × 3 layers = 6)

## Per-Model Details

### deepseek_v2

- **sqnr_db**: 45.50373840332031
- **cosine_similarity**: 0.9999860525131226
- **max_abs_error**: 0.0078125
- **mean_abs_error**: 0.001238056574948132
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 2.08

### deepseek_v3

- **sqnr_db**: 28.83037567138672
- **cosine_similarity**: 0.9993456602096558
- **max_abs_error**: 0.104736328125
- **mean_abs_error**: 0.006209942977875471
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.62

### gpt_oss

- **sqnr_db**: 25.589065551757812
- **cosine_similarity**: 0.9986194968223572
- **max_abs_error**: 0.07421875
- **mean_abs_error**: 0.013340490870177746
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.19

### qwen2_moe

- **sqnr_db**: 36.431575775146484
- **cosine_similarity**: 0.9998863935470581
- **max_abs_error**: 0.046875
- **mean_abs_error**: 0.0030269471462816
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.1

### qwen3_moe

- **sqnr_db**: 40.49770736694336
- **cosine_similarity**: 0.9999553561210632
- **max_abs_error**: 0.0126953125
- **mean_abs_error**: 0.0024280105717480183
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.04

### qwen3_5_moe

- **sqnr_db**: 19.49864959716797
- **cosine_similarity**: 0.9943850636482239
- **max_abs_error**: 0.287109375
- **mean_abs_error**: 0.020220158621668816
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.19

### qwen3_next

- **sqnr_db**: 18.975154876708984
- **cosine_similarity**: 0.9936676621437073
- **max_abs_error**: 0.3203125
- **mean_abs_error**: 0.01973189227283001
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.13

### qwen3_vl_moe

- **sqnr_db**: 33.40561294555664
- **cosine_similarity**: 0.999771773815155
- **max_abs_error**: 0.1953125
- **mean_abs_error**: 0.013111336156725883
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 256]
- **elapsed_seconds**: 0.04

### mistral4

- **sqnr_db**: 34.479034423828125
- **cosine_similarity**: 0.9998217821121216
- **max_abs_error**: 0.052734375
- **mean_abs_error**: 0.003810788271948695
- **dispatch_count**: 6
- **expected_dispatches**: 6
- **baseline_shape**: [1, 16, 1000]
- **elapsed_seconds**: 0.28
