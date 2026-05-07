# Fuzz results — scaled_dot_product_attention

- kernel: `scaled_dot_product_attention`
- iterations attempted: 250
- iterations completed (status=OK): 228
- divergences found: 58
- unsupported-on-MPS: 0
- other errors: 0
- elapsed: 1.0s
- MPS-vs-CPU max relative error: 0.3169
- MPS-vs-CUDA-mock max relative error: N/A (no NVIDIA hardware present; mocked detection cannot produce real numerics)
- recommended upstream filing target: **pytorch/pytorch**

## Tolerance model
- gpucheck per-dtype defaults (fp32=1e-4, fp16=1e-2, bf16=5e-2), MPS multiplier 2x, atol scaled by sqrt(k/128) where k = max(S, D). (rtol is not k-scaled by gpucheck, so we use the scaled atol as the single per-dtype threshold.)
- divergence rule: `max_rel_err > k_scaled_threshold`.

## Top 3 minimal repros
1. shape=(1, 1, 7, 32)  dtype=float32  stride=contiguous  shape_cat=prime  max_rel_err=0.0005017  threshold=0.0001
2. shape=(1, 1, 31, 8)  dtype=float32  stride=slice  shape_cat=prime  max_rel_err=0.0001163  threshold=9.843e-05
3. shape=(1, 1, 31, 16)  dtype=float32  stride=contiguous  shape_cat=prime  max_rel_err=0.0002737  threshold=9.843e-05

## Status breakdown
- OK: 228
- SKIP_EMPTY: 22

