# Fuzz results — argmax

- kernel: `torch.argmax`
- iterations attempted: 250
- iterations completed (status=OK): 250
- divergences found: 0
- unsupported-on-MPS / CPU: 0
- other errors: 0
- elapsed: 0.6s
- MPS-vs-CPU max relative error (value-at-argmax): 0
- MPS-vs-CUDA-mock max relative error: N/A (no NVIDIA GPU; CUDA detection mocked off)
- recommended upstream filing target: **none**

## Method
- Compared the *value at the picked index*: `x[mps_argmax]` vs `x[cpu_fp32_argmax]`. This is tie-tolerant: when MPS and CPU pick different indices that happen to hold the same max, the metric is 0.
- Tolerance from `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')` (per-dtype atol/rtol scaled by sqrt(k/128), MPS overlay multiplier 2x).
- Divergence rule: `max_rel_err > rtol`.
- CUDA detection mocked via `unittest.mock.patch('torch.cuda.is_available', return_value=False)` (no NVIDIA GPU on this Mac).

## Top 3 minimal repros
_None — no divergences found within budget._

## Status breakdown
- OK: 250

