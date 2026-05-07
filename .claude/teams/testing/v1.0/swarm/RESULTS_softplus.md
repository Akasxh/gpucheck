# torch.nn.functional.softplus — MPS fuzz report (v2)

- **Kernel:** `softplus` (`torch.nn.functional.softplus`)
- **Status:** OK
- **Aborted early:** False
- **Iterations attempted:** 1000
- **Iterations completed:** 1000
- **Iterations unsupported (skipped):** 0
- **Iterations errored:** 0
- **Divergences (FILABLE, >=3 seeds):** 0
- **Divergences (RECALIBRATION):** 0
- **Divergences (other, <3 seeds, not filable):** 0
- **Max abs err (global):** 1.953e-03
- **Max rel err (global, raw):** 5.814e-03
- **Max rel err (denom>=1e-6 filter):** 5.814e-03
- **Recommended filing target:** `none`
- **Elapsed:** 2.68 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), seeds: [0, 1, 2, 3, 4]

## v2 divergence rules

- `max_abs_err > 10×atol` always counts as a divergence candidate.
- `max_rel_err > 10×rtol` counts ONLY if `denom_magnitude >= 1e-6` (filters near-zero artifacts where `softplus(x)≈0` for very negative `x`).
- A config is FILABLE only if the divergence reproduces across ≥3 of the 5 seeds {0,1,2,3,4}.
- 1×–5× tolerance ⇒ TOLERANCE_RECALIBRATION (xfail or tighten).
- ≤1× tolerance ⇒ OK.

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 182, 'non_tile_aligned': 175, 'prime': 173, 'power_of_2_boundary': 159, 'large': 166, 'mixed': 145} |
| dtype | {'float32': 322, 'float16': 347, 'bfloat16': 331} |
| stride category | {'row_major': 130, 'column_major': 154, 'broadcast': 132, 'transpose': 150, 'slice': 154, 'non_contig': 146, 'gather': 134} |

## Top 3 repros

### Repro #1

- **shape:** `(1024, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `transpose`
- **softplus params:** beta=2.0, threshold=5.0
- **seeds seen:** [2]
- **seeds with DIVERGENCE:** []
- **seeds with RECALIBRATION:** [2]
- **max abs err:** 2.384e-07  (atol=2.00e-04)
- **max rel err:** 5.351e-04  (rtol=2.00e-04, denom_mag=4.480e-05)
- **max ratio (×tol):** 2.68
- **CPU layout:** `shape=(1024, 64), strides=(1, 1024), contig=False`
- **MPS layout:** `shape=(1024, 64), strides=(1, 1024), contig=False`

### Repro #2

- **shape:** `(1024, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `gather`
- **softplus params:** beta=2.0, threshold=50.0
- **seeds seen:** [2]
- **seeds with DIVERGENCE:** []
- **seeds with RECALIBRATION:** [2]
- **max abs err:** 4.768e-07  (atol=2.00e-04)
- **max rel err:** 3.245e-04  (rtol=2.00e-04, denom_mag=6.548e-05)
- **max ratio (×tol):** 1.62
- **CPU layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **MPS layout:** `shape=(1024, 64), strides=(64, 1), contig=True`

### Repro #3

- **shape:** `(2048,)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `column_major`
- **softplus params:** beta=2.0, threshold=50.0
- **seeds seen:** [3]
- **seeds with DIVERGENCE:** []
- **seeds with RECALIBRATION:** [3]
- **max abs err:** 2.384e-07  (atol=2.00e-04)
- **max rel err:** 3.229e-04  (rtol=2.00e-04, denom_mag=7.245e-05)
- **max ratio (×tol):** 1.61
- **CPU layout:** `shape=(2048,), strides=(1,), contig=True`
- **MPS layout:** `shape=(2048,), strides=(1,), contig=True`

## Method notes

- Reference: `torch.nn.functional.softplus` on CPU (comparison against MPS, both promoted to FP32 for error metrics).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`. softplus is elementwise — no `k_dim` scaling.
- softplus(x) = log(1 + exp(beta*x)) / beta with `x>threshold` returning x as-is. Both branches are exercised via `BETA_CHOICES` and `THRESHOLD_CHOICES`.
- denom_magnitude is the |reference| value at the worst-rel-err location, used to discard near-zero-denominator artifacts (softplus(x)→0 as x→-∞).
- CUDA backend is mocked (no NVIDIA GPU present); MPS-vs-CUDA is N/A.
