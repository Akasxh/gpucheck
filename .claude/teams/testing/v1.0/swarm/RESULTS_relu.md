# RESULTS_relu — gpucheck stride/shape/dtype fuzz (v2)

- **kernel:** `torch.relu`
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **divergences_filable:** 0  (>10x tol AND reproducible across >=3 seeds)
- **divergences_recalibration:** 0  (1-10x tol on >=3 seeds — recommend xfail entry)
- **max_abs_err (global, MPS vs CPU):** 0
- **max_rel_err (global, safe-denom):** 0
- **MPS vs CUDA-mock max relative error:** N/A (no NVIDIA GPU on host)
- **recommended upstream filing target:** `none`
- **elapsed_seconds:** 4.88 (budget 690s, aborted=False)
- **n_configs:** 200, **seeds:** [0, 1, 2, 3, 4], **iters_total_target:** 1000
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)

## Top 3 minimal repros

_No configurations exceeded gpucheck's per-dtype tolerance (with MPS multiplier) on >=3 of 5 seeds — clean run for relu._
## Method

- 200 unique (shape, dtype, stride) configs generated with deterministic config-rng (seed=0xcafebabe).
- Each config run with input seeds 0,1,2,3,4 → 1000 target iterations.
- Reference: `torch.relu` on CPU; comparison promoted to fp64.
- Tolerances: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (MPS multiplier applied).
- Divergence filtering (v2):
  - `max_rel_err > 10x rtol` counts only when `denom_magnitude >= 1e-6` at the offending element.
  - `max_abs_err > 10x atol` always counts.
  - **FILABLE**: > 10x AND reproducible across >= 3 seeds.
  - **RECALIBRATION**: 1-10x on >= 3 seeds (recommend xfail).
  - **OK**: < 1x tolerance.
- Stride categories drawn: row_major, column_major, broadcast, transpose, slice, non_contig, gather.
