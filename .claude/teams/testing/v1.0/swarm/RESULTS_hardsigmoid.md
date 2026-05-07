# torch.nn.functional.hardsigmoid - MPS fuzz report

- **Kernel:** `torch.nn.functional.hardsigmoid`
- **Status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **iters_errored:** 0
- **divergences_filable:** 0
- **divergences_recalibration:** 0
- **max_abs_err:** 0.000e+00
- **max_rel_err (denom>=1e-6):** 0.000e+00
- **Elapsed:** 1.37 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **Seeds:** [0, 1, 2, 3, 4] x 200 iters/seed
- **Total over-tolerance hits (any band):** 0

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 153, 'non_tile_aligned': 174, 'prime': 153, 'power_of_2_boundary': 165, 'large': 180, 'mixed': 175} |
| dtype | {'float32': 343, 'float16': 320, 'bfloat16': 337} |
| stride category | {'row_major': 153, 'column_major': 128, 'broadcast': 160, 'transpose': 139, 'slice': 133, 'non_contig': 150, 'gather': 137} |

## Divergence classification

- **FILABLE_CANDIDATE:** max_abs_err > 10*atol, OR (max_rel_err > 10*rtol AND denom_magnitude >= 1e-6).
- **FILABLE (in `divergences_filable`):** a FILABLE_CANDIDATE signature `(dtype, stride_category, shape_bucket, shape)` that reproduces across **>=3 of 5 seeds**.
- **TOLERANCE_RECALIBRATION:** error in the 1x..5x tol band -> recommend an `[tool.gpucheck.mps.xfail]` or multiplier nudge.
- **OK:** below 1x tol.

## Top 3 Recalibration Hits

_None - no FILABLE or RECALIBRATION hits in this run._
## Method notes

- Reference: `torch.nn.functional.hardsigmoid` on CPU (comparison done in fp32 promotion).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (MPS overlay applied).
- `max_rel_err` is computed only over samples where |reference| >= 1e-6, removing near-zero-denominator artifacts.
- hardsigmoid saturates to 0 for x<=-3 and to 1 for x>=3, so exact-zero outputs are common and would otherwise inflate rel-err with /eps division.
- 1000 iters across seeds 0..4 (200 per seed); FILABLE requires >=3 seeds reproducing the same (dtype, stride, bucket, shape).
- CUDA backend mocked (no NVIDIA GPU on host); MPS-vs-CUDA cross-device check is N/A.
