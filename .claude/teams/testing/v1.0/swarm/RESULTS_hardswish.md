# torch.nn.functional.hardswish - MPS v2 fuzz report

- **kernel:** `torch.nn.functional.hardswish`
- **status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **divergences_filable:** 0
- **divergences_recalibration:** 0
- **max_abs_err:** 0.000e+00
- **max_rel_err:** 0.000e+00
- **elapsed_seconds:** 5.55 (budget 675s)
- **seeds:** [0, 1, 2, 3, 4]
- **n_configs_target:** 1000
- **torch:** 2.11.0 on darwin/arm64 (Apple Silicon)
- **CUDA vs MPS:** N/A (no NVIDIA GPU on host; CUDA backend is mocked)
- **recommended_filing_target:** `none`

## v2 divergence rubric

- `max_abs_err > 10x atol` -> FILABLE_CANDIDATE (always counts).
- `max_rel_err > 10x rtol AND denom_at_peak_rel >= 1e-6` -> FILABLE_CANDIDATE.
- `1x..5x tol` -> RECALIBRATION (recommend xfail entry).
- `< 1x tol` -> OK.
- A config is FILABLE only when >=3 of the 5 seeds hit the FILABLE_CANDIDATE bar.

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 180, 'non_tile_aligned': 151, 'prime': 183, 'power_of_2_boundary': 149, 'large': 181, 'mixed': 156} |
| dtype | {'float32': 325, 'float16': 332, 'bfloat16': 343} |
| stride category | {'row_major': 128, 'column_major': 159, 'broadcast': 158, 'transpose': 130, 'slice': 139, 'non_contig': 152, 'gather': 134} |

## Top 3 repros

_No divergences exceeded the v2 thresholds (atol/rtol from `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`)._
## Method notes

- Reference: `torch.nn.functional.hardswish` on CPU (FP32 promotion for accuracy comparison).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`.
- `denom_at_peak_rel` is `|y_cpu_fp32|` at the element where the MPS-vs-CPU rel-err peaks. Values below `1e-6` are treated as near-zero-denominator artifacts and excluded from divergence (per v2 spec).
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- CUDA backend is mocked (no NVIDIA GPU on host); cross-device MPS-vs-CUDA comparison reported as N/A by spec.
