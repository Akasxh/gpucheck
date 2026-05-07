# torch.nn.functional.silu — MPS fuzz report (v2)

- **Kernel:** `torch.nn.functional.silu`
- **Status:** OK
- **Iters attempted:** 1000 / target 1000
- **Iters completed:** 1000
- **Iters unsupported:** 0
- **Divergences (FILABLE, >=3 seeds):** 0
- **Divergences (TOLERANCE_RECALIBRATION):** 0
- **Max abs err:** 1.221e-04
- **Max rel err (denom>=1e-6):** 5.200e-04
- **Filing target:** `none`
- **Elapsed:** 1.92 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), seeds: [0, 1, 2, 3, 4]

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'power_of_2_boundary': 165, 'large': 180, 'prime': 153, 'non_tile_aligned': 174, 'degenerate': 153, 'mixed': 175} |
| dtype | {'float32': 343, 'float16': 320, 'bfloat16': 337} |
| stride category | {'broadcast': 160, 'gather': 137, 'slice': 133, 'row_major': 153, 'transpose': 139, 'non_contig': 150, 'column_major': 128} |

## Per-axis max relative error (denom-filtered)

| dtype | max_rel_err |
|---|---|
| `bfloat16` | 0.000e+00 |
| `float16` | 5.200e-04 |
| `float32` | 2.356e-07 |

| stride_category | max_rel_err |
|---|---|
| `broadcast` | 5.200e-04 |
| `column_major` | 5.200e-04 |
| `gather` | 5.200e-04 |
| `non_contig` | 5.200e-04 |
| `row_major` | 5.200e-04 |
| `slice` | 5.200e-04 |
| `transpose` | 5.200e-04 |

## Top 3 minimal repros

_No FILABLE or RECALIBRATION-bucket records — all errors fell within 1x the dtype tolerance (assertions/tolerances.py with MPS 2x multiplier)._
## Method notes

- Reference: `torch.nn.functional.silu` on CPU promoted to FP32 for the error comparison.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`.
- Classification: max_abs > 10x atol => DIVERGENCE; max_rel > 10x rtol with |b|>=1e-6 => DIVERGENCE; otherwise the [1x, 5x] band is TOLERANCE_RECALIBRATION; below 1x is OK.
- A DIVERGENCE only becomes FILABLE if its (stride, dtype, shape) signature reproduces under >=3 distinct seeds.
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- Fuzzer: `_fuzz_silu_v2.py` in this directory.