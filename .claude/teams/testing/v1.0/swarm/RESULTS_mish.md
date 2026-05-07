# torch.nn.functional.mish - MPS fuzz report (v2)

- **Kernel:** `torch.nn.functional.mish`
- **Status:** OK
- **Iterations attempted:** 1000 / 1000 target
- **Iterations completed:** 1000
- **Iterations unsupported:** 0
- **Iterations errored:** 0
- **Iterations empty (numel==0, skipped):** 96
- **Reproduction extra iters:** 0
- **Divergences (FILABLE, >=3 seeds, >10x tol, denom-filtered):** 0
- **Divergences (TOLERANCE_RECALIBRATION, 1-5x tol or non-reproducible):** 0
- **MPS-vs-CPU max absolute error:** 4.883e-04
- **MPS-vs-CPU max relative error (denom-filtered):** 8.881e-04
- **MPS-vs-CUDA max relative error:** N/A (CUDA mocked - no NVIDIA GPU on host)
- **Recommended filing target:** `none`
- **Elapsed:** 1.7 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), seeds: [0, 1, 2, 3, 4]

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 185, 'non_tile_aligned': 149, 'prime': 159, 'power_of_2_boundary': 166, 'large': 176, 'mixed': 165} |
| dtype | {'float32': 353, 'float16': 326, 'bfloat16': 321} |
| stride category | {'row_major': 164, 'column_major': 155, 'broadcast': 134, 'transpose': 139, 'slice': 143, 'non_contig': 128, 'gather': 137} |

## Filtering rules (v2)

- A divergence candidate requires `max_abs_err > 10x atol` OR
  (`max_rel_err > 10x rtol` AND `denom_magnitude >= 1e-6`).
- A FILABLE divergence has the same (shape, dtype, stride_category) tuple
  reproducing at >10x tolerance on >=3 of seeds {0,1,2,3,4}.
- A TOLERANCE_RECALIBRATION entry sits in the 1-5x band, or shows >10x on
  fewer than 3 seeds (single-seed spikes / non-reproducible).
- Anything below 1x tolerance is OK and not surfaced.

## Top 3 minimal repros

_No divergences exceeded gpucheck's per-dtype tolerance (MPS 2x multiplier applied; see `assertions/tolerances.py`)._
## Method notes

- Reference: `torch.nn.functional.mish` on CPU (FP32-promoted comparison).
- Tolerance source: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (CUDA base tol x MPS 2x multiplier).
- mish is elementwise; no sqrt(k/128) matmul scaling applied.
- Stride categories sampled: row_major, column_major, broadcast,
  transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- `torch.mps.synchronize()` is called after each MPS forward to defeat
  async kernel completion before pulling the result back to CPU for compare.
- CUDA mocked: no NVIDIA GPU on host - cross-device CUDA-vs-MPS comparison is N/A.
