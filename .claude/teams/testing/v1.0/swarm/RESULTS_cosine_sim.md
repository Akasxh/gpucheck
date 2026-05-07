# torch.nn.functional.cosine_similarity — MPS fuzz report

- **Kernel:** `torch.nn.functional.cosine_similarity`
- **Status:** OK
- **Iterations attempted:** 250
- **Iterations completed:** 250
- **Iterations unsupported (skipped):** 0
- **Divergences found:** 0
- **MPS vs CPU max relative error:** 2.366e-03
- **MPS vs CPU max absolute error:** 1.526e-05
- **MPS vs CUDA-mock max relative error:** N/A (CUDA mocked — no NVIDIA GPU on host)
- **Recommended filing target:** `none`
- **Elapsed:** 1.01 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), seed: 0xc051

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 42, 'non_tile_aligned': 43, 'prime': 40, 'power_of_2_boundary': 56, 'large': 31, 'mixed': 38} |
| dtype | {'float32': 82, 'float16': 92, 'bfloat16': 76} |
| stride category | {'row_major': 39, 'column_major': 37, 'broadcast': 20, 'transpose': 35, 'slice': 42, 'non_contig': 39, 'gather': 38} |

## Top 3 minimal repros

_No divergences exceeded gpucheck's per-dtype tolerance (matmul-class: atol scaled by sqrt(k_dim/128); MPS overlay applies a 2× multiplier — see assertions/tolerances.py:35,103)._
## Method notes

- Reference: `torch.nn.functional.cosine_similarity` on CPU (FP32 promotion in error computation).
- Reduction dim: last dim of the sampled shape (k_dim = shape[-1]).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')` — matmul-class scaling enabled, MPS 2× multiplier applied.
- A divergence is flagged only when **both** max-abs-err > atol *and* max-rel-err > rtol — keeps the bar high for outputs near zero (cosine of near-orthogonal vectors).
- NaN/inf entries dropped from error stats (cos of a zero vector is nan).
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- CUDA backend is mocked (no NVIDIA GPU present); cross-device MPS-vs-CUDA comparison reported as N/A by spec.
