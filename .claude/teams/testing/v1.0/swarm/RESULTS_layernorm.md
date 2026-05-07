# torch.nn.functional.layer_norm - MPS fuzz v2 report

- **Kernel:** `torch.nn.functional.layer_norm`
- **Status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported (MPS rejected):** 0
- **iters_hard_error:** 0
- **divergences_filable (>=3 seeds, >10x tol):** 1
- **divergences_recalibration (1x..5x tol, unique configs):** 42
- **divergences_unconfirmed (>10x tol, <3 seeds, unique configs):** 36
- **max_abs_err:** 1.562e-02
- **max_rel_err (denom>=1e-6):** 1.000e+00
- **max_rel_err (unfiltered, near-zero artifacts shown):** 7.539e+07
- **elapsed:** 2.06 s (budget 675 s)
- **torch:** 2.11.0 on darwin/arm64 (Apple Silicon)
- **seeds:** [0, 1, 2, 3, 4] x 200 iters = 1000 attempted

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 172, 'non_tile_aligned': 152, 'prime': 180, 'power_of_2_boundary': 159, 'large': 175, 'mixed': 162} |
| dtype | {'float32': 338, 'float16': 330, 'bfloat16': 332} |
| stride category | {'row_major': 134, 'column_major': 129, 'broadcast': 152, 'transpose': 154, 'slice': 155, 'non_contig': 140, 'gather': 136} |

## Top 3 repros (FILABLE first; then top unconfirmed)

### Repro #1

- **shape:** `(2048, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `slice`
- **affine (weight+bias) used:** False
- **max_abs_err:** 7.153e-07  (atol=1.41e-04, ratio=0.01x)
- **max_rel_err:** 2.584e-03  (rtol=2.00e-04, ratio=12.92x, denom=5.978e-06)
- **CPU layout:** `shape=(2048, 64), strides=(256, 2), contig=False`
- **MPS layout:** `shape=(2048, 64), strides=(256, 2), contig=False`
- **tensor seed:** 2038451502
- **seeds reproducing divergence:** [1, 3, 4] (>=3 == FILABLE)

### Repro #2

- **shape:** `(1, 16, 2)`  (bucket: `degenerate`)
- **dtype:** `float32`
- **stride category:** `slice`
- **affine (weight+bias) used:** False
- **max_abs_err:** 4.192e-03  (atol=2.50e-05, ratio=167.68x)
- **max_rel_err:** 2.085e-02  (rtol=2.00e-04, ratio=104.24x, denom=2.011e-01)
- **CPU layout:** `shape=(1, 16, 2), strides=(256, 8, 2), contig=False`
- **MPS layout:** `shape=(1, 16, 2), strides=(256, 8, 2), contig=False`
- **tensor seed:** 969828932
- **seeds reproducing divergence:** [0] (>=3 == FILABLE)

### Repro #3

- **shape:** `(1, 16, 2)`  (bucket: `degenerate`)
- **dtype:** `float32`
- **stride category:** `column_major`
- **affine (weight+bias) used:** True
- **max_abs_err:** 4.599e-04  (atol=2.50e-05, ratio=18.40x)
- **max_rel_err:** 2.228e-03  (rtol=2.00e-04, ratio=11.14x, denom=1.496e-01)
- **CPU layout:** `shape=(1, 16, 2), strides=(2, 2, 1), contig=True`
- **MPS layout:** `shape=(1, 16, 2), strides=(2, 2, 1), contig=True`
- **tensor seed:** 1982945636
- **seeds reproducing divergence:** [0] (>=3 == FILABLE)

## TOLERANCE_RECALIBRATION recommendations (xfail / atol bumps)

Configs whose error landed in (1x .. 5x] tolerance on at least one seed. Recommend an entry in `[tool.gpucheck.mps.xfail]` or a per-dtype atol bump.

| suggested_xfail_id | shape | dtype | stride | abs/atol | rel/rtol |
|---|---|---|---|---|---|
| `layer_norm.float32.non_contig` | `(1, 16, 2)` | `float32` | `non_contig` | 3.86x | 1.49x |
| `layer_norm.float32.row_major` | `(1, 16, 2)` | `float32` | `row_major` | 4.77x | 0.61x |
| `layer_norm.float32.gather` | `(1, 2)` | `float32` | `gather` | 0.61x | 1.79x |
| `layer_norm.float32.column_major` | `(127, 129)` | `float32` | `column_major` | 0.01x | 1.14x |
| `layer_norm.float32.column_major` | `(128, 1024)` | `float32` | `column_major` | 0.00x | 4.07x |
| `layer_norm.float32.gather` | `(128, 1024)` | `float32` | `gather` | 0.00x | 4.87x |
| `layer_norm.float32.transpose` | `(128, 1024)` | `float32` | `transpose` | 0.00x | 1.27x |
| `layer_norm.float32.slice` | `(129, 63)` | `float32` | `slice` | 0.00x | 1.08x |
| `layer_norm.float32.broadcast` | `(13, 31)` | `float32` | `broadcast` | 0.51x | 1.54x |
| `layer_norm.float32.transpose` | `(16, 127)` | `float32` | `transpose` | 0.00x | 1.13x |
| `layer_norm.float32.column_major` | `(16, 2)` | `float32` | `column_major` | 2.88x | 0.36x |
| `layer_norm.float32.slice` | `(16, 2)` | `float32` | `slice` | 3.18x | 0.50x |
| `layer_norm.float32.slice` | `(16, 64)` | `float32` | `slice` | 0.00x | 1.35x |
| `layer_norm.float32.column_major` | `(16, 65)` | `float32` | `column_major` | 0.00x | 2.01x |
| `layer_norm.float32.broadcast` | `(2, 3, 4, 5)` | `float32` | `broadcast` | 1.70x | 7.11x |
| `layer_norm.float32.column_major` | `(2048, 64)` | `float32` | `column_major` | 0.01x | 1.81x |
| `layer_norm.float32.gather` | `(2048, 64)` | `float32` | `gather` | 0.01x | 4.84x |
| `layer_norm.float32.slice` | `(2048, 64)` | `float32` | `slice` | 0.01x | 4.68x |
| `layer_norm.float32.transpose` | `(2048, 64)` | `float32` | `transpose` | 0.01x | 4.00x |
| `layer_norm.float32.column_major` | `(256, 512)` | `float32` | `column_major` | 0.00x | 3.81x |
| `layer_norm.float32.non_contig` | `(256, 512)` | `float32` | `non_contig` | 0.00x | 4.35x |
| `layer_norm.float32.transpose` | `(256, 512)` | `float32` | `transpose` | 0.00x | 1.28x |
| `layer_norm.float32.gather` | `(31, 127)` | `float32` | `gather` | 0.00x | 2.72x |
| `layer_norm.float32.column_major` | `(32, 127)` | `float32` | `column_major` | 0.00x | 1.20x |
| `layer_norm.float32.non_contig` | `(32, 127)` | `float32` | `non_contig` | 0.00x | 1.44x |

## Method notes

- Reference: `torch.nn.functional.layer_norm` on CPU, output cast to float32 for error metrics.
- Backend: `gpucheck.backends.mps.MPSBackend`; `backend.synchronize()` (device-level sync, deadlock-safe per pytorch#162872) called before pulling MPS output back to CPU.
- Tolerance: `compute_tolerance(dtype, k_dim=last_dim, device_type='mps')`; k_dim scales atol per gpucheck's CUTLASS-style sqrt(k/128) error model.
- Divergence filter (v2 spec):
  - `max_abs_err > 10x atol` -> always counts.
  - `max_rel_err > 10x rtol` -> counts only when `|y_cpu_at_argmax_rel| >= 1e-6`; otherwise dropped as a near-zero-denom artifact.
  - **FILABLE** = same `(shape, dtype, stride)` config diverged on >=3 of the 5 seeds (0..4).
- Recalibration bucket: error in (1x .. 5x] tol on at least one seed -> recommend xfail or atol bump rather than a bug filing.
- Stride categories from `gpucheck.fuzzing.strides`: row_major, column_major, broadcast, transpose, slice, non_contig, gather.
- CUDA backend mocked on this host (no NVIDIA GPU); MPS-vs-CUDA comparison N/A by spec.

**Recommended filing target:** `pytorch/pytorch`
