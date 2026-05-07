# Scatter kernel fuzz results

- **kernel**: `torch.Tensor.scatter` and `torch.Tensor.scatter_add` (out-of-place, dim-axis index/src)
- **iterations attempted**: 250
- **iterations completed**: 250
- **iterations skipped (unsupported / degenerate)**: 0
- **iterations errored (input build)**: 0
- **divergences found**: 0
- **MPS-vs-CPU max relative error (fp64 reference)**: 5.054e-04
- **MPS-vs-CUDA-mock max relative error**: N/A (no NVIDIA GPU on this host; CUDA detection mocked, CUDA scatter not executed)
- **wall time**: 1.7s (budget 450s)
- **recommended upstream filing target**: `none`

## Method

- Reference: same op on `cpu` device, same dtype/shape/stride layout, same seed.
- Error metric: `max(|mps - cpu| / max(|cpu|, atol_bar))` after upcast to fp64 on CPU. The `atol_bar` denominator floor matches the torch.allclose semantic `|a-b| <= atol + rtol*|b|`, preventing near-zero reference values from inflating relative error.
- Tolerance bar: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=shape[dim], device_type='mps')` — applies the MPS overlay (2× per `_MPS_TOLERANCE_MULTIPLIERS`) and the matmul-class `sqrt(k/128)` scaling using the scatter axis length as `k`.
- Divergence requires **both** rel > rtol_bar AND abs > atol_bar to fire — no double-counting noise floors.
- Stride categories sampled: row_major, transpose, slice, non_contig, broadcast.
- Dtypes sampled: float32, float16, bfloat16.
- Ops sampled: scatter, scatter_add. `scatter` uses a collision-free identity index along `dim` (so the result is bit-deterministic mod fp precision and CPU/MPS must agree exactly); `scatter_add` uses a random index (collisions are well-defined since addition is commutative; small fp drift from atomic ordering is expected and bounded by the tolerance bar).
- Per-op divergence count: {'scatter': 0, 'scatter_add': 0}.

## Top divergences (minimal repro)

_No divergences observed._

## Notes on CUDA mock

- gpucheck's CUDA detection path (`gpucheck.arch`) was not exercised in this run; we report it as `N/A` rather than fabricate a number.
- A future run with `monkeypatch`-ed `pynvml`/`torch.cuda` can compare MPS-vs-CUDA-reference where a CUDA result has been pre-recorded; without recorded CUDA outputs, a mock cannot produce numerical results so this column is correctly N/A.
