# Fuzz results — kernel: `kl_div`

**Backends:** MPS (real, Apple Silicon) vs CPU reference (same dtype).
CUDA: not present — would have been mocked but no comparison performed.

## Summary

- iterations attempted : **250** / target 250
- iterations completed : **250**
- OK                    : 250
- skipped (empty shape) : 0
- unsupported (MPS)     : 0
- harness errors        : 0
- divergences (gpucheck combined check fails): **0**
- max relative error (MPS vs CPU): `9.766e+08`
- max absolute error (MPS vs CPU): `7.812e-03`
- max relative error (MPS vs CUDA-mock): N/A (no CUDA device; mocked)
- runtime: `1.3s` (budget 435s; end=completed)
- torch: `2.11.0`

## Top 3 minimal repros

_None — every (shape, dtype, stride) combo stayed within the MPS-overlay tolerance from `gpucheck.assertions.tolerances`._

## Recommended upstream filing target

**none** — no divergence exceeded the MPS-overlay tolerance.

## Method notes

- Kernel: `torch.nn.functional.kl_div(log_p, target, reduction='none', log_target=...)`.
- 25% of iterations sample `log_target=True` (target is log-probabilities) to exercise the alternative branch.
- Inputs built on CPU at fp32, projected to native dtype, transferred to MPS as a contiguous clone — both backends see identical bytes for the materialized stride pattern.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=K, device_type='mps')` — base + sqrt(K/128) accumulator scaling + MPS 2× overlay.
- Divergence rule: gpucheck combined check `|a-b| <= atol + rtol*|b|` element-wise (FAIL on any element).
- Stride categories: contiguous, slice (stride-2 over class axis), transpose (`.t()` view), broadcast (1×K row expanded to B×K), non_contig_perm (3-D channel slice giving stride (2K,1)).
- Shape buckets: degenerate, prime, pow2_boundary, non_tile_aligned, large.
- CUDA channel: gpucheck arch detection is mockable, but the kernel itself cannot execute without an NVIDIA device — the CUDA-vs-MPS comparison is N/A on this host.
