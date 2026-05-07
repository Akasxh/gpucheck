# Fuzz results — kernel: `softmax` (v2)

**Backends:** MPS (Apple Silicon) vs CPU reference (same dtype). CUDA: not present.

## Summary

- **kernel**: `torch.softmax(x, dim=-1)`
- **iters_attempted**: 1000 (samples × seeds = 200 × 5)
- **iters_completed**: 1000
- **iters_skipped_empty**: 0
- **iters_unsupported**: 0
- **iters_errored**: 0
- **divergences_filable** (≥3-of-5 seeds at ≥10× tol, denom-gated): **0**
- **divergences_recalibration** (1×–<10× tol, or <3-seed flaky): **0**
- samples OK : 200
- samples UNSUPPORTED : 0
- **max_abs_err** (MPS vs CPU, any iter): `2.441e-04`
- **max_rel_err** (MPS vs CPU, any iter): `7.812e-03`
- runtime: `2.6s` (budget 690s; end=completed)
- torch: `2.11.0`, seeds: `[0, 1, 2, 3, 4]`

## Top 3 minimal repros

_None — every (shape, dtype, stride) sample stayed below 1× the MPS-overlay tolerance from `compute_tolerance(..., device_type='mps')`._

## Filing recommendation

**none** — MPS softmax stayed within the v1.0 MPS-overlay tolerance.

## Method notes

- Reference: `torch.softmax(x.cpu(), dim=-1)` at the SAME dtype as MPS.
- Tolerance: `compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')`.
- Per-iter violation: `abs_ratio >= 10` OR (`rel_ratio >= 10` AND `|ref| at worst-rel index >= 1e-6`).
- Sample classification: FILABLE if ≥3-of-5 seeds violate; RECALIBRATION if any seed reaches 1×–<10× tol (or fewer than 3 seeds violate at ≥10×, i.e. value-dependent flakiness); else OK.
- Stride categories sampled: row_major, column_major, broadcast, transpose, slice, non_contig, gather
- Shape buckets: degenerate, prime, power_of_2_boundary, non_tile_aligned, large
