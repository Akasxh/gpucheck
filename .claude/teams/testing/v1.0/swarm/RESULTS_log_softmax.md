# Fuzz results - kernel: `torch.nn.functional.log_softmax`

**Backends:** MPS (real, Apple Silicon) vs CPU reference (same dtype).
CUDA: not present - mocked / not compared.

## Summary

- kernel: `torch.nn.functional.log_softmax`
- iters_attempted: **1000** / target 1000
- iters_completed: **1000**
- iters_unsupported: 0
- iters_skipped_empty: 0
- divergences_filable: **0**
- divergences_recalibration: **8**
- max_abs_err: `6.250e-02`
- max_rel_err: `2.362e-01`
- runtime: `3.09s` (budget 675s, timed_out=False)
- torch: `2.11.0`, host: darwin/arm64 (Apple Silicon)
- seeds: [0, 1, 2, 3, 4], configs planned: 200

## Divergence-filtering rules (spec)

- **OK** if max_abs_err < 1x atol AND max_rel_err < 1x rtol.
- **TOLERANCE_RECALIBRATION** if 1x <= worst factor <= 5x (recommend xfail entry).
- **FILABLE** only if max_abs_factor > 10x OR (max_rel_factor > 10x AND denom_magnitude >= 1e-6) AND the same (shape, dtype, stride_category) reproduces in >=3 of 5 seeds.

## Top 3 minimal repros

### Repro #1 (RECALIBRATION)

- **shape:** `(1024, 3)`  (bucket: `mixed`)
- **dtype:** `bfloat16`
- **stride category:** `column_major`
- **max abs err:** 3.125e-02  (atol=1.53e-02, factor=2.04x)
- **max rel err:** 1.474e-01  (rtol=1.00e-01, factor=1.47x)
- **denom_magnitude:** 2.319e-02
- **seed reproducibility:** 0/5 seeds = divergence_candidate, 5/5 = recalibration
- **CPU layout:** `shape=(1024, 3), strides=(1, 1024), contig=False`
- **MPS layout:** `shape=(1024, 3), strides=(1, 1024), contig=False`
- **worst-seed:** 0

### Repro #2 (RECALIBRATION)

- **shape:** `(1024, 3)`  (bucket: `mixed`)
- **dtype:** `bfloat16`
- **stride category:** `column_major`
- **max abs err:** 3.125e-02  (atol=1.53e-02, factor=2.04x)
- **max rel err:** 1.474e-01  (rtol=1.00e-01, factor=1.47x)
- **denom_magnitude:** 2.319e-02
- **seed reproducibility:** 0/5 seeds = divergence_candidate, 5/5 = recalibration
- **CPU layout:** `shape=(1024, 3), strides=(1, 1024), contig=False`
- **MPS layout:** `shape=(1024, 3), strides=(1, 1024), contig=False`
- **worst-seed:** 0

### Repro #3 (RECALIBRATION)

- **shape:** `(33, 128, 4)`  (bucket: `mixed`)
- **dtype:** `bfloat16`
- **stride category:** `column_major`
- **max abs err:** 3.125e-02  (atol=1.77e-02, factor=1.77x)
- **max rel err:** 2.362e-01  (rtol=1.00e-01, factor=2.36x)
- **denom_magnitude:** 6.787e-02
- **seed reproducibility:** 0/5 seeds = divergence_candidate, 5/5 = recalibration
- **CPU layout:** `shape=(33, 128, 4), strides=(4, 132, 1), contig=False`
- **MPS layout:** `shape=(33, 128, 4), strides=(4, 132, 1), contig=False`
- **worst-seed:** 0

## Recalibration candidates (xfail recommendations)

| dtype | shape | stride | abs_factor | rel_factor | n_div_seeds | n_recal_seeds |
|---|---|---|---|---|---|---|
| bfloat16 | (1024, 3) | column_major | 2.04 | 1.47 | 0 | 5 |
| bfloat16 | (1024, 3) | column_major | 2.04 | 1.47 | 0 | 5 |
| bfloat16 | (33, 128, 4) | column_major | 1.77 | 2.36 | 0 | 5 |
| float16 | (1024, 3) | column_major | 1.28 | 1.18 | 0 | 4 |
| float16 | (1024, 3) | non_contig | 1.28 | 0.91 | 0 | 3 |
| float16 | (1024, 3) | non_contig | 1.28 | 0.91 | 0 | 3 |
| float16 | (1024, 3) | row_major | 1.28 | 0.63 | 0 | 4 |
| bfloat16 | (2, 5, 7, 11) | slice | 1.07 | 0.15 | 0 | 5 |

## Method notes

- Reference: `torch.nn.functional.log_softmax(x_cpu, dim=-1)` at the SAME dtype as MPS.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=shape[-1], device_type='mps')` (k_dim from the reduction axis).
- Per-seed classification: ok / recalibration / divergence_candidate. Per-config: 5 seeds total, FILABLE iff >=3 seeds are divergence_candidate.
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- log_softmax over the last dim is exercised (well-defined for any shape with shape[-1] >= 1).
