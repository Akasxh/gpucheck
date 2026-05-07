# torch.nn.functional.gelu — MPS fuzz v2 report

- **kernel:** `torch.nn.functional.gelu`
- **device under test:** `mps` (reference: `cpu_fp32`, CUDA backend: `mocked`)
- **iters_attempted:** 5000
- **iters_completed:** 5000
- **iters_unsupported:** 0
- **iters_empty (numel==0):** 465
- **iters_skipped_dtype (bf16 unsupported):** 0
- **bf16 supported on MPS for gelu:** True
- **n_configs_planned:** 1000 × seeds=[0, 1, 2, 3, 4]
- **n_configs_run (saw ≥1 seed completed):** 907
- **divergences_filable (CRITICAL on ≥3 seeds):** 0
- **divergences_recalibration (RECAL on ≥3 seeds, not filable):** 91
- **max_abs_err:** 7.812e-03
- **max_rel_err:** 2.577e-02
- **elapsed:** 8.47 s (budget 690s, timed_out=False)
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **config_seed:** 0xCAFEFACE, data_seeds: [0, 1, 2, 3, 4]

## Divergence rules (v2)

Per-run status:

- `FILABLE_HIT` — `abs_err > 10·atol` AND `rel_err > 10·rtol AND |y_ref|@argmax_rel ≥ 1e-6`.
- `CRIT_ABS_ONLY` — only the abs criterion fires.
- `CRIT_REL_ONLY` — only the denom-gated rel criterion fires.
- `RECAL` — `abs_err ∈ [atol, 5·atol)` OR (`rel_err ∈ [rtol, 5·rtol)` AND `|y_ref|@argmax_rel ≥ 1e-6`).
- `OK` — below 1× tolerance.

Per-config verdict (across the 5 data seeds):

- **FILABLE** — `FILABLE_HIT` on ≥ 3 of 5 seeds.
- **RECALIBRATION** — `RECAL` (or `CRIT_*_ONLY` ≥ 3 seeds) on ≥ 3 of 5 seeds and not FILABLE. Single-criterion crit hits roll into RECAL because the spec's FILABLE bar requires both checks to fire.

Tolerances from `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (includes the 2× MPS multiplier).
Convention matches `RESULTS_softmax.md` / `RESULTS_rmsnorm.md` (swarm-v2).

## Top 3 repros

### Repro #1 — RECALIBRATION

- **shape:** `(127, 129)` (bucket: `non_tile_aligned`)
- **dtype:** `float32`
- **stride_category:** `transpose`
- **per-seed status:** `{0: 'CRIT_REL_ONLY', 1: 'RECAL', 2: 'RECAL', 3: 'RECAL', 4: 'RECAL'}`
- **filable_hit / crit_abs_only / crit_rel_only / recal / n_runs:** 0 / 0 / 1 / 4 / 5
- **max_abs_err:** 4.768e-07 (atol=2.00e-04, 10×=2.00e-03)
- **max_rel_err:** 8.475e-03 (rtol=2.00e-04, 10×=2.00e-03)
- **max_denom_magnitude (|y_ref|@argmax_rel):** 6.960e-04

### Repro #2 — RECALIBRATION

- **shape:** `(1024, 64)` (bucket: `large`)
- **dtype:** `float32`
- **stride_category:** `gather`
- **per-seed status:** `{0: 'CRIT_REL_ONLY', 1: 'CRIT_REL_ONLY', 2: 'CRIT_REL_ONLY', 3: 'RECAL', 4: 'RECAL'}`
- **filable_hit / crit_abs_only / crit_rel_only / recal / n_runs:** 0 / 0 / 3 / 2 / 5
- **max_abs_err:** 4.768e-07 (atol=2.00e-04, 10×=2.00e-03)
- **max_rel_err:** 8.475e-03 (rtol=2.00e-04, 10×=2.00e-03)
- **max_denom_magnitude (|y_ref|@argmax_rel):** 1.398e-04

### Repro #3 — RECALIBRATION

- **shape:** `(1024, 64)` (bucket: `large`)
- **dtype:** `float32`
- **stride_category:** `row_major`
- **per-seed status:** `{0: 'CRIT_REL_ONLY', 1: 'CRIT_REL_ONLY', 2: 'CRIT_REL_ONLY', 3: 'RECAL', 4: 'RECAL'}`
- **filable_hit / crit_abs_only / crit_rel_only / recal / n_runs:** 0 / 0 / 3 / 2 / 5
- **max_abs_err:** 4.768e-07 (atol=2.00e-04, 10×=2.00e-03)
- **max_rel_err:** 8.475e-03 (rtol=2.00e-04, 10×=2.00e-03)
- **max_denom_magnitude (|y_ref|@argmax_rel):** 1.398e-04

## RECALIBRATION candidates (suggest xfail entries)

| cid | shape | dtype | stride | recal/crit_abs/crit_rel of n_runs | max_abs | max_rel |
|-----|-------|-------|--------|-----------------------------------|---------|---------|
| 7 | `(127, 129)` | `float32` | `transpose` | 4/0/1 of 5 | 4.768e-07 | 8.475e-03 |
| 17 | `(1024, 64)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 62 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 93 | `(256, 255)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 142 | `(1024, 64)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 144 | `(1024, 64)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 145 | `(256, 256)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 155 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 189 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 196 | `(1024, 64)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 218 | `(33, 128, 4)` | `float32` | `transpose` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 227 | `(127, 129)` | `float32` | `row_major` | 4/0/1 of 5 | 4.768e-07 | 8.475e-03 |
| 236 | `(256, 256)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 239 | `(33, 128, 4)` | `float32` | `non_contig` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 281 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 311 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 313 | `(1024, 64)` | `float32` | `non_contig` | 0/0/5 of 5 | 4.768e-07 | 8.475e-03 |
| 317 | `(33, 128, 4)` | `float32` | `non_contig` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 340 | `(1024, 64)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 349 | `(33, 128, 4)` | `float32` | `non_contig` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 367 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 413 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 476 | `(256, 256)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 484 | `(33, 128, 4)` | `float32` | `non_contig` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 513 | `(33, 128, 4)` | `float32` | `transpose` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 530 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 555 | `(128, 129)` | `float32` | `row_major` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 557 | `(1024, 64)` | `float32` | `slice` | 1/0/4 of 5 | 4.768e-07 | 8.475e-03 |
| 570 | `(256, 256)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 612 | `(1024, 64)` | `float32` | `slice` | 1/0/4 of 5 | 4.768e-07 | 8.475e-03 |
| 664 | `(128, 129)` | `float32` | `transpose` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 698 | `(1024, 64)` | `float32` | `column_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 712 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 714 | `(33, 128, 4)` | `float32` | `transpose` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 716 | `(256, 256)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 718 | `(1024, 64)` | `float32` | `non_contig` | 0/0/5 of 5 | 4.768e-07 | 8.475e-03 |
| 722 | `(1024, 64)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 740 | `(33, 128, 4)` | `float32` | `column_major` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 748 | `(256, 256)` | `float32` | `column_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 758 | `(1024, 64)` | `float32` | `transpose` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 774 | `(1024, 64)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 777 | `(256, 256)` | `float32` | `row_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 831 | `(33, 128, 4)` | `float32` | `column_major` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 843 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 844 | `(127, 129)` | `float32` | `transpose` | 4/0/1 of 5 | 4.768e-07 | 8.475e-03 |
| 873 | `(256, 256)` | `float32` | `gather` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 893 | `(1024, 64)` | `float32` | `column_major` | 2/0/3 of 5 | 4.768e-07 | 8.475e-03 |
| 916 | `(1024, 64)` | `float32` | `slice` | 1/0/4 of 5 | 4.768e-07 | 8.475e-03 |
| 996 | `(33, 128, 4)` | `float32` | `non_contig` | 3/0/2 of 5 | 4.768e-07 | 8.475e-03 |
| 223 | `(127, 129)` | `float32` | `slice` | 3/0/1 of 5 | 4.768e-07 | 6.993e-03 |

_… 41 more recalibration entries omitted._

## Method notes

- Reference: `torch.nn.functional.gelu` on CPU, comparison promoted to fp32.
- MPS execution synced via `MPSBackend`-style `torch.mps.synchronize()` (see `src/gpucheck/backends/mps.py`).
- bf16 probed once at startup; if unsupported, those configs are skipped (counted under iters_skipped_dtype).
- CUDA cross-device comparison is N/A (no NVIDIA GPU on host).
- Stride categories from `gpucheck.fuzzing.strides.CATEGORIES`: ['row_major', 'column_major', 'broadcast', 'transpose', 'slice', 'non_contig', 'gather'].
