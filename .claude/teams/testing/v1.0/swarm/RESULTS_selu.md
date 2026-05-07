# torch.nn.functional.selu — MPS fuzz report (v2)

- **kernel:** `torch.nn.functional.selu`
- **status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **divergences_filable:** 0
- **divergences_recalibration:** 14
- **max_abs_err:** 3.125e-02
- **max_rel_err:** 3.358e-02
- **n_distinct_configs_seen:** 185 / 200
- **master_seeds:** [0, 1, 2, 3, 4]
- **filing_rule:** FILABLE = divergence reproduces under >=3 master seeds
- **recommended_filing_target:** `none`
- **elapsed:** 1.15 s (budget 690s)
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), config_seed: 0xcefe5e10

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 165, 'non_tile_aligned': 145, 'prime': 150, 'power_of_2_boundary': 200, 'large': 205, 'mixed': 135} |
| dtype | {'float32': 310, 'float16': 310, 'bfloat16': 380} |
| stride category | {'row_major': 125, 'column_major': 110, 'broadcast': 150, 'transpose': 150, 'slice': 110, 'non_contig': 170, 'gather': 185} |

## Top 3 minimal repros

### Repro #1  (DIVERGENCE_CANDIDATE, signal=rel)

- **shape:** `(128, 129)`  (bucket: `power_of_2_boundary`)
- **dtype:** `float32`
- **stride category:** `transpose`
- **max_abs_err:** 2.384e-07  (atol=2.00e-04, abs_ratio=0.00×)
- **max_rel_err:** 3.358e-02  (rtol=2.00e-04, rel_ratio=167.90×)
- **denom_mag:** 2.940e-06  (threshold for rel-divergence: ≥1e-6)
- **reproducibility:** divergent in 1/5 master seeds, recalibration in 2/5
- **CPU layout:** `shape=(128, 129), strides=(1, 128), contig=False`
- **MPS layout:** `shape=(128, 129), strides=(1, 128), contig=False`
- **example data_seed:** 511371832 (master_seed=4, cidx=53)

### Repro #2  (DIVERGENCE_CANDIDATE, signal=rel)

- **shape:** `(1024, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `row_major`
- **max_abs_err:** 2.384e-07  (atol=2.00e-04, abs_ratio=0.00×)
- **max_rel_err:** 3.034e-02  (rtol=2.00e-04, rel_ratio=151.69×)
- **denom_mag:** 5.085e-06  (threshold for rel-divergence: ≥1e-6)
- **reproducibility:** divergent in 2/5 master seeds, recalibration in 3/5
- **CPU layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **MPS layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **example data_seed:** 511981595 (master_seed=4, cidx=130)

### Repro #3  (DIVERGENCE_CANDIDATE, signal=rel)

- **shape:** `(256, 255)`  (bucket: `power_of_2_boundary`)
- **dtype:** `float32`
- **stride category:** `non_contig`
- **max_abs_err:** 2.384e-07  (atol=2.00e-04, abs_ratio=0.00×)
- **max_rel_err:** 1.597e-02  (rtol=2.00e-04, rel_ratio=79.85×)
- **denom_mag:** 2.063e-06  (threshold for rel-divergence: ≥1e-6)
- **reproducibility:** divergent in 2/5 master seeds, recalibration in 2/5
- **CPU layout:** `shape=(256, 255), strides=(2, 512), contig=False`
- **MPS layout:** `shape=(256, 255), strides=(2, 512), contig=False`
- **example data_seed:** 511298352 (master_seed=3, cidx=170)

## TOLERANCE_RECALIBRATION configs (1×–10× over tolerance, or <3 seed reproduction)

| cidx | shape | dtype | stride | abs_ratio× | rel_ratio× | denom_mag | div_seeds | recal_seeds |
|------|-------|-------|--------|------------|------------|-----------|-----------|-------------|
| 53 | `(128, 129)` | float32 | transpose | 0.00 | 167.90 | 2.94e-06 | 1/5 | 2/5 |
| 130 | `(1024, 64)` | float32 | row_major | 0.00 | 151.69 | 5.09e-06 | 2/5 | 3/5 |
| 170 | `(256, 255)` | float32 | non_contig | 0.00 | 79.85 | 2.06e-06 | 2/5 | 2/5 |
| 153 | `(256, 256)` | float32 | transpose | 0.00 | 54.17 | 2.75e-06 | 1/5 | 4/5 |
| 164 | `(2048,)` | float32 | column_major | 0.00 | 20.90 | 1.65e-05 | 1/5 | 1/5 |
| 168 | `(1024, 64)` | float32 | gather | 0.00 | 16.67 | 7.94e-06 | 1/5 | 4/5 |
| 131 | `(128, 129)` | float32 | gather | 0.00 | 14.64 | 8.51e-06 | 1/5 | 2/5 |
| 4 | `(256, 256)` | float32 | gather | 0.00 | 13.34 | 5.17e-05 | 1/5 | 3/5 |
| 10 | `(256, 256)` | float32 | column_major | 0.00 | 10.18 | 5.71e-05 | 1/5 | 4/5 |
| 107 | `(33, 128, 4)` | float32 | broadcast | 0.00 | 6.36 | 4.57e-05 | 0/5 | 3/5 |
| 5 | `(33, 128, 4)` | float32 | slice | 0.00 | 6.15 | 9.21e-05 | 0/5 | 4/5 |
| 101 | `(33, 128, 4)` | float32 | transpose | 0.00 | 4.33 | 1.19e-04 | 0/5 | 2/5 |
| 186 | `(127, 16)` | float32 | slice | 0.00 | 2.16 | 1.47e-05 | 0/5 | 1/5 |
| 160 | `(63, 16)` | float32 | gather | 0.00 | 1.18 | 2.81e-04 | 0/5 | 1/5 |

**Recommendation:** for any config in this table that is reproducibly 1×–5× over tolerance, add a curated entry to `[tool.gpucheck.mps.xfail]` rather than further inflating the global MPS multiplier. (See `assertions/tolerances.py:35` for the current 2× overlay.)

## Method notes

- **Reference:** `torch.nn.functional.selu` on CPU (FP32 promotion in error calc).
- **Tolerance:** `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (2× MPS overlay applied).
- **SELU is element-wise**, so no `sqrt(k/128)` matmul scaling.
- **Divergence buckets** (per the v2 spec):
    - `OK` — error ≤ 1× tolerance.
    - `RECALIBRATION` — error 1×–10× tolerance, OR rel-error >10× rtol but denom_mag < 1e-6 (a near-zero artifact, not real numerics).
    - `DIVERGENCE_CANDIDATE` — abs > 10× atol, OR (rel > 10× rtol AND denom_mag ≥ 1e-6).
    - `FILABLE` — a `DIVERGENCE_CANDIDATE` reproduced under ≥3 of the 5 master seeds.
- **Stride categories drawn:** row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- **CUDA backend mocked** (no NVIDIA GPU on host); MPS-vs-CUDA comparison N/A.
- **Reproducibility:** the 200 (shape, dtype, stride) configs are deterministic from `CONFIG_RNG_SEED=0xcefe5e10`. Each config is run once per master_seed in {0, 1, 2, 3, 4} (5 x 200 = 1000 iterations).
