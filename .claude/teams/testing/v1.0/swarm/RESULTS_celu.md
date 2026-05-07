# torch.nn.functional.celu — MPS fuzz report (v2)

- **Kernel:** `torch.nn.functional.celu`
- **Status:** OK
- **Iters attempted:** 1000
- **Iters completed:** 1000
- **Iters unsupported:** 0
- **Divergences (FILABLE — ≥3 seeds, >10× tol):** 0
- **Divergences (RECALIBRATION — 1-10× tol):** 12
- **max_abs_err (global):** 3.052e-05
- **max_rel_err (global):** 4.918e-02
- **Configs × seeds:** 200 × 5 = 1000 planned
- **Seeds:** [0, 1, 2, 3, 4]
- **Elapsed:** 1.1 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **Filing target:** `none`

## Filtering rules

- `max_abs_err > 10× atol` → DIVERGENCE (always).
- `max_rel_err > 10× rtol` → DIVERGENCE only when `denom_at_rel_max >= 1e-6` (else near-zero artefact).
- `1× ≤ err ≤ 10× tol` → RECALIBRATION (xfail-eligible).
- `< 1× tol` → OK.
- A config is **FILABLE** iff DIVERGENCE on ≥3 of 5 seeds.

## Sampling distribution (completed iters)

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 50, 'non_tile_aligned': 180, 'prime': 200, 'power_of_2_boundary': 160, 'large': 150, 'mixed': 170} |
| dtype | {'float32': 280, 'float16': 280, 'bfloat16': 350} |
| stride category | {'row_major': 100, 'column_major': 110, 'broadcast': 165, 'transpose': 130, 'slice': 145, 'non_contig': 145, 'gather': 115} |

## Top 3 repros

### Repro #1

- **shape:** `(256, 256)` (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `transpose`
- **alpha:** 2.0
- **max_abs_err:** 1.751e-07 (atol=2.00e-04, ratio=0.00×)
- **max_rel_err:** 1.623e-03 (rtol=2.00e-04, ratio=8.11×)
- **denom_at_rel_max:** 4.701e-05
- **recalibration seeds:** [0, 2, 3] (3/5) [no FILABLE — top repro pulled from RECAL bucket]

### Repro #2

- **shape:** `(256, 256)` (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `gather`
- **alpha:** 2.0
- **max_abs_err:** 1.751e-07 (atol=2.00e-04, ratio=0.00×)
- **max_rel_err:** 1.623e-03 (rtol=2.00e-04, ratio=8.11×)
- **denom_at_rel_max:** 4.701e-05
- **recalibration seeds:** [0, 2, 3] (3/5) [no FILABLE — top repro pulled from RECAL bucket]

### Repro #3

- **shape:** `(256, 256)` (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `non_contig`
- **alpha:** 1.0
- **max_abs_err:** 8.754e-08 (atol=2.00e-04, ratio=0.00×)
- **max_rel_err:** 1.872e-03 (rtol=2.00e-04, ratio=9.36×)
- **denom_at_rel_max:** 3.248e-05
- **recalibration seeds:** [0, 1, 2, 3] (4/5) [no FILABLE — top repro pulled from RECAL bucket]

## RECALIBRATION configs (xfail candidates)

| dtype | shape | stride | alpha | abs_err | atol | abs_ratio | seeds_hit |
|---|---|---|---|---|---|---|---|
| float32 | (256, 256) | transpose | 2.0 | 1.75e-07 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (256, 256) | gather | 2.0 | 1.75e-07 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (256, 256) | non_contig | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 4/5 |
| float32 | (33, 128, 4) | non_contig | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (1024, 64) | column_major | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 4/5 |
| float32 | (33, 128, 4) | gather | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (1024, 64) | transpose | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 4/5 |
| float32 | (1024, 64) | gather | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 4/5 |
| float32 | (128, 129) | transpose | 1.0 | 8.75e-08 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (1024, 64) | gather | 0.5 | 4.38e-08 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (1024, 64) | transpose | 0.5 | 4.38e-08 | 2.0e-04 | 0.00× | 3/5 |
| float32 | (256, 256) | slice | 0.5 | 4.28e-08 | 2.0e-04 | 0.00× | 4/5 |

## Method notes

- Reference: `torch.nn.functional.celu` on CPU, FP32-promoted for error metrics.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (carries the v1.0 MPS 2× overlay).
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (`gpucheck.fuzzing.strides`).
- Alpha sweep: 1.0 (default), 0.5, 2.0 — exposes alpha-handling bugs distinct from input-handling.
- CUDA backend mocked (no NVIDIA GPU on host); cross-device MPS-vs-CUDA comparison N/A by spec.
