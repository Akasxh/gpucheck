# torch.nn.functional.elu - MPS fuzz report

- **Kernel:** `torch.nn.functional.elu`
- **Status:** OK
- **Iterations attempted:** 1000
- **Iterations completed:** 1000
- **Iterations unsupported (skipped):** 0
- **Iterations planned:** 1000 (200 configs x 5 seeds)
- **Divergences (FILABLE, repro >=3 seeds):** 0
- **Divergences (TOLERANCE_RECALIBRATION):** 14
- **MPS vs CPU max absolute error:** 1.526e-05
- **MPS vs CPU max relative error (denom >= 1e-6):** 1.639e-02
- **Recommended filing target:** `none`
- **Elapsed:** 1.39 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), seeds: [0, 1, 2, 3, 4], meta-seed: 0xe10

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 120, 'non_tile_aligned': 185, 'prime': 185, 'power_of_2_boundary': 170, 'large': 165, 'mixed': 175} |
| dtype | {'float32': 325, 'float16': 400, 'bfloat16': 275} |
| stride category | {'row_major': 135, 'column_major': 155, 'broadcast': 145, 'transpose': 145, 'slice': 150, 'non_contig': 125, 'gather': 145} |

## Top 3 minimal repros

### Repro #1  -- `NOT_FILABLE`

- **shape:** `(256, 256)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `gather`
- **seeds reproducing DIVERGE:** `[4]`  (>=3 -> filable)
- **max abs err:** 8.754e-08  (atol=2.00e-04, ratio=0.00x)
- **max rel err:** 2.660e-03  (rtol=2.00e-04, ratio=13.30x)
- **denom magnitude at max-rel:** 8.666e-06  (>=1e-6 ? True)
- **CPU layout:** `shape=(256, 256), strides=(256, 1), contig=True`
- **MPS layout:** `shape=(256, 256), strides=(256, 1), contig=True`
- **example seed:** 4

### Repro #2  -- `NOT_FILABLE`

- **shape:** `(1024, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `gather`
- **seeds reproducing DIVERGE:** `[4]`  (>=3 -> filable)
- **max abs err:** 8.754e-08  (atol=2.00e-04, ratio=0.00x)
- **max rel err:** 2.660e-03  (rtol=2.00e-04, ratio=13.30x)
- **denom magnitude at max-rel:** 8.666e-06  (>=1e-6 ? True)
- **CPU layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **MPS layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **example seed:** 4

### Repro #3  -- `NOT_FILABLE`

- **shape:** `(1024, 64)`  (bucket: `large`)
- **dtype:** `float32`
- **stride category:** `row_major`
- **seeds reproducing DIVERGE:** `[4]`  (>=3 -> filable)
- **max abs err:** 8.754e-08  (atol=2.00e-04, ratio=0.00x)
- **max rel err:** 2.660e-03  (rtol=2.00e-04, ratio=13.30x)
- **denom magnitude at max-rel:** 8.666e-06  (>=1e-6 ? True)
- **CPU layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **MPS layout:** `shape=(1024, 64), strides=(64, 1), contig=True`
- **example seed:** 4

## Method notes

- Reference: `torch.nn.functional.elu` on CPU, FP32-promoted for the error metric.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (MPS 2x overlay applied).
- Divergence classification per spec:
    * **OK:** abs_ratio < 1.0 and rel_ratio < 1.0
    * **TOLERANCE_RECALIBRATION:** 1.0 <= ratio <= 10.0 (either)
    * **DIVERGE:** ratio > 10.0 (either); rel-err filtered when denom < 1e-6 (near-zero artifact)
    * **FILABLE:** DIVERGE config reproduced under >=3 distinct seeds from {0,1,2,3,4}
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (`gpucheck.fuzzing.strides`).
- Sampling: 200 (bucket, shape, dtype, stride) configs sampled with meta-seed; each replayed under all 5 seeds = 1000 iterations.
- ELU is elementwise; no k_dim scaling applied.
- CUDA backend is mocked (no NVIDIA GPU on host); cross-device MPS-vs-CUDA comparison N/A.
