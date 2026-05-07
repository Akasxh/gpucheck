# torch.nn.functional.hardtanh - MPS fuzz report (v2 schema)

- **Kernel:** `torch.nn.functional.hardtanh`
- **Agent:** `kernel-fuzzer-hardtanh`
- **Status:** OK
- **Iterations attempted:** 1000
- **Iterations completed:** 1000
- **Iterations unsupported:** 0
- **Iterations errored:** 0
- **Divergences (FILABLE, >=3 seeds, >10x tol):** 0
- **Divergences (RECALIBRATION, 1-10x tol):** 0
- **Max abs err (global):** 0.000e+00
- **Max rel err (filtered |denom|>=1e-06): 0.000e+00**
- **Max rel err (raw, unfiltered):** 0.000e+00
- **Wall time:** 1.3 s
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **CUDA:** N/A_mocked_no_nvidia_hardware
- **Recommended filing target:** `none`

## Method

- Seeds tested per config: [0, 1, 2, 3, 4]
- Configs planned: 200
- Total iterations: configs * seeds = 1000
- Reference: `torch.nn.functional.hardtanh` on CPU (FP32 promotion).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`.
- **Filable** = max_abs_err > 10x atol OR (max_rel_err > 10x rtol AND |denom| >= 1e-6),
  AND reproduced in >= 3 of 5 seeds.
- **Recalibration** = score in [1x, 10x] tol on at least one seed,
  OR a single-seed >10x outlier (not reproduced).
- Stride categories tested: row_major, column_major, broadcast, transpose,
  slice, non_contig, gather (gpucheck.fuzzing.strides).
- Hardtanh bounds sampled: (-1,1), (-3,3), (0,6), (-2,2), (-0.5,0.5).

## Top 3 minimal repros

_No divergences exceeded gpucheck's per-dtype tolerance (MPS 2x overlay applied)._
## Notes

- CUDA backend is N/A (no NVIDIA GPU on host); only MPS-vs-CPU compared.
- bf16 is supported by hardtanh on MPS (verified across iterations).
- `_filtered_rel_err` excludes positions where |reference| < 1e-6 to avoid
  near-zero-denom artifacts inflating relative error.
