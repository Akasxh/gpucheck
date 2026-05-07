# swish — MPS fuzz report

- **Kernel:** `torch.nn.functional.silu  # swish == x * sigmoid(x)`
- **Status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **divergences_filable:** 0
- **divergences_recalibration:** 0
- **single-seed-only filable (not reproducible across 3+ seeds):** 0
- **single-seed-only recalibration:** 0
- **max_abs_err (global):** 1.221e-04
- **max_rel_err (global):** 5.200e-04
- **elapsed_seconds:** 1.51
- **seeds:** [0, 1, 2, 3, 4]  (iters_per_seed=200)
- **filable threshold:** > 10.0× tolerance AND (abs OR (rel AND |y_cpu|≥1e-06)), reproduced ≥3 seeds
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **MPS vs CUDA-mock:** N/A (no NVIDIA GPU on host)
- **Recommended filing target:** `none`

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 150, 'non_tile_aligned': 185, 'prime': 145, 'power_of_2_boundary': 210, 'large': 150, 'mixed': 160} |
| dtype | {'float32': 375, 'float16': 245, 'bfloat16': 380} |
| stride category | {'row_major': 150, 'column_major': 185, 'broadcast': 135, 'transpose': 155, 'slice': 115, 'non_contig': 140, 'gather': 120} |

## Top 3 minimal repros

_No configuration exceeded the per-dtype tolerance band on ≥3 seeds. swish (== silu) on MPS appears numerically faithful to CPU within gpucheck's MPS-overlay tolerance._
## Method notes

- swish == silu == x · sigmoid(x) (the v2 swarm spec lists swish and silu separately; this run uses `torch.nn.functional.silu`).
- Reference: `F.silu` on CPU at the input dtype, comparison done in fp32.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` (MPS 2× overlay applied). Elementwise op — no matmul √(k/128) scaling.
- Filable: max_abs > 10×atol, OR (max_rel > 10×rtol AND max|y_cpu| ≥ 1e-06); reproduced across ≥3 of 5 seeds.
- Recalibration: 1×–5× tolerance with same denom guard; suggests an xfail registry entry rather than a torch-issue file.
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (gpucheck.fuzzing.strides).
- CUDA backend mocked (no NVIDIA GPU). MPS-vs-CUDA reported as N/A.
- Configs are deterministic across seeds: index `i` always maps to the same (shape, dtype, stride) tuple, so per-seed runs are directly comparable for reproducibility scoring.
