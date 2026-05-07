# torch.tanh - MPS fuzz report (v2)

- **Kernel:** `torch.tanh`
- **Status:** OK
- **iters_attempted:** 1000
- **iters_completed:** 1000
- **iters_unsupported:** 0
- **divergences_filable:** 0
- **divergences_recalibration:** 0
- **max_abs_err (global):** 1.788e-07
- **max_rel_err (global):** 2.156e-07
- **elapsed:** 1.21 s
- **recommended filing target:** `none`
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon), master_seed: 0xc0ffee, input_seeds: [0, 1, 2, 3, 4]

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 164, 'non_tile_aligned': 178, 'prime': 168, 'power_of_2_boundary': 158, 'large': 159, 'mixed': 173} |
| dtype | {'float32': 343, 'float16': 313, 'bfloat16': 344} |
| stride category | {'row_major': 138, 'column_major': 152, 'broadcast': 149, 'transpose': 150, 'slice': 150, 'non_contig': 135, 'gather': 126} |

## Divergence classification (v2 spec)

- **FILABLE** — ratio > 10x tolerance (abs always; rel only when |y_ref| >= 1e-6) AND reproducible across >= 3 seeds.
- **RECALIBRATION** — 1x < ratio <= 5x tolerance, OR a >10x FILABLE_CANDIDATE that failed the >= 3-seed reproducibility gate.
- **OK** — ratio <= 1x tolerance.

## Top 3 repros

_No divergences exceeded the 1x tolerance threshold (with MPS 2x multiplier from `assertions/tolerances.py:35`)._
## Method notes

- Reference: `torch.tanh` on CPU (FP32 promotion for accuracy comparison).
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')` — applies the PROVISIONAL 2x MPS overlay.
- `tanh` is elementwise with output bounded in [-1, 1]; no `k_dim` scaling applies. Near-zero outputs (`tanh(0) = 0`) make rel-err noisy, so the v2 spec's `denom_magnitude >= 1e-6` guard is essential here.
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (see `gpucheck.fuzzing.strides`).
- bf16 IS exercised — torch.tanh supports bf16 on MPS in torch 2.11.
- CUDA backend is mocked (no NVIDIA GPU present); cross-device MPS-vs-CUDA comparison reported as N/A by spec.
