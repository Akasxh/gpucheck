# torch.nn.functional.rrelu - MPS fuzz report

- **Kernel:** `torch.nn.functional.rrelu`
- **Status:** UNSUPPORTED
- **Iterations attempted:** 1000
- **Iterations completed:** 80  (of which MPS-rrelu actually produced an output: 0)
- **Iterations unsupported (MPS):** 920
- **Iterations unsupported (CPU):** 0
- **Iterations unsupported (build):** 0
- **Divergences FILABLE (>=3 seeds, >10x tol):** 0
- **Divergences RECALIBRATION (>=3 seeds, 1-10x tol):** 0
- **MPS vs CPU max abs err:** 0.000e+00
- **MPS vs CPU max rel err (denom>=1e-06):** 0.000e+00
- **MPS vs CUDA-mock max rel err:** N/A (CUDA mocked - no NVIDIA GPU on host)
- **Recommended filing target:** pytorch/pytorch (MPS op-coverage: implement aten::rrelu_with_noise)
- **Elapsed:** 0.6 s  (wall budget: 540 s)
- **torch:** 2.11.0, host: darwin/arm64 (Apple Silicon)
- **Seeds:** [0, 1, 2, 3, 4], configs/seed: 200
- **rrelu eval slope:** 0.229167 ((lower+upper)/2 in `training=False`)

## Sampling distribution

| dimension | counts |
|---|---|
| shape bucket | {'degenerate': 180, 'non_tile_aligned': 150, 'prime': 215, 'power_of_2_boundary': 175, 'large': 130, 'mixed': 150} |
| dtype | {'float32': 350, 'float16': 330, 'bfloat16': 320} |
| stride category | {'row_major': 125, 'column_major': 165, 'broadcast': 130, 'transpose': 140, 'slice': 170, 'non_contig': 140, 'gather': 130} |

## Top 3 minimal repros

_No numeric repros: `aten::rrelu_with_noise` is not implemented for the MPS backend in torch 2.11.0. Every iteration of every dtype (fp32/fp16/bf16) and every training mode raises `NotImplementedError` before any output is produced. There is therefore no MPS-vs-CPU numeric divergence to report._
## Method notes

- Reference: `torch.nn.functional.rrelu(x, lower=1/8, upper=1/3, training=False)` on CPU. In `training=False` mode rrelu is deterministic: slope = (lower+upper)/2 for negative inputs, identity for non-negative.
- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, device_type='mps')`. rrelu is elementwise -> no sqrt(k/128) matmul scaling.
- Divergence filter (per task spec):
    - `max_rel_err > 10x rtol` counts only if `denom_magnitude >= 1e-6` (mask near-zero refs).
    - `max_abs_err > 10x atol` always counts.
    - **FILABLE** = same (shape, dtype, stride) config exceeds 10x threshold across >=3 of 5 seeds.
    - **RECALIBRATION** = same config in 1x-10x band across >=3 seeds.
    - Below 1x = OK.
- Stride categories: row_major, column_major, broadcast, transpose, slice, non_contig, gather (`gpucheck.fuzzing.strides`).
- CUDA backend mocked (no NVIDIA GPU present); MPS-vs-CUDA comparison reported as N/A by spec.

## Why UNSUPPORTED

`torch.nn.functional.rrelu`, `torch.rrelu`, and `torch.nn.RReLU` all dispatch to `aten::rrelu_with_noise`, which has no MPS implementation in torch 2.11.0:

```
NotImplementedError: The operator 'aten::rrelu_with_noise' is not currently implemented for the MPS device.
```

This is a coverage gap, not a numerical bug. Recommended follow-ups for gpucheck:

1. Add `rrelu` (and `rrelu_with_noise`) to the `[tool.gpucheck.mps.xfail]` registry citing pytorch's MPS-coverage tracking issue (pytorch/pytorch#77764).
2. File or upvote a feature request against pytorch/pytorch for an MPS implementation of `aten::rrelu_with_noise`.
3. Until then, gpucheck users on Apple Silicon should xfail rrelu tests rather than inflate tolerances.