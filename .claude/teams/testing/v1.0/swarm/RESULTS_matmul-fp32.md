# matmul-fp32 — fuzz results

- Kernel: `matmul-fp32` (torch.matmul on MPS vs CPU)
- Iterations attempted: 250
- Iterations completed: 250
- Iterations UNSUPPORTED (op rejected by MPS): 0
- Iterations ERRORED (harness or other): 0
- Divergences (abs_err > atol + rtol * |ref|): 0
- Wall clock: 12.1s

## MPS vs CPU max relative error

- Overall: 0.00330704
- float32: 4.45612e-07
- float16: 0.000474471
- bfloat16: 0.00330704

## MPS vs CUDA-mock

N/A — CUDA detection is mocked on this host (no NVIDIA GPU). We did not run the kernel on CUDA, so a numerical comparison is not available.

## Top 3 minimal repros

None — no divergence exceeded the dtype tolerance.

## Recommended upstream filing target: `none`

