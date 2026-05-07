# gemm-3d Fuzz Results

## Summary

- Kernel: `gemm-3d` (torch.matmul, 3D batched: (B,M,K) @ (B,K,N))
- Backends: MPS (real) vs CPU reference; CUDA mocked (no NVIDIA GPU available)
- Iterations attempted: 250
- Iterations completed: 250
- Unsupported skipped: 0
- Process errors: 0
- Divergences (over MPS-overlay tolerance): 0
- Wall time: 11.9s

## Max relative error (MPS vs CPU)

- overall: 3.611e-01
- float32: 2.057e-03
- float16: 3.611e-01
- bfloat16: 2.752e-02

## Max relative error (MPS vs CUDA)

- N/A (no NVIDIA GPU on this host; CUDA detection was mocked, no kernels run)

## Top 3 minimal repros

_None — every iteration stayed within tolerance._

## Recommended upstream filing target

**none** — no divergences observed beyond the MPS-overlay tolerance from `gpucheck.assertions.tolerances`.
