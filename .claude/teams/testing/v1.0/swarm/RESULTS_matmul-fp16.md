# matmul-fp16 fuzz results

- **kernel:** matmul-fp16
- **iterations attempted:** 250
- **iterations completed:** 250
- **OK:** 250
- **DIVERGENCE:** 0
- **UNSUPPORTED:** 0
- **ERROR:** 0
- **SKIP:** 0
- **MPS-vs-CPU max relative error (overall):** 3.958e-01
- **MPS-vs-CUDA-mock max relative error:** N/A (no NVIDIA hardware; CUDA detection is mocked, kernel cannot run)
- **recommended upstream target:** none

## Top 3 minimal repros (by max relative error)

_No divergences observed at gpucheck MPS-overlay tolerances._

## Methodology notes
- Source RNG seeded fp32 on CPU, then cast/transferred so MPS and CPU see identical inputs.
- Reference Y is computed in fp64 (for fp32 ops) or fp32 (for fp16/bf16 ops) on CPU.
- Pass condition: `|Y_mps - Y_cpu| <= atol + rtol*|Y_cpu|` element-wise.
- Tolerance: `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='mps')` — base + sqrt(K/128) scaling + MPS 2× overlay.
- CUDA channel: gpucheck arch detection is mockable (pynvml/torch dual backend), but the kernel itself cannot execute without an NVIDIA device — so the CUDA-vs-MPS comparison is N/A on this host.
- Stride patterns: contiguous, slice (stride-2 over K), transpose (`.t()` view), broadcast (1×K and K×1 expanded).
