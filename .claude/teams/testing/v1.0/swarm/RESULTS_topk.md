# Fuzz results — torch.topk

- **Kernel:** `torch.topk`
- **Backends compared:** MPS (real, Apple Silicon) vs CPU (reference)
- **CUDA backend:** mocked / unavailable on this host (no NVIDIA GPU)
- **Iterations attempted:** 250
- **Iterations completed (OK+DIV):** 250
  - OK: 250
  - Divergences: 0
  - Unsupported (op not on MPS): 0
  - Errors: 0
  - Skipped (empty dim): 0
- **Wall time:** 3.5s (budget 420s)
- **Max relative error MPS-vs-CPU:** 0
- **Max relative error MPS-vs-CUDA-mock:** N/A (no CUDA device on host; detection mocked, kernel not executed)
- **Recommended upstream filing target:** `none`

## Top divergences
_None._
