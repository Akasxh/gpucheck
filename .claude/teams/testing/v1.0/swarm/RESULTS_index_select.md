# index_select fuzz — MPS vs CPU

- **kernel**: `torch.index_select`
- **torch**: 2.11.0
- **MPS available**: True
- **CUDA backend**: mocked-only (no NVIDIA GPU on host)
- **iterations attempted**: 250
- **iterations completed (ok+divergent)**: 223
- **iterations unsupported on MPS**: 0
- **divergences**: 0
- **elapsed**: 2.9s
- **status counts**: {'ok': 223, 'skipped_empty': 27}

## MPS-vs-CPU max relative error

<= per-dtype tolerance (no divergences)

## MPS-vs-CUDA-mock max relative error

N/A — no NVIDIA GPU present; CUDA detection mocked but kernel cannot execute.

## Top 3 minimal repros

_None — no divergences observed in this run._

## Recommended upstream filing target

`none`

No divergences exceeding gpucheck's MPS-overlaid per-dtype tolerance. `torch.index_select` on MPS agrees with CPU within tolerance across the tested shape/dtype/stride matrix.
