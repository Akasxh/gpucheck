# flash-attn-v1 — gpucheck fuzz report

- Kernel: `flash-attn-v1` (torch.nn.functional.scaled_dot_product_attention)
- Backends: MPS (real, Apple Silicon) vs CPU fp32 math reference
- CUDA backend: MOCKED (no NVIDIA GPU on host) — N/A in this run
- Iterations attempted: **250** / 250
- Iterations completed: **250**
- Unsupported / skipped (op or stride): 0
- Process errors: 0
- Divergences (combined `|a-b| > atol + rtol*|b|`): **0**

## Error stats (safe rel err = |a-b| / max(|a|,|b|,1))
- max safe rel err (MPS vs CPU fp32): `3.8376e-03`
- median safe rel err: `3.4274e-04`
- p95 safe rel err: `3.6229e-03`
- max rel err (MPS vs CUDA mock): N/A (no NVIDIA GPU; CUDA detection mocked)

## Top 3 minimal repros
_None — all completed configurations within gpucheck tolerance._

## Recommended upstream filing target
**none** — no actionable divergence at gpucheck tolerances (per-dtype, k-scaled, MPS overlay applied).

_Generated 2026-05-01 10:16:49 on Apple Silicon (torch=2.11.0, MPS available=True)._
