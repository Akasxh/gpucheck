# Fuzz Report — scaled_dot_product_attention (flash-attn v2 / MPS SDPA)

**Method:** stride/contiguity + shape + dtype fuzzing on MPS, CPU as fp32 reference.
MPS-vs-CUDA-mock: **N/A** (no NVIDIA GPU on host; CUDA detection mocked elsewhere — kernel itself cannot be exercised under mock).

## Run summary

- Kernel: `scaled_dot_product_attention (flash-attn v2 / MPS SDPA)`
- Iterations attempted: **250** (target 250)
- Iterations completed (ok+divergence): **250**
- ok: 250 | divergences: 0 | unsupported: 0 | errors: 0
- Wall time: 23.00s (budget 450s)
- Seed: 0xf1a54
- Torch: 2.11.0, MPS available: True

## MPS-vs-CPU error envelope

- Overall **max relative error**: `0.11004430055618286`
- Overall **max absolute error**: `0.012948155403137207`

Per-dtype max rel err (ok+divergence only):
  - `bfloat16` max-rel-err = `0.11`
  - `float16` max-rel-err = `0.04766`
  - `float32` max-rel-err = `0.001035`

## Divergences

Total: **0**.

_No divergences found within tolerance budget._

## Unsupported / errored configurations

_None._

## Recommended upstream filing target

`none`
