# empiricist-mac-benchmarks summary (W1)

**Mac M5 / 32 GB / torch 2.11.0 / mlx 0.31.2 / gpucheck 1.0.0rc1**

## Peak measured

- **fp32 matmul (MPS)**: 3.54 TFLOPs at 4096³ (38.8 ms median, n=10)
- **fp16/bf16 matmul (MPS)**: 14.1 TFLOPs at 4096³
- **MLX fp32 matmul beats MPS at 4096³ — 9.24 TFLOPs (2.61× faster)**; fp16/bf16 at parity

## Speedup range MPS vs CPU

0.07× (256³ fp32 matmul — Apple AMX wins on small shapes) → 3950× (N4×64×128 bf16 conv2d). Typical medium-shape MPS wins: 5-18× on attention/conv2d/gelu fp16/bf16.

**MPS LOSES to CPU on:** matmul 256³, layernorm/softmax 256-seq, conv2d 64x64 (small fp32 cells).

## Quality

Zero NaN/Inf, zero hangs, zero crashes across 114 PyTorch cells through gpucheck's deadlock-safe event_timer.

## Three signal items for v1.1

1. **PyTorch CPU has no half-precision GEMM on Apple Silicon** — fp16/bf16 matmul ≥1024³ takes >1 s/iter. fp16 conv2d throws on CPU. (Affects fallback expectations.)
2. **MPS conv2d N4_64_128_128x128_3x3 has 80-135% CV** — first iter ~6ms, settles to ~1ms. v1.1 docs should recommend **5 warmups for conv2d on MPS** (current default 3).
3. ⭐ **MPS matmul 1024³ fp32 is 4× slower than expected** (3.29 ms, 653 GFLOPs) vs MLX (1.34 ms, 1.6 TFLOPs) and CPU AMX (1.08 ms). fp16/bf16 at same shape are 3× faster than fp32. **Likely an MPSGraph fp32 GEMM dispatch / kernel-pick anomaly. WORTH FILING UPSTREAM.**

## Process notes

- Original lambda-factory pattern timed only input allocation, not kernel — inflated MPS by ~25×. Caught via 4096³ sanity probe; fixed mid-run; final numbers post-fix.
- Files: `EVIDENCE/empiricist-mac-benchmarks.md`, `mac_benchmarks.json` (126 rows: 114 PyTorch + 12 MLX)
