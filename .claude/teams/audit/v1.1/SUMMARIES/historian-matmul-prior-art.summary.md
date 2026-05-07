# historian-matmul-prior-art summary

24 sources across 8 layers. Most-relevant 8:

## Awni Hannun (Apple MLX lead) admission

[ml-explore/mlx#243](https://github.com/ml-explore/mlx/issues/243):
> "for the speech kwt example this size matmul comes up and we are really slow compared to MPS on it (about 3x I think)"

**First-party Apple-employee acknowledgement** of shape-specific 3× MPS matmul cliffs. Same magnitude as our finding.

## ml-explore/mlx#1828 + #1295 (MEDIUM)

Both: shape-specific performance cliffs in MLX matmul. Pattern: certain (M, N, K, dtype) combinations 3-5× slower than neighbors. Same class, different framework.

## Apple Forums #105534 (~2018, HIGH)

MPSMatrixMultiplication shape-sensitivity, documented 5× cliff for shapes not divisible by 8.

**Our case is NOT the N%8 cliff** (1024 is divisible by 128). So it's a NEW INSTANCE of an OLD CLASS at a stricter cache-key granularity that nobody has previously root-caused.

## Hollemans 2017 ([machinethink.net](https://machinethink.net/blog/mps-matrix-multiplication/), HIGH)

Public docs from 2017 on MPSMatrixMultiplication kernel-pick variability. **Phenomenon has been known for ~9 years** but never deeply triaged.

## ggml-metal source (MEDIUM, architectural)

llama.cpp's ggml-metal **deliberately avoids MPSGraph for matmul** and ships its own hand-written Metal kernels. No comments cite the cliff bug specifically, but the architectural decision suggests they don't trust MPSGraph for inference matmul.

## Apple WWDC docs (MEDIUM)

WWDC 2024 session 10160 + WWDC 2023 #10050 both document `MPSGraphCache` cache-key as `(operation_name, input_shapes, input_dtypes)` — confirms our C++ source-dive's finding via separate channel.

## HN thread (Axiom 2026, MEDIUM-HIGH)

Hacker News discussion confirming `(shape, dtype)` cache key behavior in MPS in 2026.

## What this means for the issue body

- Phenomenon is a **known class** (9-year-old). MPS matmul kernel-pick has shape cliffs.
- Our SPECIFIC case (1024³ fp32, post-warmup-3 sticky) is a **new instance** that hasn't been root-caused publicly.
- We can cite Awni Hannun's admission (MLX#243) as corroborating evidence of the magnitude class.
- Maintainer reading the issue immediately knows this is a real class of bug, not a one-off.
- Filing strengthens IF MTLCaptureManager confirms different pipeline-state IDs (which would distinguish our finding from the generic "warmup not enough" mundane explanation).
