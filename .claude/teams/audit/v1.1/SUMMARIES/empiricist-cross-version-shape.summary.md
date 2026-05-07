# empiricist-cross-version-shape summary

**Method:** subprocess-isolated, 10 fresh-process replicates per cell, "slow" if median > 2.0 ms on 1024³ fp32.

## Bug status per torch version

| Version | Slow / 10 | Cold/Warm | Verdict |
|---|---|---|---|
| **2.10.0** | 4 / 10 (40%) | 2.89× | bug present |
| **2.11.0** | 4 / 10 (40%) | 2.84× | bug present |
| **nightly 2.13.0.dev20260507** | 4 / 10 (40%) | 2.84× | **bug present (NOT fixed in nightly)** |

**Bug is bimodal at process level (~40% per process)**, statistically indistinguishable across 3 versions. Original investigation's "stuck on slow" was sample-size-of-1 artifact — skeptic was right.

## Top-5 affected (shape, dtype) cells

1. square `1024×1024×1024 fp32` — 2.91×
2. square `1792×1792×1792 bf16` — 2.76×
3. fixed-nk `4096×1024×1024 bf16` — 2.67×
4. fixed-nk `4096×1024×1024 fp16` — 2.66×
5. fixed-nk `768×1024×1024 fp32` — 2.57×

## Underlying pattern

- **20 of 63 cells** exceed 2× ratio
- Spans **ALL 3 dtypes** (fp16/bf16 dominate top-20: 16/20 cells)
- Spans **both axes** (square + fixed-NK)
- Danger zone: **M ∈ [768, 3072]**
- Clean: small (256-512) and giant (4096³)

**Original "1024³ fp32 uniquely sticky" claim is REFUTED.**

Best model: MPSGraph picks one of **≥3 kernel classes (slow/medium/fast) probabilistically per fresh process**.

## Workaround validation

"Touch any other shape" works ONLY for 1024³ fp32:
- 7/9 pokes unblock
- 2/9 (1023³ fp32, 777×1111×999 fp32) do NOT unblock

For 1792³ bf16 + 4096×1024×1024 bf16, most pokes make timing **WORSE** (e.g., 1792³ bf16 + 1024³ bf16 poke goes 1.47 → 3.99 ms).

**No universal workaround exists.**

## Implications for issue body

- Drop "1024³ fp32 specifically" framing — it's a class
- Drop "touch off-shape" advice — only works for 1 specific cell
- Reframe as: **MPSGraph runtime kernel-selection has stochastic per-process variance, ~40% probability of picking a 2-3× slower kernel for shapes M ∈ [768, 3072] × {fp16, bf16, fp32}**
- Bug present in nightly → maintainer can't claim "already fixed"
- Bimodality → users hit it with 40% probability per cold-start, MOST users will see the slow path at least once
