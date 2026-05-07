# synthesist-bugs-inventory summary (W2)

**59 distinct issues** (ISS-01 → ISS-59).

## Distribution

- HIGH: **17**
- MEDIUM: 22
- LOW: 19
- N/A: 1 (ISS-24 refuted)

## Top-5 fix-first (Quadrant 1: high-impact × easy)

1. **ISS-08** — `assertions/close.py:13-19` top-level `import torch` defeats lazy-import contract (HIGH, TRIVIAL)
2. **ISS-26** — Tolerance config loader has zero tests (HIGH, TRIVIAL — exact code in mutator-survivors)
3. **ISS-29 / 30 / 31** — Three MIGRATION.md broken signatures (HIGH, TRIVIAL — every example fails on copy-paste)
4. **ISS-25** — `reporting.py` has 83/221 surviving mutants = 38% of total (HIGH, EASY)
5. **ISS-09** — Two `compute_tolerance` functions colliding by name (HIGH, EASY)

## Cross-audit contradictions (5)

- **C1** — Silent fp64 downcast on MPS: linguist-v3 hypothesized; tracer-runtime **REFUTED** on torch 2.11. Drop catcher from v1.1 plan.
- **C2** — CLAUDE.md "Known Weaknesses" stale (archaeologist + detector + api-dx-grade converge that several gaps are already fixed)
- **C3** — `flush_l2=True` warn-every-call: complementary not contradictory
- **C4** — Memory leak warning vs failure: scope mismatch with intentional design
- **C5** — README MPS-as-first-class vs CUDA-hard-coded snippets: REAL inconsistency

## Upstream-fileable (2)

- **ISS-56** — MPS matmul 1024³ fp32 4× slower than MLX/CPU AMX (empiricist's repro). File at github.com/pytorch/pytorch.
- **ISS-57** — PyTorch CPU half-precision GEMM gap on Apple Silicon.

## Mac/Metal cluster

**14 issues**. Minimum v1.1 Mac track: ISS-05, 19, 20, 21, 32, 35, 36, 58.
