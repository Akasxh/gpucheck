# SYNTHESIS — gpucheck v1.0 MPS Backend (final 3-round consolidation)

**Slug:** v1.0
**Owner:** research-lead → research-synthesist (consolidation across 3 rounds)
**Binding for:** engineering-lead's CHARTER.md for v1.1 (and any v1.0.0rc2 patch release)

This document consolidates **three rounds of research**:

| round | scope | output | citations |
|---|---|---|---|
| R1 | initial 8 sub-questions × 17 specialists | `SYNTHESIS_v1.md` | 44 distinct primary |
| R2 | novelty injection from skeptic+adversary attack vectors × 8 specialists | `SYNTHESIS_v2.md` | ~85 net-new |
| R3 | depth on lowest-confidence claims × 6 specialists | this consolidation | actionable artifacts |
| **total** | **~129 distinct primary citations**, well exceeding the ≥30 floor |

---

## Headline finding (final)

**gpucheck v1.0.0rc1 ships now.** v1.1 (planned, ~2026-Q3) lands six concrete improvements that R3 specialists drafted but did not commit:

1. **Per-(kernel, dtype) tolerance overlay** replaces the global 2× MPS multiplier. Driven by R2 empiricist's M5 measurement (`EVIDENCE/empiricist-v2.md`) refuting 2× for 5 of 12 (kernel, dtype) pairs, and R3 empiricist's extension to 8 more kernels (`EVIDENCE/empiricist-v3-extended.md`). Concrete overlay table:

   | kernel × dtype | required multiplier | source |
   |---|---|---|
   | matmul / fp32 | 16× | empiricist-v2 P99 |
   | matmul / fp16 | 20× | empiricist-v2 P99 |
   | matmul / bf16 | 32× | empiricist-v2 P99 |
   | conv2d / fp16 | 5× | empiricist-v2 P99 |
   | conv2d / bf16 | 8× | empiricist-v2 P99 |
   | attention / all | 2× | empiricist-v2 holds |
   | layernorm / all | 2× | empiricist-v2 holds |
   | other 8 kernels (R3) | per `empiricist-v3-extended.md` table | R3 |

2. **Apple-tile-aware shape fuzzing.** v1.0's `ShapeStrategy` uses CUDA tiles only (32/64/128). R3 cartographer (`EVIDENCE/cartographer-v3-fuzzer-patch.md`) provides a +91/-8 line unified diff to `src/gpucheck/fuzzing/shapes.py` that adds `TILE_SIZES_MPS = (8, 16, 32, 64, 128)` and `POWER_OF_2_BOUNDARIES_MPS = (7, 8, 9, 15, 16, 17, …, 79, 80, 81, …)` keyed on a new `device_type` parameter. CUDA path bit-for-bit preserved. 3 property tests pin the new boundary coverage.

3. **xfail registry expansion 12 → 43 entries.** R3 github-miner (`EVIDENCE/github-miner-v3-xfail-config.md`) provides ready-to-paste TOML for `[tool.gpucheck.mps.xfail].ops` plus a Python `xfail_metadata` table mapping each entry to its PyTorch issue URL + dtype + shape pattern. 31 new entries include 5 OOB-indexing-silently-returns-zero issues, 4 unsigned-dtype garbage cases, 5 non-contiguous failure-class issues (4 closed-but-stale landmines flagged separately).

4. **Silent-downcast catcher (opt-out, raise-by-default).** R3 linguist (`EVIDENCE/linguist-v3-downcast.md`) designed `Backend.silently_downcasts_dtype(dtype, ndim) -> bool` (rank-aware: fp64 *tensor* raises loudly, fp64 *0-d scalar* silently degrades to fp32 per OperationUtils.mm:120-158). Hooks into `assert_close` before the GPU fast-path with `MPSDtypeWarning` / raise. Tri-state `mps_strict_dtype` kw + `[tool.gpucheck.mps.strict_dtype]` config. **Recommended OPT-OUT** (raise-by-default).

5. **Runtime deadlock probe (ship in v1.0).** R3 tracer (`EVIDENCE/tracer-v3-deadlock.md`) — pytorch#162872's Event API deadlock is **STILL NOT FIXED in HEAD** (R2 archaeologist; R3 tracer re-verified). 3-line probe via `threading.Event` + daemon thread + 2s timeout. Recommended public API: `gpucheck.diagnostics.mps_event_deadlock.{probe_mps_event_deadlock(timeout_ms=2000), assert_no_event_deadlock(timeout_ms=2000)}` + session-scoped pytest fixture. Cost ~150 LOC. Verdict: SHIP IN v1.0.

6. **Cross-version compat finding.** R3 archaeologist (`EVIDENCE/archaeologist-v3-crossver.md`) triaged 4 mixed-precision test failures observed on torch 2.10 (vs pass on 2.11). **Recommendation: do NOT pin `torch>=2.11`.** Cost of a bad pin: stranding 2.10 wheels (Linux arm64, x86_64 CUDA, macOS universal2). The 2.10 vs 2.11 difference is in test setup, not gpucheck behavior — fix the tests, keep the floor at 2.6 (Linux) / 2.10 (macOS arm64 wheel availability).

---

## Status of the 8 original sub-questions (final)

| # | sub-question | R1 | R2 | R3 | confidence |
|---|---|---|---|---|---|
| 1 | torch.mps API surface | mapped | re-mapped + tile sizes from MLX | n/a | HIGH |
| 2 | open MPS bug landscape | 12-bug top list | 31-bug long tail (12→43) | TOML config ready | HIGH |
| 3 | op-coverage gaps | enumerated | 3 of 21 deterministic gates found | silent-downcast API | HIGH |
| 4 | determinism contract | "docs silent, treat as none" | strengthened: MSL §6.15.4 atomic_relaxed by spec | api: detect-don't-promise | HIGH |
| 5 | tolerance model | "2× starting" PROVISIONAL | REFUTED: per-(kernel, dtype) needed | overlay dictionary written | HIGH |
| 6 | llama.cpp / ggml precedent | mapped | n/a | n/a | HIGH |
| 7 | Triton/FlashAttention on Metal | partial port status | n/a | n/a | MEDIUM |
| 8 | event-timing quirks | deadlock pytorch#162872 | confirmed UNFIXED in HEAD | runtime probe code | HIGH |

---

## Engineering hand-off (v1.1 CHARTER)

`engineering-lead` reads this file as the binding v1.1 spec. The CHARTER for the v1.1 implementation should:

1. Adopt the per-(kernel, dtype) overlay table from §1 above.
2. Apply the unified diff in `cartographer-v3-fuzzer-patch.md` to `fuzzing/shapes.py`.
3. Paste the TOML block from `github-miner-v3-xfail-config.md` into pyproject `[tool.gpucheck.mps.xfail]`.
4. Build the silent-downcast hook per `linguist-v3-downcast.md` (new module).
5. Build the deadlock probe per `tracer-v3-deadlock.md` (new module). **In v1.0, not v1.1.**
6. Reconcile cross-version test failures per `archaeologist-v3-crossver.md` (no pin change).

---

## Citations summary

- R1: 44 distinct primary
- R2: ~85 net-new (after dedup)
- R3: actionable artifacts (patch diffs, TOML configs, code sketches)
- **Total: ~129 distinct primary citations**

Citation classes: PyTorch GitHub issues (75+), PyTorch source at `aten/src/ATen/{mps,native/mps}/` (15+), MLX source at `mlx/backend/metal/` (10+), Apple Metal Shading Language Specification (verified in v2 librarian, was REPORTED-NOT-VERIFIED in v1), WWDC 2024+2025 sessions (4), arxiv papers on adversarial DNN fuzzing (17), llama.cpp + ggml (6).

---

## File pointers

- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/QUESTION.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/HYPOTHESES.md`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS_v1.md` (R1)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS_v2.md` (R2 consolidation)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/SYNTHESIS.md` (this file — final)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/*-v2.md` (8 files)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/EVIDENCE/*-v3*.md` (6 files)
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/drift_histogram.json`
- `/Users/cero/Code/gpucheck/.claude/teams/research/v1.0/TURN_LOG.md`
