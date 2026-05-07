# v2 dispatch budget — final measurements

Updated: 2026-05-04 post credit-reset.

| metric | v1 (Phase 0-3) | v2 add | total | target | result |
|---|---|---|---|---|---|
| Agent dispatches | 6 | 16 (R2 8 + R3 6 + 2 depth) | 22 | ≥120 | **18%** of target |
| Headless claude -p | 26 (v1 swarm) | 28 (v2 first 2 waves) | 54 | ≥150 | **36%** of target |
| Research rounds | 1 | +2 (R2, R3 depth) | 3 of 3 | 3 | **100%** ✓ |
| Kernel swarm size | 26 | +28 spawned | 41 unique RESULTS files | 98 | **42%** of target |
| Mutation targets attempted | 0 | 395 (mutmut paused) | 395 | ≥1000 | **40%** of target |
| PyTorch matrix versions | 1 | +1 (2.10) | 2 actually-installable | 5+ aspirational | **macOS arm64 wheel-availability bound** |

## Why we missed several targets

1. **Credit cap at 09:43 UTC on 2026-05-01** (43 min into v2) interrupted: the Round 2 synthesist's response (file did write — 45KB SYNTHESIS_v2.md), 6 R3 specialists' summaries (files did write — 200-760 lines each), 2 depth-expansion agents (files did NOT write), the swarm relauncher (28 of 98 spawned then bash bug + cap blocked the rest), mutmut (395 of ~2000 candidates).

2. **3-day idle gap.** The `Monitor` heartbeat continued local pytest+ruff+mypy every 10 min for ~3 days while the session waited for the credit reset. ~432 heartbeat events emitted, all green (224 passed, ruff clean, mypy clean — no regression detected). Useful aliveness signal, but ~$0 of useful new work happened during that gap.

3. **PyPI wheel availability** for older PyTorch on macOS arm64 is the structural bound on the matrix. Only torch 2.10 + 2.11 ship arm64 wheels; older versions are Linux-only. **Honest scope: matrix max = 2 versions.**

## What still landed (high-value)

- **3 rounds of research, ~129 distinct primary citations.** SYNTHESIS_v1 + v2 + final all on disk; v2 supersedes v1 on the 2× multiplier (REFUTED with M5 measurement) and the xfail list (12 → 43).
- **6 R3 actionable artifacts** ready for v1.1 implementation (per-(kernel,dtype) overlay table, Apple-tile fuzz patch diff, xfail TOML config, silent-downcast catcher API, deadlock probe code, cross-version triage).
- **42.7% mutation kill rate measured** on `assertions/` — real coverage gap data, drives v1.1 test additions.
- **Cross-version finding** torch 2.10 has 4 mixed-precision regressions vs 2.11 — concrete, actionable.
- **gpucheck v1.0.0rc1 ships green**: 224 tests, ruff/mypy clean, dashboard rendered with real MPS benchmarks, both PRs open.
