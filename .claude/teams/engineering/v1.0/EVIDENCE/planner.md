# planner.md

**Specialist**: engineering-planner (Phase A)
**Date**: 2026-05-01
**Output mode**: PLAN.md sections per track + dependency graph

## Atomic decomposition

Per-track tasks are listed in `../PLAN.md` §Track-A through §Track-D. Each task has: spec, blast radius, rollback sketch, acceptance criterion.

## Dependency graph

- A and C both touch `assertions/tolerances.py`. They touch DIFFERENT mechanisms:
  - C touches the override-stack data structure (`_tolerance_overrides`).
  - A adds a new function signature (`compute_tolerance(..., device_type=None)`) + a separate `_MPS_TOLERANCE_MULTIPLIERS` dict + a `_mps_xfail_set` registry.
- A and D both touch `pyproject.toml`. Different sections:
  - A adds `[project.optional-dependencies] mps`, `[project.optional-dependencies] apple`, `[tool.gpucheck.mps.xfail]`.
  - D adds nothing to pyproject.toml (its CI mitigations are in `.github/workflows/ci.yml` and `uv.lock`).
- Tracks B and others are file-disjoint.

## Phase 3 merge order recommendation

**C → A → B → D**.

Rationale:
- C is smallest (~30 LOC) and lowest risk; ContextVar conversion is mechanical and the existing API is preserved.
- A is the headline; lands second so docs team can update the README in parallel.
- B is independent and adds test pressure; lands third.
- D is largest test surface; lands last so the test suite is comprehensive at merge.

## Risks

1. **A.4 (gpu_benchmark MPS path)** — cannot deterministically test the deadlock-avoidance in CI; we verify by code review + a single integration smoke test on the dev machine.
2. **D.2 (HTML reporter)** — Playwright E2E is out of scope (no time budget). We verify by parsing the HTML with `html.parser` and checking expected content.
3. **C.2 (thread-safety regression test)** — true thread races are non-deterministic. Test must be carefully ordered (`barrier.wait()` before all threads enter context) to make the bug reliably observable on the unfixed code AND pass on the fixed code.

## Verdict

PASS. PLAN.md is complete.
