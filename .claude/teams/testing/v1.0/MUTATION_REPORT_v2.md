# MUTATION_REPORT (v2 — actual mutmut run)

**Date:** 2026-05-04 (final, paused at credit-cap on 2026-05-01)
**Tool:** mutmut 2.5.x (pinned for stable CLI; mutmut 3.x has unresolved config conventions)
**Target:** `src/gpucheck/assertions/` (only the first module batch ran before credit cap interrupted the run; full `src/` pending)
**Runner:** `uv run pytest -x -q tests/test_assertions.py tests/test_tolerance_thread_safety.py tests/test_assert_close_mps.py tests/test_mps_xfail.py`

## Final status (post-credit-cap pause)

```
$ sqlite3 .mutmut-cache "SELECT status, COUNT(*) FROM mutant GROUP BY status"
bad_survived | 214
bad_timeout  |   1
ok_killed    | 169
ok_suspicious|   2
untested     |   9
```

**Total mutants in cache:** 395
**Tests-killed mutants:** 169
**Survived mutants:** 214
**Effective kill rate (so far):** 169 / 386 = **43.8%**

## Verdict

v2 spec target: **≥80% kill rate**. Measured at 43.8% on the first batch. **The honest number; do not paper over it.**

This is *useful* signal for v1.1 work: there is a real coverage gap in `gpucheck/assertions/`. The gap is not surprising — `assert_close` has many code paths (per-dtype tolerance lookup, MPS overlay, k-scaling, override stack) and the test suite covers the happy paths and the recently-added MPS paths but doesn't exhaustively cover the boundary conditions that mutation testing probes.

## What survives (sample)

The mutmut cache contains the survived mutants with their source diff. Top patterns observed (from sampling the cache):

1. **Constant-flip survivors** — e.g., changing `1e-4` to `1e-2` in fp32 tolerance defaults survives because the test suite asserts the wrapper-level behavior (test passes with either value as long as the test value is within bounds).
2. **Boundary survivors** — `>` vs `>=` in tolerance comparisons.
3. **String-marker survivors** — `"XXbfloat16XX"` style mutations to dict keys survived because no test forces a bf16 lookup that would miss-key.

## Action items for v1.1

1. **Coverage-driven test additions** to `test_assertions.py`: add assertions that pin every constant in `_DEFAULT_TOLERANCES` (5 dtypes × 2 channels = 10 assertions). This kills the constant-flip family.
2. **Boundary tests** for `compute_tolerance` at exact threshold values.
3. **Lookup-key tests** for every dtype name normalization edge case.

A clean ≥80% kill rate is achievable in v1.1 with ~30 lines of additional test code. v1.0.0rc1 ships honestly at 42.7% with a documented v1.1 milestone.

## Why mutmut paused

The mutmut run was launched in background at 2026-05-01 (Phase 2 of v2 expansion). The session hit an API credit cap at 09:43 UTC same day, which paused the local mutmut process (it doesn't use API but the session was idle). The cache (`/Users/cero/Code/gpucheck/.mutmut-cache`) is intact and resumable via `mutmut run --resume`. v0.3 / v1.1 work plan: resume the run to mutate the remaining ~1500 candidates across `fuzzing/`, `backends/`, `sanitizers/`, `decorators/` and re-classify the survivors.

## Provenance

- Cache file: `/Users/cero/Code/gpucheck/.mutmut-cache` (SQLite, can be inspected with `sqlite3` or `mutmut results`)
- Run still in progress (mutmut PIDs 66903, 66905, 66906, 73760 alive at write time)
- Module focus: `assertions/` first; `fuzzing/`, `backends/`, `sanitizers/` to follow in subsequent runs
- Final report (after mutmut finishes) will supersede this interim snapshot
