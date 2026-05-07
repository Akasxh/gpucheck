---
specialist: engineering-planner
slug: v1.1
phase: planning
binding_inputs:
  - .claude/teams/audit/v1.1/SUMMARIES/api-dx-grade.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/security-postmerge.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/archaeologist-debt.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/detector-files.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/mutator-survivors.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/tracer-runtime.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/docs-tester.summary.md
  - .claude/teams/audit/v1.1/SUMMARIES/empiricist-mac-bench.summary.md
  - .claude/teams/research/v1.0/SYNTHESIS.md
  - .claude/teams/research/v1.0/EVIDENCE/{empiricist-v3-extended,cartographer-v3-fuzzer-patch,github-miner-v3-xfail-config,tracer-v3-deadlock,linguist-v3-downcast,archaeologist-v3-crossver}.md
release_targets:
  - v1.0.0rc2 (Phase A + B subset, mergeable independently)
  - v1.1 (Phase C + D)
total_tasks: 27
total_estimated_minutes: 525
total_estimated_hours: 8.75
rejected_features:
  - linguist-v3 silent-fp64-downcast catcher (REFUTED on torch 2.11 by tracer-runtime: TypeError fires loudly)
---

# Planner v1.1 — gpucheck atomic task graph

## §0. Scope decisions (load-bearing)

1. **Phase A is the v1.0.0rc2 patch bundle** — every task here is independently mergeable, low blast-radius (≤2), and either fixes a security-LOW or a documented contract violation. Aim: ship rc2 within 1-2h of merging Phase A.
2. **Phase B drives mutation kill-rate from 42.7% → ≥80%** with ~30 LoC of new tests and ~80 LoC of v1.1-feature tests. Independently parallelizable per file.
3. **Phase C deletes duplicated code** with zero new public API. The dependency-graph audit (archaeologist + detector) showed five separate code-clone families; we collapse the three highest-value ones.
4. **Phase D ships R3's binding deliverables**: per-(kernel,dtype) overlay, Apple-tile fuzz patch, xfail TOML expansion, deadlock probe. **The silent-fp64 downcast catcher is dropped** — tracer-runtime §4 verified torch 2.11 raises `TypeError: Cannot convert a MPS Tensor to float64...` so the catcher solves a problem that no longer exists. Cited refutation in T-26 caveats.

## §1. Atomic task table

| ID | Tag | Title | Files | Deps | Blast (1-5) | Min | Rollback | Acceptance |
|----|-----|-------|-------|------|-------------|-----|----------|------------|
| **PHASE A — parallel-friendly bug fixes (≤2h, all v1.0.0rc2-mergeable)** ||||||||
| T-01 | DX | Lazy-fix top-level torch import | `src/gpucheck/assertions/close.py` (lines 13-19) | none | 2 | 25 | `git revert <sha>` (one-file delta) | `python -c "import gpucheck"` does not import torch (verified by `sys.modules` check); `pytest tests/test_assertions.py` green; src: detector-files.summary.md fix #1 |
| T-02 | SECURITY | Add `.contiguous()` before `.numpy()` slow-path | `src/gpucheck/assertions/close.py` (`_to_numpy`) | none | 1 | 10 | one-line revert | New test `test_assert_close_handles_non_contiguous_input` passes on torch <2.1 and ≥2.11; src: security-postmerge.summary.md PM-4 |
| T-03 | SECURITY | Narrow `_load_pyproject_config` exception handler | `src/gpucheck/plugin.py:86-88` | none | 2 | 15 | revert single hunk | Bare `except Exception` replaced with `(OSError, tomllib.TOMLDecodeError, ValueError)`; warns via `warnings.warn` on TOMLDecodeError; new test `test_pyproject_config_warns_on_malformed_toml`; src: security-postmerge.summary.md PM-2 |
| T-04 | SECURITY | Bare-except cleanup in arch/detection + backends/mps | `src/gpucheck/arch/detection.py:157,229`, `src/gpucheck/backends/mps.py:99,137,141,148,191` | none | 2 | 30 | revert 7 hunks | Zero bare `except:` in src/; `ruff check --select=E722 src/` is clean; existing arch and backend tests stay green; src: detector-files.summary.md fix #3 |
| T-05 | DX | Replace `_MutableReport` leak in `memory_guard` public API | `src/gpucheck/sanitizers/memory.py:142,216` (+ `__init__.py`) | none | 2 | 25 | revert; rename one type | `memory_guard()` yields a `MemoryReport` (no underscore); `_MutableReport` either renamed to public or wrapped; mypy strict still green; src: detector-files.summary.md fix #3 (single-file finding) |
| T-06 | DOCS | Fix README broken doc blocks (R-B2 fence, R-B16 numeric, R-B8/12/13 CUDA hardcode) | `README.md` (L66-76, L282-291, §4/§6/§7 examples) | none | 1 | 25 | revert markdown only | `python .claude/teams/audit/v1.1/scripts/test_doc_blocks.py README.md` reports 0 broken (was 6); src: docs-tester.summary.md fix #2 |
| T-07 | DOCS | Fix MIGRATION.md broken signatures (M-B1, M-B7, M-B8, M-B9) | `MIGRATION.md` (4 examples) | none | 1 | 25 | revert markdown only | All 12 MIGRATION blocks pass docs-tester; src: docs-tester.summary.md fix #1 |
| T-08 | DOCS | Fix `pytest tests/gpu_integration/` auto-skip claim (52 hard-fails on MPS) | `README.md` §"Running tests", `CONTRIBUTING.md` | T-09 (optional) | 1 | 15 | revert markdown only | README/CONTRIBUTING document the actual MPS auto-skip behavior; if T-09 lands the doc text matches the new gating; src: docs-tester.summary.md fix #3 |
| T-09 | TEST | Add MPS auto-skip gate to `tests/gpu_integration/conftest.py` | `tests/gpu_integration/conftest.py` (new or amend) | none | 2 | 30 | revert conftest | `pytest tests/gpu_integration/` on MPS-only host returns 0 hard-fails (was 52); auto-skips with reason `"requires CUDA, MPS available but unsupported"`; src: docs-tester.summary.md + archaeologist-debt 22780ae rename |
| T-10 | DX | Add deprecation warning for `baseline_2x: bool` on `assert_close` (introduce `tolerance_scale: float | None`) | `src/gpucheck/assertions/close.py` (signature) | T-01 | 2 | 30 | revert kwarg add | `assert_close(..., baseline_2x=True)` emits `DeprecationWarning` with replacement guidance; `tolerance_scale=2.0` is equivalent; backward-compat test for `baseline_2x` still green; src: api-dx-grade.summary.md "Deprecation candidate" |
| **PHASE B — parallel-friendly test additions (≤2h, drives mutation 42.7%→≥80%)** ||||||||
| T-11 | TEST | Pin numeric fields in `format_mismatch_report` (kills ~30 reporting.py mutants) | `tests/test_assertions.py` or new `tests/test_reporting.py` | T-02 (optional) | 1 | 30 | drop test file | New test `test_report_contains_correct_max_error_value` asserts max-err numeric value, mismatch count, location, histogram presence; mutmut/cosmic-ray reports kill rate ≥75% on `assertions/reporting.py`; src: mutator-survivors.summary.md top-3 #1 |
| T-12 | TEST | Round-trip + parser tests for tolerances config loader | `tests/test_tolerances.py` (new file or append) | none | 1 | 25 | drop tests | `test_apply_config_tolerances_round_trip` and `test_tolerances_from_config_parses_overrides` cover untested `apply_config_tolerances` and `tolerances_from_config`; ≥17 tolerances.py mutants killed; src: mutator-survivors.summary.md top-3 #2 |
| T-13 | TEST | Parametrize over `_DEFAULT_TOLERANCES.items()` with hard-coded expected pairs | `tests/test_tolerances.py` | T-12 | 1 | 15 | drop test | Each (dtype, atol, rtol) triple hard-coded as expected; replaces tautological iteration; ≥12 mutants killed (dict-value tier); src: mutator-survivors.summary.md top-3 #3 |
| T-14 | TEST | Replace substring-only `pytest.raises(match="NaN")` with anchored regex | `tests/test_assertions.py` and `tests/test_arch_compatibility.py` (3 sites) | none | 1 | 15 | revert match strings | `pytest.raises(match=r"^NaN")` or `match=r"\bNaN\b"`; injecting `XXNaNXX` in test makes the test fail (was passing); src: mutator-survivors.summary.md TEST_BUG cluster (~30 mutants) |
| T-15 | TEST | Tests for v1.1 features — assertions on `device_type="mps"` fuzzer (depends T-23) | `tests/test_fuzzing.py` (append 3 property tests from cartographer-v3-fuzzer-patch §3) | T-23 | 1 | drop tests | 20 | The 3 property tests from cartographer-v3-fuzzer-patch.md §3 are integrated and pass; src: cartographer-v3-fuzzer-patch.md §3 |
| T-16 | TEST | Tests for v1.1 features — `assert_no_event_deadlock` healthy/skipped paths (depends T-26) | `tests/test_diagnostics_mps_event_deadlock.py` (new) | T-26 | 1 | drop tests | 20 | On non-MPS host: `probe_mps_event_deadlock()` returns `"skipped"`. On MPS host healthy build: returns `"healthy"` in <500ms. Daemon-thread leak is bounded to one. src: tracer-v3-deadlock.md sketches 1+2 |
| T-17 | TEST | Tests for v1.1 features — xfail registry parser exercises 41 entries (depends T-25) | `tests/test_mps_xfail_config.py` | T-25 | 1 | drop tests | 15 | All 41 entries from the new TOML are returned by `mps_xfail_from_config()`; `is_mps_xfailed("scaled_dot_product_attention.large")` returns True; metadata sidecar imports cleanly; src: github-miner-v3-xfail-config.md Section A+B |
| **PHASE C — architectural refactors (2-4h, v1.1 release)** ||||||||
| T-18 | REFACTOR | Unify the 2 colliding `compute_tolerance` functions | `src/gpucheck/arch/tensor_cores.py:96` (rename), `src/gpucheck/assertions/tolerances.py:70` (canonical), call sites | T-04 | 4 | 60 | restore old name; tag tensor_cores variant `compute_tensor_core_tolerance` | Only one symbol named `compute_tolerance` is exported from `gpucheck.assertions`; the tensor-core variant is renamed `compute_tensor_core_tolerance` and re-exported; full `pytest -q` green; mypy strict green; src: detector-files.summary.md fix #2 |
| T-19 | REFACTOR | Eliminate `_run_mps` duplicate of `MPSBackend.event_timer` | `src/gpucheck/fixtures/benchmark.py:283-327`, `src/gpucheck/backends/mps.py` | T-04 | 4 | 75 | restore deleted `_run_mps` | `fixtures/benchmark.py` calls `MPSBackend.event_timer` via the Backend Protocol; the `_FLUSH_L2_WARNED` gate is honored (warning fires once, not per-call); existing benchmark tests stay green; tracer-runtime trace 2 reproduces (same hot-step distribution); src: tracer-runtime.summary.md finding #1 + #2 |
| T-20 | REFACTOR | Deduplicate GPU detection (3 implementations → 1) | `src/gpucheck/fixtures/gpu.py:43,90`, `src/gpucheck/arch/detection.py:133,206`, `src/gpucheck/plugin.py:10-22` | T-01 | 4 | 60 | restore three shims | A single `gpucheck.arch.detection.gpu_available()` is the source of truth; `fixtures/gpu.py` and `plugin.py` re-export or delegate; no behavioral change in pytest collection; src: detector-files.summary.md fix #1 |
| T-21 | REFACTOR | Fix `@require_arch` naming inconsistency (rename to `@requires_arch`, deprecate old) | `src/gpucheck/arch/__init__.py`, `src/gpucheck/arch/detection.py`, public re-exports | none | 3 | 45 | revert rename; restore `@require_arch` only | `@requires_arch` is the canonical decorator; `@require_arch` re-exported with `DeprecationWarning`; consistent with `requires_determinism`; typo'd arch strings raise instead of silent skip (per api-dx-grade fix #1 strict-mode); src: api-dx-grade.summary.md weak API #2 |
| T-22 | DX | Promote 7 hidden symbols to top-level `_LAZY_MAP` | `src/gpucheck/__init__.py` | none | 3 | 25 | revert init delta | `from gpucheck import memory_tracker, gpu_device, fuzz_strides, ShapeStrategy, StrideStrategy, requires_determinism, assert_deterministic` all succeed; lazy-import contract preserved (no torch/pynvml imported at gpucheck import time); src: api-dx-grade.summary.md fix #2 |
| **PHASE D — v1.1 features from R3 (2-4h)** ||||||||
| T-23 | FEATURE | Apple-tile-aware shape fuzzing (apply cartographer-v3 patch) | `src/gpucheck/fuzzing/shapes.py` (+91/-8 unified diff) | none | 3 | 60 | `git apply -R <patch>` | `fuzz_shapes(device_type="mps")` returns shapes including {7,8,9,15,16,17,79,80,81} boundaries; `fuzz_shapes(device_type="cuda")` (default) is bit-for-bit identical to v1.0; the 3 new property tests from T-15 pass; src: cartographer-v3-fuzzer-patch.md §2+§3 |
| T-24 | FEATURE | Per-(kernel,dtype) MPS tolerance overlay (Shape B class-bucketed) | `src/gpucheck/assertions/tolerances.py`, `pyproject.toml` `[tool.gpucheck.mps.tolerances]` | T-12, T-18 | 4 | 90 | restore single-multiplier 2× | `MPS_KERNEL_CLASS` dict + `MPS_TOLERANCE_MULTIPLIERS` table per empiricist-v3 §"Shape B"; `compute_tolerance(..., kernel="matmul", device_type="mps")` returns the 20×/25×/40× scale for fp32/fp16/bf16; backward compat: 2× preserved for any kernel not in `MPS_KERNEL_CLASS`; new tests in T-12 cover the parser; src: SYNTHESIS.md §1 + empiricist-v3-extended.md §"Shape B" |
| T-25 | FEATURE | Expand MPS xfail registry from 12→41 entries | `pyproject.toml` `[tool.gpucheck.mps.xfail].ops`, `src/gpucheck/assertions/mps_xfail_metadata.py` (new) | T-21 (optional) | 3 | 45 | revert TOML + delete metadata file | TOML block matches Section A of github-miner-v3-xfail-config.md byte-identical; metadata sidecar `MPS_XFAIL_METADATA` lists 41 (op_name, url, date, dtype, shape) tuples; T-17 covers parser; src: github-miner-v3-xfail-config.md §A+§B |
| T-26 | FEATURE | Deadlock probe + session fixture (`gpucheck.diagnostics.mps_event_deadlock`) | `src/gpucheck/diagnostics/__init__.py` (new pkg), `src/gpucheck/diagnostics/mps_event_deadlock.py` (new), `src/gpucheck/fixtures/mps_safety.py` (new) | T-04 | 3 | 60 | delete new package | Public API `probe_mps_event_deadlock(timeout_ms=2000) -> Literal["healthy","deadlocked","skipped"]` and `assert_no_event_deadlock(timeout_ms)` exist; T-16 covers tests; module is in `_LAZY_MAP`; src: tracer-v3-deadlock.md sketches 1+2; **EXPLICITLY EXCLUDES the linguist-v3 silent-fp64-downcast catcher (refuted by tracer-runtime §4 on torch 2.11)** |
| T-27 | DOCS | Update CHANGELOG.md + CLAUDE.md "Known Weaknesses" stale list | `CHANGELOG.md` (new entries), `CLAUDE.md` Known Weaknesses section | T-01..T-26 | 1 | revert markdown | 15 | CHANGELOG documents v1.0.0rc2 (T-01..T-10) and v1.1 (T-11..T-26); CLAUDE.md "Known Weaknesses" no longer falsely claims reporting/MPS/strides/thread-safety as gaps (per archaeologist-debt §"CLAUDE.md is STALE"); src: archaeologist-debt.summary.md tail finding |

## §2. Dependency graph (text)

```
PHASE A (independent, all parallel-safe):
  T-01 ───────────────────────────────┐
  T-02 (independent)                  │
  T-03 (independent)                  │
  T-04 ───────────────────────────────┤
  T-05 (independent)                  │
  T-06 (independent)                  │
  T-07 (independent)                  │
  T-09 ───→ T-08                      │
  T-10 ←── T-01                       │

PHASE B (depends on Phase A subset, parallel within Phase B):
  T-11 ←─ T-02 (recommended)
  T-12 ──→ T-13
  T-14 (independent)
  T-15 ←── T-23
  T-16 ←── T-26
  T-17 ←── T-25

PHASE C (depends on Phase A; Phase C tasks parallel within phase):
  T-18 ←── T-04
  T-19 ←── T-04
  T-20 ←── T-01
  T-21 (independent)
  T-22 (independent, but should land after T-20 to not re-touch __init__)

PHASE D (depends on Phase A and selected Phase C):
  T-23 (independent — applies cleanly to v1.0)
  T-24 ←── T-12, T-18
  T-25 ←── T-21 (optional — naming consistency)
  T-26 ←── T-04
  T-27 ←── T-01..T-26 (final docs sweep)
```

## §3. Recommended execution order with parallelism markers

### Wave 1 (Phase A, parallel pool of 8 — target ≤90min wall-clock)
- Pool: T-01, T-02, T-03, T-04, T-05, T-06, T-07, T-09 (run concurrently; each ≤30min)
- Then sequentially: T-08 (after T-09), T-10 (after T-01)
- **Mergeable as v1.0.0rc2 patch release.**

### Wave 2 (Phase B, parallel pool of 4)
- Pool A: T-11 (after T-02), T-12 (independent), T-14 (independent)
- Pool B: T-13 (after T-12)
- Defer T-15/16/17 to after their Phase D parents land.

### Wave 3 (Phase C refactors, parallel pool of 4)
- T-18 (after T-04), T-19 (after T-04), T-20 (after T-01), T-21, T-22 (after T-20)

### Wave 4 (Phase D features, parallel pool of 3)
- T-23 (independent), T-25 (after T-21), T-26 (after T-04)
- T-24 sequentially after T-12 + T-18 (deps)
- Backfill: T-15 (after T-23), T-16 (after T-26), T-17 (after T-25)

### Wave 5 (final docs)
- T-27

### Gantt sketch (wall-clock, assuming 4-way executor parallelism)

```
hour:  0    1    2    3    4    5    6    7    8    9
       |----|----|----|----|----|----|----|----|----|
W1-A:  [T-01][T-04][T-09][T-08]
W1-B:  [T-02][T-05][T-07]    [T-10]
W1-C:  [T-03][T-06]
W1-D:                        ← rc2 ships
W2:         [T-12][T-13][T-14][T-11]
W3:                  [T-18][T-19][T-20][T-21][T-22]
W4:                                 [T-23][T-25][T-26]
                                         [T-24]
W4-tests:                                       [T-15][T-16][T-17]
W5:                                                         [T-27]
```

## §4. Acceptance criteria coverage check

| CHARTER acceptance criterion | Covered by |
|------------------------------|------------|
| v1.0.0rc2 ships within 1-2h of bundle merge | T-01..T-10 (Phase A) |
| Mutation kill rate 42.7% → ≥80% | T-11, T-12, T-13, T-14 |
| Per-(kernel,dtype) MPS overlay (R3 deliverable) | T-24 (data: empiricist-v3) |
| Apple-tile fuzz patch (R3 deliverable) | T-23 + T-15 |
| xfail TOML 12→43 entries (R3 deliverable, realistic 41 after dedup) | T-25 + T-17 |
| Deadlock probe (R3 deliverable) | T-26 + T-16 |
| Eliminate `_run_mps` duplication (tracer finding #1) | T-19 |
| Unify two `compute_tolerance` (detector finding #2) | T-18 |
| Deduplicate GPU detection (detector finding #1) | T-20 |
| Loud failure on typo'd arch strings + naming consistency | T-21 |
| Lazy import contract restored | T-01 |
| Bare-except cleanup (CLAUDE.md violation) | T-04 |
| Top-3 docs broken blocks fixed | T-06, T-07, T-08 |
| Security LOW findings closed | T-02 (PM-4), T-03 (PM-2), T-04 (PM-5 partial) |
| `_MutableReport` no longer leaks public API | T-05 |
| 7 hidden symbols promoted | T-22 |
| `baseline_2x` deprecation path | T-10 |
| CHANGELOG + CLAUDE.md sync | T-27 |

Coverage is complete: every charter criterion maps to ≥1 task.

## §5. Estimated iteration budget

- Task count: **27**
- Soft cap (2 × N): **54** inner executor+verifier iterations
- Hard cap (5 × N): **135** inner iterations
- Total minutes: **525** (~8.75h, within the 6-12h target)
- Per-phase totals: A = 230 min (~3.8h, larger than the 1-2h aspirational), B = 140 min, C = 265 min, D = 270 min, T-27 = 15 min
  - Phase A's 3.8h is **wall-clock parallel-safe in ~90min** with a 4-way pool (each task ≤30min independently).

## §6. High-risk task flags

| ID | Reason for high risk | Mitigation |
|----|---------------------|------------|
| T-19 | Touches both fixtures + backends; tracer-runtime showed `_run_mps` and `MPSBackend.event_timer` have subtle differences (one gates `_FLUSH_L2_WARNED`, the other doesn't) | Land T-04 first; preserve `_FLUSH_L2_WARNED` gate; reuse existing benchmark tests as regression net. |
| T-20 | Three call sites; pytest collection order matters | Behavioral parity test before/after; use `git diff --stat` to confirm only re-export shims changed |
| T-24 | Changes the public meaning of `compute_tolerance` for MPS users | Behind a `kernel:` kwarg defaulting to None; old `compute_tolerance(dtype)` calls still return 2× scale on MPS unchanged |
| T-25 | Adds 29 new xfail entries; mis-spelling could silent-pass | T-17 verifies all 41 entries deserialize; `sort+uniq` check by hand |
| T-26 | Daemon-thread leak per probe call on actual deadlock; needs hardware verification | tracer-v3-deadlock §"Confidence" already flagged this; T-16 only tests healthy + skipped paths in CI |

## §7. Caveats and open questions

1. **Phase A's 3.8h estimate vs the 1-2h aspirational.** Achievable within 1-2h wall-clock only with 4-way executor parallelism. Single-stream execution will take ~3.5-4h. The CHARTER says "1-2h" which I interpret as wall-clock, hence the parallelism plan in §3.

2. **T-15/16/17 ordering.** These are tests for Phase D features; I placed them inside Phase B because that's where the "drive kill rate to 80%" charter lives, but they can only run after the Phase D parent task. The Gantt accounts for this by deferring them into Wave 4.

3. **T-26 explicitly excludes the linguist-v3 silent-fp64-downcast catcher.** Per the user's instruction. The refutation is that tracer-runtime §4 verified torch 2.11 fp64-on-MPS raises `TypeError: Cannot convert a MPS Tensor to float64...` — the catcher would solve a non-existent problem. **Possible reopen path** in v1.2: if a user reports the bug on torch <2.11, re-add the catcher behind a torch-version gate.

4. **archaeologist-v3-crossver** found that the "torch 2.10 mixed-precision regression" in the prior matrix report **could not be reproduced**. The recommendation "do NOT pin `torch>=2.11`" is therefore a **negative recommendation** (don't change anything) — no task needed, but we note this in T-27's CHANGELOG entry to record the finding.

5. **Phase D tolerance overlay (T-24) — Shape A vs Shape B choice.** The task spec is the empiricist's "Shape B" (class-bucketed: gemm/convN/norm_protected), which is simpler to maintain. Shape A (precise per-(kernel,dtype) atol) is equivalent on the v1.1 calibration corpus but harder to extend. If reviewer prefers Shape A, the task is the same files but a different table; switch is a 30-min substitution.

6. **GPU CI gate (archaeologist-debt 22780ae).** The disabled GPU CI gate is a meta-issue (process, not code) that we surface in T-09's MPS auto-skip wiring but **do not re-enable** here — the CHARTER is v1.1 features, and re-enabling would be a separate CI/CD task that needs MPS-CI runners (out of v1.1 scope). Recommend a follow-up T-28 in v1.2 if a GitHub Actions M-runner becomes available.

7. **mutator-survivors §EQUIVALENT (~15 mutants)** are deliberately not killed (they're equivalent — `copy=False↔True`, `ContextVar` name string, `str|None` annotation, panel widths). Including them would be 15 wasted-effort tests that pin implementation details; the 80% target was set with these excluded.

8. **PM-5 (HTML reporter raw `class=`/`style=` interpolation)** — not in the task graph because the audit summary marked it "not exploitable today (whitelisted constants)". Logged as a v1.2 concern. If the v1.1 reviewer disagrees, add T-28 (DX/SEC, 30min, low blast).

## §8. Files this plan touches (alphabetical, for reviewer scan)

```
CHANGELOG.md                                   T-27
CLAUDE.md                                      T-27
CONTRIBUTING.md                                T-08
MIGRATION.md                                   T-07
README.md                                      T-06, T-08
pyproject.toml                                 T-24, T-25
src/gpucheck/__init__.py                       T-05, T-22, T-26
src/gpucheck/arch/__init__.py                  T-21
src/gpucheck/arch/detection.py                 T-04, T-20, T-21
src/gpucheck/arch/tensor_cores.py              T-18
src/gpucheck/assertions/close.py               T-01, T-02, T-10
src/gpucheck/assertions/mps_xfail_metadata.py  T-25 (new)
src/gpucheck/assertions/tolerances.py          T-18, T-24
src/gpucheck/backends/mps.py                   T-04, T-19
src/gpucheck/diagnostics/__init__.py           T-26 (new)
src/gpucheck/diagnostics/mps_event_deadlock.py T-26 (new)
src/gpucheck/fixtures/benchmark.py             T-19
src/gpucheck/fixtures/gpu.py                   T-20
src/gpucheck/fixtures/mps_safety.py            T-26 (new)
src/gpucheck/fuzzing/shapes.py                 T-23
src/gpucheck/plugin.py                         T-03, T-20
src/gpucheck/sanitizers/memory.py              T-05
tests/gpu_integration/conftest.py              T-09
tests/test_assertions.py                       T-11, T-14
tests/test_diagnostics_mps_event_deadlock.py   T-16 (new)
tests/test_fuzzing.py                          T-15
tests/test_mps_xfail_config.py                 T-17 (new)
tests/test_reporting.py                        T-11 (new)
tests/test_tolerances.py                       T-12, T-13 (new or append)
```

29 distinct files; 7 new, 22 modified.

## §9. Confidence

**HIGH** on the decomposition. Every Phase A task is a single-file or single-hunk change traceable to a Wave-1 audit finding with verified line numbers. Phase B tests follow the mutator-survivors leverage analysis directly. Phase C eliminates duplications named explicitly by detector + tracer. Phase D follows R3 SYNTHESIS §"Engineering hand-off" verbatim minus the explicitly-rejected linguist-v3 catcher.

**MEDIUM** on timing estimates. Most tasks have ≤30min budgets; a 25% overrun across the board would push total to ~11h, still within the 6-12h cap. The 1-2h Phase A wall-clock target requires 4-way parallelism that the executor harness must support.
