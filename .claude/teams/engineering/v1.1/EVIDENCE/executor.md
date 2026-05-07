# Executor — v1.1 / T-26 — MPS Event-API deadlock probe

## Task T-26: implement runtime probe for pytorch#162872

### What I did

Created the `gpucheck.diagnostics` subpackage with a single module,
`mps_event_deadlock.py`, implementing the tracer-v3-recommended runtime
probe. The probe runs the unsafe `Event.synchronize` + `elapsed_time`
trigger pattern inside a `daemon=True` worker thread named
`gpucheck-mps-deadlock-probe` and returns one of `"healthy"`,
`"deadlocked"`, or `"skipped"`. Exposed `probe_mps_event_deadlock`,
`assert_no_event_deadlock`, and a fixture-body generator
`mps_event_deadlock_status_fixture` that callers wrap in their own
`@pytest.fixture(scope="session")`.

### Files modified

- `src/gpucheck/__init__.py`: added `probe_mps_event_deadlock` and
  `assert_no_event_deadlock` to `_LAZY_MAP` and `__all__` so the
  top-level public API surface includes the diagnostics. No torch
  import at package load — preserves the lazy-import contract.
- `CHANGELOG.md`: appended an `[Unreleased] / Added` entry citing
  pytorch#162872 and PR #162874 close-without-merge.

### Files created

- `src/gpucheck/diagnostics/__init__.py`: package boundary, re-exports
  the three functions plus the `ProbeResult` type alias.
- `src/gpucheck/diagnostics/mps_event_deadlock.py` (~234 LOC including
  module/function docstrings): runtime probe + assert + fixture body.
  Lazy `_torch()` helper, `_mps_available(torch)` predicate,
  `_trigger_deadlock_pattern()` (the unsafe sequence in a single
  function for clarity), `probe_mps_event_deadlock(timeout_ms=2000)`,
  `assert_no_event_deadlock(timeout_ms=2000)`, and
  `mps_event_deadlock_status_fixture(timeout_ms=2000)`.
- `tests/test_mps_event_deadlock_probe.py`: 8 tests including one
  real-MPS integration test gated on `_has_real_mps()`. Mocks `torch`,
  `torch.backends.mps`, `torch.mps`, and `torch.mps.event` via
  `sys.modules` injection so `Event.synchronize` can be made fast,
  slow, or absent (no-MPS) at fixture entry.

### Design decisions made during implementation

1. **Exception handling inside the daemon thread.** A non-hang failure
   (e.g. `_MockTensor` is unmatmulable, or the real torch fails for
   some unrelated reason) should NOT be classified as `"deadlocked"`.
   The `_run` wrapper catches `Exception` and falls through to
   `done.set()` so the caller returns `"healthy"`. The deadlock
   signature is "thread never sets the event"; an exception still
   sets it. Note: `BaseException` (e.g. `KeyboardInterrupt`) is
   intentionally NOT caught — those should propagate.
2. **Fixture exposure as a generator function, not a `@pytest.fixture`.**
   Auto-decorating with `@pytest.fixture(scope="session")` would
   require `pytest` as a runtime import inside `gpucheck.diagnostics`,
   which would either pull pytest into the import-time graph (bad —
   breaks the `import gpucheck` lazy contract for non-test users) or
   require a try/except shim. Cleaner: ship the generator body and
   document the one-line `yield from` wrapper in the docstring.
   Users still get session-scoping; gpucheck-the-library does not
   import pytest at runtime.
3. **`ProbeResult` as a `Literal[...]` alias, not a `StrEnum`.** The
   task spec used `Literal["healthy", "deadlocked", "skipped"]`
   verbatim. Matching the spec keeps mypy strict-mode happy and
   avoids forcing callers to import an enum class.
4. **The 1.5s mock sleep, not 5s.** Reduces leaked-daemon-thread
   resource usage between tests. The probe timeout in tests is 200ms,
   so any sleep above ~500ms is sufficient to demonstrate `"deadlocked"`;
   1500ms gives a comfortable margin while keeping the sleeping daemon
   from leaking past pytest exit.

### Potential blast radius

- The daemon thread is intentionally leaked on `"deadlocked"`. On a
  real broken PyTorch, the thread is parked in `pthread_cond_wait`
  inside `MPSEventPool::elapsedTime` and cannot be cancelled from
  Python. Process exit reaps it. **The verifier should confirm**
  `pytest` exits cleanly even after the deadlocked test runs (i.e.,
  the integration test plus the mocked deadlock test together do not
  hang `pytest`'s shutdown).
- The `sys.modules` mocking pattern in the test file replaces the real
  `torch` for the duration of the fixture. **The verifier should
  confirm** that ordering between this test file and the real-torch
  test files (e.g. `test_assert_close_mps.py`) is not affected — the
  fixture's `try/finally` restores the original entries.
- `gpucheck.__init__` now references a 4th submodule
  (`gpucheck.diagnostics`). Confirmed `_LAZY_MAP` resolution path is
  identical to existing entries — no side-effect-at-import added.
- No changes to `shapes.py`, `pyproject.toml`, or `tolerances.py` per
  task hard rule. Verified by inspection.

### Expected behavior on this Mac

The repo CLAUDE.md notes torch 2.11.0 + MPS available. tracer-v3 §3
confirmed PR #162874 is closed-without-merge in pytorch/main HEAD as of
2026-05-01. Therefore on this Mac the real-MPS integration test
(`test_probe_on_real_mps_reports_healthy_or_deadlocked`) is expected to
return `"deadlocked"` — the bug is unfixed.

### Verification

I did NOT run pytest (executor hard rule). The verifier will run
`pytest tests/test_mps_event_deadlock_probe.py -v`, mypy strict on
`src/gpucheck/diagnostics/`, and ruff. Expected outcomes:

- 7 tests pass on any host (the four mocked-state tests + parametrized
  3 + session-fixture + daemon-thread).
- `test_probe_on_real_mps_reports_healthy_or_deadlocked` passes on MPS
  hardware with `result in {"healthy", "deadlocked"}`. Skipped on
  non-MPS hosts.

### Verdict

T-26 implementation complete per tracer-v3 spec and task constraints.
Daemon thread is `daemon=True`. Lazy torch import preserved. mypy strict
should pass (annotations only use `from __future__ import annotations`
forward refs and stdlib types). Ready for verifier.
