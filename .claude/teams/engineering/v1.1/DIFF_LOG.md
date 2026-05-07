# DIFF_LOG — gpucheck v1.1 Phase 2

Schema: `## Iteration <N> — Task <task_id>: <title>`

## Iteration 1 — T-26: MPS Event-API deadlock probe

- **File**: `src/gpucheck/diagnostics/__init__.py` (new)
- **Change**: Package `__init__` re-exports `probe_mps_event_deadlock`,
  `assert_no_event_deadlock`, `mps_event_deadlock_status_fixture`, and the
  `ProbeResult` literal.
- **Reason**: Minimal package boundary so callers can `from
  gpucheck.diagnostics import probe_mps_event_deadlock` without reaching
  through the submodule path.
- **Acceptance criterion addressed**: tracer-v3 §"Recommended public API" —
  ship `probe_mps_event_deadlock()` and `assert_no_event_deadlock()` in
  `gpucheck.diagnostics.mps_event_deadlock` for v1.0 (now v1.1).

## Iteration 2 — T-26: MPS Event-API deadlock probe

- **File**: `src/gpucheck/diagnostics/mps_event_deadlock.py` (new, ~150 LOC)
- **Change**: Implements the runtime probe. Lazy `import torch`; daemon
  thread (`daemon=True`, named `gpucheck-mps-deadlock-probe`) runs the
  unsafe `start.record / end.record / end.synchronize / start.elapsed_time`
  pattern; main thread `done.wait(timeout=timeout_ms / 1000)` returns
  `"healthy"` on completion, `"deadlocked"` on timeout, `"skipped"` if
  torch is missing or MPS unavailable. `assert_no_event_deadlock` wraps
  the probe and raises `RuntimeError` only on `"deadlocked"`.
  `mps_event_deadlock_status_fixture` is a generator that yields the
  probe result once for users to wrap in a `pytest.fixture(scope="session")`.
- **Reason**: Direct implementation of tracer-v3 sketch 1 + sketch 2.
  Daemon thread is intentionally leaked on deadlock per tracer-v3 §"Why a
  daemon thread is safe" (no `pthread_cancel` from Python). Exception
  handling inside `_run` ensures non-hang failures (CUDA-style stub on
  non-MPS hosts, missing op) are NOT misclassified as deadlocks.
- **Acceptance criterion addressed**: tracer-v3 §"Verdict" items 1-3
  (probe + assert + fixture-body); pytorch#162872 still open per task
  prompt; daemon thread mandatory per task hard rule.

## Iteration 3 — T-26: MPS Event-API deadlock probe

- **File**: `tests/test_mps_event_deadlock_probe.py` (new)
- **Change**: 8 tests. Three fixtures install a fake `torch` shim into
  `sys.modules` with controllable `Event.synchronize` delay (50ms healthy,
  1500ms deadlocked, no-MPS). Tests:
  `test_probe_returns_skipped_without_mps`,
  `test_probe_returns_healthy_when_thread_completes_fast`,
  `test_probe_returns_deadlocked_on_timeout`,
  `test_assert_no_event_deadlock_raises_on_deadlocked`
  (parametrized over all three states),
  `test_session_fixture_caches_result_within_session` (verifies the
  generator yields exactly one probe result and exhausts),
  `test_probe_daemon_thread_does_not_block_process_exit` (asserts the
  spawned thread is `daemon=True`),
  `test_probe_on_real_mps_reports_healthy_or_deadlocked` (skipif no MPS).
- **Reason**: Each test in the task spec maps to one test function. The
  `sys.modules` shim is the only way to mock `torch.mps.event.Event`
  without touching the existing real-torch test suite.
- **Acceptance criterion addressed**: All six bullet points under "Tests"
  in the task spec.

## Iteration 4 — T-26: MPS Event-API deadlock probe

- **File**: `src/gpucheck/__init__.py` (modified, +9 lines in LAZY_MAP +
  __all__)
- **Change**: Added `probe_mps_event_deadlock` and
  `assert_no_event_deadlock` to `_LAZY_MAP` pointing at
  `gpucheck.diagnostics.mps_event_deadlock`. Added the names to
  `__all__`.
- **Reason**: Allows `from gpucheck import probe_mps_event_deadlock` /
  `assert_no_event_deadlock` per the task's "add `mps_event_deadlock` to
  lazy export map" instruction. Existing lazy-import contract preserved:
  `torch` is not imported at gpucheck package import.
- **Acceptance criterion addressed**: Task instruction "may also touch
  `src/gpucheck/__init__.py` (add `mps_event_deadlock` to lazy export
  map)".

## Iteration 5 — T-26: MPS Event-API deadlock probe

- **File**: `CHANGELOG.md` (modified, +18 lines under [Unreleased] / Added)
- **Change**: New bullet describing `probe_mps_event_deadlock`,
  `assert_no_event_deadlock`, and the `mps_event_deadlock_status_fixture`
  helper. Cites pytorch#162872 and PR #162874 close-without-merge.
- **Reason**: Task instruction "append v1.1 entry". No existing
  `[1.1.0]` section yet — placed under the existing `[Unreleased]`
  section, which is conventional Keep-a-Changelog flow until a
  v1.1 release stamp lands.
- **Acceptance criterion addressed**: Task instruction "may also touch
  `CHANGELOG.md` (append v1.1 entry)".
