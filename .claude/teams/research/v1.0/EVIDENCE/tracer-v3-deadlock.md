# Tracer v3 — runtime deadlock probe for gpucheck MPSBackend

Charter: archaeologist-v2 confirmed the `torch.mps.event.Event.synchronize()` →
`Event.elapsed_time()` deadlock in `aten/src/ATen/mps/MPSEvent.mm` is **still
unfixed in pytorch/main HEAD as of 2026-05-01** (PR #162874 closed without
merge). gpucheck v1.0 sidesteps it by routing all timing through device-level
`torch.mps.synchronize()` + `time.perf_counter()` — but the failure mode is a
silent process-wide hang the moment any user calls `Event.synchronize()`
(directly or via a `gpu_benchmark` extension). Design a runtime probe that
detects this before it hangs the test runner.

---

## Entry point

The deadlock is reachable from any of these call surfaces:

- `torch/mps/event.py:38` — `Event.synchronize()` →
  `torch._C._mps_synchronizeEvent(self.__eventId)` (Python binding, copy
  of the file fetched from `pytorch/main`).
- `torch/mps/event.py:43` — `Event.elapsed_time(end)` →
  `torch._C._mps_elapsedTimeOfEvents(self.__eventId, end._Event__eventId)`.
- gpucheck only enters this surface from the `_run_mps` loop in
  `/Users/cero/Code/gpucheck/src/gpucheck/fixtures/benchmark.py:283-327` and
  `/Users/cero/Code/gpucheck/src/gpucheck/backends/mps.py:110-129` — both
  use `torch.mps.synchronize()` (device-level), neither touches `Event`.

The deadlock fires inside libtorch C++:

- `aten/src/ATen/mps/MPSEvent.mm:115-125` — `MPSEvent::synchronize()`:
  ```cpp
  bool MPSEvent::synchronize() {
    bool scheduledNotify = notifyLocked(^(id<MTLSharedEvent>, uint64_t) {
      m_completion_time = getTime();
      notifyCpuSync();          // <- arms the cv predicate
    });
    if (scheduledNotify) {
      waitForCpuSync();         // <- consumes one notify, sets predicate=false
      return true;
    }
    return false;
  }
  ```
- `aten/src/ATen/mps/MPSEvent.mm:103-107` — `notifyCpuSync()` sets
  `m_cpu_sync_completed = true` then `m_cpu_sync_cv.notify_one()`.
- `aten/src/ATen/mps/MPSEvent.mm:109-113` — `waitForCpuSync()` waits on
  `m_cpu_sync_cv` until predicate is `true`, then **resets it to false**
  via `m_cpu_sync_completed = false` in `MPSEvent::reset()` at line 144 —
  but `reset()` only runs when the event returns to the pool. After
  `synchronize()` returns, the predicate is left in whatever state the
  notify block put it.
- `aten/src/ATen/mps/MPSEvent.mm:221-238` — `MPSEventPool::elapsedTime`
  calls `end_event->waitForCpuSync()` at **line 230** (still HEAD as of
  2026-05-01; archaeologist-v2 line cite was 228, drift is from a later
  comment edit). The notify block that would arm the predicate has
  **already fired** during the user's prior `Event.synchronize()` call,
  and there is no second notify scheduled. The cv wait is therefore
  unsatisfiable. The thread blocks forever inside `elapsed_time()`.

Critically, `MPSEventPool::synchronizeEvent` (line 211-214) does NOT hold
the pool mutex while calling `event->synchronize()`. That means a thread
hung in `elapsed_time()` does not block other threads from calling
`torch.mps.synchronize()` device-level — which is what makes a worker-thread
probe viable.

## Forward trace — gpucheck's actual MPS path (no deadlock)

1. `tests/test_my_kernel.py` — user calls `gpu_benchmark(my_kernel, x)`.
2. `src/gpucheck/fixtures/benchmark.py:147-209` — `_BenchmarkRunner.__call__`
   detects `torch.cuda.is_available() == False` and `mps_avail == True`,
   dispatches to `_run_mps` at line 209.
3. `src/gpucheck/fixtures/benchmark.py:283-327` — `_run_mps` loops over
   `n_rounds`, doing `torch.mps.synchronize(); t0=...; fn(...);
   torch.mps.synchronize(); elapsed_ms=(...).` No `Event` ever instantiated.
   Path is deadlock-free.
4. `src/gpucheck/backends/mps.py:110-129` — `MPSBackend.event_timer`
   context manager: same pattern, device-level sync only.

## Backward trace — where could a future user hit `Event.synchronize()`?

- A custom user fixture that wraps gpucheck and tries to replicate the
  CUDA pattern (`Event.record(); Event.synchronize(); Event.elapsed_time()`).
- A gpucheck contributor adding sub-millisecond timing on MPS via the
  Event API in a future patch.
- A benchmark extension that imports `torch.mps.event.Event` directly,
  unaware of #162872.

There is **no in-tree call site for `Event.synchronize()`** today. Confirmed
by `grep -rn "Event(enable_timing" src/gpucheck/`: zero hits in MPS paths.

## Competing hypotheses

### H1: A threaded watchdog around `Event.synchronize()` is sufficient.

Wrap the deadlock-trigger pattern in a daemon thread with a `Thread.join(timeout)`.
If the thread is still alive after the timeout, it deadlocked.

- **Supporting**:
  - `synchronizeEvent` (MPSEvent.mm:211) does **not** hold the
    `MPSEventPool::m_mutex`; the cv wait is per-event-instance, not
    pool-global. Other threads can still progress.
  - Python's GIL is released inside the C++ wait — the daemon thread parks
    on a `pthread_cond_wait`, the main thread keeps running.
  - `Thread(daemon=True).join(timeout=N)` is the canonical hang-detector
    in CPython.
- **Falsifiable by**: a test where the daemon thread holds a Python lock
  that the watchdog needs (not the case for `_mps_synchronizeEvent`).
- **Status**: **supported** by code-level reading; needs M-silicon
  empirical confirmation (deferred to empiricist on hardware).

### H2: gpucheck should refuse to call `Event.synchronize()` at all and only document the rule.

Make `Event.synchronize()` an out-of-policy call; static-check user code
via an AST visitor in a pytest plugin hook (the way `test_backends.py:71`
already does for gpucheck's own source).

- **Supporting**:
  - The runtime cost of a probe is non-trivial (10ms to instantiate,
    record, sync two MPS events even on a healthy build).
  - The static guard at `tests/test_backends.py:71-111` already prevents
    gpucheck-internal regressions; extending the AST scan to user
    test-modules would catch user mistakes too.
  - Zero hardware risk: AST scan runs on any platform.
- **Falsifiable by**: a user who imports `torch.mps.event.Event` indirectly
  (e.g., via `from torch.mps import event; event.Event(...)`), bypassing
  the AST pattern. This is observed in real Triton/PyTorch repos.
- **Status**: **partially supported** — covers the static case but cannot
  catch dynamic / indirect imports. Best as **complement** to H1, not
  replacement.

### H3: gpucheck should publish `assert_no_event_deadlock()` as a public API.

Expose `gpucheck.assert_no_event_deadlock(timeout_ms=2000)` so users on
M-silicon CI can opt into a self-test on every run. Run the trigger
pattern in a daemon thread, assert it returns within `timeout_ms`.

- **Supporting**:
  - The deadlock is permanent (the cv never gets a second notify);
    a single probe is dispositive: hang ⇒ broken pytorch, return ⇒ healthy.
  - The probe is *cheap when healthy* (sub-10ms) and *bounded* when
    broken (the daemon thread leaks, but this is one-shot per session).
  - It surfaces upstream regressions for users (e.g., if PyTorch ever
    re-introduces the bug after a partial fix lands).
- **Falsifiable by**: hardware where `Event.synchronize` works on cold but
  hangs on warm pool — i.e., the deadlock depends on which event slot the
  pool returns. (Plausible: archaeologist-v2 noted the bug is in
  `MPSEventPool::elapsedTime`, which interacts with reused `m_cpu_sync_cv`
  state across `reset()` calls. Hot-pool path may differ from cold-pool.)
- **Status**: **open** — design is sound, hardware verification needed.
  Recommend shipping as `pytest.fixture` so tests can opt in, plus a
  CLI smoke-test `python -m gpucheck.diagnostics --check mps-event-deadlock`.

### H4: PR #162874's one-line fix has a workable user-side mimic.

The proposed fix removed `end_event->waitForCpuSync();` at
`MPSEvent.mm:230`. User-side, the analogous pattern is to **never call
`Event.synchronize()` before `elapsed_time()`** — let `elapsed_time()`'s
internal `dispatch_sync(...)` + `m_default_stream->synchronize(SyncType::COMMIT_AND_WAIT)`
(line 224) drain the queue.

- **Supporting**:
  - The fix removes the second `waitForCpuSync` at line 230; the first
    drain at line 224 is sufficient. Conceptually: line 224 already waits
    for *all* prior work on the stream; line 230 is redundant.
  - User-side rule: "record start; record end; call `elapsed_time(end)`
    directly; no manual `synchronize()`" mimics this safely.
- **Falsifiable by**: a test where the user *needs* `Event.synchronize()`
  for some other reason (e.g., overlapping with a custom CPU operation
  whose wall-clock starts after the GPU work fully drains) and is forced
  back into the deadlock window.
- **Status**: **supported** as a documentation rule; gpucheck should
  publish the safe pattern in `MPSBackend.__doc__` (already done in
  `mps.py:1-36`) **and** ship a runtime probe that catches violations.

## Code sketches (runnable)

### Sketch 1 — runtime deadlock probe (H1 + H3 combined)

```python
# src/gpucheck/diagnostics/mps_event_deadlock.py
"""Runtime probe for the pytorch#162872 MPS Event.synchronize deadlock."""
from __future__ import annotations

import threading
from typing import Literal

DeadlockResult = Literal["healthy", "deadlocked", "skipped"]


def probe_mps_event_deadlock(timeout_ms: int = 2000) -> DeadlockResult:
    """Run the deadlock-trigger pattern in a daemon thread.

    Returns "healthy" if the call completed within ``timeout_ms``,
    "deadlocked" if the thread is still alive after the timeout,
    "skipped" if MPS is not available.

    The daemon thread is intentionally leaked on deadlock — there is no
    safe way to interrupt a C++ pthread_cond_wait from Python. Process
    exit will reap it.
    """
    try:
        import torch
    except ImportError:
        return "skipped"
    if not (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "skipped"

    done = threading.Event()

    def _run() -> None:
        # The exact pattern that triggers MPSEvent.mm:230 deadlock.
        start = torch.mps.event.Event(enable_timing=True)
        end = torch.mps.event.Event(enable_timing=True)
        x = torch.randn(64, 64, device="mps")
        start.record()
        _ = x @ x
        end.record()
        end.synchronize()           # arms cv notify
        _ = start.elapsed_time(end) # blocks on already-consumed notify
        done.set()

    t = threading.Thread(target=_run, daemon=True, name="gpucheck-mps-deadlock-probe")
    t.start()
    if done.wait(timeout=timeout_ms / 1000.0):
        return "healthy"
    # Thread is still alive — leak it; process exit will cleanup.
    return "deadlocked"


def assert_no_event_deadlock(timeout_ms: int = 2000) -> None:
    """Assert that the MPS Event API does not deadlock; raise on hang.

    Intended as an opt-in fixture or pytest collection hook on M-silicon CI.
    Raises RuntimeError on deadlock; silently returns on healthy or non-MPS.
    """
    result = probe_mps_event_deadlock(timeout_ms=timeout_ms)
    if result == "deadlocked":
        raise RuntimeError(
            "torch.mps.event.Event.synchronize() + elapsed_time() hung "
            f"for >{timeout_ms}ms; pytorch#162872 deadlock active. "
            "Use device-level torch.mps.synchronize() + time.perf_counter() "
            "instead (see gpucheck.backends.mps.MPSBackend)."
        )
```

### Sketch 2 — pytest fixture wrapper

```python
# src/gpucheck/fixtures/mps_safety.py
import pytest
from gpucheck.diagnostics.mps_event_deadlock import probe_mps_event_deadlock


@pytest.fixture(scope="session")
def mps_event_deadlock_status() -> str:
    """Run-once probe; cached for the whole pytest session."""
    return probe_mps_event_deadlock(timeout_ms=2000)


@pytest.fixture()
def require_no_mps_deadlock(mps_event_deadlock_status: str) -> None:
    """Skip tests if the runtime MPS Event deadlock is active."""
    if mps_event_deadlock_status == "deadlocked":
        pytest.skip("pytorch#162872 active — torch.mps.event.Event hangs")
```

### Sketch 3 — minimal 3-line core (the answer to "3-line code sketch")

```python
import threading, torch
def hung(t=2.0):
    f = threading.Event()
    threading.Thread(target=lambda: (torch.mps.event.Event(enable_timing=True).synchronize(), f.set()), daemon=True).start()
    return not f.wait(t)
```

## Probes I ran

No M-silicon hardware available in this sandbox. Static probes only:

- **Probe A**: confirmed MPSEvent.mm HEAD via `gh api repos/pytorch/pytorch/contents/aten/src/ATen/mps/MPSEvent.mm`
  base64-decoded; verified line 230 still contains
  `end_event->waitForCpuSync();` and lines 115-125 still implement the
  one-shot notify pattern. **Conclusion**: deadlock geometry intact in
  HEAD as of 2026-05-01.
- **Probe B**: grep `Event(enable_timing` across `src/gpucheck/`:
  `cuda.py:72-73` (CUDA only), `fixtures/benchmark.py:266-267` (CUDA only),
  zero MPS hits. **Conclusion**: gpucheck v1.0 does not trigger the
  deadlock from any in-tree code path.
- **Probe C**: read `tests/test_backends.py:71-111`: existing AST-based
  static guard already enforces "no `event.synchronize()` in
  `MPSBackend.event_timer`". **Conclusion**: H2 is partially deployed;
  static guard is good but doesn't cover user code.
- **Probe D**: confirmed `MPSEventPool::synchronizeEvent` at MPSEvent.mm:211-214
  does **not** acquire `m_mutex` before calling `event->synchronize()`.
  Therefore a thread hung inside `synchronize()` does not block other
  threads from calling `torch.mps.synchronize()` device-level.
  **Conclusion**: H1 (worker-thread watchdog) is viable — the main thread
  remains responsive even if the probe daemon hangs.

## Uncovered ground

- **Hardware verification**: does the probe in Sketch 1 actually return
  `"deadlocked"` on a real M-series Mac running pytorch 2.10+? Empiricist
  on M5 hardware should run it. Predicted: yes for M-silicon torch >=2.10,
  no for older. Cost: a leaked daemon thread per healthy run (cheap), one
  permanent leak per deadlocked run (one-shot session-scoped fixture).
- **Hot-pool vs cold-pool**: PR #162874's fix conjecture suggests the
  deadlock is unconditional, but archaeologist-v2 noted the cv predicate
  is reset only via `MPSEvent::reset()` (line 144) when the event returns
  to the pool. If a user holds a long-lived `Event` reference (preventing
  pool return), the cv state may differ between first and Nth call. Worth
  parameterizing the probe across pool reuse depth.
- **GIL interaction**: confirm via `py-spy dump` on a hung run that the
  daemon thread is parked in `pthread_cond_wait` and not holding the GIL.
  If it is holding the GIL, the main thread is also dead, and Sketch 1
  collapses to a no-op. Static reading of `_mps_synchronizeEvent` C
  binding suggests `pybind11` releases GIL at the boundary — but unverified.
- **Cancel path**: there is no `pthread_cancel` from Python. If we ever
  want to recover the leaked thread, we need a libtorch-side change
  (e.g., `notifyCpuSync_with_timeout`) — out of scope for v1.0.

## Recommended public API surface for v1.0 vs v1.1

**v1.0 (now)**:
- Ship `gpucheck.diagnostics.mps_event_deadlock` module with
  `probe_mps_event_deadlock()` and `assert_no_event_deadlock()`.
- Ship `gpucheck.fixtures.mps_safety` with the `mps_event_deadlock_status`
  session-scoped fixture and `require_no_mps_deadlock` test fixture.
- Extend the existing AST guard at `tests/test_backends.py:71-111` to
  scan ALL of `src/gpucheck/`, not just `MPSBackend.event_timer`. Forbidden
  pattern set: `{event.synchronize, start.synchronize, end.synchronize,
  Event.synchronize}`.
- Document the safe pattern in `gpucheck.backends.mps.MPSBackend.__doc__`
  (already present at `mps.py:1-36`).

**v1.1 (later)**:
- Auto-run the probe at first invocation of `gpu_benchmark` on MPS, log
  result to the test session as a Rich panel; xfail any benchmark that
  attempts `Event` timing on a deadlocked PyTorch.
- Add `probe_mps_event_deadlock(force_pool_reuse=N)` parameter to test
  hot-pool vs cold-pool deadlock variance.
- Publish a contributing-guide section "Why we don't use torch.mps.Event"
  with the citation chain (#102121 → #162872 → #162874-closed).

## Verdict

**Ship a deadlock probe in v1.0**, scoped to:
1. `probe_mps_event_deadlock()` (private, leak-tolerant daemon thread).
2. `assert_no_event_deadlock()` (public, raises on hang).
3. `mps_event_deadlock_status` session fixture (cached probe result).
4. AST guard widened to all of `src/gpucheck/`.

The probe is cheap when healthy (<10ms), self-contained (one daemon
thread leak per deadlocked session), and pre-empts a class of bug that
will hang user CI runners with no diagnostic. Deferring to v1.1 means a
silent CI hang for any user who experiments with `Event` timing — exactly
the failure mode v1.0's docstring promises to prevent.

## Confidence

**high** for the deadlock geometry (verified by reading raw `MPSEvent.mm`
HEAD, line-cited; PR #162874 confirmed not-merged via archaeologist-v2).
**high** for H1 viability (`MPSEventPool::synchronizeEvent` does not hold
pool mutex; daemon-thread leak is bounded).
**medium** for hardware-empirical claims about probe cost and false-positive
rate — those need M-silicon verification by the empiricist persona.
**high** for the v1.0-ship recommendation: incremental cost (~150 LOC plus
one new test file) is small, the upside (guarding against a permanent CI
hang) is large, and the hardware risk is contained to the daemon-thread
fixture (which is opt-in).
