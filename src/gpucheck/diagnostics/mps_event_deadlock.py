"""Runtime probe for the pytorch#162872 MPS ``Event.synchronize()`` deadlock.

# Background

[pytorch#162872](https://github.com/pytorch/pytorch/issues/162872) is an
unconditional deadlock in the MPS Event API on Apple Silicon. The bug
geometry, verified line-by-line against ``aten/src/ATen/mps/MPSEvent.mm``
HEAD as of 2026-05-01 by tracer-v3, is:

1. ``MPSEvent::synchronize()`` schedules a single ``notifyCpuSync()``
   block on the shared event, then ``waitForCpuSync()`` consumes that
   one notify and the predicate is left in whatever state the notify
   block put it (the cv predicate is reset only inside
   ``MPSEvent::reset()`` at line 144, which runs only when the event
   returns to the pool).
2. ``MPSEventPool::elapsedTime`` then calls
   ``end_event->waitForCpuSync()`` *again* at line 230 — but no second
   notify has been scheduled, so the cv wait is unsatisfiable and the
   thread blocks forever inside ``elapsed_time()``.
3. PR #162874, which removed the duplicate ``waitForCpuSync`` at line
   230, was *closed without merge*. The geometry remains intact in
   ``pytorch/main`` HEAD; the deadlock will trigger on any MPS build
   that calls the canonical CUDA-style timing pattern.

gpucheck v1.0's :class:`gpucheck.backends.mps.MPSBackend` already
sidesteps the bug by routing all timing through device-level
``torch.mps.synchronize()`` plus ``time.perf_counter()``. The probe in
this module exists so that **users** can detect the bug if a future
gpucheck contributor (or a custom user fixture) drifts back to the
unsafe pattern, and so that M-silicon CI can xfail tests that depend on
working ``Event.elapsed_time`` until upstream lands a fix.

# Why a daemon thread is safe

``MPSEventPool::synchronizeEvent`` (MPSEvent.mm:211-214) does **not**
hold the pool mutex while calling ``event->synchronize()``. A thread
hung in ``elapsed_time()`` therefore does not block other threads from
calling ``torch.mps.synchronize()`` device-level. Python's GIL is
released inside the C++ ``pthread_cond_wait`` (verified by reading the
``_mps_synchronizeEvent`` pybind11 binding), so the main thread keeps
running while the probe daemon parks in a kernel wait.

There is no safe way to ``pthread_cancel`` from Python — if the probe
detects a deadlock, the daemon thread is **intentionally leaked**.
Process exit reaps it. The leak is one-shot per session because the
intended usage is the ``mps_event_deadlock_status`` session-scoped
fixture (see :mod:`gpucheck.fixtures.mps_safety`), which caches the
probe result.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator

ProbeResult = Literal["healthy", "deadlocked", "skipped"]


def _torch() -> Any:
    """Lazy ``torch`` import — never resolved at module load time."""
    import torch

    return torch


def _mps_available(torch: Any) -> bool:
    """Return True iff this host has an MPS-capable PyTorch build."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def _trigger_deadlock_pattern() -> None:
    """Run the exact code path that fires pytorch#162872.

    This is the *unsafe* pattern (deliberately) — recording two
    timing-enabled events, running a kernel between them, then calling
    ``end.synchronize()`` followed by ``start.elapsed_time(end)``. On a
    broken PyTorch build the second call blocks forever inside
    ``MPSEventPool::elapsedTime``; on a fixed build it returns the
    elapsed time in milliseconds.

    Called only from inside :func:`probe_mps_event_deadlock`'s daemon
    thread so the host process stays responsive.
    """
    torch = _torch()
    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)
    x = torch.randn(64, 64, device="mps")
    start.record()
    _ = x @ x
    end.record()
    end.synchronize()  # arms the cv predicate; consumes the one notify
    _ = start.elapsed_time(end)  # blocks on already-consumed notify on broken builds


def probe_mps_event_deadlock(timeout_ms: int = 2000) -> ProbeResult:
    """Detect whether ``torch.mps.event.Event`` deadlocks (pytorch#162872).

    Runs the deadlock-trigger pattern (``start.record`` + ``end.record``
    + ``end.synchronize`` + ``start.elapsed_time``) inside a daemon
    thread with a hard timeout. The daemon thread is intentionally
    leaked on deadlock — there is no safe way to interrupt a C++
    ``pthread_cond_wait`` from Python. Process exit reaps it.

    Parameters
    ----------
    timeout_ms:
        Maximum time in milliseconds to wait for the trigger pattern to
        complete. Default 2000ms; on a healthy build the pattern returns
        in <10ms, so any setting above ~500ms is fine.

    Returns
    -------
    ``"healthy"``
        The trigger pattern completed within ``timeout_ms``. PyTorch's
        MPS Event API is functioning correctly on this host.
    ``"deadlocked"``
        The daemon thread was still alive after ``timeout_ms`` —
        pytorch#162872 (or an equivalent regression) is active. The
        daemon thread is leaked.
    ``"skipped"``
        ``torch`` is not importable, or this host has no MPS-capable
        PyTorch build. No probe was attempted.
    """
    try:
        torch = _torch()
    except ImportError:
        return "skipped"
    if not _mps_available(torch):
        return "skipped"

    done = threading.Event()

    def _run() -> None:
        try:
            _trigger_deadlock_pattern()
        except Exception:  # noqa: BLE001
            # Any non-hang failure (CUDA-style stub, missing op, etc.) is
            # NOT a deadlock — flag the thread as completed so the caller
            # returns "healthy". The deadlock signature is "thread never
            # sets the event"; an exception still sets it.
            pass
        finally:
            done.set()

    t = threading.Thread(
        target=_run,
        daemon=True,
        name="gpucheck-mps-deadlock-probe",
    )
    t.start()
    if done.wait(timeout=timeout_ms / 1000.0):
        return "healthy"
    return "deadlocked"


def assert_no_event_deadlock(timeout_ms: int = 2000) -> None:
    """Raise :class:`RuntimeError` if the MPS Event API is deadlocked.

    Thin wrapper around :func:`probe_mps_event_deadlock` for callers
    that prefer assertion semantics. ``"skipped"`` and ``"healthy"``
    return silently; ``"deadlocked"`` raises with a citation to
    pytorch#162872 and a pointer to the safe device-level pattern.

    Parameters
    ----------
    timeout_ms:
        Forwarded to :func:`probe_mps_event_deadlock`.

    Raises
    ------
    RuntimeError
        If the probe returns ``"deadlocked"``.
    """
    result = probe_mps_event_deadlock(timeout_ms=timeout_ms)
    if result == "deadlocked":
        raise RuntimeError(
            "torch.mps.event.Event.synchronize() + elapsed_time() hung "
            f"for >{timeout_ms}ms; pytorch#162872 deadlock active. "
            "Use device-level torch.mps.synchronize() + time.perf_counter() "
            "instead (see gpucheck.backends.mps.MPSBackend.event_timer)."
        )


def mps_event_deadlock_status_fixture(timeout_ms: int = 2000) -> Iterator[ProbeResult]:
    """Body of the session-scoped ``mps_event_deadlock_status`` fixture.

    This is the *generator function* the pytest fixture wraps. It is
    exposed here so users can register the fixture in their own
    ``conftest.py`` (gpucheck does not auto-register fixtures from this
    module to keep the import-time cost of importing
    ``gpucheck.diagnostics`` zero):

    .. code-block:: python

        # tests/conftest.py
        import pytest
        from gpucheck.diagnostics.mps_event_deadlock import (
            mps_event_deadlock_status_fixture,
        )

        @pytest.fixture(scope="session")
        def mps_event_deadlock_status():
            yield from mps_event_deadlock_status_fixture()

    Once registered, any test in the session can request the fixture
    and receive the cached probe result without re-running the daemon
    thread.

    Parameters
    ----------
    timeout_ms:
        Forwarded to :func:`probe_mps_event_deadlock` on the single
        invocation per session.

    Yields
    ------
    ProbeResult
        The probe result for this session. Cached by pytest's session
        scope — the daemon thread runs at most once per ``pytest``
        process.
    """
    yield probe_mps_event_deadlock(timeout_ms=timeout_ms)


__all__ = [
    "ProbeResult",
    "assert_no_event_deadlock",
    "mps_event_deadlock_status_fixture",
    "probe_mps_event_deadlock",
]
