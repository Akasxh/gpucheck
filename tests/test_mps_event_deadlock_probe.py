"""Tests for the MPS Event-API deadlock probe (T-26 / pytorch#162872).

The probe runs an unsafe trigger pattern (``Event.synchronize()`` +
``Event.elapsed_time()``) inside a daemon thread with a timeout. These
tests exercise the threading + timeout state machine on CPU using mock
``torch.mps.event.Event`` shims; one final test runs the *real* probe
on MPS hardware (skipped if no MPS).
"""

from __future__ import annotations

import os
import sys
import threading
import time
import types
from typing import TYPE_CHECKING, Any

import pytest

from gpucheck.diagnostics.mps_event_deadlock import (
    assert_no_event_deadlock,
    mps_event_deadlock_status_fixture,
    probe_mps_event_deadlock,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


# ---------------------------------------------------------------------------
# Helpers — install / restore a mock torch shim into sys.modules
# ---------------------------------------------------------------------------

class _MockTensor:
    """Mock tensor that supports ``@`` so the trigger body runs end-to-end."""

    def __matmul__(self, other: Any) -> _MockTensor:
        del other
        return self

    def __rmatmul__(self, other: Any) -> _MockTensor:
        del other
        return self


class _MockEvent:
    """Mock that controls how long the trigger pattern runs."""

    _delay_ms: int = 0

    def __init__(self, *, enable_timing: bool = False) -> None:
        del enable_timing

    def record(self) -> None:
        return None

    def synchronize(self) -> None:
        if _MockEvent._delay_ms > 0:
            time.sleep(_MockEvent._delay_ms / 1000.0)

    def elapsed_time(self, end: _MockEvent) -> float:
        del end
        return 0.0


def _install_mock_torch(
    *,
    mps_available: bool,
    delay_ms: int,
) -> tuple[Any, Any]:
    """Install a fake ``torch`` into ``sys.modules``.

    Returns the (mock torch module, mock event submodule) pair so the
    caller can restore them in a teardown step.
    """
    _MockEvent._delay_ms = delay_ms

    # types.ModuleType has dynamic attributes; mypy's strict mode flags
    # every assignment as attr-defined. The ignore-comments are required
    # only on this dynamic surface — sys.modules entries are typed as Any
    # via the dict signature.
    fake_torch = types.ModuleType("torch")
    fake_backends = types.ModuleType("torch.backends")
    fake_mps_backends = types.ModuleType("torch.backends.mps")
    fake_mps_backends.is_available = lambda: mps_available  # type: ignore[attr-defined]
    fake_backends.mps = fake_mps_backends  # type: ignore[attr-defined]
    fake_torch.backends = fake_backends  # type: ignore[attr-defined]

    fake_mps = types.ModuleType("torch.mps")
    fake_mps_event = types.ModuleType("torch.mps.event")
    fake_mps_event.Event = _MockEvent  # type: ignore[attr-defined]
    fake_mps.event = fake_mps_event  # type: ignore[attr-defined]
    fake_torch.mps = fake_mps  # type: ignore[attr-defined]

    fake_torch.randn = lambda *_a, **_k: _MockTensor()  # type: ignore[attr-defined]

    return fake_torch, fake_mps_event


@pytest.fixture()
def mock_torch_healthy() -> Iterator[None]:
    """Mock torch where ``Event.synchronize`` returns in 50ms (healthy)."""
    saved = {k: sys.modules.get(k) for k in (
        "torch",
        "torch.backends",
        "torch.backends.mps",
        "torch.mps",
        "torch.mps.event",
    )}
    fake_torch, _ = _install_mock_torch(mps_available=True, delay_ms=50)
    sys.modules["torch"] = fake_torch
    sys.modules["torch.backends"] = fake_torch.backends
    sys.modules["torch.backends.mps"] = fake_torch.backends.mps
    sys.modules["torch.mps"] = fake_torch.mps
    sys.modules["torch.mps.event"] = fake_torch.mps.event
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture()
def mock_torch_deadlocked() -> Iterator[None]:
    """Mock torch where ``Event.synchronize`` sleeps past the timeout."""
    saved = {k: sys.modules.get(k) for k in (
        "torch",
        "torch.backends",
        "torch.backends.mps",
        "torch.mps",
        "torch.mps.event",
    )}
    # Sleep 1.5 seconds — well past the 200ms test timeout but bounded so
    # the leaked daemon thread is reaped quickly when the test session ends.
    fake_torch, _ = _install_mock_torch(mps_available=True, delay_ms=1500)
    sys.modules["torch"] = fake_torch
    sys.modules["torch.backends"] = fake_torch.backends
    sys.modules["torch.backends.mps"] = fake_torch.backends.mps
    sys.modules["torch.mps"] = fake_torch.mps
    sys.modules["torch.mps.event"] = fake_torch.mps.event
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@pytest.fixture()
def mock_torch_no_mps() -> Iterator[None]:
    """Mock torch where MPS is not available."""
    saved = {k: sys.modules.get(k) for k in (
        "torch",
        "torch.backends",
        "torch.backends.mps",
    )}
    fake_torch, _ = _install_mock_torch(mps_available=False, delay_ms=0)
    sys.modules["torch"] = fake_torch
    sys.modules["torch.backends"] = fake_torch.backends
    sys.modules["torch.backends.mps"] = fake_torch.backends.mps
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_probe_returns_skipped_without_mps(mock_torch_no_mps: None) -> None:
    """When ``torch.backends.mps.is_available()`` is False, return 'skipped'."""
    del mock_torch_no_mps
    result = probe_mps_event_deadlock(timeout_ms=200)
    assert result == "skipped"


def test_probe_returns_healthy_when_thread_completes_fast(
    mock_torch_healthy: None,
) -> None:
    """50ms-completing trigger pattern returns 'healthy' with 1000ms timeout."""
    del mock_torch_healthy
    result = probe_mps_event_deadlock(timeout_ms=1000)
    assert result == "healthy"


def test_probe_returns_deadlocked_on_timeout(
    mock_torch_deadlocked: None,
) -> None:
    """5000ms-sleeping trigger pattern returns 'deadlocked' with 200ms timeout."""
    del mock_torch_deadlocked
    result = probe_mps_event_deadlock(timeout_ms=200)
    assert result == "deadlocked"


@pytest.mark.parametrize(
    ("expected_state", "fixture_name"),
    [
        ("deadlocked", "mock_torch_deadlocked"),
        ("healthy", "mock_torch_healthy"),
        ("skipped", "mock_torch_no_mps"),
    ],
)
def test_assert_no_event_deadlock_raises_on_deadlocked(
    expected_state: str,
    fixture_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """``assert_no_event_deadlock`` raises only on 'deadlocked'."""
    request.getfixturevalue(fixture_name)
    if expected_state == "deadlocked":
        with pytest.raises(RuntimeError, match="pytorch#162872"):
            assert_no_event_deadlock(timeout_ms=200)
    else:
        # healthy + skipped both return without raising
        assert_no_event_deadlock(timeout_ms=1000)


def test_session_fixture_caches_result_within_session(
    mock_torch_no_mps: None,
) -> None:
    """The session fixture body runs the probe exactly once and yields it.

    We can't easily exercise pytest's session scope from inside a single
    test, so we verify the fixture-body generator's contract: it yields
    one ``ProbeResult`` value derived from a single call to
    :func:`probe_mps_event_deadlock`. We mock MPS as unavailable so the
    test is fast and deterministic on every host (``"skipped"``) and
    does not leak a daemon thread on real MPS where pytorch#162872
    cascades across probe calls.
    """
    del mock_torch_no_mps
    gen = mps_event_deadlock_status_fixture(timeout_ms=200)
    first = next(gen)
    assert first == "skipped"
    # Generator is exhausted after the single yield.
    with pytest.raises(StopIteration):
        next(gen)


def test_probe_daemon_thread_does_not_block_process_exit(
    mock_torch_deadlocked: None,
) -> None:
    """A leaked daemon thread must not keep the process alive.

    We mock the deadlocked path so the spawned probe thread is
    guaranteed to be alive at assertion time (the trigger sleeps for
    ~1.5s, well past our 50ms probe timeout). We can't actually call
    ``sys.exit`` inside a test, but we can verify the thread is
    ``daemon=True`` so the runtime's shutdown sequence will reap it.
    """
    del mock_torch_deadlocked
    main_threads_before = {t.ident for t in threading.enumerate()}
    result = probe_mps_event_deadlock(timeout_ms=50)
    assert result == "deadlocked"
    leaked = [
        t for t in threading.enumerate()
        if t.ident not in main_threads_before
        and t.name == "gpucheck-mps-deadlock-probe"
    ]
    assert leaked, "expected at least one leaked daemon thread"
    for t in leaked:
        assert t.daemon, "probe daemon thread must be daemon=True (leaked thread blocks exit)"


# ---------------------------------------------------------------------------
# Real-MPS integration probe (skipped if no MPS)
# ---------------------------------------------------------------------------

def _has_real_mps() -> bool:
    try:
        import torch
    except ImportError:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


@pytest.mark.skipif(not _has_real_mps(), reason="MPS hardware not available")
@pytest.mark.skipif(
    os.environ.get("GPUCHECK_RUN_REAL_MPS_PROBE") != "1",
    reason=(
        "Real-MPS probe leaks a daemon thread that may interfere with the "
        "rest of the pytest session on broken PyTorch builds (per "
        "tracer-v3 §'Uncovered ground' the GIL release inside "
        "_mps_synchronizeEvent is unverified). Set "
        "GPUCHECK_RUN_REAL_MPS_PROBE=1 to opt in."
    ),
)
def test_probe_on_real_mps_reports_healthy_or_deadlocked() -> None:
    """Run the probe against the real PyTorch MPS Event API.

    On Apple Silicon with PyTorch >=2.10, this is expected to return
    "deadlocked" until pytorch#162872 lands a fix. Older PyTorch
    versions or post-fix builds return "healthy". Either is a valid
    pass; the only failure mode is the probe returning anything outside
    that set or hanging the test process.

    Opt-in via ``GPUCHECK_RUN_REAL_MPS_PROBE=1``.
    """
    result = probe_mps_event_deadlock(timeout_ms=4000)
    assert result in {"healthy", "deadlocked"}, (
        f"unexpected probe result on real MPS: {result!r}"
    )
