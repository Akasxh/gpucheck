"""GPU memory tracking fixture for gpucheck."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Generator

import pytest


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """GPU memory state at a point in time (bytes)."""

    used: int
    free: int
    total: int

    @property
    def used_mb(self) -> float:
        return self.used / (1024 * 1024)


@dataclass(frozen=True, slots=True)
class MemoryReport:
    """Summary of GPU memory usage during a test."""

    before: MemorySnapshot
    after: MemorySnapshot
    peak: int  # bytes — peak memory usage during the test
    leaked: int  # bytes — memory not freed after the test
    leak_detected: bool

    @property
    def peak_mb(self) -> float:
        return self.peak / (1024 * 1024)

    @property
    def leaked_mb(self) -> float:
        return self.leaked / (1024 * 1024)

    def __str__(self) -> str:
        status = "LEAK" if self.leak_detected else "OK"
        return (
            f"MemoryReport({status}: before={self.before.used_mb:.1f}MB, "
            f"after={self.after.used_mb:.1f}MB, peak={self.peak_mb:.1f}MB, "
            f"leaked={self.leaked_mb:.1f}MB)"
        )


# Leak threshold: 1 MB — anything below is noise from allocator fragmentation
_LEAK_THRESHOLD_BYTES = 1 * 1024 * 1024


def _snapshot_pynvml(device_id: int = 0) -> MemorySnapshot | None:
    # Synchronize GPU before taking pynvml snapshot to ensure pending ops complete
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.synchronize(device_id)
    except (ImportError, RuntimeError):
        pass

    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return MemorySnapshot(used=mem.used, free=mem.free, total=mem.total)
    except pynvml.NVMLError:
        return None


def _snapshot_torch(device_id: int = 0) -> MemorySnapshot | None:
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize(device_id)
        free, total = torch.cuda.mem_get_info(device_id)
        used = total - free
        return MemorySnapshot(used=used, free=free, total=total)
    except RuntimeError:
        return None


def _get_peak_torch(device_id: int = 0) -> int | None:
    """Return peak allocated memory from torch's CUDA allocator."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.max_memory_allocated(device_id))
    except RuntimeError:
        return None


def _snapshot(device_id: int = 0) -> MemorySnapshot | None:
    """Take a memory snapshot, preferring pynvml over torch."""
    snap = _snapshot_pynvml(device_id)
    if snap is not None:
        return snap
    return _snapshot_torch(device_id)


def _reset_peak_torch(device_id: int = 0) -> None:
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device_id)
    except (ImportError, RuntimeError):
        pass


class MemoryTracker:
    """Tracks GPU memory across a test's lifetime."""

    def __init__(self, device_id: int = 0, leak_threshold: int = _LEAK_THRESHOLD_BYTES) -> None:
        self.device_id = device_id
        self.leak_threshold = leak_threshold
        self._before: MemorySnapshot | None = None
        self._after: MemorySnapshot | None = None
        self._peak_torch: int | None = None
        self._report: MemoryReport | None = None

    def start(self) -> None:
        _reset_peak_torch(self.device_id)
        self._before = _snapshot(self.device_id)

    def stop(self) -> MemoryReport:
        self._after = _snapshot(self.device_id)
        self._peak_torch = _get_peak_torch(self.device_id)

        if self._before is None or self._after is None:
            warnings.warn(
                "Could not capture GPU memory snapshots (no pynvml or torch available)",
                RuntimeWarning,
                stacklevel=2,
            )
            zero = MemorySnapshot(used=0, free=0, total=0)
            self._report = MemoryReport(
                before=zero, after=zero, peak=0, leaked=0, leak_detected=False
            )
            return self._report

        leaked = max(0, self._after.used - self._before.used)
        peak = self._peak_torch if self._peak_torch is not None else self._after.used

        self._report = MemoryReport(
            before=self._before,
            after=self._after,
            peak=peak,
            leaked=leaked,
            leak_detected=leaked > self.leak_threshold,
        )
        return self._report

    @property
    def report(self) -> MemoryReport | None:
        return self._report


# Backward-compat alias
_MemoryTracker = MemoryTracker


@pytest.fixture()
def memory_tracker() -> Generator[MemoryTracker, None, None]:
    """Track GPU memory usage and detect leaks during a test.

    The tracker automatically records memory before the test body runs.
    After the test, it captures the final state and generates a report.

    Usage::

        def test_no_leak(memory_tracker):
            # memory_tracker.start() is called automatically
            x = torch.randn(1024, 1024, device="cuda")
            del x
            torch.cuda.empty_cache()
            report = memory_tracker.stop()
            assert not report.leak_detected
    """
    tracker = _MemoryTracker()
    tracker.start()
    yield tracker
    # If user already called stop(), don't overwrite
    if tracker.report is None:
        report = tracker.stop()
        if report.leak_detected:
            warnings.warn(
                f"GPU memory leak detected: {report.leaked_mb:.1f}MB not freed",
                RuntimeWarning,
                stacklevel=2,
            )
