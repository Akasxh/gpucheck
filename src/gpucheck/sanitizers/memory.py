"""GPU memory leak detection and tracking."""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator


@dataclass(frozen=True, slots=True)
class SanitizerMemoryReport:
    """Result of a memory leak check."""

    leaked_bytes: int
    peak_bytes: int
    allocations: int
    deallocations: int

    @property
    def leaked_mb(self) -> float:
        return self.leaked_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def has_leak(self) -> bool:
        return self.leaked_bytes > 0


def _get_torch_memory_stats() -> dict[str, int]:
    """Read torch.cuda memory stats, returning zeros if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            stats = torch.cuda.memory_stats()
            return {
                "allocated_bytes": stats.get("allocated_bytes.all.current", 0),
                "peak_bytes": stats.get("allocated_bytes.all.peak", 0),
                "num_alloc": stats.get("allocation.all.current", 0),
                "num_free": stats.get("free.all.current", 0),
            }
    except ImportError:
        pass
    return {"allocated_bytes": 0, "peak_bytes": 0, "num_alloc": 0, "num_free": 0}


def _get_pynvml_memory() -> int:
    """Return GPU memory used in bytes via pynvml, 0 if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used: int = info.used
        pynvml.nvmlShutdown()
        return used
    except Exception:
        return 0


def _sync_and_gc(device_id: int = 0) -> None:
    """Force GC and sync GPU."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            with torch.cuda.device(device_id):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
    except ImportError:
        pass


def check_memory_leaks(
    fn: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> SanitizerMemoryReport:
    """Run *fn* and report GPU memory allocations and leaks.

    Uses torch.cuda.memory_stats when available, falls back to pynvml.
    """
    _sync_and_gc()

    torch_available = False
    try:
        import torch

        torch_available = torch.cuda.is_available()
    except ImportError:
        pass

    if torch_available:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        before = torch.cuda.memory_allocated()
        stats_before = torch.cuda.memory_stats()
        alloc_before = stats_before.get("allocation.all.current", 0)

        fn(*args, **kwargs)

        _sync_and_gc()
        after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        stats_after = torch.cuda.memory_stats()
        alloc_after = stats_after.get("allocation.all.current", 0)
        free_count = stats_after.get("free.all.current", 0)

        return SanitizerMemoryReport(
            leaked_bytes=max(0, after - before),
            peak_bytes=peak,
            allocations=max(0, alloc_after - alloc_before),
            deallocations=free_count,
        )

    # Fallback: pynvml (process-level, less precise)
    before_bytes = _get_pynvml_memory()
    fn(*args, **kwargs)
    _sync_and_gc()
    after_bytes = _get_pynvml_memory()

    return SanitizerMemoryReport(
        leaked_bytes=max(0, after_bytes - before_bytes),
        peak_bytes=max(before_bytes, after_bytes),
        allocations=0,
        deallocations=0,
    )


@contextmanager
def memory_guard(threshold_bytes: int = 0) -> Generator[_MutableReport, None, None]:
    """Context manager that tracks GPU memory and yields a :class:`SanitizerMemoryReport`.

    Usage::

        with memory_guard() as report:
            run_kernel(...)
        assert not report.has_leak

    The report is populated when the context exits. An optional *threshold_bytes*
    raises ``RuntimeError`` if leaked bytes exceed that value.
    """
    _sync_and_gc()

    torch_available = False
    try:
        import torch

        torch_available = torch.cuda.is_available()
    except ImportError:
        pass

    # We build the report as a mutable list, then replace the sentinel.
    # Since SanitizerMemoryReport is frozen we need this indirection.
    report = SanitizerMemoryReport(leaked_bytes=0, peak_bytes=0, allocations=0, deallocations=0)
    # Use object.__new__ trick to yield the same reference, then mutate via
    # __dict__ bypass — or just use a mutable wrapper.
    _holder: dict[str, Any] = {}

    if torch_available:
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        _holder["before"] = torch.cuda.memory_allocated()
        _holder["stats_before"] = torch.cuda.memory_stats()

    else:
        _holder["before"] = _get_pynvml_memory()

    # Yield a _MutableReport so caller can inspect after block
    mut = _MutableReport()
    yield mut  # type: ignore[misc]

    _sync_and_gc()

    if torch_available:
        import torch

        after = torch.cuda.memory_allocated()
        peak = torch.cuda.max_memory_allocated()
        stats_after = torch.cuda.memory_stats()
        alloc_before = _holder["stats_before"].get("allocation.all.current", 0)
        alloc_after = stats_after.get("allocation.all.current", 0)
        free_count = stats_after.get("free.all.current", 0)

        mut._fill(
            leaked_bytes=max(0, after - _holder["before"]),
            peak_bytes=peak,
            allocations=max(0, alloc_after - alloc_before),
            deallocations=free_count,
        )
    else:
        after = _get_pynvml_memory()
        mut._fill(
            leaked_bytes=max(0, after - _holder["before"]),
            peak_bytes=max(_holder["before"], after),
            allocations=0,
            deallocations=0,
        )

    if threshold_bytes > 0 and mut.leaked_bytes > threshold_bytes:
        raise RuntimeError(
            f"GPU memory leak detected: {mut.leaked_bytes} bytes "
            f"(threshold: {threshold_bytes})"
        )


class _MutableReport:
    """Mutable stand-in for :class:`SanitizerMemoryReport`, filled after context exit."""

    __slots__ = ("leaked_bytes", "peak_bytes", "allocations", "deallocations")

    def __init__(self) -> None:
        self.leaked_bytes: int = 0
        self.peak_bytes: int = 0
        self.allocations: int = 0
        self.deallocations: int = 0

    def _fill(
        self,
        *,
        leaked_bytes: int,
        peak_bytes: int,
        allocations: int,
        deallocations: int,
    ) -> None:
        self.leaked_bytes = leaked_bytes
        self.peak_bytes = peak_bytes
        self.allocations = allocations
        self.deallocations = deallocations

    @property
    def leaked_mb(self) -> float:
        return self.leaked_bytes / (1024 * 1024)

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)

    @property
    def has_leak(self) -> bool:
        return self.leaked_bytes > 0

    def to_report(self) -> SanitizerMemoryReport:
        return SanitizerMemoryReport(
            leaked_bytes=self.leaked_bytes,
            peak_bytes=self.peak_bytes,
            allocations=self.allocations,
            deallocations=self.deallocations,
        )
