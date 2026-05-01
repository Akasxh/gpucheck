"""CUDA backend conforming to the gpucheck :class:`Backend` Protocol.

This is a thin facade over the existing ``torch.cuda.*`` and ``pynvml``
helpers used elsewhere in the project. Existing call sites in
``fixtures/benchmark.py``, ``fixtures/profiler.py``, ``arch/detection.py``
keep their direct ``torch.cuda.*`` invocations for v1.0 — this module exists
so that **new** code (especially MPS-aware test code) can write
backend-agnostic loops::

    backend = get_backend("cuda")  # or "mps"
    with backend.event_timer() as t:
        kernel(x, y)
    elapsed = t.elapsed_ms

A v1.1 refactor will migrate the legacy call sites to consume this Protocol.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from gpucheck.arch.detection import GPUInfo


def _torch() -> Any:
    """Lazy-import torch so the package stays importable without it."""
    import torch

    return torch


@dataclass
class _CUDAEventTimer:
    """EventTimer backed by ``torch.cuda.Event(enable_timing=True)``."""

    device_id: int = 0
    elapsed_ms: float = field(default=0.0)


class CUDABackend:
    """Backend implementation targeting NVIDIA GPUs via ``torch.cuda``."""

    name: str = "cuda"

    def is_available(self) -> bool:
        try:
            torch = _torch()
        except ImportError:
            return False
        return bool(torch.cuda.is_available())

    def device_count(self) -> int:
        if not self.is_available():
            return 0
        return int(_torch().cuda.device_count())

    def synchronize(self, device_id: int = 0) -> None:
        torch = _torch()
        torch.cuda.synchronize(device_id)

    @contextmanager
    def event_timer(
        self, device_id: int = 0,
    ) -> Generator[_CUDAEventTimer, None, None]:
        torch = _torch()
        timer = _CUDAEventTimer(device_id=device_id)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield timer
        finally:
            end.record()
            torch.cuda.synchronize(device_id)
            timer.elapsed_ms = float(start.elapsed_time(end))

    def mem_stats(self, device_id: int = 0) -> dict[str, int]:
        torch = _torch()
        try:
            free, total = torch.cuda.mem_get_info(device_id)
        except RuntimeError:
            return {"used": 0, "total": 0, "free": 0}
        used = total - free
        return {"used": int(used), "total": int(total), "free": int(free)}

    def flush_l2(self, device_id: int = 0, buf: Any = None) -> None:
        # Use existing helper to keep behavior identical.
        from gpucheck.fixtures.benchmark import _flush_l2_cache, _get_l2_cache_size

        size = _get_l2_cache_size()
        if size <= 0:
            return
        _flush_l2_cache(size, buf=buf)

    def arch_info(self, device_id: int = 0) -> GPUInfo:
        from gpucheck.arch.detection import detect_gpus

        gpus = detect_gpus()
        if not gpus or device_id >= len(gpus):
            raise RuntimeError(
                f"CUDA backend reports no GPU at index {device_id} "
                f"(detected {len(gpus)})"
            )
        return gpus[device_id]


__all__ = ["CUDABackend"]
