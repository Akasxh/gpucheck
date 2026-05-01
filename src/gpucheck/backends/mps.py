"""MPS backend for Apple Silicon GPUs via ``torch.mps.*``.

# Deadlock context (load-bearing)

PyTorch issue [pytorch#162872](https://github.com/pytorch/pytorch/issues/162872)
documents a hang in the canonical CUDA-style timing pattern on
Apple Silicon::

    start = torch.mps.event.Event(enable_timing=True)
    end = torch.mps.event.Event(enable_timing=True)
    start.record(); kernel(); end.record()
    end.synchronize()                # <- HANGS on PyTorch 2.10+ Apple Silicon
    elapsed = start.elapsed_time(end)

gpucheck v1.0 therefore times MPS work with **device-level**
``torch.mps.synchronize()`` plus ``time.perf_counter()``. This is correct per
the PyTorch 2.11 docs (verified in research SYNTHESIS §3) and avoids the
deadlock. The ~1ms overhead vs CUDA events is acceptable — gpucheck reports
millisecond-resolution timings, not microsecond.

# Memory accounting

PyTorch issue
[pytorch#164299](https://github.com/pytorch/pytorch/issues/164299) notes that
``torch.mps.current_allocated_memory()`` and
``torch.mps.driver_allocated_memory()`` lag Activity Monitor for some
allocation patterns. This MPSBackend therefore returns BOTH numbers (so the
caller can pick) and adds an optional ``rss`` key sourced from ``psutil`` if
that package is importable. ``rss`` is the most accurate leak proxy on MPS.

# Tolerances and xfail

This module does **not** carry MPS tolerance multipliers — those live in
``gpucheck.assertions.tolerances`` so that all dtype-aware tolerance logic
shares a single source of truth.
"""

from __future__ import annotations

import platform
import subprocess
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

    from gpucheck.arch.detection import GPUInfo


def _torch() -> Any:
    import torch

    return torch


_FLUSH_L2_WARNED = False


@dataclass
class _MPSEventTimer:
    """EventTimer using device-level sync + wall-clock time.

    Avoids ``torch.mps.event.Event.synchronize()`` per pytorch#162872.
    """

    device_id: int = 0
    elapsed_ms: float = field(default=0.0)


class MPSBackend:
    """Backend implementation for Apple Silicon GPUs."""

    name: str = "mps"

    def is_available(self) -> bool:
        try:
            torch = _torch()
        except ImportError:
            return False
        return bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )

    def device_count(self) -> int:
        if not self.is_available():
            return 0
        # PyTorch's MPS device API only exposes a single logical device.
        # torch.mps.device_count() exists on 2.6+, fall back to 1.
        torch = _torch()
        fn = getattr(torch.mps, "device_count", None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                return 1
        return 1

    def synchronize(self, device_id: int = 0) -> None:
        # device_id is ignored — MPS exposes one logical device.
        del device_id
        torch = _torch()
        # Device-level sync; safe per pytorch#162872.
        torch.mps.synchronize()

    @contextmanager
    def event_timer(
        self, device_id: int = 0,
    ) -> Generator[_MPSEventTimer, None, None]:
        """Time a block with device-level sync + wall clock.

        DO NOT use ``torch.mps.event.Event.synchronize()`` here:
        pytorch#162872 deadlocks the calling thread.
        """
        timer = _MPSEventTimer(device_id=device_id)
        torch = _torch()
        # Drain any prior in-flight work so its time isn't counted in ours.
        torch.mps.synchronize()
        t0 = time.perf_counter()
        try:
            yield timer
        finally:
            # Block until the kernel(s) launched in the body actually finish.
            torch.mps.synchronize()
            timer.elapsed_ms = (time.perf_counter() - t0) * 1000.0

    def mem_stats(self, device_id: int = 0) -> dict[str, int]:
        del device_id  # MPS = single device
        torch = _torch()
        stats: dict[str, int] = {}
        try:
            stats["used"] = int(torch.mps.current_allocated_memory())
        except Exception:
            stats["used"] = 0
        try:
            stats["driver_allocated"] = int(torch.mps.driver_allocated_memory())
        except Exception:
            stats["driver_allocated"] = 0
        # recommended_max_memory exists on 2.6+
        rec_fn = getattr(torch.mps, "recommended_max_memory", None)
        if callable(rec_fn):
            try:
                stats["total"] = int(rec_fn())
            except Exception:
                stats["total"] = 0
        else:
            stats["total"] = 0
        # psutil RSS — best leak proxy per pytorch#164299
        try:
            import psutil

            stats["rss"] = int(psutil.Process().memory_info().rss)
        except ImportError:
            pass
        return stats

    def flush_l2(self, device_id: int = 0, buf: Any = None) -> None:
        """MPS does not expose an L2-cache flush primitive; this is a no-op.

        Emits a one-time :class:`UserWarning` so callers know their
        ``flush_l2=True`` request was ignored.
        """
        global _FLUSH_L2_WARNED  # noqa: PLW0603
        del device_id, buf
        if not _FLUSH_L2_WARNED:
            warnings.warn(
                "MPS backend does not implement L2 cache flush; "
                "benchmark stability may be lower than on CUDA",
                UserWarning,
                stacklevel=2,
            )
            _FLUSH_L2_WARNED = True

    def arch_info(self, device_id: int = 0) -> GPUInfo:
        del device_id
        from gpucheck.arch.detection import GPUInfo

        chip = _detect_apple_chip()
        os_ver = platform.mac_ver()[0] or ""
        # Memory total: prefer recommended_max_memory; fall back to RSS-zero.
        torch = _torch()
        rec_fn = getattr(torch.mps, "recommended_max_memory", None)
        if callable(rec_fn):
            try:
                total_bytes = int(rec_fn())
            except Exception:
                total_bytes = 0
        else:
            total_bytes = 0
        free_bytes = max(0, total_bytes - int(
            getattr(torch.mps, "current_allocated_memory", lambda: 0)()
        ))

        return GPUInfo(
            device_id=0,
            name=chip or "Apple Silicon",
            compute_capability=(0, 0),
            architecture="Apple-Silicon",
            memory_total_mb=total_bytes // (1024 * 1024),
            memory_free_mb=free_bytes // (1024 * 1024),
            driver_version=os_ver,
            cuda_version="",
            supports_fp16=True,
            supports_bf16=True,
            supports_fp8=False,  # No FP8 tensor cores on Apple Silicon as of M5
            supports_tf32=False,  # TF32 is NVIDIA-only
            tensor_core_generation=None,  # Apple GPUs have no tensor cores
            max_shared_memory_per_block=32 * 1024,  # Apple GPU threadgroup memory cap (typical)
            backend="mps",
        )


def _detect_apple_chip() -> str:
    """Return e.g. ``"Apple M4 Pro"`` or empty string on failure.

    Uses ``sysctl machdep.cpu.brand_string``; we deliberately do **not** call
    ``xcrun metal`` (security finding N1) and do **not** read ``task_info``
    (N3) — both are out-of-scope per CHARTER waivers.
    """
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            timeout=2.0,
        )
        return out.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


__all__ = ["MPSBackend"]
