"""GPU device fixture for gpucheck."""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from gpucheck.arch.detection import GPUInfo


@dataclass(frozen=True, slots=True)
class GPUDevice:
    """Describes an available GPU device."""

    device_id: int
    name: str
    compute_capability: tuple[int, int]
    memory_total: int  # bytes
    memory_free: int  # bytes

    @property
    def memory_total_mb(self) -> float:
        return self.memory_total / (1024 * 1024)

    @property
    def memory_free_mb(self) -> float:
        return self.memory_free / (1024 * 1024)

    def __str__(self) -> str:
        return (
            f"GPU({self.device_id}: {self.name}, "
            f"cc={self.compute_capability[0]}.{self.compute_capability[1]}, "
            f"mem={self.memory_total_mb:.0f}MB)"
        )


def _to_device(info: GPUInfo) -> GPUDevice:
    """Adapt a richer ``arch.detection.GPUInfo`` into the local ``GPUDevice``.

    ``GPUInfo`` carries memory in MB; ``GPUDevice`` exposes raw bytes.
    The 1MB granularity loss is acceptable for fixture-level reporting
    (``__str__`` formats only to whole-MB anyway).
    """
    return GPUDevice(
        device_id=info.device_id,
        name=info.name,
        compute_capability=info.compute_capability,
        memory_total=info.memory_total_mb * 1024 * 1024,
        memory_free=info.memory_free_mb * 1024 * 1024,
    )


def detect_gpu() -> GPUDevice | None:
    """Auto-detect a GPU, preferring pynvml (lighter) over torch.

    Delegates to :func:`gpucheck.arch.detection.detect_gpus` so there is
    exactly one detection codepath in the codebase (T-20); ``detect_gpus``
    is ``lru_cache``-backed, so repeated calls are O(1) and the
    "no detection backend available" warning fires at most once per session.
    The first detected device is adapted to the ``GPUDevice`` shape.
    """
    from gpucheck.arch.detection import detect_gpus

    gpus = detect_gpus()
    if not gpus:
        return None
    return _to_device(gpus[0])


def _cleanup_gpu() -> None:
    """Best-effort GPU memory cleanup."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    except RuntimeError as exc:
        warnings.warn(f"GPU cleanup failed: {exc}", RuntimeWarning, stacklevel=2)


@pytest.fixture()
def gpu_device() -> Generator[GPUDevice, None, None]:
    """Provide a GPU device for the test, skip if none available.

    Function-scoped. Cleans up GPU memory after the test completes.
    """
    device = detect_gpu()
    if device is None:
        pytest.skip("No GPU available")

    yield device

    _cleanup_gpu()
