"""GPU device fixture for gpucheck."""

from __future__ import annotations

import gc
import warnings
from dataclasses import dataclass
from typing import Generator

import pytest


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


def _detect_gpu_pynvml() -> GPUDevice | None:
    """Detect GPU using pynvml (no torch dependency)."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None

    try:
        count = pynvml.nvmlDeviceGetCount()
        if count == 0:
            return None

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        # Compute capability
        major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        if isinstance(major, tuple):
            cc = (major[0], major[1])
        else:
            # Older pynvml versions return two separate values
            minor = 0
            cc = (major, minor)

        return GPUDevice(
            device_id=0,
            name=name,
            compute_capability=cc,
            memory_total=mem_info.total,
            memory_free=mem_info.free,
        )
    except pynvml.NVMLError:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def _detect_gpu_torch() -> GPUDevice | None:
    """Detect GPU using torch.cuda."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    try:
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        mem_free, mem_total = torch.cuda.mem_get_info(device_id)

        return GPUDevice(
            device_id=device_id,
            name=props.name,
            compute_capability=(props.major, props.minor),
            memory_total=mem_total,
            memory_free=mem_free,
        )
    except (RuntimeError, AssertionError):
        return None


def detect_gpu() -> GPUDevice | None:
    """Auto-detect a GPU, preferring pynvml (lighter) over torch."""
    device = _detect_gpu_pynvml()
    if device is not None:
        return device
    return _detect_gpu_torch()


def _cleanup_gpu() -> None:
    """Best-effort GPU memory cleanup."""
    gc.collect()
    try:
        import torch  # type: ignore[import-untyped]

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
