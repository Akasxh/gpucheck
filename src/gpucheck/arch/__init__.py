"""GPU architecture detection and capability querying."""

from __future__ import annotations

from gpucheck.arch.compatibility import require_arch, require_capability
from gpucheck.arch.detection import GPUInfo, detect_gpus
from gpucheck.arch.tensor_cores import supports_tensor_cores, warn_tensor_core_fallback


def gpu_available() -> bool:
    """Return True if at least one GPU is detected."""
    return len(detect_gpus()) > 0


def gpu_count() -> int:
    """Return the number of detected GPUs."""
    return len(detect_gpus())


def detect_gpu() -> GPUInfo | None:
    """Return the first detected GPU, or None if no GPU is available."""
    gpus = detect_gpus()
    return gpus[0] if gpus else None


__all__ = [
    "GPUInfo",
    "detect_gpu",
    "detect_gpus",
    "gpu_available",
    "gpu_count",
    "require_arch",
    "require_capability",
    "supports_tensor_cores",
    "warn_tensor_core_fallback",
]
