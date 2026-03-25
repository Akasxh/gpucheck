"""gpucheck fixtures — GPU device, benchmarking, and memory tracking."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "BenchmarkResult": ("gpucheck.fixtures.benchmark", "BenchmarkResult"),
    "gpu_benchmark": ("gpucheck.fixtures.benchmark", "gpu_benchmark"),
    "GPUDevice": ("gpucheck.fixtures.gpu", "GPUDevice"),
    "gpu_device": ("gpucheck.fixtures.gpu", "gpu_device"),
    "MemoryReport": ("gpucheck.fixtures.profiler", "MemoryReport"),
    "MemorySnapshot": ("gpucheck.fixtures.profiler", "MemorySnapshot"),
    "MemoryTracker": ("gpucheck.fixtures.profiler", "MemoryTracker"),
    "memory_tracker": ("gpucheck.fixtures.profiler", "memory_tracker"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'gpucheck.fixtures' has no attribute {name!r}")


__all__ = [
    "BenchmarkResult",
    "GPUDevice",
    "MemoryReport",
    "MemorySnapshot",
    "MemoryTracker",
    "gpu_benchmark",
    "gpu_device",
    "memory_tracker",
]
