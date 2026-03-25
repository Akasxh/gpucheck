"""gpucheck — pytest plugin for GPU kernel testing."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"


_LAZY_MAP: dict[str, tuple[str, str]] = {
    "assert_close": ("gpucheck.assertions", "assert_close"),
    "dtypes": ("gpucheck.decorators", "dtypes"),
    "shapes": ("gpucheck.decorators", "shapes"),
    "devices": ("gpucheck.decorators", "devices"),
    "parametrize_gpu": ("gpucheck.decorators", "parametrize_gpu"),
    "FLOAT_DTYPES": ("gpucheck.decorators", "FLOAT_DTYPES"),
    "HALF_DTYPES": ("gpucheck.decorators", "HALF_DTYPES"),
    "ALL_DTYPES": ("gpucheck.decorators", "ALL_DTYPES"),
    "FP8_DTYPES": ("gpucheck.decorators", "FP8_DTYPES"),
    "SMALL_SHAPES": ("gpucheck.decorators", "SMALL_SHAPES"),
    "MEDIUM_SHAPES": ("gpucheck.decorators", "MEDIUM_SHAPES"),
    "LARGE_SHAPES": ("gpucheck.decorators", "LARGE_SHAPES"),
    "EDGE_SHAPES": ("gpucheck.decorators", "EDGE_SHAPES"),
    "fuzz_shapes": ("gpucheck.fuzzing", "fuzz_shapes"),
    "GPUInfo": ("gpucheck.arch", "GPUInfo"),
    "detect_gpu": ("gpucheck.arch", "detect_gpu"),
    "gpu_available": ("gpucheck.arch", "gpu_available"),
    "gpu_count": ("gpucheck.arch", "gpu_count"),
    "BenchmarkResult": ("gpucheck.fixtures.benchmark", "BenchmarkResult"),
    "GPUDevice": ("gpucheck.fixtures.gpu", "GPUDevice"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(module_path)
        # Cache in globals to avoid repeated import on next access
        value = getattr(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'gpucheck' has no attribute {name!r}")


if TYPE_CHECKING:
    from gpucheck.arch import GPUInfo, detect_gpu, gpu_available, gpu_count
    from gpucheck.assertions import assert_close
    from gpucheck.decorators import (
        ALL_DTYPES,
        EDGE_SHAPES,
        FLOAT_DTYPES,
        FP8_DTYPES,
        HALF_DTYPES,
        LARGE_SHAPES,
        MEDIUM_SHAPES,
        SMALL_SHAPES,
        devices,
        dtypes,
        parametrize_gpu,
        shapes,
    )
    from gpucheck.fixtures.benchmark import BenchmarkResult
    from gpucheck.fixtures.gpu import GPUDevice
    from gpucheck.fuzzing import fuzz_shapes

__all__ = [
    "__version__",
    "assert_close",
    "dtypes",
    "shapes",
    "devices",
    "parametrize_gpu",
    "fuzz_shapes",
    "GPUInfo",
    "detect_gpu",
    "gpu_available",
    "gpu_count",
    "BenchmarkResult",
    "GPUDevice",
    "FLOAT_DTYPES",
    "HALF_DTYPES",
    "ALL_DTYPES",
    "FP8_DTYPES",
    "SMALL_SHAPES",
    "MEDIUM_SHAPES",
    "LARGE_SHAPES",
    "EDGE_SHAPES",
]
