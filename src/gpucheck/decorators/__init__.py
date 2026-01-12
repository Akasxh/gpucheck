"""Parametrize decorators for GPU kernel testing."""

from __future__ import annotations

from gpucheck.decorators.devices import devices
from gpucheck.decorators.dtypes import (
    ALL_DTYPES,
    FLOAT_DTYPES,
    FP8_DTYPES,
    HALF_DTYPES,
    dtypes,
)
from gpucheck.decorators.parametrize import parametrize_gpu
from gpucheck.decorators.shapes import (
    EDGE_SHAPES,
    LARGE_SHAPES,
    MEDIUM_SHAPES,
    SMALL_SHAPES,
    shapes,
)

__all__ = [
    "dtypes",
    "shapes",
    "devices",
    "parametrize_gpu",
    "FLOAT_DTYPES",
    "HALF_DTYPES",
    "ALL_DTYPES",
    "FP8_DTYPES",
    "SMALL_SHAPES",
    "MEDIUM_SHAPES",
    "LARGE_SHAPES",
    "EDGE_SHAPES",
]
