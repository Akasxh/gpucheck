"""Fuzzing module — shape generation, edge-case inputs, and Hypothesis strategies."""

from __future__ import annotations

import importlib
from typing import Any

from gpucheck.fuzzing.inputs import edge_inputs, mixed_inputs, random_inputs
from gpucheck.fuzzing.shapes import ShapeStrategy, fuzz_shapes

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "gpu_shapes": ("gpucheck.fuzzing.strategies", "gpu_shapes"),
    "gpu_tensors": ("gpucheck.fuzzing.strategies", "gpu_tensors"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'gpucheck.fuzzing' has no attribute {name!r}")


__all__ = [
    "fuzz_shapes",
    "random_inputs",
    "edge_inputs",
    "mixed_inputs",
    "ShapeStrategy",
    "gpu_shapes",
    "gpu_tensors",
]
