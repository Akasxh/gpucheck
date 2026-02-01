"""Fuzzing module — shape generation, edge-case inputs, and Hypothesis strategies."""

from __future__ import annotations

from gpucheck.fuzzing.inputs import edge_inputs, mixed_inputs, random_inputs
from gpucheck.fuzzing.shapes import ShapeStrategy, fuzz_shapes

__all__ = [
    "fuzz_shapes",
    "random_inputs",
    "edge_inputs",
    "mixed_inputs",
    "ShapeStrategy",
]
