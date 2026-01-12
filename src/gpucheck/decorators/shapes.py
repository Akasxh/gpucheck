"""Parametrize tests across tensor shapes."""

from __future__ import annotations

from typing import Any, Callable

import pytest

Shape = tuple[int, ...]

# ---------------------------------------------------------------------------
# Predefined groups
# ---------------------------------------------------------------------------

SMALL_SHAPES: tuple[Shape, ...] = (
    (32,),
    (64, 64),
    (128,),
    (32, 32),
)

MEDIUM_SHAPES: tuple[Shape, ...] = (
    (256, 256),
    (512, 512),
    (1024,),
    (128, 256),
)

LARGE_SHAPES: tuple[Shape, ...] = (
    (2048, 2048),
    (4096, 4096),
    (8192,),
    (1024, 1024, 4),
)

EDGE_SHAPES: tuple[Shape, ...] = (
    (1, 1),
    (1, 128),
    (128, 1),
    (7, 13),
    (127, 127),
    (1, 1, 1),
    (0, 128),
)


def _shape_id(s: Shape) -> str:
    """Generate a clean test ID for a shape, e.g. '128x256'."""
    return "x".join(str(d) for d in s)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def shapes(*shape_args: Shape) -> Callable[..., Any]:
    """Parametrize a test across tensor shapes.

    Accepts shape tuples directly or spread a predefined group.

    Examples::

        @shapes((128, 128), (256, 256))
        def test_relu(shape): ...

        @shapes(*EDGE_SHAPES)
        def test_edge_cases(shape): ...
    """
    ids = [_shape_id(s) for s in shape_args]
    return pytest.mark.parametrize("shape", list(shape_args), ids=ids)


__all__ = [
    "shapes",
    "SMALL_SHAPES",
    "MEDIUM_SHAPES",
    "LARGE_SHAPES",
    "EDGE_SHAPES",
]
