"""Tests that `_to_numpy` handles non-contiguous torch tensors.

Stride-fuzzed / sliced / transposed tensors break ``.numpy()`` on torch <2.1
with a ``RuntimeError`` ("input array is not C-contiguous").  T-02 adds
``.contiguous()`` to the slow path; these tests pin the fix.

Source: security-postmerge PM-4; planner T-02.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

torch = pytest.importorskip("torch")

from gpucheck.assertions.close import _to_numpy, assert_close  # noqa: E402


# Three canonical non-contiguous stride patterns.
def _slice_pattern() -> torch.Tensor:
    """Sliced row of a 2D tensor — non-contiguous when row stride != col."""
    base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    sliced = base[:, ::2]  # stride 2 columns: not C-contiguous
    assert not sliced.is_contiguous()
    return sliced


def _transpose_pattern() -> torch.Tensor:
    """Transposed 2D tensor — non-contiguous (row/col strides swapped)."""
    base = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    t = base.t()
    assert not t.is_contiguous()
    return t


def _broadcast_pattern() -> torch.Tensor:
    """Broadcast view via ``.expand`` — has zero strides on broadcast dims."""
    base = torch.arange(8, dtype=torch.float32).reshape(8, 1)
    expanded = base.expand(8, 4)
    assert not expanded.is_contiguous()
    return expanded


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("slice", _slice_pattern),
        ("transpose", _transpose_pattern),
        ("broadcast", _broadcast_pattern),
    ],
)
def test_to_numpy_handles_non_contiguous_input(
    name: str, factory: Callable[[], torch.Tensor]
) -> None:
    """`_to_numpy` must not raise on stride-fuzzed inputs."""
    tensor = factory()
    arr = _to_numpy(tensor)
    assert isinstance(arr, np.ndarray)
    # Shape preserved through the conversion.
    assert arr.shape == tuple(tensor.shape), (
        f"{name}: shape mismatch — got {arr.shape}, expected {tuple(tensor.shape)}"
    )
    # Values preserved through the conversion.
    np.testing.assert_array_equal(arr, tensor.contiguous().numpy())


@pytest.mark.parametrize(
    ("name", "factory"),
    [
        ("slice", _slice_pattern),
        ("transpose", _transpose_pattern),
        ("broadcast", _broadcast_pattern),
    ],
)
def test_assert_close_handles_non_contiguous_input(
    name: str, factory: Callable[[], torch.Tensor]
) -> None:
    """End-to-end: `assert_close` should not raise RuntimeError when comparing
    non-contiguous tensors against their contiguous equivalents."""
    a = factory()
    b = a.contiguous().clone()
    # No RuntimeError: comparison flows through `_to_numpy`'s slow path.
    assert_close(a, b)
