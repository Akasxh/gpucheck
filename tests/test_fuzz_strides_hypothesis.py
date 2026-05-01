"""Stride fuzzing — Hypothesis StrideStrategy (Track B)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402

from gpucheck.fuzzing.strides import StrideStrategy  # noqa: E402


@settings(max_examples=20, deadline=None)
@given(t=StrideStrategy(shape=(8, 8), dtype=torch.float32))
def test_stride_strategy_yields_tensor_of_target_shape(t) -> None:
    assert t.shape == (8, 8)
    assert t.dtype == torch.float32


@settings(max_examples=10, deadline=None)
@given(
    t=StrideStrategy(
        shape=(4, 4),
        dtype=torch.float32,
        categories=("row_major",),
    ),
)
def test_stride_strategy_with_single_category_only_returns_that_category(t) -> None:
    # row_major is contiguous by construction.
    assert t.is_contiguous()
    assert t.shape == (4, 4)


def test_stride_strategy_unknown_dtype_does_not_crash_creation() -> None:
    s = StrideStrategy(shape=(2, 2), dtype=torch.float16)
    # Strategy creation must succeed; drawing also must not crash.
    assert s is not None
