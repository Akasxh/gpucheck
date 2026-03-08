"""Shape fuzzing with gpucheck.

Demonstrates using fuzz_shapes to stress-test kernels with adversarial shapes.
Run with: pytest examples/shape_fuzzing_example.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from gpucheck.fuzzing.shapes import ShapeStrategy, fuzz_shapes


# ---------------------------------------------------------------------------
# Deterministic fuzzing: pre-generated shape list
# ---------------------------------------------------------------------------

FUZZED_SHAPES = fuzz_shapes(ndim=2, n=20, seed=42)


@pytest.mark.parametrize("shape", FUZZED_SHAPES, ids=[f"{s[0]}x{s[1]}" for s in FUZZED_SHAPES])
def test_softmax_fuzzed_shapes(shape: tuple[int, ...]) -> None:
    """Test numpy softmax across fuzzed shapes."""
    if any(d == 0 for d in shape):
        pytest.skip("empty tensor")

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)

    # Numerically stable softmax along last axis
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    result = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Softmax outputs should sum to 1 along last axis
    row_sums = np.sum(result, axis=-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    # All values should be in [0, 1]
    assert np.all(result >= 0)
    assert np.all(result <= 1)


# ---------------------------------------------------------------------------
# Hypothesis-based fuzzing (if hypothesis is installed)
# ---------------------------------------------------------------------------


def test_reduction_with_hypothesis_shapes() -> None:
    """Test a reduction op with hypothesis-generated shapes."""
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import given, settings

    strat = ShapeStrategy(ndim=2, min_size=1, max_size=256)

    @settings(max_examples=20, deadline=None)
    @given(shape=strat)
    def _inner(shape: tuple[int, ...]) -> None:
        x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
        # Sum reduction should be deterministic
        s1 = np.sum(x)
        s2 = np.sum(x)
        assert s1 == s2

    _inner()


# ---------------------------------------------------------------------------
# 1D fuzzing
# ---------------------------------------------------------------------------

FUZZED_1D = fuzz_shapes(ndim=1, n=15, seed=99)


@pytest.mark.parametrize("shape", FUZZED_1D, ids=[str(s[0]) for s in FUZZED_1D])
def test_cumsum_1d_fuzzed(shape: tuple[int, ...]) -> None:
    """Test cumulative sum across fuzzed 1D shapes."""
    if shape[0] == 0:
        pytest.skip("empty tensor")

    x = np.random.default_rng(0).standard_normal(shape).astype(np.float64)
    result = np.cumsum(x)

    # Last element of cumsum should equal total sum
    assert result[-1] == pytest.approx(np.sum(x), rel=1e-10)

    # cumsum should be monotonically non-decreasing for positive input
    pos = np.abs(x)
    cs = np.cumsum(pos)
    assert np.all(np.diff(cs) >= 0)
