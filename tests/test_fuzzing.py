"""Tests for the fuzzing module: fuzz_shapes, ShapeStrategy."""

from __future__ import annotations

from typing import Any

import pytest

from gpucheck.fuzzing.shapes import (
    PRIMES,
    TILE_SIZES,
    ShapeStrategy,
    _degenerate_shapes,
    _non_tile_aligned_shapes,
    fuzz_shapes,
)


class TestFuzzShapesGeneratesEdgeCases:
    """fuzz_shapes should include shapes that commonly trigger kernel bugs."""

    def test_returns_requested_count(self) -> None:
        result = fuzz_shapes(ndim=2, n=30, seed=0)
        assert len(result) == 30

    def test_contains_degenerate_shapes(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=0)
        # Should include a shape with a zero dimension
        assert any(0 in s for s in result)
        # Should include all-ones shape
        assert (1, 1) in result

    def test_contains_non_aligned(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=0)
        # Should include shapes not divisible by any tile size
        non_aligned = [
            s for s in result
            if all(d % t != 0 for d in s if d > 0 for t in TILE_SIZES)
        ]
        assert len(non_aligned) > 0

    def test_all_tuples(self) -> None:
        result = fuzz_shapes(ndim=3, n=20, seed=0)
        for s in result:
            assert isinstance(s, tuple)
            assert len(s) == 3

    def test_deterministic_with_seed(self) -> None:
        r1 = fuzz_shapes(ndim=2, n=50, seed=42)
        r2 = fuzz_shapes(ndim=2, n=50, seed=42)
        assert r1 == r2


class TestFuzzShapesNonAligned:
    """Non-tile-aligned shapes should be explicitly generated."""

    def test_non_aligned_shapes_generated(self) -> None:
        shapes = _non_tile_aligned_shapes(ndim=2, max_size=256)
        assert len(shapes) > 0
        # Each shape should have dimension not in TILE_SIZES
        for s in shapes:
            assert any(d not in TILE_SIZES for d in s)


class TestFuzzShapesDegenerate:
    """Degenerate shapes with zeros and ones."""

    def test_1d_degenerate(self) -> None:
        shapes = _degenerate_shapes(1)
        assert (0,) in shapes
        assert (1,) in shapes

    def test_2d_degenerate(self) -> None:
        shapes = _degenerate_shapes(2)
        # Should have zero in each position
        assert (0, 16) in shapes
        assert (16, 0) in shapes
        # Should have all-ones
        assert (1, 1) in shapes

    def test_0d_degenerate(self) -> None:
        shapes = _degenerate_shapes(0)
        assert shapes == [()]


class TestRandomInputsShape:
    """fuzz_shapes respects ndim and size constraints."""

    def test_ndim_respected(self) -> None:
        for ndim in (1, 2, 3, 4):
            result = fuzz_shapes(ndim=ndim, n=10, seed=0)
            for s in result:
                assert len(s) == ndim

    def test_max_size_respected(self) -> None:
        max_size = 128
        result = fuzz_shapes(ndim=2, max_size=max_size, n=30, seed=0)
        for s in result:
            for d in s:
                assert d <= max_size


class TestEdgeInputsContainsNanInf:
    """Verify the fuzzer produces shapes that would exercise NaN/Inf paths.

    The fuzzer generates shapes, not values — but degenerate shapes
    (zeros, ones) are the ones most likely to produce NaN/Inf in kernels
    (e.g., division by zero, empty reductions).
    """

    def test_zero_dim_shapes_present(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=0)
        zero_shapes = [s for s in result if any(d == 0 for d in s)]
        assert len(zero_shapes) > 0

    def test_unit_shapes_present(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=0)
        assert (1, 1) in result


class TestShapeStrategyShrinks:
    """ShapeStrategy should work with hypothesis when available."""

    def test_strategy_produces_valid_shapes(self) -> None:
        pytest.importorskip("hypothesis")
        strat = ShapeStrategy(ndim=2, min_size=1, max_size=64)
        example = strat.example()
        assert isinstance(example, tuple)
        assert len(example) == 2
        assert all(isinstance(d, int) for d in example)

    def test_strategy_repr(self) -> None:
        strat = ShapeStrategy(ndim=3, min_size=1, max_size=512)
        r = repr(strat)
        assert "ndim=3" in r
        assert "max_size=512" in r

    def test_strategy_with_hypothesis_given(self) -> None:
        hypothesis = pytest.importorskip("hypothesis")
        from hypothesis import given, settings

        strat = ShapeStrategy(ndim=2, min_size=1, max_size=64)

        @settings(max_examples=10, deadline=None)
        @given(shape=strat)
        def _check(shape: tuple[int, ...]) -> None:
            assert len(shape) == 2
            assert all(isinstance(d, int) for d in shape)

        _check()
