"""Tests for the analysis module: roofline, regression detection."""

from __future__ import annotations

import pytest

from gpucheck.analysis.roofline import (
    RooflinePoint,
    classify_bottleneck,
    compute_roofline_point,
)
from gpucheck.analysis.regression import (
    RegressionResult,
    detect_regression,
    mann_whitney_u,
)


# ---------------------------------------------------------------------------
# classify_bottleneck
# ---------------------------------------------------------------------------


class TestClassifyBottleneck:
    """classify_bottleneck should label compute/memory/balanced correctly."""

    def test_memory_bound(self) -> None:
        # AI well below ridge point → memory bound
        point = RooflinePoint(
            flops=1e9,
            bandwidth=100e9,
            arithmetic_intensity=0.01,  # very low
            peak_flops=10e12,
            peak_bandwidth=1e12,
            ridge_point=10.0,  # peak_flops / peak_bw
        )
        assert classify_bottleneck(point) == "memory"

    def test_compute_bound(self) -> None:
        # AI well above ridge point → compute bound
        point = RooflinePoint(
            flops=5e12,
            bandwidth=10e9,
            arithmetic_intensity=500.0,
            peak_flops=10e12,
            peak_bandwidth=1e12,
            ridge_point=10.0,
        )
        assert classify_bottleneck(point) == "compute"

    def test_balanced(self) -> None:
        # AI right at the ridge point
        point = RooflinePoint(
            flops=5e12,
            bandwidth=500e9,
            arithmetic_intensity=10.0,
            peak_flops=10e12,
            peak_bandwidth=1e12,
            ridge_point=10.0,
        )
        assert classify_bottleneck(point) == "balanced"


# ---------------------------------------------------------------------------
# compute_roofline_point
# ---------------------------------------------------------------------------


class TestComputeRooflinePoint:
    """compute_roofline_point should calculate correct metrics."""

    def test_basic_computation(self) -> None:
        point = compute_roofline_point(
            flops=1e9,
            bytes_transferred=1e8,
            elapsed_seconds=0.001,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert point.flops == pytest.approx(1e12)  # 1e9 / 0.001
        assert point.bandwidth == pytest.approx(1e11)  # 1e8 / 0.001
        assert point.arithmetic_intensity == pytest.approx(10.0)  # 1e9 / 1e8
        assert point.ridge_point == pytest.approx(10.0)  # 10e12 / 1e12

    def test_zero_time(self) -> None:
        point = compute_roofline_point(
            flops=1e9,
            bytes_transferred=1e8,
            elapsed_seconds=0.0,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert point.flops == 0.0
        assert point.bandwidth == 0.0

    def test_zero_bytes(self) -> None:
        point = compute_roofline_point(
            flops=1e9,
            bytes_transferred=0.0,
            elapsed_seconds=0.001,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert point.arithmetic_intensity == float("inf")

    def test_utilization_properties(self) -> None:
        point = compute_roofline_point(
            flops=5e9,
            bytes_transferred=1e8,
            elapsed_seconds=0.001,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert 0.0 <= point.compute_utilization <= 1.0
        assert 0.0 <= point.bandwidth_utilization <= 1.0


# ---------------------------------------------------------------------------
# detect_regression
# ---------------------------------------------------------------------------


class TestDetectRegressionSignificant:
    """detect_regression should flag statistically significant slowdowns."""

    def test_clear_regression(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        current = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 2.02, 1.98, 2.01, 1.99]
        result = detect_regression(baseline, current, threshold=0.05, min_effect=1.1)
        assert result.is_regression is True
        assert result.effect_size > 1.5
        assert result.p_value < 0.05

    def test_result_fields(self) -> None:
        baseline = [1.0] * 10
        current = [3.0] * 10
        result = detect_regression(baseline, current)
        assert isinstance(result, RegressionResult)
        assert result.baseline_median == pytest.approx(1.0)
        assert result.current_median == pytest.approx(3.0)
        assert result.method == "mann_whitney_u"


class TestDetectRegressionNoChange:
    """detect_regression should not flag when performance is stable."""

    def test_same_distribution(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        current = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        result = detect_regression(baseline, current)
        assert result.is_regression is False
        assert result.effect_size == pytest.approx(1.0, abs=0.1)

    def test_slight_improvement(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95]
        current = [0.9, 0.85, 0.88, 0.92, 0.87]
        result = detect_regression(baseline, current)
        assert result.is_regression is False
        assert result.effect_size < 1.0


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------


class TestMannWhitneyUBasic:
    """mann_whitney_u should compute U statistic and p-value."""

    def test_identical_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        u, p = mann_whitney_u(a, b)
        # Identical samples → large p-value (not significant)
        assert p > 0.05

    def test_clearly_different_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        b = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        u, p = mann_whitney_u(a, b)
        assert p < 0.05

    def test_empty_sample(self) -> None:
        u, p = mann_whitney_u([], [1.0, 2.0])
        assert u == 0.0
        assert p == 1.0

    def test_returns_tuple(self) -> None:
        u, p = mann_whitney_u([1.0, 2.0], [3.0, 4.0])
        assert isinstance(u, float)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
