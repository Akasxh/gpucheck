"""Tests for the analysis module: roofline, regression detection."""

from __future__ import annotations

import pytest

from gpucheck.analysis.regression import (
    RegressionResult,
    detect_regression,
    mann_whitney_u,
)
from gpucheck.analysis.roofline import (
    RooflinePoint,
    classify_bottleneck,
    compute_roofline_point,
)

# ---------------------------------------------------------------------------
# classify_bottleneck
# ---------------------------------------------------------------------------


class TestClassifyBottleneck:
    """classify_bottleneck should label compute/memory/balanced correctly."""

    def test_memory_bound(self) -> None:
        # AI well below typical ridge → memory bound
        point = RooflinePoint(
            arithmetic_intensity=0.01,
            achieved_throughput=0.01,
            peak_throughput=float("inf"),
            efficiency_pct=0.0,
            achieved_flops=1e9,
            achieved_bandwidth=100e9,
        )
        assert classify_bottleneck(point) == "memory_bound"

    def test_compute_bound(self) -> None:
        # AI well above typical ridge → compute bound
        point = RooflinePoint(
            arithmetic_intensity=500.0,
            achieved_throughput=5000.0,
            peak_throughput=float("inf"),
            efficiency_pct=0.0,
            achieved_flops=5e12,
            achieved_bandwidth=10e9,
        )
        assert classify_bottleneck(point) == "compute_bound"

    def test_balanced(self) -> None:
        # AI in the middle range
        point = RooflinePoint(
            arithmetic_intensity=10.0,
            achieved_throughput=5000.0,
            peak_throughput=float("inf"),
            efficiency_pct=0.0,
            achieved_flops=5e12,
            achieved_bandwidth=500e9,
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
        # achieved_flops = 1e9 / 0.001 = 1e12
        assert point.achieved_flops == pytest.approx(1e12)
        # achieved_bandwidth = 1e8 / 0.001 = 1e11
        assert point.achieved_bandwidth == pytest.approx(1e11)
        # AI = 1e9 / 1e8 = 10.0
        assert point.arithmetic_intensity == pytest.approx(10.0)

    def test_zero_time(self) -> None:
        point = compute_roofline_point(
            flops=1e9,
            bytes_transferred=1e8,
            elapsed_seconds=0.0,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert point.achieved_flops == 0.0
        assert point.achieved_bandwidth == 0.0

    def test_utilization_properties(self) -> None:
        point = compute_roofline_point(
            flops=5e9,
            bytes_transferred=1e8,
            elapsed_seconds=0.001,
            peak_flops=10e12,
            peak_bandwidth=1e12,
        )
        assert 0.0 <= point.compute_utilization <= 1.0


# ---------------------------------------------------------------------------
# detect_regression
# ---------------------------------------------------------------------------


class TestDetectRegressionSignificant:
    """detect_regression should flag statistically significant slowdowns."""

    def test_clear_regression(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        current = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0, 2.02, 1.98, 2.01, 1.99]
        result = detect_regression(current, baseline, threshold=0.05, min_effect=1.1)
        assert result.is_regression is True
        assert result.effect_size > 1.5
        assert result.pvalue < 0.05

    def test_result_fields(self) -> None:
        baseline = [1.0] * 10
        current = [3.0] * 10
        result = detect_regression(current, baseline)
        assert isinstance(result, RegressionResult)
        assert result.baseline_median == pytest.approx(1.0)
        assert result.current_median == pytest.approx(3.0)


class TestDetectRegressionNoChange:
    """detect_regression should not flag when performance is stable."""

    def test_same_distribution(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        current = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.01, 0.99]
        result = detect_regression(current, baseline)
        assert result.is_regression is False

    def test_slight_improvement(self) -> None:
        baseline = [1.0, 1.1, 0.9, 1.05, 0.95]
        current = [0.9, 0.85, 0.88, 0.92, 0.87]
        result = detect_regression(current, baseline)
        assert result.is_regression is False


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------


class TestMannWhitneyUBasic:
    """mann_whitney_u should compute U statistic and p-value."""

    def test_identical_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [1.0, 2.0, 3.0, 4.0, 5.0]
        u, p = mann_whitney_u(a, b)
        # Identical samples -> large p-value (not significant)
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
