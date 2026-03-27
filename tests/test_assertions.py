"""Tests for the assertion module: assert_close, compute_tolerance, mismatch report."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gpucheck.assertions.close import assert_close
from gpucheck.assertions.reporting import format_mismatch_report
from gpucheck.assertions.tolerances import (
    _DEFAULT_TOLERANCES,
    compute_tolerance,
    tolerance_context,
)
from tests.conftest import make_close_tensors, make_divergent_tensors, make_exact_tensors

# ---------------------------------------------------------------------------
# assert_close
# ---------------------------------------------------------------------------


class TestAssertCloseExactMatch:
    """Exact-match arrays must always pass."""

    def test_1d(self) -> None:
        a, b = make_exact_tensors((128,))
        assert_close(a, b)

    def test_2d(self) -> None:
        a, b = make_exact_tensors((32, 64))
        assert_close(a, b)

    def test_scalar(self) -> None:
        a = np.array(3.14, dtype=np.float32)
        assert_close(a, a.copy())


class TestAssertCloseWithinTolerance:
    """Arrays within the specified tolerance must pass."""

    def test_small_noise_float32(self) -> None:
        a, b = make_close_tensors((64, 64), noise_scale=1e-7)
        assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_explicit_tolerances(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.001, 2.001, 3.001], dtype=np.float32)
        assert_close(a, b, atol=0.01, rtol=0.01)


class TestAssertCloseFailsOutsideTolerance:
    """Arrays far apart must raise AssertionError."""

    def test_large_mismatch(self) -> None:
        a, b = make_divergent_tensors((32, 32))
        with pytest.raises(AssertionError, match="not close"):
            assert_close(a, b)

    def test_tight_tolerance(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([1.1, 2.1], dtype=np.float32)
        with pytest.raises(AssertionError):
            assert_close(a, b, atol=1e-5, rtol=1e-5)


class TestAssertCloseDtypeAwareDefaults:
    """Default tolerances should vary by dtype string."""

    def test_float32_defaults(self) -> None:
        atol, rtol = compute_tolerance("float32")
        assert atol == pytest.approx(1e-4)
        assert rtol == pytest.approx(1e-4)

    def test_float16_defaults(self) -> None:
        atol, rtol = compute_tolerance("float16")
        assert atol == pytest.approx(1e-2)
        assert rtol == pytest.approx(1e-2)

    def test_bfloat16_defaults(self) -> None:
        atol, rtol = compute_tolerance("bfloat16")
        assert atol == pytest.approx(5e-2)
        assert rtol == pytest.approx(5e-2)

    def test_unknown_dtype_falls_back_to_float32(self) -> None:
        atol, rtol = compute_tolerance("weird_dtype_xyz")
        expected_atol, expected_rtol = _DEFAULT_TOLERANCES["float32"]
        assert atol == expected_atol
        assert rtol == expected_rtol


class TestAssertCloseNanHandling:
    """NaN handling: default raises, nan_equal=True allows matching positions."""

    def test_nan_in_actual_raises(self) -> None:
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError, match="NaN"):
            assert_close(a, b)

    def test_nan_in_expected_raises(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="NaN"):
            assert_close(a, b)

    def test_matching_nans_with_flag(self) -> None:
        a = np.array([1.0, float("nan"), 3.0])
        b = np.array([1.0, float("nan"), 3.0])
        assert_close(a, b, nan_equal=True)

    def test_mismatched_nan_positions_with_flag(self) -> None:
        a = np.array([float("nan"), 2.0, 3.0])
        b = np.array([1.0, float("nan"), 3.0])
        with pytest.raises(AssertionError, match="NaN position mismatch"):
            assert_close(a, b, nan_equal=True)


class TestAssertCloseInfHandling:
    """Inf handling: matching infs pass, mismatched infs/signs fail."""

    def test_matching_inf_passes(self) -> None:
        a = np.array([1.0, float("inf"), 3.0])
        b = np.array([1.0, float("inf"), 3.0])
        assert_close(a, b)

    def test_mismatched_inf_position_fails(self) -> None:
        a = np.array([1.0, float("inf"), 3.0])
        b = np.array([1.0, 2.0, 3.0])
        with pytest.raises(AssertionError, match="Inf position mismatch"):
            assert_close(a, b)

    def test_inf_sign_mismatch_fails(self) -> None:
        a = np.array([float("inf")])
        b = np.array([float("-inf")])
        with pytest.raises(AssertionError, match="Inf sign mismatch"):
            assert_close(a, b)

    def test_negative_inf_matches(self) -> None:
        a = np.array([float("-inf"), 1.0])
        b = np.array([float("-inf"), 1.0])
        assert_close(a, b)


# ---------------------------------------------------------------------------
# compute_tolerance
# ---------------------------------------------------------------------------


class TestComputeToleranceBasic:
    """Basic tolerance computation from dtype strings."""

    def test_known_dtypes(self) -> None:
        for dtype_name, (expected_atol, expected_rtol) in _DEFAULT_TOLERANCES.items():
            atol, rtol = compute_tolerance(dtype_name)
            assert atol == expected_atol, f"{dtype_name} atol mismatch"
            assert rtol == expected_rtol, f"{dtype_name} rtol mismatch"

    def test_torch_prefix_stripped(self) -> None:
        atol, rtol = compute_tolerance("torch.float32")
        expected_atol, expected_rtol = _DEFAULT_TOLERANCES["float32"]
        assert atol == expected_atol
        assert rtol == expected_rtol


class TestComputeToleranceWithKScaling:
    """k_dim scaling: atol *= sqrt(k_dim)."""

    def test_k_dim_scaling(self) -> None:
        base_atol, base_rtol = compute_tolerance("float32")
        k = 256
        atol, rtol = compute_tolerance("float32", k_dim=k)
        # Scaling is sqrt(k / 128) per CUTLASS error model
        assert atol == pytest.approx(base_atol * math.sqrt(k / 128.0))
        assert rtol == base_rtol  # rtol unchanged

    def test_k_dim_zero_no_scaling(self) -> None:
        base_atol, _ = compute_tolerance("float32")
        atol, _ = compute_tolerance("float32", k_dim=0)
        assert atol == base_atol

    def test_tolerance_context_overrides(self) -> None:
        with tolerance_context(atol=0.5, rtol=0.5):
            atol, rtol = compute_tolerance("float32")
            assert atol == 0.5
            assert rtol == 0.5
        # Outside context, back to defaults
        atol, rtol = compute_tolerance("float32")
        assert atol == _DEFAULT_TOLERANCES["float32"][0]


# ---------------------------------------------------------------------------
# format_mismatch_report
# ---------------------------------------------------------------------------


class TestMismatchReportFormat:
    """Mismatch report should contain key diagnostic fields."""

    def test_report_contains_statistics(self) -> None:
        actual = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected = np.array([1.5, 2.5, 3.5], dtype=np.float32)
        report = format_mismatch_report(actual, expected, atol=1e-5, rtol=1e-5)

        assert "Max absolute error" in report
        assert "Mean absolute error" in report
        assert "Max relative error" in report
        assert "Mismatch count" in report
        assert "Tolerances used" in report

    def test_report_is_string(self) -> None:
        a = np.array([1.0])
        b = np.array([2.0])
        report = format_mismatch_report(a, b, atol=0.0, rtol=0.0)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_report_with_nan_inf(self) -> None:
        actual = np.array([float("nan"), float("inf"), 1.0])
        expected = np.array([1.0, 2.0, 3.0])
        report = format_mismatch_report(actual, expected, atol=1e-5, rtol=1e-5)
        assert "NaN" in report
        assert "Inf" in report


# ---------------------------------------------------------------------------
# baseline_2x tolerance doubling
# ---------------------------------------------------------------------------


class TestBaseline2xDoublesTolerance:
    """baseline_2x=True should double the dtype-default tolerances."""

    def test_baseline_2x_passes_with_doubled_tolerance(self) -> None:
        """Value within 2x tolerance but outside 1x should pass with baseline_2x."""
        base_atol, base_rtol = compute_tolerance("float32")
        # Create arrays with difference just above 1x tolerance but below 2x
        a = np.array([0.0], dtype=np.float32)
        b = np.array([base_atol * 1.5], dtype=np.float32)
        # Should fail without baseline_2x
        with pytest.raises(AssertionError):
            assert_close(a, b)
        # Should pass with baseline_2x
        assert_close(a, b, baseline_2x=True)

    def test_baseline_2x_fails_beyond_doubled_tolerance(self) -> None:
        """Value beyond 2x tolerance should still fail with baseline_2x."""
        base_atol, _ = compute_tolerance("float32")
        a = np.array([0.0], dtype=np.float32)
        b = np.array([base_atol * 2.5], dtype=np.float32)
        with pytest.raises(AssertionError):
            assert_close(a, b, baseline_2x=True)


# ---------------------------------------------------------------------------
# Mixed-precision _resolve_dtype
# ---------------------------------------------------------------------------


class TestMixedPrecisionDtype:
    """Mixed-precision comparisons should use the lower-precision dtype's tolerance."""

    def test_fp16_fp32_uses_fp16_tolerance_order1(self) -> None:
        """fp16 actual + fp32 expected -> fp16 tolerance (wider)."""
        fp16_atol, _ = compute_tolerance("float16")
        fp32_atol, _ = compute_tolerance("float32")
        # Difference between fp16 and fp32 tolerances is large (1e-2 vs 1e-4)
        a = np.array([0.0], dtype=np.float16)
        b = np.array([fp32_atol * 5], dtype=np.float32)  # above fp32 tol, below fp16 tol
        # Should pass because fp16 tolerance (1e-2) is used, not fp32 (1e-4)
        assert_close(a, b)

    def test_fp32_fp16_uses_fp16_tolerance_order2(self) -> None:
        """fp32 actual + fp16 expected -> fp16 tolerance (wider), same as reversed order."""
        fp32_atol, _ = compute_tolerance("float32")
        a = np.array([0.0], dtype=np.float32)
        b = np.array([fp32_atol * 5], dtype=np.float16)  # above fp32 tol, below fp16 tol
        # Should also pass — order should not matter
        assert_close(a, b)

    def test_both_orders_produce_same_result(self) -> None:
        """Swapping actual/expected dtypes should not change pass/fail outcome."""
        val = np.float32(5e-3)  # between fp32 tol (1e-4) and fp16 tol (1e-2)
        a16 = np.array([0.0], dtype=np.float16)
        b32 = np.array([val], dtype=np.float32)
        a32 = np.array([0.0], dtype=np.float32)
        b16 = np.array([val], dtype=np.float16)
        # Both orders should pass (fp16 tolerance used in both cases)
        assert_close(a16, b32)
        assert_close(a32, b16)
