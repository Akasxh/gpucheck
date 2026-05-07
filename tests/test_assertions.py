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
# format_mismatch_report — pinned numeric fields (T-10, kills ~30 mutants)
# ---------------------------------------------------------------------------


class TestMismatchReportPinnedNumerics:
    """Lock down the exact numeric values reported by ``format_mismatch_report``.

    These tests target ``assertions/reporting.py`` mutation survivors
    documented in `EVIDENCE/mutator-survivors.md` top-leverage #1. Each test
    fixes hard-coded expected values so that any arithmetic substitution
    (e.g. ``+`` → ``-``, ``np.nanmax`` → ``np.nanmin``, ``unravel_index``
    swap, ``mismatch_pct`` factor flip) breaks the assertion.
    """

    def test_max_abs_error_value_is_pinned(self) -> None:
        """Max absolute error == max of element-wise |actual - expected|.

        Construct ``diff = [0.5, 4.5, 1.5, 3.5]`` so the unique maximum is
        4.5 at flat index 1. The report formats with ``{:.6e}``, yielding
        ``"4.500000e+00"``. Any swap of ``np.nanmax`` → ``np.nanmin`` /
        ``np.nanmean`` / sign flip in the diff computation breaks this.
        """
        actual = np.array([1.0, 0.5, 2.5, 0.5], dtype=np.float64)
        expected = np.array([1.5, 5.0, 1.0, 4.0], dtype=np.float64)
        # Element-wise |a - b| = [0.5, 4.5, 1.5, 3.5] — exact, no FP rounding.

        report = format_mismatch_report(actual, expected, atol=0.0, rtol=0.0)

        # Pinned: the unique maximum 4.5 is rendered as "4.500000e+00".
        assert "4.500000e+00" in report, (
            "Max absolute error value drifted from 4.5; "
            "check `np.nanmax(diff)` and the `{:.6e}` formatter."
        )
        # Mean abs error is (0.5+4.5+1.5+3.5)/4 = 2.5 → "2.500000e+00".
        assert "2.500000e+00" in report, (
            "Mean absolute error value drifted from 2.5; "
            "check `np.nanmean(diff)`."
        )
        # And the table label must be present (kills label-mutation survivors).
        assert "Max absolute error" in report
        assert "Mean absolute error" in report

    def test_mismatch_count_and_location_are_pinned(self) -> None:
        """Mismatch count, percentage, and 2-D max-error location are pinned.

        With ``atol=0, rtol=0``, every element above zero diff is a
        mismatch. The 2-D layout pins the unravel_index call: maximum is
        at row 1, col 2.
        """
        actual = np.zeros((2, 3), dtype=np.float64)
        # Place the unique maximum (5.0) at row=1, col=2.
        expected = np.array(
            [[1.0, 2.0, 3.0],
             [4.0, 0.0, 5.0]],
            dtype=np.float64,
        )
        # Mismatches at 5 of 6 positions (the (1, 1) zero matches).

        report = format_mismatch_report(actual, expected, atol=0.0, rtol=0.0)

        # 5 mismatches out of 6 total → "5 / 6 (83.33%)".
        assert "5 / 6 (83.33%)" in report, (
            "Mismatch count / total / pct drifted; check "
            "`mismatch_count = int(np.sum(mismatch_mask))` and the pct factor (100.0)."
        )
        assert "Mismatch count" in report
        # Location: max diff |0 - 5| = 5 sits at (1, 2). Off-by-one or axis
        # swap in `np.unravel_index` would yield e.g. "(2, 1)" or "(0, 2)".
        assert "(1, 2)" in report, (
            "Location of max error is no longer (1, 2); check "
            "`np.unravel_index(np.nanargmax(diff), diff.shape)`."
        )
        assert "Location of max error" in report
        # And the max abs error itself is 5.0, formatted as "5.000000e+00".
        assert "5.000000e+00" in report

    def test_histogram_present_with_pinned_bucket_and_count(self) -> None:
        """The error histogram appears with a known bucket label and count.

        Construct 3 mismatches all with absolute error == 1e-3. log10(1e-3)
        is -3, so the only bucket is ``[1e-3, 1e-2)`` with count 3.
        """
        actual = np.zeros(3, dtype=np.float64)
        expected = np.full(3, 1e-3, dtype=np.float64)

        report = format_mismatch_report(actual, expected, atol=0.0, rtol=0.0)

        # Histogram panel header — kills any rename of the panel title.
        assert "Error Histogram" in report
        # Bucket label format `[1e{lo:+d}, 1e{hi:+d})`. Mutating `lo` or `hi`
        # by ±1 changes the rendered label.
        assert "[1e-3, 1e-2)" in report, (
            "Histogram bucket label drifted; check `np.floor(np.min(log_vals))` "
            "and `np.ceil(np.max(log_vals))`, plus the `{:+d}` format."
        )
        # All 3 mismatches fall into the single bucket. The histogram line
        # formats as `[1e-3, 1e-2) | ███...███ 3`. Strip ANSI escape codes
        # (Rich emits `\x1b[<digits>m`) and isolate the bucket line.
        import re
        plain_report = re.sub(r"\x1b\[[0-9;]*m", "", report)
        tail_after_bucket = plain_report.split("[1e-3, 1e-2)")[1]
        bucket_line_tail = tail_after_bucket.split("\n", 1)[0]
        digits_only = "".join(ch for ch in bucket_line_tail if ch.isdigit())
        assert digits_only == "3", (
            "Histogram count for the [1e-3, 1e-2) bucket is no longer 3 "
            f"(got digits={digits_only!r}); "
            "check `np.histogram(log_vals, bins=bins)` and the bar-rendering loop."
        )


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


class TestBaseline2xKDimScaling:
    """Regression test for BLOCKER N1.

    The ``baseline_2x`` path used to scale ``atol`` by ``sqrt(k_dim)``
    while every other code path scales by ``sqrt(k_dim/128)``. At
    k_dim=4096 the two diverge by ``sqrt(128) ≈ 11.3×`` — silently
    making ``baseline_2x=True`` 11× more permissive than canonical.
    The expectation: ``baseline_2x`` is exactly 2× the canonical
    tolerance for the same dtype + k_dim.
    """

    @pytest.mark.parametrize("k_dim", [128, 1024, 4096])
    def test_baseline_2x_is_exactly_2x_canonical(self, k_dim: int) -> None:
        # Canonical: dtype-aware + sqrt(k_dim/128) scaling.
        canonical_atol, canonical_rtol = compute_tolerance("float16", k_dim=k_dim)

        # Build two arrays with diff that sits just inside 2× canonical
        # but outside 1× canonical. baseline_2x should pass; canonical
        # should fail.
        # We deliberately craft the diff at 1.5× canonical_atol so it
        # demonstrably fails canonical and passes 2x — and then we
        # also pin the boundary at 2.5× to demonstrate it fails the 2x
        # threshold (proving 2x is not the bug-prone 256x).
        diff = canonical_atol * 1.5
        a = np.array([0.0], dtype=np.float16)
        b = np.array([diff], dtype=np.float16)

        # Sanity: canonical (no baseline_2x) rejects diff > 1× canonical.
        with pytest.raises(AssertionError):
            assert_close(a, b, k_dim=k_dim)

        # baseline_2x must accept diff < 2× canonical at this k_dim.
        assert_close(a, b, k_dim=k_dim, baseline_2x=True)

        # baseline_2x must still reject diff > 2× canonical: pin at 2.5×.
        a2 = np.array([0.0], dtype=np.float16)
        b2 = np.array([canonical_atol * 2.5], dtype=np.float16)
        with pytest.raises(AssertionError):
            assert_close(a2, b2, k_dim=k_dim, baseline_2x=True)
        # rtol unused in this craft (b2 large; expected==0 → rtol leg vanishes).
        assert canonical_rtol >= 0.0

    @pytest.mark.parametrize("k_dim", [128, 1024, 4096])
    def test_baseline_2x_does_not_use_sqrt_k(self, k_dim: int) -> None:
        """At k_dim=4096 the buggy sqrt(K) path was ``11.3× looser`` than
        the canonical sqrt(K/128) path. Pin the contract so the bug
        cannot regress: a diff at ``2.5× canonical_atol`` MUST fail.
        Under the old bug, the threshold would be ``2× sqrt(128) ≈
        22.6×`` of canonical — and 2.5× canonical would erroneously
        pass.
        """
        canonical_atol, _ = compute_tolerance("float16", k_dim=k_dim)
        a = np.array([0.0], dtype=np.float16)
        b = np.array([canonical_atol * 2.5], dtype=np.float16)
        with pytest.raises(AssertionError):
            assert_close(a, b, k_dim=k_dim, baseline_2x=True)


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
