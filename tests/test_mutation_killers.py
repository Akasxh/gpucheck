"""Mutation-killer leverage tests targeting v1.1 mutmut survivors.

This module is the home of the three high-leverage tests called out by the
v1.1 audit (`.claude/teams/audit/v1.1/EVIDENCE/mutator-survivors.md` and the
companion summary). They are deliberately segregated from the broader
`test_assertions.py` /  `test_plugin_tomli.py` suites so that the kill-rate
ratchet is auditable as a single artifact.

The three leverage clusters:

1. **Pinned numeric fields in ``format_mismatch_report``** — kills ~30
   surviving reporting.py mutants whose existing tests only do substring
   checks (e.g. ``"NaN" in report`` happily matches ``"XXNaNXX"``).
2. **Round-trip the pyproject.toml tolerance config loader** — covers
   ``apply_config_tolerances`` / ``tolerances_from_config`` /
   ``reset_config_tolerances`` end to end (~17 mutants previously had no
   direct test).
3. **Hard-coded parametrize over ``_DEFAULT_TOLERANCES``** — kills ~12
   dict-value mutants currently shielded by the tautological
   ``for k, v in _DEFAULT_TOLERANCES.items(): assert compute_tolerance(k) == v``
   loop (which always passes, regardless of the mutated value).

Total: ~10 new test cases, ~59 mutants killed.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from gpucheck.assertions.reporting import format_mismatch_report
from gpucheck.assertions.tolerances import (
    _DEFAULT_TOLERANCES,
    apply_config_tolerances,
    compute_tolerance,
    reset_config_tolerances,
    tolerances_from_config,
)

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    """Remove Rich's ANSI styling so substring checks compare plain text."""
    return _ANSI_RE.sub("", s)


# ---------------------------------------------------------------------------
# Test 1 — pin numeric fields in format_mismatch_report
# ---------------------------------------------------------------------------


class TestFormatMismatchReportPinnedFields:
    """Pin every numeric field rendered by ``format_mismatch_report``.

    The fixture is the audit-recommended canonical pattern:
    ``actual = zeros((4, 4))``, ``expected = eye(4)``. Diff is the identity:
    1.0 on the diagonal, 0.0 elsewhere. With ``atol=0, rtol=0`` every
    diagonal cell is a mismatch and the off-diagonal cells are exact
    matches. This pins:

    - max abs error VALUE (``"1.000000e+00"``, not just the substring
      ``"Max absolute error"``)
    - mean abs error VALUE (``mean([1,1,1,1,0,...]) = 4/16 = 0.25`` →
      ``"2.500000e-01"``)
    - mismatch count + percentage (``"4 / 16 (25.00%)"``)
    - location of max error (``"(0, 0)"`` — the first diagonal entry, the
      one ``np.argmax`` picks when ties exist)
    - the histogram panel header AND the bucket label/count
      (``[1e+0, 1e+1)`` with count 4)
    - the tolerances-used line (``"atol=0.00e+00, rtol=0.00e+00"``)

    Each assertion picks values that uniquely identify the canonical
    formula, so any arithmetic substitution (``+`` → ``-``, ``np.nanmax``
    → ``np.nanmin``, ``unravel_index`` axis swap, ``100.0 *`` → ``101.0 *``,
    etc.) trips at least one assertion.
    """

    @pytest.fixture()
    def canonical_report(self) -> str:
        """Build the canonical ``zeros / eye`` report once per test."""
        actual = np.zeros((4, 4), dtype=np.float64)
        expected = np.eye(4, dtype=np.float64)
        return format_mismatch_report(actual, expected, atol=0.0, rtol=0.0)

    def test_max_abs_error_value(self, canonical_report: str) -> None:
        """Max absolute error == 1.0, formatted as ``1.000000e+00``."""
        plain = _strip_ansi(canonical_report)
        # Row label must literally appear (kills XX-wrapping of the label).
        assert "Max absolute error" in plain
        # And so must the rendered value (kills nanmax → nanmin / nanmean,
        # diff sign flip, formatter mutations).
        assert "1.000000e+00" in plain, (
            "Max absolute error value drifted from 1.0; check `np.nanmax(diff)` "
            "and the `{:.6e}` formatter."
        )

    def test_mean_abs_error_value(self, canonical_report: str) -> None:
        """Mean abs error == sum(diff)/16 = 4/16 = 0.25 → ``2.500000e-01``."""
        plain = _strip_ansi(canonical_report)
        assert "Mean absolute error" in plain
        assert "2.500000e-01" in plain, (
            "Mean absolute error drifted from 0.25; check `np.nanmean(diff)`."
        )

    def test_mismatch_count_and_total_and_pct(self, canonical_report: str) -> None:
        """4 mismatches out of 16 cells → ``4 / 16 (25.00%)``."""
        plain = _strip_ansi(canonical_report)
        assert "Mismatch count" in plain
        # Pinning the full triple guards against:
        #   - mismatch_count = int(np.sum(mask)) → wrong sum
        #   - total = actual.size → wrong total
        #   - 100.0 * mismatch / total → constant flip (e.g. 101.0)
        assert "4 / 16 (25.00%)" in plain, (
            "Mismatch count / total / pct drifted from `4 / 16 (25.00%)`; "
            "check `int(np.sum(mismatch_mask))`, `actual.size`, and the 100.0 factor."
        )

    def test_location_of_max_error(self, canonical_report: str) -> None:
        """``argmax(diff)`` is flat-index 0; unravel for (4,4) is (0, 0)."""
        plain = _strip_ansi(canonical_report)
        assert "Location of max error" in plain
        # `str((0, 0))` is "(0, 0)". An axis swap would be the same here
        # (symmetric), so we additionally pin a non-symmetric case below.
        assert "(0, 0)" in plain, (
            "Location-of-max-error tuple drifted; check "
            "`np.unravel_index(np.nanargmax(diff), diff.shape)`."
        )

    def test_location_axis_order_is_row_then_col(self) -> None:
        """A non-symmetric ``actual``/``expected`` proves the (i, j) order.

        Place the unique maximum at row 1, col 3 of a 2x4 array. If
        ``np.unravel_index`` is replaced with a transpose / axis swap the
        rendered tuple becomes ``(3, 1)`` instead of ``(1, 3)``.
        """
        actual = np.zeros((2, 4), dtype=np.float64)
        expected = np.zeros((2, 4), dtype=np.float64)
        expected[1, 3] = 7.0  # unique global max
        report = _strip_ansi(format_mismatch_report(actual, expected, atol=0.0, rtol=0.0))
        assert "(1, 3)" in report
        assert "(3, 1)" not in report  # explicit anti-axis-swap guard

    def test_tolerances_line_renders_inputs(self, canonical_report: str) -> None:
        """``atol=0.00e+00, rtol=0.00e+00`` — kills tolerance-passthrough mutations."""
        plain = _strip_ansi(canonical_report)
        assert "atol=0.00e+00, rtol=0.00e+00" in plain

    def test_histogram_panel_present_for_divergent_inputs(
        self, canonical_report: str
    ) -> None:
        """Histogram panel appears whenever there is at least one finite mismatch.

        Kills the ``histogram = _error_histogram(...) → None`` mutation as
        well as label-rename mutations on the panel title.
        """
        plain = _strip_ansi(canonical_report)
        assert "Error Histogram" in plain

    def test_histogram_bucket_label_and_count(self, canonical_report: str) -> None:
        """All 4 mismatches sit in the ``[1e+0, 1e+1)`` bucket with count 4.

        log10(1.0) = 0, so floor(min)=0, ceil(max)=0 → lo==hi → hi=1 →
        bins=[0, 1] → exactly one bucket. Count is 4. The bucket label
        format ``[1e{lo:+d}, 1e{hi:+d})`` requires the explicit ``+``
        sign, so dropping ``:+d`` flips the assertion.
        """
        plain = _strip_ansi(canonical_report)
        assert "[1e+0, 1e+1)" in plain, (
            "Histogram bucket label drifted; check log10 floor/ceil and the "
            "`{:+d}` width specifier."
        )
        # And the count column. Locate the bucket line; the trailing integer
        # is the count rendered by the bar-printing loop.
        bucket_tail = plain.split("[1e+0, 1e+1)")[1].split("\n", 1)[0]
        digits = "".join(ch for ch in bucket_tail if ch.isdigit())
        assert digits == "4", (
            f"Histogram bucket count for [1e+0, 1e+1) is no longer 4 "
            f"(got digits={digits!r}); check `np.histogram(log_vals, bins=bins)`."
        )

    def test_no_histogram_panel_when_arrays_match(self) -> None:
        """When mismatch_mask is empty, ``_error_histogram`` returns ``""``,
        which is falsy, and the panel must NOT be rendered.

        Kills histogram-presence mutations (the ``if histogram:`` guard
        and the empty-mismatched fallback string).
        """
        a = np.zeros(8, dtype=np.float64)
        report = _strip_ansi(format_mismatch_report(a, a.copy(), atol=0.0, rtol=0.0))
        assert "Error Histogram" not in report


# ---------------------------------------------------------------------------
# Test 2 — round-trip the tolerance config loader
# ---------------------------------------------------------------------------


class TestToleranceConfigRoundTrip:
    """End-to-end coverage of the pyproject.toml tolerance overlay path.

    Targets ``apply_config_tolerances`` / ``tolerances_from_config`` /
    ``reset_config_tolerances`` (~17 surviving mutants). All previous
    tests for these helpers were either indirect (the
    ``test_plugin_tomli.py`` smoke test only covers the *plugin* loader,
    not the parser's edge cases) or outright absent.
    """

    def test_apply_config_tolerances_round_trip(self) -> None:
        """Apply → compute → reset → compute produces the expected values.

        Pins:
          - the config loader actually populates ``_config_overrides``
            (mutating it to a no-op would leave defaults in place)
          - ``compute_tolerance`` consults the overlay before defaults
          - ``reset_config_tolerances`` empties the overlay
        """
        cfg = {
            "tool": {
                "gpucheck": {
                    "tolerances": {
                        "float16": {"atol": 2e-3, "rtol": 2e-3},
                        "float32": {"atol": 5e-4, "rtol": 5e-4},
                    }
                }
            }
        }
        try:
            apply_config_tolerances(cfg)
            assert compute_tolerance("float16") == (2e-3, 2e-3)
            assert compute_tolerance("float32") == (5e-4, 5e-4)
        finally:
            reset_config_tolerances()
        # After reset, the defaults are reachable again.
        assert compute_tolerance("float16") == _DEFAULT_TOLERANCES["float16"]
        assert compute_tolerance("float32") == _DEFAULT_TOLERANCES["float32"]

    def test_apply_config_tolerances_is_no_op_for_empty_section(self) -> None:
        """``[tool.gpucheck.tolerances]`` entirely absent → defaults preserved.

        Kills the ``return None`` ↔ ``return result`` and similar guard
        mutations on ``tolerances_from_config``.
        """
        try:
            apply_config_tolerances({"tool": {"gpucheck": {}}})
            apply_config_tolerances({})
            assert compute_tolerance("float16") == _DEFAULT_TOLERANCES["float16"]
        finally:
            reset_config_tolerances()

    def test_tolerances_from_config_returns_none_when_absent(self) -> None:
        """No ``tolerances`` key → ``None`` (not an empty dict).

        Kills ``return None`` → ``return {}`` and ``or None`` → ``and None``
        mutations at the end of the function.
        """
        assert tolerances_from_config({}) is None
        assert tolerances_from_config({"tool": {}}) is None
        assert tolerances_from_config({"tool": {"gpucheck": {}}}) is None
        assert (
            tolerances_from_config({"tool": {"gpucheck": {"tolerances": {}}}}) is None
        )

    def test_tolerances_from_config_parses_overrides(self) -> None:
        """Every code path of the parser:

        - well-formed entry → parsed and present
        - malformed entry (missing ``rtol``) → silently skipped
        - non-dict entry → silently skipped
        - multiple well-formed entries → all applied
        - non-empty result → returned as a dict, not ``None``
        """
        cfg = {
            "tool": {
                "gpucheck": {
                    "tolerances": {
                        "float16": {"atol": 2e-3, "rtol": 3e-3},
                        "float32": {"atol": 5e-4, "rtol": 5e-4},
                        "missing_rtol": {"atol": 1.0},
                        "missing_atol": {"rtol": 1.0},
                        "not_a_dict": "garbage",
                    }
                }
            }
        }
        out = tolerances_from_config(cfg)
        # Non-empty: every well-formed entry is present.
        assert out is not None
        assert out == {
            "float16": (2e-3, 3e-3),
            "float32": (5e-4, 5e-4),
        }
        # And the parser converts to ``float`` (kills a copy-as-int mutation).
        atol, rtol = out["float16"]
        assert isinstance(atol, float) and isinstance(rtol, float)

    def test_tolerances_from_config_uses_correct_section_keys(self) -> None:
        """The parser walks ``tool → gpucheck → tolerances`` exactly.

        Mutating any of those three string literals (``"tool"``,
        ``"gpucheck"``, ``"tolerances"``) to an XX-wrapped form makes the
        section look absent for a properly-formed config — so the parser
        returns ``None`` instead of the populated dict, and this test
        catches the regression.
        """
        cfg = {
            "tool": {
                "gpucheck": {"tolerances": {"float16": {"atol": 1e-3, "rtol": 1e-3}}}
            }
        }
        out = tolerances_from_config(cfg)
        assert out == {"float16": (1e-3, 1e-3)}

        # And a config that uses *different* top-level keys must NOT match.
        wrong = {
            "TOOL": {
                "GPUCHECK": {"TOLERANCES": {"float16": {"atol": 1e-3, "rtol": 1e-3}}}
            }
        }
        assert tolerances_from_config(wrong) is None


# ---------------------------------------------------------------------------
# Test 3 — hard-coded parametrize over _DEFAULT_TOLERANCES
# ---------------------------------------------------------------------------

# Hard-coded expected (atol, rtol) pairs that mirror the source-of-truth
# table in `src/gpucheck/assertions/tolerances.py`. The whole point of
# duplicating the values here is to break the tautology of iterating the
# dict against itself — any single-value mutation in the source dict will
# now diverge from these literals and fail the matching parametrize case.

_DTYPE_TOLERANCES_GROUND_TRUTH: list[tuple[str, tuple[float, float]]] = [
    ("float64", (1e-10, 1e-7)),
    ("float32", (1e-4, 1e-4)),
    ("float16", (1e-2, 1e-2)),
    ("bfloat16", (5e-2, 5e-2)),
    ("float8_e4m3fn", (0.125, 0.125)),
    ("float8_e5m2", (0.25, 0.25)),
    ("tf32", (5e-4, 5e-4)),
]


@pytest.mark.parametrize(
    ("dtype_name", "expected"), _DTYPE_TOLERANCES_GROUND_TRUTH
)
def test_default_tolerance_table_value_is_pinned(
    dtype_name: str, expected: tuple[float, float]
) -> None:
    """``_DEFAULT_TOLERANCES[name]`` matches the hard-coded literal.

    Kills the ~12 dict-value mutations (atol or rtol → 2x, 0.5x, 0.0,
    constant swap) that previously survived because the existing test
    iterated the dict against itself.
    """
    assert _DEFAULT_TOLERANCES[dtype_name] == expected


@pytest.mark.parametrize(
    ("dtype_name", "expected"), _DTYPE_TOLERANCES_GROUND_TRUTH
)
def test_compute_tolerance_returns_pinned_default(
    dtype_name: str, expected: tuple[float, float]
) -> None:
    """``compute_tolerance(name)`` agrees with the hard-coded literal.

    Catches mutations in ``compute_tolerance`` that would silently swap a
    valid lookup for the float32 fallback — for example, mutating the
    ``if name in _config_overrides`` guard to ``not in`` and combined
    with an empty overlay, the lookup short-circuits to defaults but a
    different mutation might force the float32 fallback. The hard-coded
    expected pair eliminates that escape hatch.
    """
    # Make sure no stale config overrides leak in from a parallel test.
    reset_config_tolerances()
    assert compute_tolerance(dtype_name) == expected


def test_default_tolerances_table_size_is_pinned() -> None:
    """Length of ``_DEFAULT_TOLERANCES`` equals the ground-truth list.

    Catches a deletion or duplication of an entry — both of which would
    survive a per-key parametrize that doesn't cross-check the cardinality.
    """
    assert len(_DEFAULT_TOLERANCES) == len(_DTYPE_TOLERANCES_GROUND_TRUTH)
    assert set(_DEFAULT_TOLERANCES.keys()) == {
        name for name, _ in _DTYPE_TOLERANCES_GROUND_TRUTH
    }
