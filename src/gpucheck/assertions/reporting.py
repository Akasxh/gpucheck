"""Rich-formatted mismatch reports for tensor comparisons."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy.typing as npt


def _safe_import_numpy() -> Any:
    import numpy as np

    return np


def format_mismatch_report(
    actual: npt.NDArray[Any],
    expected: npt.NDArray[Any],
    atol: float,
    rtol: float,
    *,
    actual_f64: npt.NDArray[Any] | None = None,
    expected_f64: npt.NDArray[Any] | None = None,
) -> str:
    """Build a Rich-formatted mismatch report between two numpy arrays.

    Returns a plain string containing Rich markup that can be printed with
    ``rich.print()`` or ``rich.console.Console().print()``.

    Pre-computed float64 arrays can be passed via *actual_f64* / *expected_f64*
    to avoid redundant copies when the caller already has them.
    """
    np = _safe_import_numpy()
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    if actual_f64 is None:
        actual_f64 = actual.astype(np.float64, copy=False)
    if expected_f64 is None:
        expected_f64 = expected.astype(np.float64, copy=False)

    diff = np.abs(actual_f64 - expected_f64)
    abs_expected_f64 = np.abs(expected_f64)

    # Relative error: avoid division by zero.
    with np.errstate(divide="ignore", invalid="ignore"):
        zero_fallback = np.where(diff == 0, 0.0, np.inf)
        rel_err = np.where(abs_expected_f64 != 0, diff / abs_expected_f64, zero_fallback)

    # Mismatch mask: |a - b| > atol + rtol * |b|
    threshold = atol + rtol * abs_expected_f64
    mismatch_mask = diff > threshold

    total = actual.size
    mismatch_count = int(np.sum(mismatch_mask))
    mismatch_pct = 100.0 * mismatch_count / total if total > 0 else 0.0

    all_nan = diff.size == 0 or np.all(np.isnan(diff))
    max_abs_err = float("nan") if all_nan else float(np.nanmax(diff))
    mean_abs_err = float("nan") if all_nan else float(np.nanmean(diff))
    finite_rel = rel_err[np.isfinite(rel_err)]
    max_rel_err = float(np.nanmax(finite_rel)) if finite_rel.size > 0 else float("inf")

    # Location of max absolute error.
    if all_nan or diff.size == 0:
        max_idx: tuple[int, ...] = ()
    else:
        max_idx = tuple(int(i) for i in np.unravel_index(int(np.nanargmax(diff)), diff.shape))

    # NaN / Inf stats — reuse pre-computed f64 arrays.
    nan_actual = int(np.sum(np.isnan(actual_f64)))
    nan_expected = int(np.sum(np.isnan(expected_f64)))
    inf_actual = int(np.sum(np.isinf(actual_f64)))
    inf_expected = int(np.sum(np.isinf(expected_f64)))

    # --- Build Rich output ---
    stats_table = Table(title="Error Statistics", show_header=True, header_style="bold cyan")
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", justify="right")
    stats_table.add_row("Max absolute error", f"{max_abs_err:.6e}")
    stats_table.add_row("Mean absolute error", f"{mean_abs_err:.6e}")
    stats_table.add_row("Max relative error", f"{max_rel_err:.6e}")
    stats_table.add_row("Mismatch count", f"{mismatch_count} / {total} ({mismatch_pct:.2f}%)")
    stats_table.add_row("Location of max error", str(max_idx))
    stats_table.add_row("Tolerances used", f"atol={atol:.2e}, rtol={rtol:.2e}")

    if nan_actual or nan_expected or inf_actual or inf_expected:
        stats_table.add_row("NaN (actual / expected)", f"{nan_actual} / {nan_expected}")
        stats_table.add_row("Inf (actual / expected)", f"{inf_actual} / {inf_expected}")

    # Histogram of absolute errors.
    histogram = _error_histogram(diff, mismatch_mask)

    console = Console(record=True, width=100)
    console.print(Panel(
        stats_table,
        title="[bold red]Tensor Mismatch Report[/bold red]",
        border_style="red",
    ))
    if histogram:
        console.print(Panel(
            Text(histogram),
            title="[bold yellow]Error Histogram[/bold yellow]",
            border_style="yellow",
        ))

    return console.export_text(styles=True)


def _error_histogram(diff: npt.NDArray[Any], mismatch_mask: npt.NDArray[Any]) -> str:
    """ASCII histogram of absolute error magnitudes (log-scale buckets)."""
    np = _safe_import_numpy()

    mismatched = diff[mismatch_mask]
    if mismatched.size == 0:
        return ""

    # Filter out non-finite values before computing log-scale histogram.
    finite_mismatched = mismatched[np.isfinite(mismatched)]
    if finite_mismatched.size == 0:
        return f"  All {mismatched.size} mismatched values are non-finite (NaN/Inf)"

    # Log10 buckets.
    with np.errstate(divide="ignore"):
        log_vals = np.log10(np.maximum(finite_mismatched, 1e-20))

    lo = int(np.floor(np.min(log_vals)))
    hi = int(np.ceil(np.max(log_vals)))
    if lo == hi:
        hi = lo + 1

    bins = list(range(lo, hi + 1))
    counts, _ = np.histogram(log_vals, bins=bins)

    max_count = int(np.max(counts)) if counts.size > 0 else 1
    bar_width = 40

    lines: list[str] = []
    for i, count in enumerate(counts):
        bucket = f"[1e{bins[i]:+d}, 1e{bins[i + 1]:+d})"
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "\u2588" * bar_len
        lines.append(f"  {bucket:>22s} | {bar} {count}")

    return "\n".join(lines)
