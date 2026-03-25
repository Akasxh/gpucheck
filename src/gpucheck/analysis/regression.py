"""Performance regression detection with statistical rigour.

Provides:
- Mann-Whitney U test (pure-stdlib, no scipy)
- Cohen's d effect-size estimation
- Simplified E-Divisive change-point detection (numpy only)
- JSON baseline storage & Rich table display
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Outcome of comparing current timings against a baseline."""

    is_regression: bool
    pvalue: float
    effect_size: float  # Cohen's d (positive ⇒ current is slower)
    description: str
    current_median: float
    baseline_median: float
    change_pct: float  # (current − baseline) / baseline × 100


# Legacy alias so existing imports keep working.
RegressionResult = RegressionReport


# ---------------------------------------------------------------------------
# Mann-Whitney U (normal-approximation, tie-corrected)
# ---------------------------------------------------------------------------


def mann_whitney_u(
    baseline: Sequence[float],
    current: Sequence[float],
) -> tuple[float, float]:
    """Two-sided Mann-Whitney U test.

    Returns ``(U_statistic, p_value)``.  Uses the normal approximation with
    tie correction, which is accurate for n ≥ 8 per group.
    """
    n1, n2 = len(baseline), len(current)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and rank
    combined: list[tuple[float, int]] = []
    for v in baseline:
        combined.append((v, 0))
    for v in current:
        combined.append((v, 1))
    combined.sort(key=lambda x: x[0])

    ranks = [0.0] * len(combined)
    tie_counts: list[int] = []
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-based
        tie_counts.append(j - i)
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    r1 = sum(ranks[i] for i in range(len(combined)) if combined[i][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)

    mu = n1 * n2 / 2.0
    n = n1 + n2

    # Tie correction: Σ(t³ - t) / (n(n-1))
    tie_term = sum(t**3 - t for t in tie_counts) / (n * (n - 1)) if n > 1 else 0.0
    sigma_sq = (n1 * n2 / 12.0) * (n + 1 - tie_term)
    if sigma_sq <= 0:
        return u, 1.0

    z = (u - mu) / math.sqrt(sigma_sq)
    p = 2.0 * _normal_cdf(-abs(z))
    return u, p


# ---------------------------------------------------------------------------
# Cohen's d
# ---------------------------------------------------------------------------


def _cohens_d(current: np.ndarray, baseline: np.ndarray) -> float:
    """Pooled-variance Cohen's d.  Positive ⇒ current > baseline (slower)."""
    nc, nb = len(current), len(baseline)
    if nc < 2 or nb < 2:
        diff = float(np.mean(current) - np.mean(baseline))
        return diff  # degenerate — return raw difference

    var_c = float(np.var(current, ddof=1))
    var_b = float(np.var(baseline, ddof=1))
    pooled = math.sqrt(((nc - 1) * var_c + (nb - 1) * var_b) / (nc + nb - 2))
    if pooled < 1e-15:
        return 0.0
    return float(np.mean(current) - np.mean(baseline)) / pooled


# ---------------------------------------------------------------------------
# Simplified E-Divisive change-point detection
# ---------------------------------------------------------------------------


def _mean_abs_diff_cross(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |a_i − b_j| over all (i, j) pairs."""
    return float(np.mean(np.abs(a[:, None] - b[None, :])))


def _mean_abs_diff_within(a: np.ndarray) -> float:
    """Mean |a_i − a_j| over all i ≠ j pairs."""
    n = len(a)
    if n < 2:
        return 0.0
    diffs = np.abs(a[:, None] - a[None, :])
    return float(np.sum(diffs)) / (n * (n - 1))


def e_divisive_single(
    series: np.ndarray,
    min_segment: int = 5,
    max_samples: int = 200,
) -> int | None:
    """Find the best single change point using the E-Divisive energy statistic.

    Returns the split index τ (first element of right segment), or ``None``
    if the series is too short.

    For large series (n > *max_samples*), a uniform sub-sample of candidate
    split points is evaluated to keep runtime manageable (O(n^2) per
    candidate instead of exhaustive O(n^3)).
    """
    n = len(series)
    if n < 2 * min_segment:
        return None

    best_stat = -math.inf
    best_tau: int | None = None

    candidates = list(range(min_segment, n - min_segment + 1))
    # Sub-sample candidate split points for large series to reduce O(n^3)
    if len(candidates) > max_samples:
        step = max(1, len(candidates) // max_samples)
        candidates = candidates[::step]

    for tau in candidates:
        left, right = series[:tau], series[tau:]
        nl, nr = len(left), len(right)
        cross = _mean_abs_diff_cross(left, right)
        within_l = _mean_abs_diff_within(left)
        within_r = _mean_abs_diff_within(right)
        e = (nl * nr / (nl + nr)) * (2.0 * cross - within_l - within_r)
        if e > best_stat:
            best_stat = e
            best_tau = tau

    return best_tau


# ---------------------------------------------------------------------------
# Normal CDF (Abramowitz & Stegun, max error ≈ 1.5 × 10⁻⁷)
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if z >= 0 else -1.0
    x = abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regression(
    current_results: Sequence[float],
    baseline_results: Sequence[float],
    threshold: float = 0.05,
    *,
    min_effect: float = 1.1,
) -> RegressionReport:
    """Compare *current_results* against *baseline_results*.

    Parameters
    ----------
    current_results:
        Kernel timings from the current run (seconds).
    baseline_results:
        Kernel timings from the baseline (seconds).
    threshold:
        P-value cutoff for significance.
    min_effect:
        Minimum median-ratio (current / baseline) to flag a regression.
        Default 1.1 → current must be ≥ 10 % slower.
    """
    curr = np.asarray(list(current_results), dtype=np.float64)
    base = np.asarray(list(baseline_results), dtype=np.float64)

    if len(curr) == 0 or len(base) == 0:
        return RegressionReport(
            is_regression=False,
            pvalue=1.0,
            effect_size=0.0,
            description="Insufficient data for regression test",
            current_median=0.0,
            baseline_median=0.0,
            change_pct=0.0,
        )

    c_med = _median(list(current_results))
    b_med = _median(list(baseline_results))
    change_pct = ((c_med - b_med) / b_med * 100.0) if b_med > 0 else 0.0
    effect_ratio = c_med / b_med if b_med > 0 else float("inf")

    _, pvalue = mann_whitney_u(list(baseline_results), list(current_results))
    cohen_d = _cohens_d(curr, base)

    # E-Divisive change-point sanity check
    combined = np.concatenate([base, curr])
    min_seg = max(3, min(len(base), len(curr)) // 2)
    cp = e_divisive_single(combined, min_segment=min_seg)
    cp_near_boundary = cp is not None and abs(cp - len(base)) <= max(2, len(base) // 4)

    is_regression = (
        pvalue < threshold
        and effect_ratio >= min_effect
        and c_med > b_med
    )

    # Build human-readable description
    parts: list[str] = []
    if is_regression:
        parts.append(
            f"REGRESSION DETECTED: {change_pct:+.1f}% "
            f"(p={pvalue:.4f}, Cohen's d={cohen_d:.2f})"
        )
    else:
        parts.append(f"No regression: {change_pct:+.1f}% (p={pvalue:.4f})")

    parts.append(f"  Current:  median={c_med * 1e3:.3f} ms (n={len(curr)})")
    parts.append(f"  Baseline: median={b_med * 1e3:.3f} ms (n={len(base)})")
    if cp_near_boundary:
        parts.append(f"  E-Divisive change point at index {cp} (boundary={len(base)})")

    return RegressionReport(
        is_regression=is_regression,
        pvalue=pvalue,
        effect_size=cohen_d,
        description="\n".join(parts),
        current_median=c_med,
        baseline_median=b_med,
        change_pct=change_pct,
    )


# ---------------------------------------------------------------------------
# JSON baseline storage
# ---------------------------------------------------------------------------


def save_baseline(path: str | Path, name: str, results: Sequence[float]) -> None:
    """Persist timing results as a JSON baseline.

    The file maps benchmark names → lists of timings; existing entries for
    other benchmarks are preserved.
    """
    p = Path(path)
    data: dict[str, list[float]] = {}
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}

    data[name] = [float(v) for v in results]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_baseline(path: str | Path, name: str) -> list[float] | None:
    """Load baseline timings.  Returns ``None`` if missing."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get(name)
    if raw is None:
        return None
    return [float(v) for v in raw]


# ---------------------------------------------------------------------------
# Rich table display
# ---------------------------------------------------------------------------


def format_regression_table(reports: dict[str, RegressionReport]) -> str:
    """Render multiple :class:`RegressionReport` instances as a table.

    Uses Rich if available, otherwise falls back to plain text.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        table = Table(title="Regression Report", show_lines=True)
        table.add_column("Benchmark", style="cyan")
        table.add_column("Change", justify="right")
        table.add_column("p-value", justify="right")
        table.add_column("Effect (d)", justify="right")
        table.add_column("Status", justify="center")

        for name, r in reports.items():
            chg = f"{r.change_pct:+.1f}%"
            pv = f"{r.pvalue:.4f}"
            eff = f"{r.effect_size:.2f}"
            if r.is_regression:
                status = "[bold red]REGRESSION[/bold red]"
                chg = f"[red]{chg}[/red]"
            elif r.change_pct < -2.0:
                status = "[green]IMPROVED[/green]"
                chg = f"[green]{chg}[/green]"
            else:
                status = "[dim]OK[/dim]"
            table.add_row(name, chg, pv, eff, status)

        console = Console(record=True, width=100)
        console.print(table)
        return console.export_text()

    except ImportError:
        lines = ["=" * 80, "Regression Report", "=" * 80]
        hdr = f"{'Benchmark':<30} {'Change':>10} {'p-value':>10} {'Effect':>10} {'Status':>12}"
        lines.append(hdr)
        lines.append("-" * 80)
        for name, r in reports.items():
            st = "REGRESSION" if r.is_regression else ("IMPROVED" if r.change_pct < -2.0 else "OK")
            lines.append(
                f"{name:<30} {r.change_pct:>+9.1f}% "
                f"{r.pvalue:>10.4f} {r.effect_size:>10.2f} {st:>12}"
            )
        lines.append("=" * 80)
        return "\n".join(lines)
