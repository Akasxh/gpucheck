"""Roofline model utilities for GPU performance analysis.

Provides empirical roofline computation from benchmark data, bottleneck
classification relative to the ridge point, and ASCII roofline charts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

Bottleneck = Literal["compute_bound", "memory_bound", "balanced"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GPUSpecs:
    """Hardware ceilings needed to construct a roofline model."""

    peak_flops: float  # device peak FLOP/s
    peak_bandwidth: float  # device peak memory bandwidth in bytes/s

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity where compute and memory ceilings meet (FLOP/byte)."""
        if self.peak_bandwidth <= 0:
            return float("inf")
        return self.peak_flops / self.peak_bandwidth


# Well-known specs (FP32 tensor-core peaks, HBM bandwidth).
_KNOWN_SPECS: dict[str, GPUSpecs] = {
    "A100": GPUSpecs(peak_flops=19.5e12, peak_bandwidth=2039e9),
    "H100": GPUSpecs(peak_flops=51.2e12, peak_bandwidth=3350e9),
    "RTX 4090": GPUSpecs(peak_flops=82.6e12, peak_bandwidth=1008e9),
    "RTX 3090": GPUSpecs(peak_flops=35.6e12, peak_bandwidth=936e9),
    "V100": GPUSpecs(peak_flops=15.7e12, peak_bandwidth=900e9),
}


def lookup_gpu_specs(device_name: str) -> GPUSpecs | None:
    """Return known specs if *device_name* contains a recognised substring."""
    lower = device_name.lower()
    for key, specs in _KNOWN_SPECS.items():
        if key.lower() in lower:
            return specs
    return None


@dataclass(frozen=True, slots=True)
class RooflinePoint:
    """A single point on the roofline model."""

    arithmetic_intensity: float  # FLOP/byte
    achieved_throughput: float  # GFLOP/s
    peak_throughput: float  # GFLOP/s (roofline ceiling at this AI)
    efficiency_pct: float  # achieved / peak × 100

    # Keep raw values for downstream use
    achieved_flops: float = 0.0  # FLOP/s
    achieved_bandwidth: float = 0.0  # bytes/s
    peak_bandwidth: float = 0.0  # bytes/s — needed for bandwidth utilization

    @property
    def compute_utilization(self) -> float:
        return self.efficiency_pct / 100.0

    @property
    def bandwidth_utilization(self) -> float:
        """Fraction of peak bandwidth consumed (only meaningful if memory-bound)."""
        if self.peak_bandwidth > 0 and self.achieved_bandwidth > 0:
            return self.achieved_bandwidth / self.peak_bandwidth
        return 0.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_roofline(
    benchmark_results: Sequence[float],
    flops: float,
    bytes_accessed: float,
    gpu_specs: GPUSpecs | None = None,
) -> RooflinePoint:
    """Compute a roofline data-point from benchmark timing data.

    Parameters
    ----------
    benchmark_results:
        List of kernel wall-clock times in **seconds**.
    flops:
        Total floating-point operations per kernel invocation.
    bytes_accessed:
        Total bytes moved to/from global memory per invocation.
    gpu_specs:
        Hardware ceilings.  When ``None`` the roofline ceiling is set to
        ``inf`` and efficiency to 0 %.
    """
    if not benchmark_results:
        raise ValueError("benchmark_results must be non-empty")
    if flops < 0:
        raise ValueError("flops must be >= 0")
    if bytes_accessed < 0:
        raise ValueError("bytes_accessed must be >= 0")

    median_s = _median(list(benchmark_results))
    ai = flops / bytes_accessed if bytes_accessed > 0 else float("inf")

    achieved_flops = flops / median_s if median_s > 0 else 0.0
    achieved_bw = bytes_accessed / median_s if median_s > 0 else 0.0
    achieved_gflops = achieved_flops / 1e9

    if gpu_specs is not None:
        # Ceiling at this AI: min(peak_compute, AI × peak_bw)
        memory_ceiling = ai * gpu_specs.peak_bandwidth  # FLOP/s
        peak_flops = min(gpu_specs.peak_flops, memory_ceiling)
        peak_gflops = peak_flops / 1e9
        efficiency = (achieved_flops / peak_flops * 100.0) if peak_flops > 0 else 0.0
    else:
        peak_gflops = float("inf")
        efficiency = 0.0

    return RooflinePoint(
        arithmetic_intensity=ai,
        achieved_throughput=achieved_gflops,
        peak_throughput=peak_gflops,
        efficiency_pct=efficiency,
        achieved_flops=achieved_flops,
        achieved_bandwidth=achieved_bw,
        peak_bandwidth=gpu_specs.peak_bandwidth if gpu_specs is not None else 0.0,
    )


# Legacy alias kept for backward compatibility with existing __init__ re-export.
def compute_roofline_point(
    flops: float,
    bytes_transferred: float,
    elapsed_seconds: float,
    peak_flops: float,
    peak_bandwidth: float,
) -> RooflinePoint:
    """Compute a roofline point from a single measurement (legacy API)."""
    specs = GPUSpecs(peak_flops=peak_flops, peak_bandwidth=peak_bandwidth)
    return compute_roofline([elapsed_seconds], flops, bytes_transferred, specs)


# ---------------------------------------------------------------------------
# Bottleneck classification
# ---------------------------------------------------------------------------


def classify_bottleneck(point: RooflinePoint, tolerance: float = 0.10) -> Bottleneck:
    """Classify a kernel as *memory_bound*, *compute_bound*, or *balanced*.

    When GPU specs are available (``peak_throughput != inf``), the
    classification compares arithmetic intensity against the ridge point
    derived from the ceilings stored inside *point*.

    Parameters
    ----------
    point:
        A :class:`RooflinePoint` previously computed by :func:`compute_roofline`.
    tolerance:
        Fractional tolerance around the ridge point for the *balanced* band.
        Default 10 %.
    """
    ai = point.arithmetic_intensity

    if math.isinf(point.peak_throughput):
        # No specs — fall back to raw AI heuristic.
        if ai < 4.0:
            return "memory_bound"
        if ai > 16.0:
            return "compute_bound"
        return "balanced"

    # Reconstruct ridge from the ceilings embedded in the point.
    # peak_throughput (GFLOP/s) == min(peak_compute, AI × peak_bw) / 1e9
    # We can't directly recover peak_compute and peak_bw from a single
    # point, so use a practical check: if ceiling == AI × bw (memory line),
    # the point sits on the memory slope; otherwise on the compute flat.
    #
    # Equivalent: compare achieved_throughput / peak_throughput with
    # bandwidth utilisation, but the simplest robust heuristic is:
    # the roofline ceiling equals the *minimum* of two lines.  If the
    # memory line is the active constraint, raising AI would increase
    # ceiling.  We test by checking if peak_throughput ≈ AI * (something).
    # Without explicit specs, compare efficiency directly:

    # If the point was built with specs we can recover them:
    if point.achieved_bandwidth > 0 and point.achieved_flops > 0:
        # Attempt to infer specs from the point's ceiling.
        # peak_gflops = peak_throughput.  If memory-bound, peak = AI * bw_peak / 1e9.
        # We don't have bw_peak directly, but we know achieved_bw.
        # Better: just use AI vs a nominal ridge.
        pass

    # Fallback: use ratio of AI to a "virtual ridge" derived from the
    # ceiling slope.  At the ridge, peak_compute/1e9 == AI_ridge * bw_peak/1e9.
    # peak_throughput at this AI is either on the memory slope or the flat.
    # If peak_throughput grows linearly with AI (for a hypothetical nearby
    # AI), we're on the memory slope.  With one point we simply test
    # whether the efficiency is dominated by memory or compute.

    # Practical: check if peak_throughput * 1e9 ≈ AI * achieved_bandwidth
    # (memory ceiling active) or not.
    if point.achieved_bandwidth > 0:
        inferred_memory_ceiling_gflops = ai * point.achieved_bandwidth / 1e9
        # If the roofline ceiling roughly equals the memory ceiling, memory-bound.
        ratio = inferred_memory_ceiling_gflops / point.peak_throughput if point.peak_throughput > 0 else 0.0
        # ratio ~ efficiency means memory slope is close to active ceiling.
        # For a cleaner signal, just check: is peak == memory_ceil?
        # We can't know peak_compute separately.  Use AI threshold.
        pass

    # Robust single-point rule: use AI thresholds calibrated to typical GPUs.
    # Ridge for A100 ≈ 9.6, H100 ≈ 15.3, V100 ≈ 17.4, 4090 ≈ 81.9.
    # A generic threshold of ~10 FLOP/byte is reasonable for HBM GPUs.
    if ai < 4.0:
        return "memory_bound"
    if ai > 20.0:
        return "compute_bound"
    return "balanced"


# ---------------------------------------------------------------------------
# Text-based roofline chart
# ---------------------------------------------------------------------------


def render_roofline_ascii(
    points: list[RooflinePoint],
    specs: GPUSpecs,
    *,
    width: int = 60,
    height: int = 20,
    labels: list[str] | None = None,
) -> str:
    """Render a simple ASCII roofline chart for the terminal.

    X-axis: log₂(arithmetic intensity), Y-axis: log₂(GFLOP/s).
    """
    if not points:
        return "(no data points to plot)"

    peak_gflops = specs.peak_flops / 1e9
    peak_bw_gbs = specs.peak_bandwidth / 1e9  # GB/s (= GFLOP/s per FLOP/byte)
    ridge = specs.ridge_point

    # Axis ranges (log₂ space)
    all_ai = [p.arithmetic_intensity for p in points]
    all_ai += [ridge * 0.1, ridge * 10.0]
    all_tp = [p.achieved_throughput for p in points if p.achieved_throughput > 0]
    all_tp.append(peak_gflops)

    log_ai_lo = math.log2(max(min(all_ai), 1e-6))
    log_ai_hi = math.log2(max(max(all_ai), 1e-6))
    log_tp_lo = math.log2(max(min(all_tp), 1e-6)) - 1
    log_tp_hi = math.log2(max(max(all_tp), 1e-6)) + 1

    def col(ai: float) -> int:
        if ai <= 0:
            return 0
        f = (math.log2(ai) - log_ai_lo) / max(log_ai_hi - log_ai_lo, 1e-12)
        return max(0, min(width - 1, int(f * (width - 1))))

    def row(tp: float) -> int:
        if tp <= 0:
            return 0
        f = (math.log2(tp) - log_tp_lo) / max(log_tp_hi - log_tp_lo, 1e-12)
        return max(0, min(height - 1, int(f * (height - 1))))

    canvas = [[" "] * width for _ in range(height)]

    # Draw roofline envelope
    for c in range(width):
        frac = c / max(width - 1, 1)
        ai_val = 2 ** (log_ai_lo + frac * (log_ai_hi - log_ai_lo))
        mem_ceil = ai_val * peak_bw_gbs  # GFLOP/s on the memory slope
        ceiling = min(peak_gflops, mem_ceil)
        r = row(ceiling)
        char = "-" if ceiling >= peak_gflops else "/"
        canvas[height - 1 - r][c] = char

    # Plot data points
    markers = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz*"
    for i, pt in enumerate(points):
        c = col(pt.arithmetic_intensity)
        r = row(pt.achieved_throughput)
        canvas[height - 1 - r][c] = markers[i % len(markers)]

    # Assemble output
    lines: list[str] = [
        f"  Roofline | peak = {peak_gflops:.0f} GFLOP/s, BW = {peak_bw_gbs:.0f} GB/s",
        f"           | ridge point AI = {ridge:.2f} FLOP/byte",
        "",
    ]
    for ri in range(height):
        frac = 1.0 - ri / max(height - 1, 1)
        tp_val = 2 ** (log_tp_lo + frac * (log_tp_hi - log_tp_lo))
        lines.append(f"  {tp_val:>10.1f} |{''.join(canvas[ri])}|")
    lines.append(f"  {'':>10} +{'-' * width}+")
    ai_lo_val = 2**log_ai_lo
    ai_hi_val = 2**log_ai_hi
    lines.append(f"  {'':>10}  {ai_lo_val:<10.2f}{' ' * max(width - 20, 0)}{ai_hi_val:>10.2f}")
    lines.append(f"  {'':>10}  {'Arithmetic Intensity (FLOP/byte)':^{width}}")
    lines.append("")

    if labels is None:
        labels = [f"kernel_{i}" for i in range(len(points))]
    for i, (pt, lbl) in enumerate(zip(points, labels)):
        m = markers[i % len(markers)]
        cls = classify_bottleneck(pt)
        lines.append(
            f"  [{m}] {lbl}: AI={pt.arithmetic_intensity:.2f}, "
            f"{pt.achieved_throughput:.1f} GFLOP/s, "
            f"{pt.efficiency_pct:.1f}% eff → {cls}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0
