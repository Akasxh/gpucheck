"""Auto-classification of GPU kernel bottlenecks without profiling.

Sweeps a kernel across multiple input sizes and infers whether it is
memory-bound or compute-bound from throughput scaling behaviour.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BottleneckAnalysis:
    """Result of automatic bottleneck classification."""

    classification: str  # "memory_bound" | "compute_bound" | "balanced"
    evidence: list[str]
    throughputs_bytes_per_s: list[float]
    input_sizes: list[int]
    scaling_exponent: float  # log-log slope of throughput vs size

    @property
    def description(self) -> str:
        parts = [f"Classification: {self.classification}"]
        parts.append(f"Scaling exponent: {self.scaling_exponent:.3f}")
        for ev in self.evidence:
            parts.append(f"  - {ev}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sync_gpu() -> None:
    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _time_kernel(
    fn: Callable[..., Any],
    size: int,
    warmup: int,
    rounds: int,
) -> float:
    """Return median wall-clock time (seconds) for ``fn(size)``.

    .. note::
        This uses CPU-side ``time.perf_counter()`` with ``torch.cuda.synchronize()``
        barriers for quick bottleneck classification only. For accurate GPU kernel
        timing, use the ``gpu_benchmark`` fixture which employs CUDA events.
    """
    _sync_gpu()
    for _ in range(warmup):
        fn(size)
    _sync_gpu()

    times: list[float] = []
    for _ in range(rounds):
        _sync_gpu()
        t0 = time.perf_counter()
        fn(size)
        _sync_gpu()
        times.append(time.perf_counter() - t0)

    times.sort()
    mid = len(times) // 2
    return times[mid]


def _fit_log_log_slope(xs: list[float], ys: list[float]) -> float:
    """OLS slope in log-log space.  Returns 0 on degenerate input."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    lx = [math.log(x) for x in xs[:n] if x > 0]
    ly = [math.log(y) for y in ys[:n] if y > 0]
    n = min(len(lx), len(ly))
    if n < 2:
        return 0.0
    lx, ly = lx[:n], ly[:n]

    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    if den < 1e-15:
        return 0.0
    return num / den


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_classify_bottleneck(
    benchmark_fn: Callable[..., Any],
    input_sizes: list[int] | None = None,
    *,
    warmup: int = 3,
    rounds: int = 7,
    element_bytes: int = 4,
) -> BottleneckAnalysis:
    """Run *benchmark_fn* at several sizes and classify its bottleneck.

    Parameters
    ----------
    benchmark_fn:
        Callable accepting a single ``size: int``.  It should allocate and
        process *size* elements on the GPU.
    input_sizes:
        Sizes to sweep.  Defaults to ``[2**k for k in range(14, 23)]``.
    warmup:
        Warmup iterations per size point.
    rounds:
        Timed iterations per size point.
    element_bytes:
        Bytes per element (used to convert to bytes/s throughput).

    Returns
    -------
    BottleneckAnalysis
        Contains the classification string (``"memory_bound"``,
        ``"compute_bound"``, or ``"balanced"``), supporting evidence,
        raw throughputs, and the fitted scaling exponent.
    """
    if input_sizes is None:
        input_sizes = [2**k for k in range(14, 23)]

    throughputs: list[float] = []
    evidence: list[str] = []

    for size in input_sizes:
        t = _time_kernel(benchmark_fn, size, warmup=warmup, rounds=rounds)
        bps = (size * element_bytes) / t if t > 0 else 0.0
        throughputs.append(bps)

    slope = _fit_log_log_slope([float(s) for s in input_sizes], throughputs)

    # Plateau detection on the last third of measurements
    plateau_ratio = 0.0
    if len(throughputs) >= 3:
        tail = throughputs[len(throughputs) * 2 // 3:]
        tp_max = max(throughputs)
        if tp_max > 0 and tail:
            plateau_ratio = min(tail) / tp_max

    # Decision thresholds
    # slope ≈ 1  → throughput grows with size → memory-bound
    # slope ≈ 0  → throughput plateaus         → compute-bound
    if slope > 0.6:
        cls = "memory_bound"
        evidence.append(
            f"Throughput scales with input size (log-log slope = {slope:.2f})"
        )
        evidence.append("Kernel performance limited by memory bandwidth")
    elif slope < 0.2:
        cls = "compute_bound"
        evidence.append(
            f"Throughput plateaus with input size (log-log slope = {slope:.2f})"
        )
        if plateau_ratio > 0.85:
            evidence.append(
                f"Tail throughput is {plateau_ratio:.0%} of peak — strong plateau"
            )
        evidence.append("Kernel performance limited by compute")
    else:
        cls = "balanced"
        evidence.append(
            f"Moderate throughput scaling (log-log slope = {slope:.2f})"
        )
        evidence.append("Kernel shows mixed memory/compute characteristics")

    return BottleneckAnalysis(
        classification=cls,
        evidence=evidence,
        throughputs_bytes_per_s=throughputs,
        input_sizes=input_sizes,
        scaling_exponent=slope,
    )
