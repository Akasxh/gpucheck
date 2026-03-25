"""GPU kernel benchmarking fixture for gpucheck."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pytest


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Statistical summary of GPU kernel benchmark timings (all values in ms)."""

    median: float
    mean: float
    std: float
    min: float
    max: float
    p5: float
    p25: float
    p75: float
    p95: float
    rounds: int
    warmup_rounds: int
    outliers_removed: int
    raw_times: tuple[float, ...] = field(repr=False)

    def __str__(self) -> str:
        return (
            f"BenchmarkResult(median={self.median:.4f}ms, mean={self.mean:.4f}ms, "
            f"std={self.std:.4f}ms, min={self.min:.4f}ms, max={self.max:.4f}ms, "
            f"rounds={self.rounds})"
        )


class KernelCallable(Protocol):
    """Any callable that launches GPU work."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) from pre-sorted data."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _remove_outliers_iqr(times: list[float], factor: float = 1.5) -> list[float]:
    """Remove statistical outliers using the IQR method."""
    if len(times) < 4:
        return times
    sorted_t = sorted(times)
    q1 = _percentile(sorted_t, 25.0)
    q3 = _percentile(sorted_t, 75.0)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return [t for t in times if lower <= t <= upper]


def _get_l2_cache_size() -> int:
    """Auto-detect L2 cache size in bytes via pynvml. Falls back to 40MB."""
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            # nvmlDeviceGetAttribute is not universally available; fall back
            try:
                attr = pynvml.NVML_DEVICE_ATTRIBUTE_GPU_L2_CACHE_SIZE  # type: ignore[attr-defined]
                l2 = pynvml.nvmlDeviceGetAttribute(handle, attr)  # type: ignore[attr-defined]
                return int(l2) * 1024  # returned in KB
            except (AttributeError, pynvml.NVMLError):
                pass
        finally:
            pynvml.nvmlShutdown()
    except (ImportError, Exception):  # noqa: BLE001
        pass
    return 40 * 1024 * 1024  # 40 MB default


_torch_cache: Any = None


def _get_torch() -> Any:
    """Lazy-cached torch import for hot-loop use."""
    global _torch_cache  # noqa: PLW0603
    if _torch_cache is None:
        import torch  # type: ignore[import-untyped]

        _torch_cache = torch
    return _torch_cache


def _flush_l2_cache(l2_size: int, buf: Any = None) -> None:
    """Flush GPU L2 cache by writing a buffer the size of L2."""
    try:
        torch = _get_torch()

        if not torch.cuda.is_available():
            return
        # Use pre-allocated buffer if available, otherwise allocate
        if buf is None:
            buf = torch.empty(l2_size // 4, dtype=torch.float32, device="cuda")
        buf.fill_(0.0)
        torch.cuda.synchronize()
    except ImportError:
        # Without torch we can't easily flush; this is best-effort
        pass
    except RuntimeError:
        pass


@dataclass
class _BenchmarkRunner:
    """Callable returned by the gpu_benchmark fixture."""

    warmup: int = 10
    rounds: int = 100
    flush_l2: bool = True
    _l2_size: int = field(default=0, init=False, repr=False)
    _flush_buf: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.flush_l2:
            self._l2_size = _get_l2_cache_size()
            # Pre-allocate flush buffer for reuse across iterations
            try:
                torch = _get_torch()
                if torch.cuda.is_available() and self._l2_size > 0:
                    self._flush_buf = torch.empty(
                        self._l2_size // 4, dtype=torch.float32, device="cuda"
                    )
            except (ImportError, RuntimeError):
                pass

    def __call__(
        self,
        fn: KernelCallable,
        *args: Any,
        warmup: int | None = None,
        rounds: int | None = None,
        flush_l2: bool | None = None,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark *fn* using CUDA events for accurate GPU timing.

        Parameters
        ----------
        fn:
            The kernel / callable to benchmark.
        *args:
            Positional arguments forwarded to *fn*.
        warmup:
            Override default warmup iterations.
        rounds:
            Override default benchmark iterations.
        flush_l2:
            Override default L2 flushing behaviour.
        **kwargs:
            Keyword arguments forwarded to *fn*.
        """
        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "gpu_benchmark requires PyTorch for CUDA event timing. "
                "Install it with: pip install torch"
            ) from exc

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for benchmarking")

        n_warmup = warmup if warmup is not None else self.warmup
        n_rounds = rounds if rounds is not None else self.rounds
        do_flush = flush_l2 if flush_l2 is not None else self.flush_l2

        # Warmup
        for _ in range(n_warmup):
            fn(*args, **kwargs)
        torch.cuda.synchronize()

        # Pre-allocate CUDA events to avoid per-iteration allocation overhead
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        # Timed runs
        raw_times: list[float] = []
        for _ in range(n_rounds):
            if do_flush and self._l2_size > 0:
                _flush_l2_cache(self._l2_size, buf=self._flush_buf)

            start.record()  # type: ignore[no-untyped-call]
            fn(*args, **kwargs)
            end.record()  # type: ignore[no-untyped-call]

            torch.cuda.synchronize()
            elapsed_ms: float = start.elapsed_time(end)  # type: ignore[no-untyped-call]
            raw_times.append(elapsed_ms)

        # Outlier removal
        cleaned = _remove_outliers_iqr(raw_times)
        outliers_removed = len(raw_times) - len(cleaned)
        if not cleaned:
            warnings.warn(
                "All benchmark samples were removed as outliers; using raw times",
                RuntimeWarning,
                stacklevel=2,
            )
            cleaned = raw_times
            outliers_removed = 0

        sorted_clean = sorted(cleaned)
        n = len(sorted_clean)
        mean = sum(sorted_clean) / n
        variance = sum((t - mean) ** 2 for t in sorted_clean) / max(n - 1, 1)
        std = math.sqrt(variance)

        return BenchmarkResult(
            median=_percentile(sorted_clean, 50.0),
            mean=mean,
            std=std,
            min=sorted_clean[0],
            max=sorted_clean[-1],
            p5=_percentile(sorted_clean, 5.0),
            p25=_percentile(sorted_clean, 25.0),
            p75=_percentile(sorted_clean, 75.0),
            p95=_percentile(sorted_clean, 95.0),
            rounds=n_rounds,
            warmup_rounds=n_warmup,
            outliers_removed=outliers_removed,
            raw_times=tuple(raw_times),
        )


@pytest.fixture()
def gpu_benchmark() -> _BenchmarkRunner:
    """Provide a GPU kernel benchmarker using CUDA event timing.

    Usage::

        def test_my_kernel(gpu_benchmark):
            result = gpu_benchmark(my_kernel, input_tensor)
            assert result.median < 1.0  # ms
    """
    return _BenchmarkRunner()
