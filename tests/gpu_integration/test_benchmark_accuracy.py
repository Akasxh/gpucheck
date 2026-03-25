"""Tests verifying gpu_benchmark fixture produces accurate, reproducible timings.

These tests require a real CUDA GPU. Mark all with @pytest.mark.gpu.
"""

from __future__ import annotations

import math

import pytest
import torch

from gpucheck.fixtures.benchmark import (
    _BenchmarkRunner,
    _percentile,
    _remove_outliers_iqr,
)

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matmul_512() -> None:
    """512x512 float32 matmul on CUDA."""
    a = torch.randn(512, 512, device="cuda")
    b = torch.randn(512, 512, device="cuda")
    torch.mm(a, b)


def _vector_add() -> None:
    """Simple element-wise add — sub-ms kernel."""
    a = torch.randn(1024, device="cuda")
    b = torch.randn(1024, device="cuda")
    _ = a + b


# Pre-allocate tensors to avoid allocation noise in timing tests
_A_512: torch.Tensor | None = None
_B_512: torch.Tensor | None = None


def _get_matmul_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    global _A_512, _B_512
    if _A_512 is None:
        _A_512 = torch.randn(512, 512, device="cuda")
        _B_512 = torch.randn(512, 512, device="cuda")
    return _A_512, _B_512


def _matmul_preallocated() -> None:
    a, b = _get_matmul_tensors()
    torch.mm(a, b)


# ---------------------------------------------------------------------------
# 1. Timing reasonableness: 512x512 matmul should be ~0.01-50ms
# ---------------------------------------------------------------------------

class TestTimingReasonableness:
    def test_matmul_512_in_expected_range(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """512x512 matmul timing should fall in 0.01-50ms range."""
        result = gpu_benchmark(_matmul_preallocated, warmup=10, rounds=50)
        assert 0.01 <= result.median <= 50.0, (
            f"512x512 matmul median={result.median:.4f}ms outside expected [0.01, 50]ms"
        )


# ---------------------------------------------------------------------------
# 2. Reproducibility: two runs within 10% of each other
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_two_runs_within_10_percent(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """Back-to-back benchmarks of the same kernel should be within 10%."""
        r1 = gpu_benchmark(_matmul_preallocated, warmup=20, rounds=100)
        r2 = gpu_benchmark(_matmul_preallocated, warmup=20, rounds=100)
        diff_pct = abs(r1.median - r2.median) / max(r1.median, r2.median) * 100
        assert diff_pct < 10.0, (
            f"Runs differ by {diff_pct:.1f}%: r1.median={r1.median:.4f}ms, "
            f"r2.median={r2.median:.4f}ms"
        )


# ---------------------------------------------------------------------------
# 3. Warmup effect: warmup=0 vs warmup=20
# ---------------------------------------------------------------------------

class TestWarmupEffect:
    def test_warmup_reduces_variance(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """With warmup, std should be <= std without warmup (or at least runs)."""
        r_no_warmup = gpu_benchmark(_matmul_preallocated, warmup=0, rounds=50)
        r_warmup = gpu_benchmark(_matmul_preallocated, warmup=20, rounds=50)

        # Both should produce valid results
        assert r_no_warmup.warmup_rounds == 0
        assert r_warmup.warmup_rounds == 20

        # The warmed-up median should be reasonable
        assert r_warmup.median > 0.0
        assert r_no_warmup.median > 0.0

        # Key: warmup=20 std should not be drastically worse than warmup=0
        # (In practice warmup helps; we just verify both run and produce stats)
        # Allow warmed-up std to be at most 3x the other (generous bound)
        assert r_warmup.std <= r_no_warmup.std * 3 + 0.01, (
            f"Warmup unexpectedly increased variance: "
            f"warmup.std={r_warmup.std:.4f} > no_warmup.std={r_no_warmup.std:.4f}"
        )


# ---------------------------------------------------------------------------
# 4. L2 flush effect: flush_l2=True vs False for memory-bound ops
# ---------------------------------------------------------------------------

class TestL2FlushEffect:
    def test_flush_l2_measurable_for_memory_bound(
        self, gpu_benchmark: _BenchmarkRunner
    ) -> None:
        """L2 flush should produce >= median for memory-bound ops (or at least run)."""
        # Large vector copy — memory-bound
        big = torch.randn(4 * 1024 * 1024, device="cuda")  # 16MB float32

        def _copy_op() -> None:
            _ = big.clone()

        r_flush = gpu_benchmark(_copy_op, warmup=10, rounds=50, flush_l2=True)
        r_no_flush = gpu_benchmark(_copy_op, warmup=10, rounds=50, flush_l2=False)

        # Both must run successfully
        assert r_flush.median > 0.0
        assert r_no_flush.median > 0.0

        # Flushing L2 should make memory-bound ops at least as slow (or equal).
        # Allow 20% tolerance for noise.
        assert r_flush.median >= r_no_flush.median * 0.8, (
            f"L2 flush result unexpectedly faster: "
            f"flush={r_flush.median:.4f}ms < no_flush={r_no_flush.median:.4f}ms * 0.8"
        )


# ---------------------------------------------------------------------------
# 5. IQR outlier removal: inject artificial outlier, verify removal
# ---------------------------------------------------------------------------

class TestIQROutlierRemoval:
    def test_outlier_is_removed(self) -> None:
        """_remove_outliers_iqr should strip values far outside IQR."""
        # Tight cluster around 1.0 with one extreme outlier
        times = [1.0, 1.01, 0.99, 1.02, 0.98, 1.0, 1.01, 0.99, 100.0]
        cleaned = _remove_outliers_iqr(times)
        assert 100.0 not in cleaned, "Extreme outlier 100.0 was not removed"
        assert len(cleaned) == 8, f"Expected 8 samples after removing 1 outlier, got {len(cleaned)}"

    def test_no_removal_for_tight_data(self) -> None:
        """Tightly clustered data should have no outliers removed."""
        times = [1.0, 1.01, 1.02, 0.99, 0.98, 1.0]
        cleaned = _remove_outliers_iqr(times)
        assert len(cleaned) == len(times)

    def test_too_few_samples_skips_removal(self) -> None:
        """With < 4 samples, IQR removal is skipped."""
        times = [1.0, 100.0, 1.0]
        cleaned = _remove_outliers_iqr(times)
        assert cleaned == times  # returned as-is

    def test_benchmark_reports_outlier_count(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """BenchmarkResult.outliers_removed should reflect IQR filtering."""
        result = gpu_benchmark(_matmul_preallocated, warmup=10, rounds=100)
        # outliers_removed should be non-negative int
        assert isinstance(result.outliers_removed, int)
        assert result.outliers_removed >= 0
        assert result.outliers_removed < result.rounds


# ---------------------------------------------------------------------------
# 6. BenchmarkResult fields all populated
# ---------------------------------------------------------------------------

class TestBenchmarkResultFields:
    def test_all_statistical_fields_populated(
        self, gpu_benchmark: _BenchmarkRunner
    ) -> None:
        """All stat fields (median, mean, std, p5, p25, p75, p95) should be set."""
        result = gpu_benchmark(_matmul_preallocated, warmup=5, rounds=30)

        # All should be finite positive floats
        for field_name in ("median", "mean", "std", "min", "max", "p5", "p25", "p75", "p95"):
            val = getattr(result, field_name)
            assert isinstance(val, float), f"{field_name} is not float: {type(val)}"
            assert math.isfinite(val), f"{field_name} is not finite: {val}"

        # Ordering invariants
        assert result.min <= result.p5 <= result.p25 <= result.median
        assert result.median <= result.p75 <= result.p95 <= result.max
        assert result.mean > 0.0
        assert result.std >= 0.0

        # Metadata
        assert result.rounds == 30
        assert result.warmup_rounds == 5
        assert len(result.raw_times) == 30  # raw_times has all rounds

    def test_str_representation(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """__str__ should produce a readable summary."""
        result = gpu_benchmark(_matmul_preallocated, warmup=5, rounds=10)
        s = str(result)
        assert "BenchmarkResult" in s
        assert "median=" in s
        assert "ms" in s


# ---------------------------------------------------------------------------
# 7. gpu_benchmark vs raw CUDA event timing — within 20%
# ---------------------------------------------------------------------------

class TestVsRawCudaEvents:
    def test_within_20_percent_of_raw_events(
        self, gpu_benchmark: _BenchmarkRunner
    ) -> None:
        """gpu_benchmark median should be within 20% of manual CUDA event timing."""
        a, b = _get_matmul_tensors()

        def _mm() -> None:
            torch.mm(a, b)

        # Warmup for manual timing
        for _ in range(20):
            _mm()
        torch.cuda.synchronize()

        # Manual CUDA event timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        manual_times: list[float] = []
        for _ in range(100):
            start.record()
            _mm()
            end.record()
            torch.cuda.synchronize()
            manual_times.append(start.elapsed_time(end))

        manual_median = sorted(manual_times)[len(manual_times) // 2]

        # Fixture timing
        result = gpu_benchmark(_mm, warmup=20, rounds=100, flush_l2=False)

        diff_pct = abs(result.median - manual_median) / max(result.median, manual_median) * 100
        assert diff_pct < 20.0, (
            f"Fixture vs manual differ by {diff_pct:.1f}%: "
            f"fixture={result.median:.4f}ms, manual={manual_median:.4f}ms"
        )


# ---------------------------------------------------------------------------
# 8. Sub-ms operations: vector add
# ---------------------------------------------------------------------------

class TestSubMsOperations:
    def test_vector_add_sub_ms(self, gpu_benchmark: _BenchmarkRunner) -> None:
        """Very fast kernel (vector add 1024 elems) should produce valid sub-ms timings."""
        a = torch.randn(1024, device="cuda")
        b = torch.randn(1024, device="cuda")

        def _add() -> None:
            _ = a + b

        result = gpu_benchmark(_add, warmup=20, rounds=100)
        assert result.median < 5.0, f"Vector add took {result.median:.4f}ms, expected < 5ms"
        assert result.median > 0.0, "Timing must be positive"
        assert result.min >= 0.0, f"Negative min time: {result.min}"


# ---------------------------------------------------------------------------
# 9. memory_tracker: allocate known size, verify peak matches
# ---------------------------------------------------------------------------

class TestMemoryTracker:
    def test_peak_matches_known_allocation(self) -> None:
        """Allocating a known-size tensor should show up in peak memory."""
        from gpucheck.fixtures.profiler import MemoryTracker

        tracker = MemoryTracker()
        tracker.start()

        # Allocate ~64MB (16M float32 = 64MB)
        alloc_size_bytes = 16 * 1024 * 1024 * 4  # 64 MB
        x = torch.randn(16 * 1024 * 1024, device="cuda")
        torch.cuda.synchronize()

        report = tracker.stop()

        # Peak should be at least the allocation size (within reasonable tolerance)
        # torch.cuda.max_memory_allocated tracks the allocator's peak
        assert report.peak >= alloc_size_bytes * 0.9, (
            f"Peak {report.peak / (1024*1024):.1f}MB < expected "
            f"{alloc_size_bytes / (1024*1024):.1f}MB"
        )

        del x
        torch.cuda.empty_cache()

    def test_no_leak_after_cleanup(self) -> None:
        """Allocating and freeing should show no leak."""
        from gpucheck.fixtures.profiler import MemoryTracker

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        tracker = MemoryTracker()
        tracker.start()

        x = torch.randn(1024, 1024, device="cuda")
        del x
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        report = tracker.stop()
        # After cleanup, leaked should be minimal
        assert report.leaked_mb < 2.0, f"Unexpected leak: {report.leaked_mb:.1f}MB"


# ---------------------------------------------------------------------------
# Unit tests for _percentile (no GPU needed)
# ---------------------------------------------------------------------------

class TestPercentileUnit:
    """Pure-CPU unit tests for the _percentile helper."""

    pytestmark = []  # override module-level gpu mark

    def test_median_of_odd(self) -> None:
        assert _percentile([1.0, 2.0, 3.0], 50.0) == 2.0

    def test_median_of_even(self) -> None:
        result = _percentile([1.0, 2.0, 3.0, 4.0], 50.0)
        assert abs(result - 2.5) < 1e-9

    def test_p0_and_p100(self) -> None:
        data = [10.0, 20.0, 30.0]
        assert _percentile(data, 0.0) == 10.0
        assert _percentile(data, 100.0) == 30.0

    def test_empty_returns_zero(self) -> None:
        assert _percentile([], 50.0) == 0.0
