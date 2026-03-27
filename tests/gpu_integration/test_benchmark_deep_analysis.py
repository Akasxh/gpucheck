"""Deep accuracy analysis of gpucheck's gpu_benchmark fixture.

Tests:
1. Compare gpu_benchmark timings against raw CUDA events for the same kernel
2. Test with fast kernels (<1ms), medium (1-10ms), slow (>10ms)
3. Measure the overhead the benchmark fixture adds
4. Test if L2 flush actually works (compare with/without)
5. Test IQR outlier removal — does it improve consistency?
6. Measure coefficient of variation across multiple runs
7. Compare against triton.testing.do_bench for same operations

Writes findings to benchmark_accuracy.md

Run: python tests/test_benchmark_deep_analysis.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.cuda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gpucheck.fixtures.benchmark import (
    _BenchmarkRunner,
    _remove_outliers_iqr,
)

# ──────────────────────────────────────────────────────────────
# Pre-allocated tensors (avoid alloc noise in timing)
# ──────────────────────────────────────────────────────────────

_FAST_A: torch.Tensor | None = None
_FAST_B: torch.Tensor | None = None
_MED_A: torch.Tensor | None = None
_MED_B: torch.Tensor | None = None
_SLOW_A: torch.Tensor | None = None
_SLOW_B: torch.Tensor | None = None
_SCALAR: torch.Tensor | None = None
_BIG: torch.Tensor | None = None


def _init_tensors() -> None:
    global _FAST_A, _FAST_B, _MED_A, _MED_B, _SLOW_A, _SLOW_B, _SCALAR, _BIG
    _FAST_A = torch.randn(256, device="cuda")
    _FAST_B = torch.randn(256, device="cuda")
    _MED_A = torch.randn(1024, 1024, device="cuda")
    _MED_B = torch.randn(1024, 1024, device="cuda")
    _SLOW_A = torch.randn(4096, 4096, device="cuda")
    _SLOW_B = torch.randn(4096, 4096, device="cuda")
    _SCALAR = torch.tensor(1.0, device="cuda")
    _BIG = torch.randn(8192, 8192, device="cuda")
    torch.cuda.synchronize()


def _fast_prealloc() -> None:
    _ = _FAST_A + _FAST_B  # type: ignore[operator]


def _med_prealloc() -> None:
    _ = torch.matmul(_MED_A, _MED_B)  # type: ignore[arg-type]


def _slow_prealloc() -> None:
    _ = torch.matmul(_SLOW_A, _SLOW_B)  # type: ignore[arg-type]


def _tiny_prealloc() -> None:
    _ = _SCALAR + _SCALAR  # type: ignore[operator]


def _membound_prealloc() -> None:
    _ = _BIG * 2.0 + 1.0  # type: ignore[operator]


# ──────────────────────────────────────────────────────────────
# Raw CUDA event timing (ground truth, no IQR, no fixture)
# ──────────────────────────────────────────────────────────────

def raw_cuda_bench(
    fn: Any,
    warmup: int = 10,
    rounds: int = 100,
    flush_l2: bool = False,
) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times: list[float] = []

    for _ in range(rounds):
        if flush_l2:
            buf = torch.empty(
                40 * 1024 * 1024 // 4, dtype=torch.float32, device="cuda"
            )
            buf.fill_(0.0)
            torch.cuda.synchronize()

        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return times


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _cv(data: list[float]) -> float:
    if len(data) < 2:
        return 0.0
    m = statistics.mean(data)
    if m == 0:
        return 0.0
    return (statistics.stdev(data) / m) * 100.0


# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True, scope="module")
def _init_cuda_tensors() -> None:
    """Allocate GPU tensors once before any test in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    _init_tensors()


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

def test_1_accuracy_vs_raw_cuda(results: dict[str, Any]) -> None:
    """Compare gpu_benchmark timings against raw CUDA events."""
    print("\n" + "=" * 60)
    print("TEST 1: gpu_benchmark vs raw CUDA events")
    print("=" * 60)

    runner = _BenchmarkRunner(warmup=10, rounds=100, flush_l2=True)
    data: list[dict[str, Any]] = []

    for label, fn in [
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
        ("slow(>10ms)", _slow_prealloc),
    ]:
        bench = runner(fn, warmup=10, rounds=100)
        raw = raw_cuda_bench(fn, warmup=10, rounds=100, flush_l2=True)
        raw_med = statistics.median(raw)
        raw_mean = statistics.mean(raw)
        raw_std = statistics.stdev(raw) if len(raw) > 1 else 0.0

        med_diff = ((bench.median - raw_med) / raw_med) * 100 if raw_med else 0
        mean_diff = ((bench.mean - raw_mean) / raw_mean) * 100 if raw_mean else 0

        entry = {
            "label": label,
            "gpucheck_median": round(bench.median, 4),
            "raw_median": round(raw_med, 4),
            "median_diff_pct": round(med_diff, 2),
            "gpucheck_mean": round(bench.mean, 4),
            "raw_mean": round(raw_mean, 4),
            "mean_diff_pct": round(mean_diff, 2),
            "gpucheck_std": round(bench.std, 4),
            "raw_std": round(raw_std, 4),
        }
        data.append(entry)
        print(f"  {label}: gpucheck={bench.median:.4f}ms raw={raw_med:.4f}ms diff={med_diff:+.2f}%")

    results["accuracy_vs_raw"] = data


def test_2_fixture_overhead(results: dict[str, Any]) -> None:
    """Measure wall-clock overhead of the benchmark fixture."""
    print("\n" + "=" * 60)
    print("TEST 2: Fixture overhead")
    print("=" * 60)

    data: list[dict[str, Any]] = []
    ROUNDS = 50

    for label, fn in [
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
    ]:
        runner = _BenchmarkRunner(warmup=5, rounds=ROUNDS, flush_l2=True)

        t0 = time.perf_counter()
        _ = runner(fn, warmup=5, rounds=ROUNDS)
        wall_fixture = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = raw_cuda_bench(fn, warmup=5, rounds=ROUNDS, flush_l2=True)
        wall_raw = time.perf_counter() - t0

        overhead = wall_fixture - wall_raw
        overhead_pct = (overhead / wall_raw) * 100 if wall_raw else 0.0

        entry = {
            "label": label,
            "fixture_wall_s": round(wall_fixture, 4),
            "raw_wall_s": round(wall_raw, 4),
            "overhead_s": round(overhead, 4),
            "overhead_pct": round(overhead_pct, 2),
            "rounds": ROUNDS,
        }
        data.append(entry)
        print(f"  {label}: fixture={wall_fixture:.3f}s raw={wall_raw:.3f}s overhead={overhead:.4f}s ({overhead_pct:+.1f}%)")

    results["overhead"] = data


def test_3_l2_flush_effect(results: dict[str, Any]) -> None:
    """Test if L2 flush actually changes timings."""
    print("\n" + "=" * 60)
    print("TEST 3: L2 cache flush effect")
    print("=" * 60)

    data: list[dict[str, Any]] = []

    for label, fn in [
        ("membound_eltwise_8k", _membound_prealloc),
        ("matmul_1k", _med_prealloc),
    ]:
        runner_f = _BenchmarkRunner(warmup=10, rounds=100, flush_l2=True)
        runner_nf = _BenchmarkRunner(warmup=10, rounds=100, flush_l2=False)

        res_f = runner_f(fn, warmup=10, rounds=100)
        res_nf = runner_nf(fn, warmup=10, rounds=100)

        diff = ((res_f.median - res_nf.median) / res_nf.median) * 100 if res_nf.median else 0
        cv_f = _cv(list(res_f.raw_times))
        cv_nf = _cv(list(res_nf.raw_times))

        entry = {
            "label": label,
            "with_flush_median": round(res_f.median, 4),
            "without_flush_median": round(res_nf.median, 4),
            "diff_pct": round(diff, 2),
            "with_flush_cv": round(cv_f, 2),
            "without_flush_cv": round(cv_nf, 2),
        }
        data.append(entry)
        print(f"  {label}: flush={res_f.median:.4f}ms no_flush={res_nf.median:.4f}ms diff={diff:+.2f}%")

    results["l2_flush"] = data


def test_4_iqr_outlier_removal(results: dict[str, Any]) -> None:
    """Test IQR outlier removal effectiveness."""
    print("\n" + "=" * 60)
    print("TEST 4: IQR outlier removal")
    print("=" * 60)

    data: list[dict[str, Any]] = []

    for label, fn in [
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
        ("slow(>10ms)", _slow_prealloc),
    ]:
        raw = raw_cuda_bench(fn, warmup=10, rounds=200, flush_l2=True)
        cleaned = _remove_outliers_iqr(raw)
        removed = len(raw) - len(cleaned)

        raw_cv = _cv(raw)
        clean_cv = _cv(cleaned)
        improvement = ((raw_cv - clean_cv) / raw_cv) * 100 if raw_cv else 0.0

        entry = {
            "label": label,
            "raw_cv": round(raw_cv, 2),
            "cleaned_cv": round(clean_cv, 2),
            "improvement_pct": round(improvement, 1),
            "outliers_removed": removed,
            "total_samples": len(raw),
        }
        data.append(entry)
        print(f"  {label}: raw_cv={raw_cv:.2f}% cleaned_cv={clean_cv:.2f}% improvement={improvement:.1f}% removed={removed}/{len(raw)}")

    results["iqr_outlier"] = data


def test_5_cv_across_runs(results: dict[str, Any]) -> None:
    """Measure CV across multiple independent runs."""
    print("\n" + "=" * 60)
    print("TEST 5: Coefficient of variation across runs")
    print("=" * 60)

    data: list[dict[str, Any]] = []
    N_RUNS = 5

    for label, fn in [
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
    ]:
        run_medians: list[float] = []
        intra_cvs: list[float] = []

        for _ in range(N_RUNS):
            runner = _BenchmarkRunner(warmup=10, rounds=100, flush_l2=True)
            res = runner(fn, warmup=10, rounds=100)
            run_medians.append(res.median)
            intra_cvs.append(_cv(list(res.raw_times)))

        cv_medians = _cv(run_medians)
        mean_intra = statistics.mean(intra_cvs)

        entry = {
            "label": label,
            "run_medians": [round(m, 4) for m in run_medians],
            "cv_of_medians": round(cv_medians, 2),
            "mean_intra_cv": round(mean_intra, 2),
        }
        data.append(entry)
        print(f"  {label}: medians={[f'{m:.4f}' for m in run_medians]} inter_cv={cv_medians:.2f}% intra_cv={mean_intra:.2f}%")

    results["cv_across_runs"] = data


def test_6_triton_comparison(results: dict[str, Any]) -> None:
    """Compare against triton.testing.do_bench."""
    print("\n" + "=" * 60)
    print("TEST 6: Comparison with triton.testing.do_bench")
    print("=" * 60)

    try:
        from triton.testing import do_bench
    except ImportError:
        print("  SKIPPED: triton not installed")
        results["triton_comparison"] = "skipped"
        return

    data: list[dict[str, Any]] = []

    for label, fn in [
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
        ("slow(>10ms)", _slow_prealloc),
    ]:
        runner = _BenchmarkRunner(warmup=10, rounds=100, flush_l2=True)
        gpucheck_res = runner(fn, warmup=10, rounds=100)

        triton_ms = do_bench(fn, warmup=10, rep=100)

        diff = ((gpucheck_res.median - triton_ms) / triton_ms) * 100 if triton_ms else 0

        entry = {
            "label": label,
            "gpucheck_median": round(gpucheck_res.median, 4),
            "triton_median": round(float(triton_ms), 4),
            "diff_pct": round(diff, 2),
        }
        data.append(entry)
        print(f"  {label}: gpucheck={gpucheck_res.median:.4f}ms triton={triton_ms:.4f}ms diff={diff:+.2f}%")

    results["triton_comparison"] = data


def test_7_kernel_time_ranges(results: dict[str, Any]) -> None:
    """Verify fixture handles different time scales."""
    print("\n" + "=" * 60)
    print("TEST 7: Kernel time range coverage")
    print("=" * 60)

    runner = _BenchmarkRunner(warmup=10, rounds=50, flush_l2=True)
    data: list[dict[str, Any]] = []

    for label, fn in [
        ("tiny(<0.1ms)", _tiny_prealloc),
        ("fast(<1ms)", _fast_prealloc),
        ("medium(1-10ms)", _med_prealloc),
        ("slow(>10ms)", _slow_prealloc),
    ]:
        res = runner(fn, warmup=10, rounds=50)
        cv = _cv(list(res.raw_times))
        entry = {
            "label": label,
            "median_ms": round(res.median, 4),
            "mean_ms": round(res.mean, 4),
            "std_ms": round(res.std, 4),
            "cv_pct": round(cv, 2),
            "min_ms": round(res.min, 4),
            "max_ms": round(res.max, 4),
            "outliers_removed": res.outliers_removed,
        }
        data.append(entry)
        print(f"  {label}: median={res.median:.4f}ms cv={cv:.2f}% range=[{res.min:.4f},{res.max:.4f}]")

    results["time_ranges"] = data


# ──────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────

def generate_report(results: dict[str, Any], output_path: Path) -> None:
    lines: list[str] = []
    w = lines.append

    w("# gpucheck `gpu_benchmark` Fixture: Deep Accuracy Analysis")
    w("")
    w(f"**Device:** {torch.cuda.get_device_name(0)}")
    w(f"**CUDA:** {torch.version.cuda}")
    w(f"**PyTorch:** {torch.__version__}")
    w(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w("")

    # --- Test 1 ---
    w("## 1. gpu_benchmark vs Raw CUDA Event Timing")
    w("")
    w("Both use `torch.cuda.Event` with same warmup/rounds/L2-flush.")
    w("Only difference: gpucheck applies IQR outlier removal before computing statistics.")
    w("")
    w("| Kernel | gpucheck median (ms) | raw median (ms) | median diff | gpucheck mean | raw mean | mean diff |")
    w("|--------|---------------------|----------------|------------|--------------|---------|----------|")
    for r in results.get("accuracy_vs_raw", []):
        w(f"| {r['label']} | {r['gpucheck_median']:.4f} | {r['raw_median']:.4f} | {r['median_diff_pct']:+.2f}% | {r['gpucheck_mean']:.4f} | {r['raw_mean']:.4f} | {r['mean_diff_pct']:+.2f}% |")
    w("")
    diffs = [abs(r["median_diff_pct"]) for r in results.get("accuracy_vs_raw", [])]
    max_diff = max(diffs) if diffs else 0
    if max_diff < 5:
        w(f"**Finding:** Maximum median deviation is **{max_diff:.2f}%** -- highly accurate. "
          f"Small differences are from IQR outlier removal (gpucheck cleans, raw does not).")
    elif max_diff < 15:
        w(f"**Finding:** Maximum median deviation is **{max_diff:.2f}%** -- reasonably accurate.")
    else:
        w(f"**Finding:** Maximum median deviation is **{max_diff:.2f}%** -- investigate overhead sources.")
    w("")

    # --- Test 2 ---
    w("## 2. Fixture Overhead")
    w("")
    w("Wall-clock time: full `_BenchmarkRunner.__call__` vs bare CUDA event loop.")
    w("")
    w("| Kernel | fixture wall (s) | raw wall (s) | overhead (s) | overhead % |")
    w("|--------|-----------------|-------------|-------------|-----------|")
    for o in results.get("overhead", []):
        w(f"| {o['label']} | {o['fixture_wall_s']:.4f} | {o['raw_wall_s']:.4f} | {o['overhead_s']:.4f} | {o['overhead_pct']:+.1f}% |")
    w("")
    oh = results.get("overhead", [])
    if oh:
        avg = statistics.mean([abs(o["overhead_pct"]) for o in oh])
        w(f"**Finding:** Average overhead is **{avg:.1f}%**. "
          f"{'Negligible -- IQR computation and stats are cheap.' if avg < 10 else 'Measurable -- dominated by L2 flush or stats.'}")
    w("")

    # --- Test 3 ---
    w("## 3. L2 Cache Flush Effectiveness")
    w("")
    w("Memory-bound kernels should run slower with L2 flush (cold cache). Compute-bound less affected.")
    w("")
    w("| Kernel | with flush (ms) | without flush (ms) | diff | flush CV | no-flush CV |")
    w("|--------|----------------|-------------------|------|---------|------------|")
    for r in results.get("l2_flush", []):
        w(f"| {r['label']} | {r['with_flush_median']:.4f} | {r['without_flush_median']:.4f} | {r['diff_pct']:+.2f}% | {r['with_flush_cv']:.2f}% | {r['without_flush_cv']:.2f}% |")
    w("")
    l2 = results.get("l2_flush", [])
    membound = [r for r in l2 if "membound" in r["label"]]
    if membound:
        d = membound[0]["diff_pct"]
        if d > 0:
            w(f"**Finding:** L2 flush makes membound kernel **{d:.1f}% slower** -- expected and correct. "
              "Without flush, L2 keeps hot data from prior iteration, giving artificially fast times. "
              "Flushing gives realistic cold-cache measurement.")
        else:
            w(f"**Finding:** L2 flush had **{d:.1f}%** effect on membound kernel. "
              "Working set may exceed L2, or flush buffer is undersized.")
    w("")

    # --- Test 4 ---
    w("## 4. IQR Outlier Removal Effectiveness")
    w("")
    w("Does IQR (factor=1.5) reduce coefficient of variation?")
    w("")
    w("| Kernel | raw CV | cleaned CV | improvement | outliers removed |")
    w("|--------|--------|-----------|-------------|-----------------|")
    for r in results.get("iqr_outlier", []):
        w(f"| {r['label']} | {r['raw_cv']:.2f}% | {r['cleaned_cv']:.2f}% | {r['improvement_pct']:.1f}% | {r['outliers_removed']}/{r['total_samples']} |")
    w("")
    iqr = results.get("iqr_outlier", [])
    if iqr:
        avg_imp = statistics.mean([r["improvement_pct"] for r in iqr])
        avg_rm = statistics.mean([r["outliers_removed"] / r["total_samples"] * 100 for r in iqr])
        w(f"**Finding:** IQR removal improves CV by **{avg_imp:.1f}%** on average, removing ~{avg_rm:.1f}% of samples. ")
        if avg_imp > 10:
            w("Highly effective at filtering OS scheduling jitter and GPU clock fluctuations.")
        elif avg_imp > 0:
            w("Modest improvement -- raw data is already fairly clean on this GPU.")
        else:
            w("No improvement -- data is clean or IQR factor too aggressive.")
    w("")

    # --- Test 5 ---
    w("## 5. Coefficient of Variation Across Independent Runs")
    w("")
    w("5 independent benchmark runs (100 rounds each), measuring stability of median.")
    w("")
    w("| Kernel | medians (ms) | inter-run CV | mean intra-run CV |")
    w("|--------|-------------|-------------|------------------|")
    for r in results.get("cv_across_runs", []):
        meds = ", ".join(f"{m:.4f}" for m in r["run_medians"])
        w(f"| {r['label']} | [{meds}] | {r['cv_of_medians']:.2f}% | {r['mean_intra_cv']:.2f}% |")
    w("")
    cvd = results.get("cv_across_runs", [])
    if cvd:
        mx = max(r["cv_of_medians"] for r in cvd)
        if mx < 3:
            w(f"**Finding:** Max inter-run CV is **{mx:.2f}%** -- highly reproducible.")
        elif mx < 10:
            w(f"**Finding:** Max inter-run CV is **{mx:.2f}%** -- moderate variability. Consider more rounds.")
        else:
            w(f"**Finding:** Max inter-run CV is **{mx:.2f}%** -- high variability. Check GPU throttling.")
    w("")

    # --- Test 6 ---
    w("## 6. Comparison with `triton.testing.do_bench`")
    w("")
    tri = results.get("triton_comparison")
    if isinstance(tri, str):
        w("**Skipped:** triton not installed.")
    else:
        w("Both tools use CUDA events. triton uses percentile-based timing with its own warmup heuristic.")
        w("")
        w("| Kernel | gpucheck median (ms) | triton do_bench (ms) | diff |")
        w("|--------|---------------------|---------------------|------|")
        for r in (tri or []):
            w(f"| {r['label']} | {r['gpucheck_median']:.4f} | {r['triton_median']:.4f} | {r['diff_pct']:+.2f}% |")
        w("")
        if tri:
            mx = max(abs(r["diff_pct"]) for r in tri)
            w(f"**Finding:** Max deviation from triton.do_bench is **{mx:.2f}%**. ")
            if mx < 10:
                w("gpucheck and triton agree closely -- both use CUDA events under the hood.")
            elif mx < 25:
                w("Moderate difference -- likely due to different L2 flush strategies and outlier handling.")
            else:
                w("Significant difference -- investigate warmup/rep defaults.")
    w("")

    # --- Test 7 ---
    w("## 7. Kernel Time Range Coverage")
    w("")
    w("| Kernel | median (ms) | std (ms) | CV | range [min, max] (ms) | outliers |")
    w("|--------|------------|---------|-----|----------------------|---------|")
    for r in results.get("time_ranges", []):
        w(f"| {r['label']} | {r['median_ms']:.4f} | {r['std_ms']:.4f} | {r['cv_pct']:.2f}% | [{r['min_ms']:.4f}, {r['max_ms']:.4f}] | {r['outliers_removed']} |")
    w("")
    tr = results.get("time_ranges", [])
    fast = [r for r in tr if "tiny" in r["label"] or "fast" in r["label"]]
    slow = [r for r in tr if "slow" in r["label"]]
    if fast and slow:
        w(f"**Finding:** Fast kernels CV={fast[0]['cv_pct']:.1f}%, slow kernels CV={slow[0]['cv_pct']:.1f}%. "
          "Fast kernels show higher relative variance due to fixed overhead (event record, sync) being a larger fraction.")
    w("")

    # --- Summary ---
    w("## Summary")
    w("")
    w("| Aspect | Rating | Notes |")
    w("|--------|--------|-------|")

    acc = results.get("accuracy_vs_raw", [])
    if acc:
        mx = max(abs(r["median_diff_pct"]) for r in acc)
        rating = "Excellent" if mx < 5 else "Good" if mx < 15 else "Fair"
        w(f"| Timing accuracy vs raw CUDA | {rating} | Max {mx:.1f}% deviation |")

    if oh:
        avg = statistics.mean([abs(o["overhead_pct"]) for o in oh])
        rating = "Excellent" if avg < 5 else "Good" if avg < 15 else "Fair"
        w(f"| Fixture overhead | {rating} | {avg:.1f}% average |")

    if membound:
        works = "Yes" if membound[0]["diff_pct"] > 1 else "Marginal"
        w(f"| L2 flush working | {works} | {membound[0]['diff_pct']:.1f}% effect on membound |")

    if iqr:
        avg_imp = statistics.mean([r["improvement_pct"] for r in iqr])
        rating = "Effective" if avg_imp > 10 else "Modest" if avg_imp > 0 else "No effect"
        w(f"| IQR outlier removal | {rating} | {avg_imp:.1f}% CV improvement |")

    if cvd:
        mx = max(r["cv_of_medians"] for r in cvd)
        rating = "Excellent" if mx < 3 else "Good" if mx < 10 else "Fair"
        w(f"| Reproducibility | {rating} | {mx:.2f}% inter-run CV |")

    if isinstance(tri, list) and tri:
        mx = max(abs(r["diff_pct"]) for r in tri)
        rating = "Excellent" if mx < 10 else "Good" if mx < 25 else "Fair"
        w(f"| Agreement with triton.do_bench | {rating} | Max {mx:.1f}% deviation |")

    w("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {output_path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA not available")
        sys.exit(1)

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")

    _init_tensors()
    results: dict[str, Any] = {}

    test_1_accuracy_vs_raw_cuda(results)
    test_2_fixture_overhead(results)
    test_3_l2_flush_effect(results)
    test_4_iqr_outlier_removal(results)
    test_5_cv_across_runs(results)
    test_6_triton_comparison(results)
    test_7_kernel_time_ranges(results)

    output = Path(
        "/home/akash/PROJECTS/github-projects-all/new-project/gpucheck/deep_analysis/benchmark_accuracy.md"
    )
    generate_report(results, output)

    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    main()
