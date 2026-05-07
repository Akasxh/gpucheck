"""JSONReporter tests (Track D — D.1)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from gpucheck.reporting.console import BenchmarkEntry, MemoryEntry, TestResult
from gpucheck.reporting.json import JSONReporter

if TYPE_CHECKING:
    from pathlib import Path


def test_json_reporter_writes_results_json(tmp_path: Path) -> None:
    rep = JSONReporter(output_dir=tmp_path)
    rep.set_gpu_info({"name": "GTX 1650"})
    rep.add_test_result(TestResult(name="t1", status="passed", duration=0.1))
    rep.add_test_result(
        TestResult(name="t2", status="failed", duration=0.2, message="boom"),
    )
    rep.add_benchmark(BenchmarkEntry(name="matmul", times=[0.001, 0.0011]))
    rep.add_memory(MemoryEntry(name="t1", peak_mb=10.0, leaked_mb=0.0, allocations=2))
    out = rep.flush()

    assert out == tmp_path / "results.json"
    data = json.loads(out.read_text())
    assert data["schema_version"] == 1
    assert data["gpu_info"] == {"name": "GTX 1650"}
    assert len(data["test_results"]) == 2
    assert data["test_results"][0]["name"] == "t1"
    assert data["test_results"][1]["status"] == "failed"
    assert len(data["benchmarks"]) == 1
    assert data["benchmarks"][0]["name"] == "matmul"
    assert data["benchmarks"][0]["samples"] == 2
    assert len(data["memory"]) == 1


def test_compare_runs_classifies_regression(tmp_path: Path) -> None:
    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"

    base_path.write_text(json.dumps({
        "benchmarks": [{"name": "matmul", "median_ms": 1.0}],
        "test_results": [{"name": "t1", "status": "passed"}],
    }))
    curr_path.write_text(json.dumps({
        "benchmarks": [{"name": "matmul", "median_ms": 1.5}],  # +50% slower
        "test_results": [{"name": "t1", "status": "failed"}],
    }))

    diff = JSONReporter.compare_runs(base_path, curr_path, regression_threshold=0.05)
    assert any(b["status"] == "regression" for b in diff["benchmarks"])
    assert any(t["was"] == "passed" and t["now"] == "failed" for t in diff["test_changes"])


def test_compare_runs_classifies_ok(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    curr = tmp_path / "c.json"
    base.write_text(json.dumps({
        "benchmarks": [{"name": "matmul", "median_ms": 1.0}],
        "test_results": [],
    }))
    curr.write_text(json.dumps({
        "benchmarks": [{"name": "matmul", "median_ms": 1.0}],
        "test_results": [],
    }))
    diff = JSONReporter.compare_runs(base, curr)
    assert diff["benchmarks"][0]["status"] == "ok"


def test_compare_runs_classifies_new_and_removed(tmp_path: Path) -> None:
    base = tmp_path / "b.json"
    curr = tmp_path / "c.json"
    base.write_text(json.dumps({
        "benchmarks": [{"name": "old", "median_ms": 1.0}],
        "test_results": [],
    }))
    curr.write_text(json.dumps({
        "benchmarks": [{"name": "new", "median_ms": 2.0}],
        "test_results": [],
    }))
    diff = JSONReporter.compare_runs(base, curr)
    statuses = {b["name"]: b["status"] for b in diff["benchmarks"]}
    assert statuses["old"] == "removed"
    assert statuses["new"] == "new"


def test_compare_runs_zero_baseline_no_div_by_zero(tmp_path: Path) -> None:
    """A 0-ms baseline must not crash via division-by-zero."""
    base = tmp_path / "b.json"
    curr = tmp_path / "c.json"
    base.write_text(json.dumps({
        "benchmarks": [{"name": "k", "median_ms": 0.0}],
        "test_results": [],
    }))
    curr.write_text(json.dumps({
        "benchmarks": [{"name": "k", "median_ms": 1.0}],
        "test_results": [],
    }))
    diff = JSONReporter.compare_runs(base, curr)
    assert diff["benchmarks"][0]["delta_pct"] == 0.0


def test_json_reporter_creates_output_dir_if_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "subdir" / "deeper"
    rep = JSONReporter(output_dir=out_dir)
    rep.flush()
    assert (out_dir / "results.json").exists()
