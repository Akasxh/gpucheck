"""Machine-readable JSON reporter for gpucheck."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from gpucheck.reporting.console import BenchmarkEntry, MemoryEntry, TestResult


@dataclass(slots=True)
class RunRecord:
    """All data for a single test run."""

    timestamp: str = ""
    gpu_info: dict[str, Any] = field(default_factory=dict)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    memory: list[dict[str, Any]] = field(default_factory=list)


class JSONReporter:
    """Writes structured JSON reports to ``.gpucheck/`` directory."""

    def __init__(self, output_dir: str | Path = ".gpucheck") -> None:
        self._output_dir = Path(output_dir)
        self._record = RunRecord(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def set_gpu_info(self, info: dict[str, Any]) -> None:
        self._record.gpu_info = info

    def add_test_result(self, result: TestResult) -> None:
        self._record.test_results.append({
            "name": result.name,
            "status": result.status,
            "duration": result.duration,
            "message": result.message,
        })

    def add_benchmark(self, entry: BenchmarkEntry) -> None:
        self._record.benchmarks.append({
            "name": entry.name,
            "median_ms": entry.median * 1000,
            "std_ms": entry.std * 1000,
            "throughput": entry.throughput,
            "samples": len(entry.times),
            "times": list(entry.times),
        })

    def add_memory(self, entry: MemoryEntry) -> None:
        self._record.memory.append({
            "name": entry.name,
            "peak_mb": entry.peak_mb,
            "leaked_mb": entry.leaked_mb,
            "allocations": entry.allocations,
        })

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def flush(self) -> Path:
        """Write the collected report to ``<output_dir>/results.json`` and return the path."""
        self._output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._output_dir / "results.json"

        payload: dict[str, Any] = {
            "schema_version": 1,
            "timestamp": self._record.timestamp,
            "gpu_info": self._record.gpu_info,
            "test_results": self._record.test_results,
            "benchmarks": self._record.benchmarks,
            "memory": self._record.memory,
        }

        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out_path

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_runs(
        baseline_path: str | Path,
        current_path: str | Path,
    ) -> dict[str, Any]:
        """Compare two JSON result files and return a diff summary.

        Returns a dict with per-benchmark deltas (median, throughput) and
        per-test status changes.
        """
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        current = json.loads(Path(current_path).read_text(encoding="utf-8"))

        base_benchmarks = {b["name"]: b for b in baseline.get("benchmarks", [])}
        curr_benchmarks = {b["name"]: b for b in current.get("benchmarks", [])}

        benchmark_diffs: list[dict[str, Any]] = []
        for name in sorted(set(base_benchmarks) | set(curr_benchmarks)):
            base = base_benchmarks.get(name)
            curr = curr_benchmarks.get(name)
            entry: dict[str, Any] = {"name": name}

            if base and curr:
                base_med = base["median_ms"]
                curr_med = curr["median_ms"]
                entry["baseline_median_ms"] = base_med
                entry["current_median_ms"] = curr_med
                entry["delta_ms"] = curr_med - base_med
                entry["delta_pct"] = (
                    ((curr_med - base_med) / base_med * 100) if base_med > 0 else 0.0
                )
                entry["status"] = "regression" if curr_med > base_med * 1.05 else "ok"
            elif curr:
                entry["status"] = "new"
                entry["current_median_ms"] = curr["median_ms"]
            else:
                entry["status"] = "removed"
                entry["baseline_median_ms"] = base["median_ms"] if base else 0

            benchmark_diffs.append(entry)

        # Test status changes
        base_tests = {t["name"]: t["status"] for t in baseline.get("test_results", [])}
        curr_tests = {t["name"]: t["status"] for t in current.get("test_results", [])}

        test_changes: list[dict[str, str]] = []
        for name in sorted(set(base_tests) | set(curr_tests)):
            old = base_tests.get(name, "absent")
            new = curr_tests.get(name, "absent")
            if old != new:
                test_changes.append({"name": name, "was": old, "now": new})

        return {
            "benchmarks": benchmark_diffs,
            "test_changes": test_changes,
        }
