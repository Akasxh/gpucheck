"""HTML dashboard tests (Track D — D.2)."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from gpucheck.reporting.html import HTMLReporter

if TYPE_CHECKING:
    from pathlib import Path


def _write_sample_json(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "timestamp": "2026-05-01T12:00:00",
        "gpu_info": {"name": "GTX 1650", "compute": "7.5"},
        "test_results": [
            {"name": "test_pass", "status": "passed", "duration": 0.1, "message": ""},
            {"name": "test_fail", "status": "failed", "duration": 0.2,
             "message": "AssertionError: nope"},
            {"name": "test_skip", "status": "skipped", "duration": 0.0,
             "message": "no gpu"},
        ],
        "benchmarks": [
            {"name": "matmul", "median_ms": 1.0, "std_ms": 0.05, "samples": 100, "times": []},
            {"name": "softmax", "median_ms": 0.5, "std_ms": 0.02, "samples": 100, "times": []},
        ],
        "memory": [
            {"name": "test_pass", "peak_mb": 10.0, "leaked_mb": 0.0, "allocations": 4},
            {"name": "test_fail", "peak_mb": 20.0, "leaked_mb": 2.5, "allocations": 8},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


class _TagCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        self.tags.append(tag)


def test_html_reporter_writes_self_contained_html(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "dashboard.html"

    HTMLReporter(json_path).render(out)
    assert out.exists()
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    # No external assets (no <link href=…> with http: or https:).
    assert 'href="http' not in text
    assert 'src="http' not in text


def test_html_reporter_contains_summary_counts(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "dashboard.html"
    HTMLReporter(json_path).render(out)
    text = out.read_text()
    # 1 passed, 1 failed, 1 skipped — summary cards must surface those.
    assert ">1<" in text  # one of the cards renders 1
    # Test names appear in the table.
    assert "test_pass" in text
    assert "test_fail" in text


def test_html_reporter_includes_benchmark_table_with_kernel_names(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "dashboard.html"
    HTMLReporter(json_path).render(out)
    text = out.read_text()
    assert "matmul" in text
    assert "softmax" in text
    assert "<svg" in text  # inline bar chart


def test_html_reporter_with_comparison_band(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "dashboard.html"
    diff = {
        "benchmarks": [
            {
                "name": "matmul", "baseline_median_ms": 1.0,
                "current_median_ms": 1.6, "delta_pct": 60.0,
                "status": "regression",
            },
        ],
    }
    HTMLReporter(json_path, comparison=diff).render(out)
    text = out.read_text()
    assert "Comparison vs Baseline" in text
    assert "REGRESSION" in text


def test_html_reporter_creates_parent_dir(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "deep" / "nested" / "dashboard.html"
    HTMLReporter(json_path).render(out)
    assert out.exists()


def test_html_reporter_html_is_well_formed(tmp_path: Path) -> None:
    """HTMLParser tolerates malformed HTML, but should at least parse and
    open balanced top-level tags (html, body)."""
    json_path = tmp_path / "results.json"
    _write_sample_json(json_path)
    out = tmp_path / "dashboard.html"
    HTMLReporter(json_path).render(out)
    text = out.read_text()
    counter = _TagCounter()
    counter.feed(text)
    assert "html" in counter.tags
    assert "body" in counter.tags
    assert "table" in counter.tags


def test_html_reporter_handles_empty_data(tmp_path: Path) -> None:
    """No tests / no benchmarks must not crash the renderer."""
    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps({
        "schema_version": 1, "timestamp": "", "gpu_info": {},
        "test_results": [], "benchmarks": [], "memory": [],
    }))
    out = tmp_path / "dash.html"
    HTMLReporter(json_path).render(out)
    text = out.read_text()
    assert "Summary" in text
    # Test Results / Benchmarks / Memory sections are skipped when empty.
    assert "Test Results" not in text
    assert "Benchmarks" not in text


def test_html_reporter_escapes_html_in_messages(tmp_path: Path) -> None:
    json_path = tmp_path / "results.json"
    json_path.write_text(json.dumps({
        "test_results": [
            {"name": "test_x", "status": "failed", "duration": 0.1,
             "message": "<script>alert('xss')</script>"},
        ],
        "benchmarks": [], "memory": [], "gpu_info": {}, "timestamp": "",
    }))
    out = tmp_path / "dash.html"
    HTMLReporter(json_path).render(out)
    text = out.read_text()
    assert "<script>alert" not in text
    assert "&lt;script&gt;" in text
