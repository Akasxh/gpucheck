"""CI reporting tests (Track D — D.1)."""

from __future__ import annotations

import io
import sys
from typing import TYPE_CHECKING

from gpucheck.reporting.ci import (
    emit_github_annotations,
    generate_pr_comment,
    write_junit_xml,
)
from gpucheck.reporting.console import TestResult

if TYPE_CHECKING:
    from pathlib import Path


def test_emit_github_annotations_writes_error_lines(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    results = [
        TestResult(
            name="tests/test_x.py::test_y", status="failed",
            duration=0.1, message="AssertionError\nexpected != actual",
            file="tests/test_x.py", line=42,
        ),
    ]
    emit_github_annotations(results)
    out = captured.getvalue()
    assert "::error" in out
    assert "file=tests/test_x.py" in out
    assert "line=42" in out
    assert "%0A" in out  # newline encoded in annotation message


def test_emit_github_annotations_skipped_warning(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    results = [TestResult(name="t1", status="skipped", message="no gpu")]
    emit_github_annotations(results)
    assert "::warning" in captured.getvalue()


def test_emit_github_annotations_passed_is_silent(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    emit_github_annotations([TestResult(name="t1", status="passed")])
    assert captured.getvalue() == ""


def test_emit_github_annotations_no_op_outside_actions(monkeypatch) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)
    emit_github_annotations([TestResult(name="t1", status="failed", message="boom")])
    assert captured.getvalue() == ""


def test_write_junit_xml_emits_valid_xml(tmp_path: Path) -> None:
    out = tmp_path / "junit.xml"
    results = [
        TestResult(name="t1", status="passed", duration=0.1),
        TestResult(name="t2", status="failed", duration=0.2, message="boom"),
        TestResult(name="t3", status="skipped", duration=0.0, message="no gpu"),
        TestResult(name="t4", status="error", duration=0.05, message="crashed"),
    ]
    path = write_junit_xml(results, output_path=out, suite_name="gpucheck")

    assert path == out
    text = out.read_text()
    assert '<testsuite' in text
    assert 'tests="4"' in text
    assert 'failures="1"' in text
    assert 'errors="1"' in text
    assert 'skipped="1"' in text
    # Validate it parses as XML.
    import xml.etree.ElementTree as ET
    root = ET.parse(out).getroot()
    assert root.tag == "testsuite"
    assert len(list(root)) == 4


def test_generate_pr_comment_lists_benchmarks_and_test_changes() -> None:
    diff = {
        "benchmarks": [
            {
                "name": "matmul",
                "baseline_median_ms": 1.0,
                "current_median_ms": 1.6,
                "delta_pct": 60.0,
                "status": "regression",
            },
            {
                "name": "softmax",
                "baseline_median_ms": 0.5,
                "current_median_ms": 0.51,
                "delta_pct": 2.0,
                "status": "ok",
            },
            {"name": "newkern", "current_median_ms": 0.3, "status": "new"},
            {"name": "removed", "baseline_median_ms": 0.7, "status": "removed"},
        ],
        "test_changes": [
            {"name": "test_x", "was": "passed", "now": "failed"},
        ],
    }
    body = generate_pr_comment(diff)
    assert "## gpucheck Benchmark Comparison" in body
    assert "matmul" in body
    assert "softmax" in body
    assert "newkern" in body
    assert "removed" in body
    assert "test_x" in body
    assert ":red_circle:" in body
    assert ":green_circle:" in body
    assert ":new:" in body


def test_generate_pr_comment_empty_diff_returns_friendly_message() -> None:
    body = generate_pr_comment({"benchmarks": [], "test_changes": []})
    assert "No changes detected." in body
