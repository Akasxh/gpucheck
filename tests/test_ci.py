"""Tests for CI integration — GitHub Actions annotations and JUnit XML."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

from gpucheck.reporting.ci import emit_github_annotations, write_junit_xml
from gpucheck.reporting.console import TestResult


class TestGitHubAnnotations:
    """Verify GitHub Actions annotation format (::error/::warning)."""

    def _capture_annotations(self, results: list[TestResult]) -> str:
        old = os.environ.get("GITHUB_ACTIONS")
        os.environ["GITHUB_ACTIONS"] = "1"
        try:
            buf = io.StringIO()
            saved = sys.stdout
            sys.stdout = buf
            emit_github_annotations(results)
            sys.stdout = saved
            return buf.getvalue()
        finally:
            if old is None:
                os.environ.pop("GITHUB_ACTIONS", None)
            else:
                os.environ["GITHUB_ACTIONS"] = old

    def test_error_with_file_and_line(self) -> None:
        results = [TestResult("test_a", "failed", message="bad", file="test.py", line=42)]
        output = self._capture_annotations(results)
        assert output == "::error file=test.py,line=42,title=FAIL: test_a::bad\n"

    def test_error_with_file_only(self) -> None:
        results = [TestResult("test_a", "failed", message="bad", file="test.py")]
        output = self._capture_annotations(results)
        assert output == "::error file=test.py,title=FAIL: test_a::bad\n"

    def test_error_without_file(self) -> None:
        results = [TestResult("test_a", "failed", message="bad")]
        output = self._capture_annotations(results)
        assert output == "::error title=FAIL: test_a::bad\n"

    def test_warning_for_skipped(self) -> None:
        results = [TestResult("test_b", "skipped", message="no gpu")]
        output = self._capture_annotations(results)
        assert output == "::warning title=SKIP: test_b::no gpu\n"

    def test_passed_emits_nothing(self) -> None:
        results = [TestResult("test_c", "passed")]
        output = self._capture_annotations(results)
        assert output == ""

    def test_noop_outside_github_actions(self) -> None:
        old = os.environ.pop("GITHUB_ACTIONS", None)
        try:
            buf = io.StringIO()
            saved = sys.stdout
            sys.stdout = buf
            emit_github_annotations([TestResult("x", "failed", message="m")])
            sys.stdout = saved
            assert buf.getvalue() == ""
        finally:
            if old is not None:
                os.environ["GITHUB_ACTIONS"] = old

    def test_newlines_escaped(self) -> None:
        results = [TestResult("t", "failed", message="line1\nline2")]
        output = self._capture_annotations(results)
        assert "%0A" in output
        assert "\nline2" not in output

    def test_error_status(self) -> None:
        results = [TestResult("t", "error", message="boom", file="f.py", line=1)]
        output = self._capture_annotations(results)
        assert output == "::error file=f.py,line=1,title=ERROR: t::boom\n"


class TestJUnitXML:
    """Verify JUnit XML report generation."""

    def test_basic_report(self) -> None:
        results = [
            TestResult("t1", "passed"),
            TestResult("t2", "failed", message="err"),
        ]
        with tempfile.TemporaryDirectory() as d:
            p = write_junit_xml(results, output_path=Path(d) / "j.xml")
            content = p.read_text()
            assert 'tests="2"' in content
            assert 'failures="1"' in content
            assert 'errors="0"' in content

    def test_skipped_and_error_counts(self) -> None:
        results = [
            TestResult("t1", "passed"),
            TestResult("t2", "skipped", message="skip"),
            TestResult("t3", "error", message="err"),
        ]
        with tempfile.TemporaryDirectory() as d:
            p = write_junit_xml(results, output_path=Path(d) / "j.xml")
            content = p.read_text()
            assert 'tests="3"' in content
            assert 'skipped="1"' in content
            assert 'errors="1"' in content
            assert 'failures="0"' in content
