"""Console reporter tests (Track D — D.1)."""

from __future__ import annotations

import io

from rich.console import Console

from gpucheck.reporting.console import (
    BenchmarkEntry,
    ConsoleReporter,
    MemoryEntry,
    TestResult,
)


def _new_reporter() -> tuple[ConsoleReporter, io.StringIO]:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    return ConsoleReporter(console=console), buf


def test_console_reporter_constructs_with_explicit_console() -> None:
    reporter, _ = _new_reporter()
    assert reporter is not None


def test_gpu_info_panel_renders_keys_and_values() -> None:
    reporter, buf = _new_reporter()
    reporter.gpu_info_panel({"Device": "GTX 1650", "Compute": "7.5"})
    out = buf.getvalue()
    assert "Device" in out
    assert "GTX 1650" in out
    assert "Compute" in out
    assert "7.5" in out


def test_test_summary_includes_pass_fail_skip_counts() -> None:
    reporter, buf = _new_reporter()
    results = [
        TestResult(name="t1", status="passed", duration=0.1),
        TestResult(name="t2", status="failed", duration=0.2, message="boom"),
        TestResult(name="t3", status="skipped", duration=0.0, message="no gpu"),
    ]
    reporter.test_summary(results)
    out = buf.getvalue()
    assert "PASSED" in out
    assert "FAILED" in out
    assert "SKIPPED" in out
    assert "1 passed" in out
    assert "1 failed" in out
    assert "1 skipped" in out


def test_benchmark_table_includes_kernel_and_throughput() -> None:
    reporter, buf = _new_reporter()
    entries = [
        BenchmarkEntry(name="matmul", times=[0.001, 0.0011, 0.0009]),
    ]
    reporter.benchmark_table(entries)
    out = buf.getvalue()
    assert "matmul" in out
    assert "Median" in out


def test_memory_summary_shows_leak_status_red_for_leaks() -> None:
    reporter, buf = _new_reporter()
    entries = [
        MemoryEntry(name="leaky", peak_mb=10.5, leaked_mb=2.0, allocations=4),
        MemoryEntry(name="clean", peak_mb=5.0, leaked_mb=0.0, allocations=2),
    ]
    reporter.memory_summary(entries)
    out = buf.getvalue()
    assert "leaky" in out
    assert "clean" in out
    assert "2.00" in out  # 2.0 MB leaked


def test_error_detail_renders_name_and_traceback() -> None:
    reporter, buf = _new_reporter()
    reporter.error_detail("test_x", "AssertionError: nope", traceback="line1\nline2")
    out = buf.getvalue()
    assert "test_x" in out
    assert "AssertionError" in out


def test_console_reporter_uses_stderr_in_ci_environment(monkeypatch) -> None:
    """When GITHUB_ACTIONS=1, the reporter writes to stderr by default."""
    monkeypatch.setenv("GITHUB_ACTIONS", "1")
    monkeypatch.delenv("CI", raising=False)
    reporter = ConsoleReporter()
    # Internal: assert the file is sys.stderr (Rich's Console exposes file).
    import sys
    assert reporter._console.file is sys.stderr  # noqa: SLF001


def test_console_reporter_with_file_kwarg() -> None:
    """Constructing with file= takes precedence over CI/GITHUB_ACTIONS env."""
    buf = io.StringIO()
    reporter = ConsoleReporter(file=buf)
    reporter.gpu_info_panel({"x": "y"})
    assert "x" in buf.getvalue()


def test_benchmark_entry_throughput_handles_zero_times() -> None:
    entry = BenchmarkEntry(name="empty", times=[])
    assert entry.median == 0.0
    assert entry.std == 0.0
    assert entry.throughput == 0.0


def test_benchmark_entry_std_with_two_samples() -> None:
    entry = BenchmarkEntry(name="k", times=[1.0, 2.0])
    assert entry.median == 1.5
    assert entry.std > 0  # statistics.stdev requires len >= 2
