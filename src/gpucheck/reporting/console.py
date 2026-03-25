"""Rich-based terminal reporter for gpucheck test results."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class TestResult:
    """Single test outcome."""

    name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float = 0.0
    message: str = ""
    file: str = ""
    line: int = 0


@dataclass(frozen=True, slots=True)
class BenchmarkEntry:
    """Aggregated benchmark timing for one kernel."""

    name: str
    times: Sequence[float]

    @property
    def median(self) -> float:
        return statistics.median(self.times) if self.times else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.times) if len(self.times) >= 2 else 0.0

    @property
    def throughput(self) -> float:
        """Iterations per second based on median."""
        med = self.median
        return 1.0 / med if med > 0 else 0.0


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """Memory usage summary for one test."""

    name: str
    peak_mb: float
    leaked_mb: float
    allocations: int


_STATUS_STYLE: dict[str, str] = {
    "passed": "bold green",
    "failed": "bold red",
    "skipped": "bold yellow",
    "error": "bold red",
}


class ConsoleReporter:
    """Rich-powered terminal reporter for gpucheck sessions."""

    def __init__(
        self, *, console: Console | None = None, verbose: bool = False, file: Any = None,
    ) -> None:
        import os
        import sys

        if console is not None:
            self._console = console
        elif file is not None:
            self._console = Console(file=file)
        elif os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
            self._console = Console(file=sys.stderr)
        else:
            self._console = Console()
        self._verbose = verbose

    # ------------------------------------------------------------------
    # GPU info
    # ------------------------------------------------------------------

    def gpu_info_panel(self, info: dict[str, Any]) -> None:
        """Print a GPU summary panel at test start."""
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold cyan")
        table.add_column("Value")

        for key, value in info.items():
            table.add_row(key, str(value))

        self._console.print(
            Panel(table, title="[bold blue]GPU Info[/bold blue]", border_style="blue")
        )

    # ------------------------------------------------------------------
    # Test results
    # ------------------------------------------------------------------

    def test_summary(self, results: Sequence[TestResult]) -> None:
        """Print a test result table with pass/fail/skip counts."""
        passed = sum(1 for r in results if r.status == "passed")
        failed = sum(1 for r in results if r.status == "failed")
        skipped = sum(1 for r in results if r.status == "skipped")

        table = Table(title="Test Results", show_lines=True)
        table.add_column("Test", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right")
        table.add_column("Message", max_width=60)

        for r in results:
            style = _STATUS_STYLE.get(r.status, "")
            status_text = Text(r.status.upper(), style=style)
            table.add_row(r.name, status_text, f"{r.duration:.4f}s", r.message)

        self._console.print(table)

        summary = Text()
        summary.append(f"{passed} passed", style="bold green")
        summary.append(" | ")
        summary.append(f"{failed} failed", style="bold red")
        summary.append(" | ")
        summary.append(f"{skipped} skipped", style="bold yellow")
        self._console.print(Panel(summary, title="Summary", border_style="cyan"))

    # ------------------------------------------------------------------
    # Benchmarks
    # ------------------------------------------------------------------

    def benchmark_table(self, entries: Sequence[BenchmarkEntry]) -> None:
        """Print a benchmark results table."""
        table = Table(title="Benchmark Results")
        table.add_column("Kernel", style="bold")
        table.add_column("Median", justify="right")
        table.add_column("Std Dev", justify="right")
        table.add_column("Throughput", justify="right")
        table.add_column("Samples", justify="right")

        for e in entries:
            table.add_row(
                e.name,
                f"{e.median * 1000:.3f} ms",
                f"{e.std * 1000:.3f} ms",
                f"{e.throughput:.1f} it/s",
                str(len(e.times)),
            )

        self._console.print(table)

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def memory_summary(self, entries: Sequence[MemoryEntry]) -> None:
        """Print GPU memory usage table."""
        table = Table(title="Memory Usage")
        table.add_column("Test", style="bold")
        table.add_column("Peak", justify="right")
        table.add_column("Leaked", justify="right")
        table.add_column("Allocations", justify="right")

        for e in entries:
            leak_style = "bold red" if e.leaked_mb > 0 else "green"
            table.add_row(
                e.name,
                f"{e.peak_mb:.2f} MB",
                Text(f"{e.leaked_mb:.2f} MB", style=leak_style),
                str(e.allocations),
            )

        self._console.print(table)

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def error_detail(self, name: str, error: str, *, traceback: str = "") -> None:
        """Print a detailed error panel with optional traceback."""
        content = Text(error, style="bold red")
        self._console.print(
            Panel(content, title=f"[bold red]FAIL: {name}[/bold red]", border_style="red")
        )

        if traceback:
            from rich.syntax import Syntax

            syntax = Syntax(traceback, "pytb", theme="monokai", word_wrap=True)
            self._console.print(
                Panel(syntax, title="[yellow]Traceback[/yellow]", border_style="yellow")
            )
