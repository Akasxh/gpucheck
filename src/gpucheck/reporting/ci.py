"""CI integration — GitHub Actions annotations, JUnit XML, and PR comments."""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gpucheck.reporting.console import TestResult

# ---------------------------------------------------------------------------
# GitHub Actions annotations
# ---------------------------------------------------------------------------

# Maps TestResult.status -> (annotation command, label prefix)
_ANNOTATION_MAP: dict[str, tuple[str, str]] = {
    "failed": ("error", "FAIL"),
    "error": ("error", "ERROR"),
    "skipped": ("warning", "SKIP"),
}


def emit_github_annotations(results: Sequence[TestResult]) -> None:
    """Write GitHub Actions `::error::` / `::warning::` annotations to stdout.

    No-op when ``GITHUB_ACTIONS`` env var is not set.
    """
    if not os.environ.get("GITHUB_ACTIONS"):
        return

    for r in results:
        entry = _ANNOTATION_MAP.get(r.status)
        if entry is None:
            continue
        cmd, label = entry
        props: list[str] = []
        if r.file:
            props.append(f"file={r.file}")
            if r.line:
                props.append(f"line={r.line}")
        safe_name = r.name.replace("::", " - ")
        props.append(f"title={label}: {safe_name}")
        msg = r.message.replace("\n", "%0A")
        sys.stdout.write(f"::{cmd} {','.join(props)}::{msg}\n")


# ---------------------------------------------------------------------------
# JUnit XML
# ---------------------------------------------------------------------------


def write_junit_xml(
    results: Sequence[TestResult],
    output_path: str | Path = ".gpucheck/junit.xml",
    suite_name: str = "gpucheck",
) -> Path:
    """Generate a JUnit-compatible XML report.

    Compatible with most CI dashboards (GitHub Actions, Jenkins, GitLab, etc.).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    failures = sum(1 for r in results if r.status == "failed")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skipped")
    total_time = sum(r.duration for r in results)

    testsuite = ET.Element("testsuite", {
        "name": suite_name,
        "tests": str(total),
        "failures": str(failures),
        "errors": str(errors),
        "skipped": str(skipped),
        "time": f"{total_time:.4f}",
    })

    for r in results:
        tc = ET.SubElement(testsuite, "testcase", {
            "name": r.name,
            "classname": suite_name,
            "time": f"{r.duration:.4f}",
        })

        if r.status == "failed":
            fail_el = ET.SubElement(tc, "failure", {"message": r.message})
            fail_el.text = r.message
        elif r.status == "error":
            err_el = ET.SubElement(tc, "error", {"message": r.message})
            err_el.text = r.message
        elif r.status == "skipped":
            ET.SubElement(tc, "skipped", {"message": r.message})

    tree = ET.ElementTree(testsuite)
    ET.indent(tree, space="  ")
    tree.write(str(path), encoding="unicode", xml_declaration=True)
    # Append trailing newline
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n")

    return path


# ---------------------------------------------------------------------------
# PR comment generation
# ---------------------------------------------------------------------------


def generate_pr_comment(
    comparison: dict[str, Any],
    *,
    header: str = "## gpucheck Benchmark Comparison",
) -> str:
    """Generate a Markdown PR comment body from a :func:`JSONReporter.compare_runs` diff.

    Suitable for posting via ``gh pr comment`` or the GitHub API.
    """
    lines: list[str] = [header, ""]

    benchmarks = comparison.get("benchmarks", [])
    test_changes = comparison.get("test_changes", [])

    if benchmarks:
        lines.append("### Benchmarks")
        lines.append("")
        lines.append("| Kernel | Baseline | Current | Delta | Status |")
        lines.append("|--------|----------|---------|-------|--------|")

        for b in benchmarks:
            status = b.get("status", "ok")
            base_ms = b.get("baseline_median_ms", "-")
            curr_ms = b.get("current_median_ms", "-")
            delta_pct = b.get("delta_pct", 0)

            base_str = f"{base_ms:.3f} ms" if isinstance(base_ms, (int, float)) else str(base_ms)

            curr_str = f"{curr_ms:.3f} ms" if isinstance(curr_ms, (int, float)) else str(curr_ms)

            if status == "regression":
                icon = ":red_circle:"
                delta_str = f"+{delta_pct:.1f}%"
            elif status == "ok":
                icon = ":green_circle:"
                delta_str = f"{delta_pct:+.1f}%"
            elif status == "new":
                icon = ":new:"
                delta_str = "N/A"
            else:
                icon = ":x:"
                delta_str = "removed"

            lines.append(
                f"| {b['name']} | {base_str} | {curr_str} | {delta_str} | {icon} {status} |"
            )

        lines.append("")

    if test_changes:
        lines.append("### Test Status Changes")
        lines.append("")
        lines.append("| Test | Was | Now |")
        lines.append("|------|-----|-----|")

        for tc in test_changes:
            lines.append(f"| {tc['name']} | {tc['was']} | {tc['now']} |")

        lines.append("")

    if not benchmarks and not test_changes:
        lines.append("No changes detected.")
        lines.append("")

    return "\n".join(lines)
