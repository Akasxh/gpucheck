"""Test result reporting — console, JSON, and CI integrations."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_MAP: dict[str, tuple[str, str]] = {
    "ConsoleReporter": ("gpucheck.reporting.console", "ConsoleReporter"),
    "JSONReporter": ("gpucheck.reporting.json", "JSONReporter"),
    "emit_github_annotations": ("gpucheck.reporting.ci", "emit_github_annotations"),
    "write_junit_xml": ("gpucheck.reporting.ci", "write_junit_xml"),
    "generate_pr_comment": ("gpucheck.reporting.ci", "generate_pr_comment"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        module_path, attr = _LAZY_MAP[name]
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'gpucheck.reporting' has no attribute {name!r}")


__all__ = [
    "ConsoleReporter",
    "JSONReporter",
    "emit_github_annotations",
    "write_junit_xml",
    "generate_pr_comment",
]
