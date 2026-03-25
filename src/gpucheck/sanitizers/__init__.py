"""GPU memory and compute sanitizers."""

from __future__ import annotations

from gpucheck.sanitizers.memory import SanitizerMemoryReport, check_memory_leaks, memory_guard
from gpucheck.sanitizers.race import SanitizerReport, run_with_sanitizer

# Backward-compat alias
MemoryReport = SanitizerMemoryReport

__all__ = [
    "MemoryReport",
    "SanitizerMemoryReport",
    "SanitizerReport",
    "check_memory_leaks",
    "memory_guard",
    "run_with_sanitizer",
]
