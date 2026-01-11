"""GPU-aware tensor comparison assertions."""

from __future__ import annotations

from gpucheck.assertions.close import assert_close
from gpucheck.assertions.tolerances import compute_tolerance, tolerance_context

__all__ = ["assert_close", "compute_tolerance", "tolerance_context"]
