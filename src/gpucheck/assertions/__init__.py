"""GPU-aware tensor comparison assertions."""

from __future__ import annotations

from gpucheck.assertions.close import assert_close
from gpucheck.assertions.tolerances import (
    KernelClass,
    apply_mps_xfail_config,
    compute_tolerance,
    is_mps_xfailed,
    mps_xfail_list,
    register_mps_xfail,
    reset_mps_xfail,
    tolerance_context,
)

__all__ = [
    "KernelClass",
    "assert_close",
    "compute_tolerance",
    "tolerance_context",
    "is_mps_xfailed",
    "mps_xfail_list",
    "apply_mps_xfail_config",
    "register_mps_xfail",
    "reset_mps_xfail",
]
