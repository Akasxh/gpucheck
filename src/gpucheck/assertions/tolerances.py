"""Dtype-aware tolerance computation and overrides for GPU kernel testing."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # dtype_name: (atol, rtol)
    # Calibrated against cuBLAS matmul on Turing/Ampere/Ada NVIDIA GPUs
    # (see assertions/tolerances and arch/tensor_cores).
    # atol covers element-wise ops; rtol covers matmul-like ops where
    # output magnitude scales with input size.
    "float64": (1e-10, 1e-7),
    "float32": (1e-4, 1e-4),
    "float16": (1e-2, 1e-2),
    "bfloat16": (5e-2, 5e-2),
    "float8_e4m3fn": (0.125, 0.125),
    "float8_e5m2": (0.25, 0.25),
    "tf32": (5e-4, 5e-4),
}

# PROVISIONAL — research SYNTHESIS §7. These multipliers are mapped from the
# real PyTorch MPS bug magnitudes documented in
# `.claude/teams/research/v1.0/SYNTHESIS.md` (pytorch#177116, #181936, #178497,
# #142836, #173525, #175189, #96602 etc.) but the precise values must be
# calibrated on Akash's actual M-generation hardware before being canonical
# (sub-Q 7 § "Calibration plan"). Until then, treat as a directional overlay.
# 2× is the FlashAttention precedent (assertions/close.py:117 baseline_2x).
_MPS_TOLERANCE_MULTIPLIERS: dict[str, float] = {
    "float32": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "float64": 1.0,  # Rarely load-bearing on MPS; keep CUDA tolerance.
    "float8_e4m3fn": 2.0,  # Apple Silicon has no FP8 tensor cores; placeholder.
    "float8_e5m2": 2.0,
    "tf32": 1.0,  # TF32 is NVIDIA-only; Apple Silicon has no analogue.
}

# Override stack: ContextVar-based for thread- and asyncio-task isolation.
# Each thread (and each asyncio task that copies the context) sees its own
# stack of (atol, rtol) overlays. The previous ``list`` implementation leaked
# overrides between threads when tests were parallelized inside a process.
# Track-A keeps this as a list (Track-C converts it to ``ContextVar``).
_tolerance_overrides: list[tuple[float, float]] = []

# MPS xfail registry — populated by `apply_mps_xfail_config` from
# ``[tool.gpucheck.mps.xfail]``. Tests can query via :func:`is_mps_xfailed`.
_mps_xfail_set: set[str] = set()


def _normalize_dtype_name(dtype: Any) -> str:
    """Extract a canonical dtype string from torch.dtype, numpy dtype, or str."""
    name = str(dtype)
    # torch dtypes look like "torch.float32"
    if name.startswith("torch."):
        name = name[len("torch."):]
    # numpy dtypes: "float64", "float32", etc.  Already fine.
    return name


def compute_tolerance(
    dtype: Any,
    *,
    k_dim: int | None = None,
    device_type: str | None = None,
) -> tuple[float, float]:
    """Return (atol, rtol) for a given dtype.

    If *k_dim* is supplied (reduction / matmul inner dimension), atol is
    scaled by ``sqrt(k_dim / 128)`` following the CUTLASS error-accumulation
    model where 128 is the standard tile dimension. This means at k_dim=128
    the tolerance is 1x the base, and scales proportionally from there.

    If *device_type* is ``"mps"``, an additional dtype-specific multiplier
    from :data:`_MPS_TOLERANCE_MULTIPLIERS` is applied. The multipliers are
    PROVISIONAL until calibrated on the user's M-generation hardware
    (see SYNTHESIS §7 calibration plan).

    Falls back to float32 tolerances for unknown dtypes.
    """
    # Check explicit override stack first (set via tolerance_context()).
    if _tolerance_overrides:
        return _tolerance_overrides[-1]

    name = _normalize_dtype_name(dtype)
    # Check config overlay first, then defaults
    if name in _config_overrides:
        atol, rtol = _config_overrides[name]
    else:
        atol, rtol = _DEFAULT_TOLERANCES.get(name, _DEFAULT_TOLERANCES["float32"])

    if k_dim is not None and k_dim > 0:
        atol = atol * math.sqrt(max(k_dim, 1) / 128.0)

    # MPS overlay (PROVISIONAL — see SYNTHESIS §7 calibration plan).
    if device_type == "mps":
        multiplier = _MPS_TOLERANCE_MULTIPLIERS.get(name, 2.0)
        atol *= multiplier
        rtol *= multiplier

    return atol, rtol


@contextmanager
def tolerance_context(
    atol: float,
    rtol: float,
) -> Generator[None, None, None]:
    """Temporarily override default tolerances returned by :func:`compute_tolerance`.

    Usage::

        with tolerance_context(atol=1e-3, rtol=1e-3):
            assert_close(a, b)
    """
    _tolerance_overrides.append((atol, rtol))
    try:
        yield
    finally:
        _tolerance_overrides.pop()


def tolerances_from_config(config: dict[str, Any]) -> dict[str, tuple[float, float]] | None:
    """Parse tolerance overrides from a ``[tool.gpucheck.tolerances]`` table.

    Expected shape in pyproject.toml::

        [tool.gpucheck.tolerances]
        float16 = {atol = 2e-3, rtol = 2e-3}

    Returns ``None`` when the section is absent or empty.
    """
    section = config.get("tool", {}).get("gpucheck", {}).get("tolerances")
    if not section:
        return None

    result: dict[str, tuple[float, float]] = {}
    for dtype_name, vals in section.items():
        if not isinstance(vals, dict) or "atol" not in vals or "rtol" not in vals:
            continue
        result[dtype_name] = (float(vals["atol"]), float(vals["rtol"]))
    return result or None


# Overlay dict for config-based overrides (separate from _DEFAULT_TOLERANCES)
_config_overrides: dict[str, tuple[float, float]] = {}


def apply_config_tolerances(config: dict[str, Any]) -> None:
    """Apply pyproject.toml tolerance overrides as an overlay (non-destructive).

    Overrides are stored separately from the built-in defaults and can be
    reverted with :func:`reset_config_tolerances`.
    """
    overrides = tolerances_from_config(config)
    if overrides:
        _config_overrides.update(overrides)


def reset_config_tolerances() -> None:
    """Remove all config-based tolerance overrides."""
    _config_overrides.clear()


# ---------------------------------------------------------------------------
# MPS xfail registry (research SYNTHESIS §2 + §7)
# ---------------------------------------------------------------------------

def mps_xfail_from_config(config: dict[str, Any]) -> set[str] | None:
    """Parse the MPS xfail list from a ``[tool.gpucheck.mps.xfail]`` block.

    Expected shape::

        [tool.gpucheck.mps.xfail]
        ops = [
          "scaled_dot_product_attention.large",
          "softmax.large_attention",
          ...
        ]

    Returns ``None`` when the section is absent or empty so callers can
    distinguish "no MPS config" from "explicit empty list".
    """
    section = config.get("tool", {}).get("gpucheck", {}).get("mps", {}).get("xfail")
    if not section:
        return None
    ops = section.get("ops")
    if not isinstance(ops, list):
        return None
    return {str(o) for o in ops}


def apply_mps_xfail_config(config: dict[str, Any]) -> None:
    """Replace the MPS xfail registry with entries from the config block."""
    parsed = mps_xfail_from_config(config)
    if parsed is None:
        return
    _mps_xfail_set.clear()
    _mps_xfail_set.update(parsed)


def reset_mps_xfail() -> None:
    """Drop all MPS xfail registrations."""
    _mps_xfail_set.clear()


def register_mps_xfail(*ops: str) -> None:
    """Add one or more op names to the MPS xfail registry (test helper)."""
    _mps_xfail_set.update(ops)


def is_mps_xfailed(op_name: str) -> bool:
    """Return ``True`` if *op_name* is in the MPS xfail registry.

    The registry is populated from ``pyproject.toml`` at session start (see
    :func:`apply_mps_xfail_config`). Tests can also push entries at runtime
    via :func:`register_mps_xfail`. The match is exact-string; the canonical
    naming convention is ``op.subcategory`` (see SYNTHESIS §7).
    """
    return op_name in _mps_xfail_set


def mps_xfail_list() -> list[str]:
    """Return the current MPS xfail list, sorted for stable iteration."""
    return sorted(_mps_xfail_set)
