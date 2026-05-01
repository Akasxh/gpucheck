"""Dtype-aware tolerance computation and overrides for GPU kernel testing."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # dtype_name: (atol, rtol)
    # Calibrated against cuBLAS matmul on Turing/Ampere GPUs.
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

# Override stack (per-context). Backed by ``contextvars.ContextVar`` so the
# stack is isolated per OS thread AND per asyncio task. Previous releases
# used a plain module-level list, which leaked overrides between threads
# when tests were run inside a single process (e.g. with a thread pool).
# The contract is unchanged from the user's perspective:
#
#     with tolerance_context(atol=1e-3, rtol=1e-3):
#         assert_close(a, b)
#
# Inside the ``with`` block, the calling thread/task observes its own
# top-of-stack overlay; a sibling thread that has not entered a
# tolerance_context block sees the underlying defaults. ContextVar.set
# returns a Token that ``ContextVar.reset`` consumes, restoring the prior
# value — this is correct under exception unwinding.
_tolerance_overrides: ContextVar[tuple[tuple[float, float], ...]] = ContextVar(
    "_tolerance_overrides", default=(),
)


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
) -> tuple[float, float]:
    """Return (atol, rtol) for a given dtype.

    If *k_dim* is supplied (reduction / matmul inner dimension), atol is
    scaled by ``sqrt(k_dim / 128)`` following the CUTLASS error-accumulation
    model where 128 is the standard tile dimension. This means at k_dim=128
    the tolerance is 1x the base, and scales proportionally from there.

    Falls back to float32 tolerances for unknown dtypes.
    """
    # Check override stack first (ContextVar for thread/task isolation).
    overrides = _tolerance_overrides.get()
    if overrides:
        return overrides[-1]

    name = _normalize_dtype_name(dtype)
    # Check config overlay first, then defaults
    if name in _config_overrides:
        atol, rtol = _config_overrides[name]
    else:
        atol, rtol = _DEFAULT_TOLERANCES.get(name, _DEFAULT_TOLERANCES["float32"])

    if k_dim is not None and k_dim > 0:
        atol = atol * math.sqrt(max(k_dim, 1) / 128.0)

    return atol, rtol


@contextmanager
def tolerance_context(
    atol: float,
    rtol: float,
) -> Generator[None, None, None]:
    """Temporarily override default tolerances returned by :func:`compute_tolerance`.

    Backed by ``contextvars.ContextVar``: the override is visible only to
    the current OS thread (and to asyncio tasks that copied the current
    context). Sibling threads observe the underlying defaults concurrently.

    Usage::

        with tolerance_context(atol=1e-3, rtol=1e-3):
            assert_close(a, b)
    """
    current = _tolerance_overrides.get()
    token = _tolerance_overrides.set(current + ((atol, rtol),))
    try:
        yield
    finally:
        _tolerance_overrides.reset(token)


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
