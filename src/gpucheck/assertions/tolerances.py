"""Dtype-aware tolerance computation and overrides for GPU kernel testing."""

from __future__ import annotations

import math
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

_DEFAULT_TOLERANCES: dict[str, tuple[float, float]] = {
    # dtype_name: (atol, rtol)
    "float64": (1e-10, 1e-7),
    "float32": (1e-5, 1.3e-6),
    "float16": (1e-3, 1e-3),
    "bfloat16": (1.6e-2, 1.6e-2),
    "float8_e4m3fn": (0.125, 0.125),
    "float8_e5m2": (0.25, 0.25),
    "tf32": (1e-4, 1e-4),
}

# Thread-local-style override stack (module-level, not truly thread-safe—fine for pytest).
_tolerance_overrides: list[tuple[float, float]] = []


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
    scaled by ``sqrt(k_dim)`` following the standard error-accumulation
    model for floating-point summation.

    Falls back to float32 tolerances for unknown dtypes.
    """
    # Check override stack first.
    if _tolerance_overrides:
        return _tolerance_overrides[-1]

    name = _normalize_dtype_name(dtype)
    atol, rtol = _DEFAULT_TOLERANCES.get(name, _DEFAULT_TOLERANCES["float32"])

    if k_dim is not None and k_dim > 0:
        atol = atol * math.sqrt(k_dim)

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


def apply_config_tolerances(config: dict[str, Any]) -> None:
    """Merge pyproject.toml tolerance overrides into the module-level default table."""
    overrides = tolerances_from_config(config)
    if overrides:
        _DEFAULT_TOLERANCES.update(overrides)
