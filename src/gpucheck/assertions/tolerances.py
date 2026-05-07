"""Dtype-aware tolerance computation and overrides for GPU kernel testing."""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


class KernelClass(str, Enum):
    """Kernel-class buckets for the per-(kernel, dtype) MPS tolerance overlay.

    Buckets follow the Shape-B class-bucketed taxonomy from the v1.1 5K
    calibration (see ``.claude/teams/audit/v1.1/EVIDENCE/calibration-final.md``).
    The classification is structural, not name-matched: callers tag kernels
    by their numerical-error profile, not by torch op identity.

    - ``MATMUL`` — GEMM-dominated, no protective normalization
      (``matmul``, ``linear``, ``bmm``, ``addmm``, ``einsum`` GEMM forms).
    - ``CONV2D`` — K-accumulating conv kernels (``conv1d``/``conv2d``/
      ``conv3d``, ``conv_transpose2d``); breached the FA-2× ceiling at
      5K-iter measurement on M5.
    - ``NORM`` — norm-protected kernels (``layer_norm``, ``rms_norm``,
      ``batch_norm``, ``group_norm``).
    - ``REDUCTION`` — reductions and softmaxes (``softmax``, ``log_softmax``,
      ``mean``, ``sum``, ``cross_entropy``).
    - ``POINTWISE`` — elementwise activations (``gelu``, ``silu``, ``relu``,
      ``tanh``, ``sigmoid``).
    - ``DEFAULT`` — fallback bucket; resolves to the flat-by-dtype overlay
      preserved from v1.0 (the ``_MPS_TOLERANCE_MULTIPLIERS`` table below).
    """

    MATMUL = "matmul"
    CONV2D = "conv2d"
    NORM = "norm"
    REDUCTION = "reduction"
    POINTWISE = "pointwise"
    DEFAULT = "default"


# Sentinel used as the dtype slot in `_MPS_KERNEL_DTYPE_MULTIPLIERS` keys to
# mark "applies to every dtype in this kernel class" (e.g. NORM is 2× across
# fp32/fp16/bf16/etc.). Wildcard lookups happen after exact-dtype lookups.
_DTYPE_WILDCARD = "*"

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

# Flat per-dtype MPS overlay (v1.0 baseline, preserved for backward compat).
#
# This table is the DEFAULT-kernel-class fallback for the v1.1 per-(kernel,
# dtype) overlay below. v1.0 callers that pass `compute_tolerance(dtype,
# device_type="mps")` without a `kernel_class=` keep getting the 2× scale
# they shipped with — see `compute_tolerance` for the resolution order.
#
# Origin: research SYNTHESIS §7. These multipliers are mapped from the real
# PyTorch MPS bug magnitudes documented in
# `.claude/teams/research/v1.0/SYNTHESIS.md` (pytorch#177116, #181936, #178497,
# #142836, #173525, #175189, #96602 etc.). 2× is the FlashAttention precedent
# (`assertions/close.py:baseline_2x`).
_MPS_TOLERANCE_MULTIPLIERS: dict[str, float] = {
    "float32": 2.0,
    "float16": 2.0,
    "bfloat16": 2.0,
    "float64": 1.0,  # Rarely load-bearing on MPS; keep CUDA tolerance.
    "float8_e4m3fn": 2.0,  # Apple Silicon has no FP8 tensor cores; placeholder.
    "float8_e5m2": 2.0,
    "tf32": 1.0,  # TF32 is NVIDIA-only; Apple Silicon has no analogue.
}

# Per-(kernel_class, dtype) MPS overlay — v1.1 5K calibration, Apple M5.
#
# Source: `.claude/teams/audit/v1.1/drift_histogram_5k.json` (21 cells × 5000
# iters), analysis `.claude/teams/audit/v1.1/EVIDENCE/calibration-final.md`.
# Each multiplier is sized to cover the *measured* P99 (and most P99.9 tails)
# with safety headroom for cross-SKU drift (M3/M4/M5 may shift ±2-3×).
#
# Headline rationale per kernel class:
#   - MATMUL: GEMM has no protective normalization; measured P99
#     {fp32: 13.43×, fp16: 17.74×, bf16: 27.57×}. Overlay {16, 20, 32}
#     covers P99 with ≥10% headroom; bf16 P99.9 (31.91×) sits exactly at
#     the 32× ceiling — at-the-edge but covered.
#   - CONV2D: K-accumulating, no normalization; v3 verdict (200-iter)
#     under-tightened. 5K shows fp32 P99=1.98×, fp16 P99=6.71×, bf16
#     P99=10.44×. Overlay {4, 8, 12} covers P99 with safety; fp32 picks
#     4× (not 2×) because the P99.9 tail of 2.52× crosses the
#     FlashAttention-2× ceiling.
#   - NORM/REDUCTION/POINTWISE: norm-protected and elementwise kernels
#     all sit comfortably under FA-2× at 5K-iter measurement. Wildcard
#     dtype row = 2× preserves the v1.0 default for these classes.
#   - DEFAULT (kernel_class omitted or unrecognized): falls through to
#     `_MPS_TOLERANCE_MULTIPLIERS` flat table (above). v1.0 behaviour.
#
# Resolution order in `compute_tolerance` for a given (kernel_class, dtype):
#   1. exact `(kernel_class, dtype_name)` key,
#   2. wildcard `(kernel_class, "*")` key,
#   3. exact `(KernelClass.DEFAULT, dtype_name)` key (none currently set —
#      reserved for future per-dtype DEFAULT overrides),
#   4. flat `_MPS_TOLERANCE_MULTIPLIERS[dtype_name]`,
#   5. hard fallback `2.0`.
_MPS_KERNEL_DTYPE_MULTIPLIERS: dict[tuple[KernelClass, str], float] = {
    # GEMM-dominated cells (5K-measured P99 in calibration-final.md).
    (KernelClass.MATMUL, "float32"): 16.0,
    (KernelClass.MATMUL, "float16"): 20.0,
    (KernelClass.MATMUL, "bfloat16"): 32.0,
    # Conv-2D cells (revised upward from v3 200-iter projection).
    (KernelClass.CONV2D, "float32"): 4.0,    # 5K P99=1.98× / P99.9=2.52× → 4× safety
    (KernelClass.CONV2D, "float16"): 8.0,    # 5K P99=6.71× / P99.9=8.24× → 8× safety
    (KernelClass.CONV2D, "bfloat16"): 12.0,  # 5K P99=10.44× / P99.9=12.32× → 12× safety
    # Norm-protected, reductions, pointwise — all dtypes sit under FA-2×
    # at 5K. Wildcard rows so any dtype (including FP8, fp64) routes to 2×.
    (KernelClass.NORM, _DTYPE_WILDCARD): 2.0,
    (KernelClass.REDUCTION, _DTYPE_WILDCARD): 2.0,
    (KernelClass.POINTWISE, _DTYPE_WILDCARD): 2.0,
}

# Override stack (per-context). Backed by ``contextvars.ContextVar`` so the
# stack is isolated per OS thread AND per asyncio task. Previous releases
# used a plain module-level list, which leaked overrides between threads
# when tests were run inside a single process. ContextVar.set returns a
# Token that ``ContextVar.reset`` consumes, restoring the prior value —
# correct under exception unwinding.
_tolerance_overrides: ContextVar[tuple[tuple[float, float], ...]] = ContextVar(
    "_tolerance_overrides", default=(),
)

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


def _resolve_mps_multiplier(
    kernel_class: KernelClass | None,
    dtype_name: str,
) -> float:
    """Resolve the MPS tolerance multiplier for a (kernel_class, dtype) pair.

    Resolution order (first match wins):

    1. ``(kernel_class, dtype_name)`` exact in
       :data:`_MPS_KERNEL_DTYPE_MULTIPLIERS`,
    2. ``(kernel_class, "*")`` wildcard row,
    3. ``(KernelClass.DEFAULT, dtype_name)`` exact (currently unused;
       reserved for future per-dtype DEFAULT overrides),
    4. flat per-dtype :data:`_MPS_TOLERANCE_MULTIPLIERS` (v1.0 behaviour),
    5. hard fallback ``2.0`` (FlashAttention precedent).

    When ``kernel_class`` is ``None``, steps 1-3 are skipped — exact
    v1.0 behaviour is preserved for callers that don't tag a kernel.
    """
    if kernel_class is not None:
        exact = _MPS_KERNEL_DTYPE_MULTIPLIERS.get((kernel_class, dtype_name))
        if exact is not None:
            return exact
        wildcard = _MPS_KERNEL_DTYPE_MULTIPLIERS.get(
            (kernel_class, _DTYPE_WILDCARD),
        )
        if wildcard is not None:
            return wildcard
        # Step 3: DEFAULT-class per-dtype override (reserved; presently empty).
        default_exact = _MPS_KERNEL_DTYPE_MULTIPLIERS.get(
            (KernelClass.DEFAULT, dtype_name),
        )
        if default_exact is not None:
            return default_exact
    return _MPS_TOLERANCE_MULTIPLIERS.get(dtype_name, 2.0)


def compute_tolerance(
    dtype: Any,
    *,
    k_dim: int | None = None,
    device_type: str | None = None,
    kernel_class: KernelClass | None = None,
) -> tuple[float, float]:
    """Return (atol, rtol) for a given dtype.

    If *k_dim* is supplied (reduction / matmul inner dimension), atol is
    scaled by ``sqrt(k_dim / 128)`` following the CUTLASS error-accumulation
    model where 128 is the standard tile dimension. This means at k_dim=128
    the tolerance is 1x the base, and scales proportionally from there.

    If *device_type* is ``"mps"``, an MPS-specific multiplier is applied.
    When *kernel_class* is supplied, the multiplier is resolved from the
    per-(kernel_class, dtype) overlay in
    :data:`_MPS_KERNEL_DTYPE_MULTIPLIERS` (v1.1 5K-iter calibration on
    Apple M5 — see ``EVIDENCE/calibration-final.md``); when omitted, the
    flat v1.0 :data:`_MPS_TOLERANCE_MULTIPLIERS` table is used unchanged
    (backward-compatible).

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

    # MPS overlay — v1.1 routes through the per-(kernel_class, dtype) table
    # when `kernel_class` is supplied, falling back to the flat v1.0 table
    # otherwise (preserving the 2× default for unmodified callers).
    if device_type == "mps":
        multiplier = _resolve_mps_multiplier(kernel_class, name)
        atol *= multiplier
        rtol *= multiplier

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
