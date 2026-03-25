"""Core assert_close() — dtype-aware tensor comparison with rich diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from gpucheck.assertions.reporting import format_mismatch_report
from gpucheck.assertions.tolerances import compute_tolerance


def _to_numpy(tensor: Any) -> npt.NDArray[Any]:
    """Convert tensor-like to numpy, handling torch, cupy, and __cuda_array_interface__."""
    if isinstance(tensor, np.ndarray):
        return tensor

    # torch.Tensor
    if hasattr(tensor, "detach"):
        t = tensor.detach().cpu()
        # Preserve float64 precision; only cast non-numpy-compatible dtypes
        if t.is_floating_point():
            if t.dtype.itemsize >= 8:
                return t.double().numpy()  # type: ignore[no-any-return]
            if t.dtype.itemsize >= 4:
                return t.numpy()  # type: ignore[no-any-return]
            # Sub-float32 types (fp16, bf16, fp8) need upcast to float32
            return t.float().numpy()  # type: ignore[no-any-return]
        return t.numpy()  # type: ignore[no-any-return]

    # cupy.ndarray or anything with .get()
    if hasattr(tensor, "get"):
        return np.asarray(tensor.get())

    # __cuda_array_interface__ (e.g. numba, rmm, dlpack-aware objects)
    if hasattr(tensor, "__cuda_array_interface__"):
        try:
            import cupy as cp

            return cp.asarray(tensor).get()  # type: ignore[no-any-return]
        except ImportError:
            pass
        # Fallback: try torch.as_tensor via dlpack if available
        try:
            import torch

            t = torch.as_tensor(tensor).detach().cpu()
            if t.is_floating_point():
                if t.dtype.itemsize >= 8:
                    return t.double().numpy()
                if t.dtype.itemsize >= 4:
                    return t.numpy()
                return t.float().numpy()
            return t.numpy()
        except (ImportError, RuntimeError):
            pass

    return np.asarray(tensor)


def _resolve_dtype(actual: Any, expected: Any) -> Any:
    """Return the dtype object from whichever input carries one."""
    for t in (actual, expected):
        if hasattr(t, "dtype"):
            return t.dtype
    return np.float32


def assert_close(
    actual: Any,
    expected: Any,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    k_dim: int | None = None,
    nan_equal: bool = False,
    baseline_2x: bool = False,
    msg: str = "",
) -> None:
    """Assert two tensors are element-wise close within tolerance.

    Parameters
    ----------
    actual, expected:
        Tensor-like objects — torch.Tensor, numpy.ndarray, cupy.ndarray,
        or anything exposing ``__cuda_array_interface__``.
    rtol, atol:
        Override relative/absolute tolerance.  When ``None``, dtype-aware
        defaults from the tolerance table are used.
    k_dim:
        Inner dimension for matmul-like ops.  Scales ``atol`` by ``sqrt(k_dim)``.
    nan_equal:
        If ``True``, matching NaN positions are treated as equal.
    baseline_2x:
        FlashAttention methodology — tolerances are doubled relative to
        dtype defaults (``2x baseline``).
    msg:
        Optional extra message prepended to failure output.
    """
    dtype = _resolve_dtype(actual, expected)

    if baseline_2x and atol is None and rtol is None:
        # FlashAttention 2x: double base tolerance BEFORE k_dim scaling
        base_atol, base_rtol = compute_tolerance(dtype)
        doubled_atol, doubled_rtol = base_atol * 2.0, base_rtol * 2.0
        # Now apply k_dim scaling on the doubled base
        if k_dim is not None and k_dim > 0:
            import math

            doubled_atol *= math.sqrt(k_dim)
        eff_atol = atol if atol is not None else doubled_atol
        eff_rtol = rtol if rtol is not None else doubled_rtol
    else:
        default_atol, default_rtol = compute_tolerance(dtype, k_dim=k_dim)
        eff_atol = atol if atol is not None else default_atol
        eff_rtol = rtol if rtol is not None else default_rtol
        if baseline_2x:
            eff_atol *= 2.0
            eff_rtol *= 2.0

    actual_np = _to_numpy(actual)
    expected_np = _to_numpy(expected)

    if actual_np.shape != expected_np.shape:
        raise AssertionError(
            f"Shape mismatch: actual {actual_np.shape} vs expected {expected_np.shape}"
        )

    actual_f64 = actual_np.astype(np.float64, copy=False)
    expected_f64 = expected_np.astype(np.float64, copy=False)

    # --- Handle NaN ---
    nan_actual = np.isnan(actual_f64)
    nan_expected = np.isnan(expected_f64)

    if nan_equal:
        # NaN in same position → ok.  NaN in only one → mismatch.
        nan_mismatch = nan_actual != nan_expected
        if np.any(nan_mismatch):
            report = format_mismatch_report(
                actual_np, expected_np, eff_atol, eff_rtol,
                actual_f64=actual_f64, expected_f64=expected_f64,
            )
            prefix = f"{msg}\n" if msg else ""
            raise AssertionError(
                f"{prefix}NaN position mismatch: "
                f"{int(np.sum(nan_mismatch))} positions differ.\n{report}"
            )
        # Mask out matched NaN positions for numeric comparison.
        compare_mask = ~(nan_actual & nan_expected)
    else:
        # Any NaN in either tensor is an immediate failure.
        if np.any(nan_actual) or np.any(nan_expected):
            report = format_mismatch_report(
                actual_np, expected_np, eff_atol, eff_rtol,
                actual_f64=actual_f64, expected_f64=expected_f64,
            )
            prefix = f"{msg}\n" if msg else ""
            raise AssertionError(
                f"{prefix}Tensors contain NaN values "
                f"(actual: {int(np.sum(nan_actual))}, expected: {int(np.sum(nan_expected))}). "
                f"Use nan_equal=True to allow matching NaN positions.\n{report}"
            )
        compare_mask = np.ones_like(actual_f64, dtype=bool)

    # --- Handle Inf ---
    inf_actual = np.isinf(actual_f64)
    inf_expected = np.isinf(expected_f64)
    inf_mismatch = inf_actual != inf_expected
    if np.any(inf_mismatch):
        report = format_mismatch_report(
                actual_np, expected_np, eff_atol, eff_rtol,
                actual_f64=actual_f64, expected_f64=expected_f64,
            )
        prefix = f"{msg}\n" if msg else ""
        raise AssertionError(
            f"{prefix}Inf position mismatch: "
            f"{int(np.sum(inf_mismatch))} positions differ.\n{report}"
        )
    # Where both are Inf, check sign matches.
    both_inf = inf_actual & inf_expected
    if np.any(both_inf):
        sign_mismatch = np.sign(actual_f64[both_inf]) != np.sign(expected_f64[both_inf])
        if np.any(sign_mismatch):
            report = format_mismatch_report(
                actual_np, expected_np, eff_atol, eff_rtol,
                actual_f64=actual_f64, expected_f64=expected_f64,
            )
            prefix = f"{msg}\n" if msg else ""
            raise AssertionError(
                f"{prefix}Inf sign mismatch at {int(np.sum(sign_mismatch))} positions.\n{report}"
            )
        compare_mask = compare_mask & ~both_inf

    # --- Numeric comparison: |a - b| <= atol + rtol * |b| ---
    if not np.any(compare_mask):
        return  # All elements were NaN/Inf and matched.

    diff = np.abs(actual_f64[compare_mask] - expected_f64[compare_mask])
    threshold = eff_atol + eff_rtol * np.abs(expected_f64[compare_mask])
    failures = diff > threshold

    if np.any(failures):
        report = format_mismatch_report(
                actual_np, expected_np, eff_atol, eff_rtol,
                actual_f64=actual_f64, expected_f64=expected_f64,
            )
        prefix = f"{msg}\n" if msg else ""
        raise AssertionError(
            f"{prefix}Tensors are not close! "
            f"(atol={eff_atol:.2e}, rtol={eff_rtol:.2e}; "
            f"override with atol=/rtol= or use k_dim=/baseline_2x=)\n{report}"
        )
