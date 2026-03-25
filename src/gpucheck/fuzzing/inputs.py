"""Input tensor generators — edge-case and random tensors for GPU kernel fuzzing."""

from __future__ import annotations

import math
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TORCH_IMPORTED = False
_torch: Any = None


def _torch_mod() -> Any:
    """Lazy-import torch."""
    global _TORCH_IMPORTED, _torch  # noqa: PLW0603
    if not _TORCH_IMPORTED:
        try:
            import torch
            _torch = torch
        except ImportError as exc:
            raise RuntimeError(
                "gpucheck.fuzzing.inputs requires PyTorch: pip install gpucheck[torch]"
            ) from exc
        _TORCH_IMPORTED = True
    return _torch


def _is_floating(dtype: Any) -> bool:
    torch = _torch_mod()
    return dtype.is_floating_point if hasattr(dtype, "is_floating_point") else torch.is_floating_point(torch.empty(0, dtype=dtype))


def _dtype_info(dtype: Any) -> tuple[float, float, float]:
    """Return (min_val, max_val, smallest_normal) for a dtype."""
    torch = _torch_mod()
    info = torch.finfo(dtype)
    return float(info.min), float(info.max), float(info.tiny)


def _is_half(dtype: Any) -> bool:
    """Check if dtype is fp16 or bf16."""
    torch = _torch_mod()
    return dtype in (torch.float16, torch.bfloat16)


def _is_fp8(dtype: Any) -> bool:
    """Check if dtype is an fp8 variant."""
    torch = _torch_mod()
    fp8_types = []
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz"):
        d = getattr(torch, name, None)
        if d is not None:
            fp8_types.append(d)
    return dtype in fp8_types


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def random_inputs(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    distribution: str = "normal",
    device: str = "cpu",
    seed: int | None = None,
    custom_fn: Callable[..., Any] | None = None,
) -> Any:
    """Generate a random tensor.

    Parameters
    ----------
    shape:
        Tensor shape.
    dtype:
        A ``torch.dtype``.
    distribution:
        ``"normal"`` (mean=0, std=1), ``"uniform"`` ([0, 1)), or ``"custom"``.
    device:
        Target device string, e.g. ``"cpu"`` or ``"cuda"``.
    seed:
        Optional RNG seed for reproducibility.
    custom_fn:
        When *distribution* is ``"custom"``, a callable
        ``(shape, dtype, device) -> Tensor``.
    """
    torch = _torch_mod()

    gen: torch.Generator | None = None  # type: ignore[name-defined]
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)

    if distribution == "normal":
        # Generate in float32, then cast (some dtypes don't support randn).
        t = torch.randn(shape, generator=gen, dtype=torch.float32, device="cpu")
    elif distribution == "uniform":
        t = torch.rand(shape, generator=gen, dtype=torch.float32, device="cpu")
    elif distribution == "custom":
        if custom_fn is None:
            raise ValueError("custom_fn must be provided when distribution='custom'")
        return custom_fn(shape, dtype, device)
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    # Cast to target dtype and move to device.
    t = t.to(dtype=dtype, device=device)
    return t


def edge_inputs(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    device: str = "cpu",
) -> list[tuple[str, Any]]:
    """Generate tensors with edge-case values for *dtype*.

    Returns a list of ``(label, tensor)`` pairs so callers can identify
    which edge case triggered a failure.

    Edge cases generated:
      - zeros, ones, negative ones
      - max / min representable values
      - negative zero, smallest normal, epsilon-scale values
      - NaN, Inf, -Inf (sprinkled at random positions)
      - Denormals (for fp16 / bf16)
      - Near-overflow values

    .. note::
        This function generates contiguous tensors only. Non-contiguous views
        (e.g. slices, transposes, expand) are a major source of GPU kernel bugs
        but cannot be expressed through shape + value alone. Consider testing
        with ``tensor.T``, ``tensor[::2]``, or ``tensor.as_strided(...)``
        separately to cover stride/contiguity edge cases.
    """
    torch = _torch_mod()
    results: list[tuple[str, Any]] = []
    is_float = _is_floating(dtype)

    def _make(val: float, label: str) -> None:
        try:
            t = torch.full(shape, val, dtype=dtype, device=device)
            results.append((label, t))
        except (RuntimeError, OverflowError):
            # Some dtypes reject certain values (e.g. unsigned + negative).
            pass

    _make(0.0, "zeros")
    _make(1.0, "ones")
    _make(-1.0, "neg_ones")

    if is_float:
        fmin, fmax, tiny = _dtype_info(dtype)

        _make(fmax, "max_val")
        _make(fmin, "min_val")
        _make(fmax * 0.99, "near_overflow")
        _make(fmin * 0.99, "near_neg_overflow")
        _make(-0.0, "neg_zero")
        _make(tiny, "smallest_normal")

        # Epsilon-scale: 1.0 + eps != 1.0, but 1.0 + eps/2 == 1.0
        eps = float(torch.finfo(dtype).eps)
        _make(eps, "epsilon")
        _make(1.0 + eps, "one_plus_eps")

        # NaN / Inf — sprinkle into a normal tensor.
        for special_val, label in [
            (float("nan"), "sprinkled_nan"),
            (float("inf"), "sprinkled_inf"),
            (float("-inf"), "sprinkled_neg_inf"),
        ]:
            base = torch.randn(shape, dtype=torch.float32, device="cpu")
            numel = base.numel()
            if numel > 0:
                # Sprinkle ~10% special values.
                mask = torch.rand(shape) < 0.1
                base[mask] = special_val
            try:
                results.append((label, base.to(dtype=dtype, device=device)))
            except (RuntimeError, OverflowError):
                pass

        # Denormals for half-precision types.
        if _is_half(dtype) or _is_fp8(dtype):
            denorm_val = tiny * 0.5
            if denorm_val > 0:
                _make(denorm_val, "denormal")
                _make(-denorm_val, "neg_denormal")
    else:
        # Integer dtypes.
        try:
            iinfo = torch.iinfo(dtype)
            _make(float(iinfo.max), "int_max")
            _make(float(iinfo.min), "int_min")
        except (RuntimeError, TypeError):
            pass

    return results


def mixed_inputs(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> list[tuple[str, Any]]:
    """Mix of normal random + edge-case tensors.

    Returns ``(label, tensor)`` pairs.
    """
    torch = _torch_mod()
    results: list[tuple[str, Any]] = []

    # Normal distribution samples.
    results.append(("normal", random_inputs(shape, dtype, distribution="normal", device=device, seed=seed)))
    results.append(("uniform", random_inputs(shape, dtype, distribution="uniform", device=device, seed=seed)))

    # Edge cases.
    results.extend(edge_inputs(shape, dtype, device=device))

    # Mixed tensor: normal values with a few edge values injected.
    if _is_floating(dtype):
        fmin, fmax, tiny = _dtype_info(dtype)
        base = torch.randn(shape, dtype=torch.float32, device="cpu")
        numel = base.numel()
        if numel >= 10:
            flat = base.view(-1)
            # Inject edge values at deterministic positions.
            specials = [0.0, fmax, fmin, float("inf"), float("nan")]
            for i, v in enumerate(specials):
                if i < flat.numel():
                    flat[i] = v
        try:
            results.append(("mixed_normal_edge", base.to(dtype=dtype, device=device)))
        except (RuntimeError, OverflowError):
            pass

    return results


__all__ = [
    "random_inputs",
    "edge_inputs",
    "mixed_inputs",
]
