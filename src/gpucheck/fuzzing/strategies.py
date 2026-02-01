"""Hypothesis strategies for GPU tensor fuzzing.

All public functions degrade gracefully when ``hypothesis`` is not installed
— they raise ``RuntimeError`` with an install hint.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_HAS_HYPOTHESIS: bool | None = None


def _check_hypothesis() -> Any:
    """Import and return hypothesis.strategies, or raise with install hint."""
    global _HAS_HYPOTHESIS  # noqa: PLW0603
    if _HAS_HYPOTHESIS is False:
        raise RuntimeError(
            "gpucheck.fuzzing.strategies requires hypothesis: "
            "pip install gpucheck[hypothesis]"
        )
    try:
        from hypothesis import strategies as st

        _HAS_HYPOTHESIS = True
        return st
    except ImportError as exc:
        _HAS_HYPOTHESIS = False
        raise RuntimeError(
            "gpucheck.fuzzing.strategies requires hypothesis: "
            "pip install gpucheck[hypothesis]"
        ) from exc


def _torch_mod() -> Any:
    try:
        import torch
        return torch
    except ImportError as exc:
        raise RuntimeError(
            "gpucheck.fuzzing.strategies requires PyTorch: "
            "pip install gpucheck[torch]"
        ) from exc


# ---------------------------------------------------------------------------
# Shape strategy
# ---------------------------------------------------------------------------

def gpu_shapes(
    ndim: int | None = None,
    *,
    min_ndim: int = 1,
    max_ndim: int = 4,
    min_size: int = 0,
    max_size: int = 4096,
) -> Any:
    """Hypothesis strategy that produces shape tuples biased towards GPU edge cases.

    Parameters
    ----------
    ndim:
        Fixed number of dimensions.  When ``None``, dimensions are drawn
        from ``[min_ndim, max_ndim]``.
    min_size / max_size:
        Bounds for each dimension value.
    """
    st = _check_hypothesis()
    from gpucheck.fuzzing.shapes import POWER_OF_2_BOUNDARIES, PRIMES, TILE_SIZES

    interesting = sorted(
        {v for v in (
            0, 1,
            *PRIMES,
            *POWER_OF_2_BOUNDARIES,
            *[t - 1 for t in TILE_SIZES],
            *[t + 1 for t in TILE_SIZES],
        ) if min_size <= v <= max_size}
    )

    dim = st.one_of(
        st.sampled_from(interesting) if interesting else st.just(1),
        st.integers(min_value=max(min_size, 0), max_value=max_size),
    )

    if ndim is not None:
        return st.tuples(*([dim] * ndim))

    return st.integers(min_value=min_ndim, max_value=max_ndim).flatmap(
        lambda nd: st.tuples(*([dim] * nd))
    )


# ---------------------------------------------------------------------------
# Tensor strategy
# ---------------------------------------------------------------------------

def gpu_tensors(
    shape: tuple[int, ...] | None = None,
    dtype: Any = None,
    *,
    device: str = "cpu",
    elements: Any = None,
    min_ndim: int = 1,
    max_ndim: int = 4,
    min_size: int = 1,
    max_size: int = 512,
) -> Any:
    """Hypothesis strategy that draws PyTorch tensors.

    Supports proper shrinking: tensors shrink towards smaller shapes and
    values closer to zero.

    Parameters
    ----------
    shape:
        Fixed shape.  When ``None``, shapes are drawn via :func:`gpu_shapes`.
    dtype:
        A ``torch.dtype``.  Defaults to ``torch.float32``.
    device:
        Device string.
    elements:
        Optional Hypothesis strategy for individual element values.
    min_ndim / max_ndim / min_size / max_size:
        Forwarded to :func:`gpu_shapes` when *shape* is ``None``.
    """
    st = _check_hypothesis()
    torch = _torch_mod()

    if dtype is None:
        dtype = torch.float32

    is_float = dtype.is_floating_point

    # Element strategy (floats shrink towards 0).
    if elements is None:
        if is_float:
            info = torch.finfo(dtype)
            # Use float32 range to avoid hypothesis issues with wide ranges.
            lo = max(float(info.min), -1e6)
            hi = min(float(info.max), 1e6)
            elements = st.floats(
                min_value=lo,
                max_value=hi,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=True,
            )
        else:
            info = torch.iinfo(dtype)
            elements = st.integers(min_value=int(info.min), max_value=int(info.max))

    shape_strategy = (
        st.just(shape) if shape is not None
        else gpu_shapes(min_ndim=min_ndim, max_ndim=max_ndim, min_size=min_size, max_size=max_size)
    )

    @st.composite
    def _draw_tensor(draw: Any) -> Any:
        s = draw(shape_strategy)
        numel = 1
        for d in s:
            numel *= d
        if numel == 0:
            return torch.empty(s, dtype=dtype, device=device)
        vals = draw(st.lists(elements, min_size=numel, max_size=numel))
        flat = torch.tensor(vals, dtype=torch.float32).to(dtype)
        return flat.reshape(s).to(device=device)

    return _draw_tensor()


__all__ = [
    "gpu_shapes",
    "gpu_tensors",
]
