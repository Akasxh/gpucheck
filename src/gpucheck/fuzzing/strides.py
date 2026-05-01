"""Stride and contiguity fuzzing for GPU kernels.

GPU kernel bugs often hide behind non-contiguous tensor layouts: a kernel
might be correct for ``tensor.contiguous()`` but mis-handle a transposed
view, a broadcast-induced stride-0 dim, or a slice with non-unit stride.
This module generates a deterministic corpus of seven stride categories,
plus a Hypothesis :class:`StrideStrategy` for property-based testing.

Categories (priority order; row-major first as the baseline)::

    row_major     -- contiguous, default torch.empty(shape)
    column_major  -- ATen 'F' layout via transpose-of-contiguous
    broadcast     -- stride-0 dim (expand)
    transpose     -- 2D stride permutation
    slice         -- regular non-unit stride (every-other)
    non_contig    -- view that is non-contiguous AND not a clean transpose
    gather        -- irregular access (gather-induced stride pattern)

Each category is independently chosen because each exercises a different
code path inside PyTorch's kernel dispatcher. A v1.0 test that passes
``row_major`` and fails ``broadcast`` has likely tripped over a missing
broadcast-aware kernel branch.

The module is **lazy** with respect to torch — it raises
:class:`RuntimeError` on first call if ``torch`` isn't installed, mirroring
the rest of ``gpucheck.fuzzing``.
"""

from __future__ import annotations

from typing import Any

CATEGORIES: tuple[str, ...] = (
    "row_major",
    "column_major",
    "broadcast",
    "transpose",
    "slice",
    "non_contig",
    "gather",
)


def _torch_mod() -> Any:
    try:
        import torch

        return torch
    except ImportError as exc:  # pragma: no cover  -- exercised when torch absent
        raise RuntimeError(
            "gpucheck.fuzzing.strides requires PyTorch: pip install gpucheck[torch]"
        ) from exc


def _row_major(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    torch = _torch_mod()
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()


def _column_major(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """Column-major: build the contiguous transposed shape, then transpose back.

    For ndim < 2 the concept is undefined; we fall back to row_major.
    """
    torch = _torch_mod()
    if len(shape) < 2:
        return _row_major(shape, dtype, device, gen)
    transposed = (shape[1], shape[0]) + shape[2:]
    base = torch.randn(transposed, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()
    return base.transpose(0, 1)


def _broadcast(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """Broadcast-induced stride-0 dim along the LAST axis.

    For shape (M, N, K), build a contiguous (M, N, 1) tensor and expand to
    (M, N, K). The last dim has stride 0 — kernels that scan strides
    naively will multiply-count or read past bounds.
    """
    torch = _torch_mod()
    if not shape:
        return torch.empty(shape, dtype=dtype, device=device)
    base_shape = shape[:-1] + (1,)
    base = torch.randn(base_shape, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()
    return base.expand(shape)


def _transpose(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """2D-style stride permutation: builds the transposed shape contiguous,
    transposes, returns. Different from :func:`_column_major` only in that
    transpose dims may be non-(0, 1) for higher-rank tensors — we transpose
    the LAST two dims for ndim >= 2.
    """
    torch = _torch_mod()
    if len(shape) < 2:
        return _row_major(shape, dtype, device, gen)
    # Transpose last two dims, e.g. (B, M, N) -> build (B, N, M) contiguous,
    # then .transpose(-1, -2) to recover (B, M, N) with permuted strides.
    transposed = shape[:-2] + (shape[-1], shape[-2])
    base = torch.randn(transposed, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()
    return base.transpose(-1, -2)


def _slice(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """Regular non-unit stride via every-other slicing.

    Build a tensor with each dim doubled, then slice ``[::2, ::2, ...]``.
    The resulting view has stride 2 in every dim.
    """
    torch = _torch_mod()
    if not shape:
        return _row_major(shape, dtype, device, gen)
    big_shape = tuple(d * 2 for d in shape)
    base = torch.randn(big_shape, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()
    slicer = tuple(slice(None, None, 2) for _ in shape)
    return base[slicer]


def _non_contig(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """Non-contiguous view that is NOT a clean transpose or slice.

    For ndim >= 3, we permute dims (1, 0, 2, ...). For ndim 2, we transpose
    and then slice the last dim by 2 — guaranteed non-contiguous and not a
    pure transpose. For ndim 1, fall back to slice.
    """
    torch = _torch_mod()
    if len(shape) <= 1:
        return _slice(shape, dtype, device, gen)
    if len(shape) == 2:
        big = (shape[1], shape[0] * 2)
        base = torch.randn(big, generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        # transpose then slice the now-first dim by 2
        return base.transpose(0, 1)[::2]
    # ndim >= 3 — permute first two dims
    transposed = (shape[1], shape[0]) + shape[2:]
    base = torch.randn(transposed, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous()
    return base.transpose(0, 1)


def _gather(shape: tuple[int, ...], dtype: Any, device: str, gen: Any) -> Any:
    """Irregular-access tensor: gather a contiguous source by a random index.

    The result is a contiguous tensor of the right shape, but it was
    materialized via gather — so kernels that combine gather + reduction
    in fused patterns may exhibit different behavior than a pure
    contiguous input. (We return the gathered view contiguous; the test
    harness's value is in *how* it was built, not the runtime layout.)
    """
    torch = _torch_mod()
    if not shape:
        return _row_major(shape, dtype, device, gen)
    src = torch.randn(shape, generator=gen, dtype=torch.float32).to(
        dtype=dtype, device=device,
    ).contiguous().flatten()
    numel = src.numel()
    idx = torch.randperm(numel, generator=gen)
    return src[idx].reshape(shape).contiguous()


_CATEGORY_FN: dict[str, Any] = {
    "row_major": _row_major,
    "column_major": _column_major,
    "broadcast": _broadcast,
    "transpose": _transpose,
    "slice": _slice,
    "non_contig": _non_contig,
    "gather": _gather,
}


def fuzz_strides_for_category(
    shape: tuple[int, ...],
    dtype: Any,
    category: str,
    *,
    device: str = "cpu",
    seed: int | None = None,
) -> Any:
    """Build a single tensor of the given stride category.

    Use this when a test is parametrized over categories (typically via
    :func:`parametrize_gpu(stride_categories=...)`).
    """
    if category not in _CATEGORY_FN:
        raise ValueError(
            f"Unknown stride category {category!r}; "
            f"expected one of {sorted(CATEGORIES)}"
        )
    torch = _torch_mod()
    gen: Any = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return _CATEGORY_FN[category](shape, dtype, device, gen)


def fuzz_strides(
    shape: tuple[int, ...],
    dtype: Any,
    *,
    n: int | None = None,
    device: str = "cpu",
    seed: int | None = None,
    categories: tuple[str, ...] | None = None,
) -> list[tuple[str, Any]]:
    """Return a deterministic ``[(category, tensor), ...]`` corpus.

    Parameters
    ----------
    shape:
        Tensor shape used for every category.
    dtype:
        torch dtype for every tensor.
    n:
        Cap on the number of items returned. ``None`` returns all
        configured categories (default 7).
    device:
        Target device string.
    seed:
        Optional torch RNG seed for reproducibility.
    categories:
        Override the default category order. Useful for tests that want
        only a subset (e.g. only the non-contiguous flavors).
    """
    cats = tuple(categories) if categories else CATEGORIES
    invalid = [c for c in cats if c not in _CATEGORY_FN]
    if invalid:
        raise ValueError(
            f"Unknown stride categories: {invalid}; "
            f"expected from {sorted(CATEGORIES)}"
        )
    out: list[tuple[str, Any]] = []
    for cat in cats:
        out.append((cat, fuzz_strides_for_category(
            shape, dtype, cat, device=device, seed=seed,
        )))
    if n is not None:
        out = out[:n]
    return out


# ---------------------------------------------------------------------------
# Hypothesis strategy
# ---------------------------------------------------------------------------

class StrideStrategy:
    """Hypothesis-compatible factory that draws a stride-perturbed tensor.

    Mirrors :class:`gpucheck.fuzzing.ShapeStrategy`'s ``__new__``-as-factory
    pattern so callers can write::

        from hypothesis import given
        @given(t=StrideStrategy(shape=(64, 64), dtype=torch.float32))
        def test_kernel_handles_strides(t): ...

    Hypothesis will draw one of the seven categories per test case and
    shrink towards ``row_major``.
    """

    def __new__(
        cls,
        shape: tuple[int, ...],
        dtype: Any = None,
        *,
        device: str = "cpu",
        categories: tuple[str, ...] | None = None,
    ) -> Any:
        try:
            from hypothesis import strategies as st
        except ImportError as exc:
            raise RuntimeError(
                "StrideStrategy requires hypothesis: pip install gpucheck[hypothesis]"
            ) from exc

        torch = _torch_mod()
        if dtype is None:
            dtype = torch.float32

        cats = tuple(categories) if categories else CATEGORIES

        @st.composite
        def _draw(draw: Any) -> Any:
            cat = draw(st.sampled_from(cats))
            seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
            return fuzz_strides_for_category(
                shape, dtype, cat, device=device, seed=seed,
            )

        return _draw()


__all__ = [
    "CATEGORIES",
    "fuzz_strides",
    "fuzz_strides_for_category",
    "StrideStrategy",
]
