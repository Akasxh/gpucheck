"""Shape fuzzer — generates tensor shapes likely to trigger GPU kernel bugs."""

from __future__ import annotations

import random
from itertools import product
from typing import Any, Literal

# Common GPU tile sizes used by CUDA/Triton kernels.
TILE_SIZES: tuple[int, ...] = (32, 64, 128)

# Apple-Silicon (MPS / Metal / MLX) tile sizes.
#
# Sourced from MLX matmul / conv2d / SDPA kernels and ggml-metal as of
# 2026-05-01. The 8x8 simdgroup_matrix MMA fragment (mlx/backend/metal/
# kernels/steel/gemm/mma.h) and BK=16 GEMM block (steel_gemm_fused.metal)
# are Apple-canonical and absent from CUDA's tile set. 32/64/128 are kept
# because they also appear on Apple (32 SIMD width, 64 dominant block,
# 128 NAX/FA-head_dim). See cartographer-v2.md §3 for full citations.
TILE_SIZES_MPS: tuple[int, ...] = (8, 16, 32, 64, 128)

PRIMES: tuple[int, ...] = (7, 13, 31, 127, 257)

POWER_OF_2_BOUNDARIES: tuple[int, ...] = (127, 128, 129, 255, 256, 257, 511, 512, 513)

# Apple-Silicon power-of-2 boundaries.
#
# Adds:
#   {7, 8, 9}   -- 8x8 MMA-fragment edges (NEW for MPS)
#   {15, 16, 17}-- BK=16 GEMM block edges (NEW for MPS)
#   {31, 32, 33}-- SIMD-width edges
#   {63, 64, 65}-- 64-block edges
#   {79, 80, 81}-- FA head_dim 80, an Apple-only optimisation shape
# Keeps the existing 127/128/129, 255/256/257, 511/512/513.
POWER_OF_2_BOUNDARIES_MPS: tuple[int, ...] = (
    7, 8, 9,
    15, 16, 17,
    31, 32, 33,
    63, 64, 65,
    79, 80, 81,
    127, 128, 129,
    255, 256, 257,
    511, 512, 513,
)

LARGE_DIMS: tuple[int, ...] = (2048, 4096, 8192)

DeviceType = Literal["cuda", "mps"]


def _tile_set(device_type: str) -> tuple[int, ...]:
    """Return the canonical tile-size tuple for *device_type*."""
    if device_type == "mps":
        return TILE_SIZES_MPS
    if device_type == "cuda":
        return TILE_SIZES
    raise ValueError(
        f"Unknown device_type {device_type!r}; expected one of 'cuda', 'mps'"
    )


def _pow2_boundary_set(device_type: str) -> tuple[int, ...]:
    """Return the canonical power-of-2-boundary tuple for *device_type*."""
    if device_type == "mps":
        return POWER_OF_2_BOUNDARIES_MPS
    if device_type == "cuda":
        return POWER_OF_2_BOUNDARIES
    raise ValueError(
        f"Unknown device_type {device_type!r}; expected one of 'cuda', 'mps'"
    )


def _degenerate_shapes(ndim: int) -> list[tuple[int, ...]]:
    """Shapes with zeros and ones — expose off-by-one and empty-tensor bugs."""
    if ndim < 1:
        return [()]
    if ndim == 1:
        return [(0,), (1,)]

    shapes: list[tuple[int, ...]] = []
    # Zero in each position
    for i in range(ndim):
        s = [16] * ndim
        s[i] = 0
        shapes.append(tuple(s))
    # All ones
    shapes.append((1,) * ndim)
    # One in each position (rest = 16)
    for i in range(ndim):
        s = [16] * ndim
        s[i] = 1
        shapes.append(tuple(s))
    return shapes


def _non_tile_aligned_shapes(
    ndim: int,
    max_size: int,
    device_type: str = "cuda",
) -> list[tuple[int, ...]]:
    """Shapes not divisible by common tile sizes for *device_type*."""
    candidates = []
    tiles = _tile_set(device_type)
    for tile in tiles:
        for offset in (-1, 1, 3):
            v = tile + offset
            if 1 <= v <= max_size:
                candidates.append(v)
    # Build ndim-tuples from unique candidates
    candidates = sorted(set(candidates))
    shapes: list[tuple[int, ...]] = []
    for v in candidates:
        shapes.append((v,) * ndim)
    return shapes


def _prime_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
    """Shapes with prime dimensions — stress non-uniform loop tails."""
    primes = [p for p in PRIMES if p <= max_size]
    return [(p,) * ndim for p in primes]


def _power_of_2_boundary_shapes(
    ndim: int,
    max_size: int,
    device_type: str = "cuda",
) -> list[tuple[int, ...]]:
    """Shapes near powers of 2 — expose fencepost errors in tiling logic."""
    vals = [v for v in _pow2_boundary_set(device_type) if v <= max_size]
    return [(v,) * ndim for v in vals]


def _large_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
    """Large shapes — stress memory and grid-size limits."""
    vals = [v for v in LARGE_DIMS if v <= max_size]
    return [(v,) * ndim for v in vals]


def _mixed_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
    """Asymmetric shapes — mix categories to trigger mismatched-stride bugs."""
    if ndim < 2:
        return []

    small = [1, 3, 7]
    large = [v for v in (1024, 2048, 4096) if v <= max_size]
    prime = [p for p in PRIMES if p <= max_size]
    pow2 = [128, 256, 512]

    shapes: list[tuple[int, ...]] = []
    for a, b in product(large[:2], small[:2]):
        base = [a, b] + [16] * (ndim - 2)
        shapes.append(tuple(base[:ndim]))
    for a, b in product(prime[:2], pow2[:2]):
        if a <= max_size and b <= max_size:
            base = [a, b] + [16] * (ndim - 2)
            shapes.append(tuple(base[:ndim]))
    return shapes


def fuzz_shapes(
    ndim: int = 2,
    *,
    min_size: int = 1,
    max_size: int = 4096,
    n: int = 50,
    seed: int | None = None,
    device_type: str = "cuda",
) -> list[tuple[int, ...]]:
    """Generate *n* shape tuples designed to find GPU kernel bugs.

    Categories (ranked by bug-finding probability):
      1. Degenerate — zeros, ones
      2. Non-tile-aligned — not divisible by tile sizes for *device_type*
      3. Prime dimensions — 7, 13, 31, 127, 257
      4. Power-of-2 boundaries — device-specific (CUDA: 127..129..513;
         MPS: adds 7..9, 15..17, 31..33, 63..65, 79..81)
      5. Large — 2048, 4096, 8192
      6. Mixed — (large, small), (prime, power_of_2)

    Parameters
    ----------
    ndim:
        Number of dimensions in each shape tuple.
    min_size:
        Minimum value for any single dimension (degenerate 0-dims are always
        included regardless).
    max_size:
        Maximum value for any single dimension.
    n:
        Number of shapes to return.  May return fewer if the pool of unique
        shapes is smaller than *n*.
    seed:
        Optional RNG seed for reproducibility.
    device_type:
        ``"cuda"`` (default, preserves v1.0 behaviour) selects the
        NVIDIA-canonical tile set {32, 64, 128} and CUDA pow2-boundary
        list. ``"mps"`` selects the Apple-canonical tile set
        {8, 16, 32, 64, 128} and MPS pow2-boundary list — this surfaces
        bugs aligned to MLX's 8x8 simdgroup_matrix MMA fragment and
        Apple's BK=16 GEMM block that the CUDA fuzzer otherwise misses.
        See ``cartographer-v2.md`` for the source citations.
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) must be <= max_size ({max_size})")
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")
    if device_type not in ("cuda", "mps"):
        raise ValueError(
            f"device_type must be 'cuda' or 'mps', got {device_type!r}"
        )

    pool: list[tuple[int, ...]] = []

    pool.extend(_degenerate_shapes(ndim))
    pool.extend(_non_tile_aligned_shapes(ndim, max_size, device_type=device_type))
    pool.extend(_prime_shapes(ndim, max_size))
    pool.extend(_power_of_2_boundary_shapes(ndim, max_size, device_type=device_type))
    pool.extend(_large_shapes(ndim, max_size))
    pool.extend(_mixed_shapes(ndim, max_size))

    # Filter by min_size (allow 0 for degenerate shapes).
    pool = [
        s for s in pool
        if all(d == 0 or d >= min_size for d in s)
    ]

    # Deduplicate while preserving priority order.
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[int, ...]] = []
    for s in pool:
        if s not in seen:
            seen.add(s)
            unique.append(s)

    if len(unique) >= n:
        return unique[:n]

    # For ndim=0, the only possible shape is (), so we can't generate more.
    if ndim == 0:
        return unique

    # Cap n at the number of unique shapes that can be generated.
    dim_range = max_size - min_size + 1
    max_possible = dim_range ** ndim
    if n > len(unique) + max_possible:
        n = len(unique) + max_possible

    # Pad with random shapes to reach n.
    rng = random.Random(seed)
    while len(unique) < n:
        shape = tuple(rng.randint(min_size, max_size) for _ in range(ndim))
        if shape not in seen:
            seen.add(shape)
            unique.append(shape)

    return unique


class ShapeStrategy:
    """Hypothesis-compatible strategy factory for GPU tensor shapes.

    Requires ``hypothesis`` to be installed.  Lazily imports it so the rest
    of the module works without the dependency.

    Returns a proper ``SearchStrategy`` instance so ``@given(shape=ShapeStrategy(...))``
    works directly.

    Usage::

        from hypothesis import given
        from gpucheck.fuzzing import ShapeStrategy

        @given(shape=ShapeStrategy(ndim=2, max_size=512))
        def test_kernel(shape): ...
    """

    def __new__(
        cls,
        ndim: int = 2,
        *,
        min_size: int = 1,
        max_size: int = 4096,
        device_type: str = "cuda",
    ) -> Any:
        return cls._build(ndim, min_size, max_size, device_type)

    @staticmethod
    def _build(
        ndim: int, min_size: int, max_size: int, device_type: str = "cuda",
    ) -> Any:
        try:
            from hypothesis import strategies as st
        except ImportError as exc:
            raise RuntimeError(
                "ShapeStrategy requires hypothesis: pip install gpucheck[hypothesis]"
            ) from exc

        if device_type not in ("cuda", "mps"):
            raise ValueError(
                f"device_type must be 'cuda' or 'mps', got {device_type!r}"
            )

        tiles = _tile_set(device_type)
        boundaries = _pow2_boundary_set(device_type)

        # Bias towards bug-triggering values.
        interesting = sorted(
            {v for v in (
                0, 1,
                *PRIMES,
                *boundaries,
                *[t - 1 for t in tiles],
                *[t + 1 for t in tiles],
                *tiles,
            ) if v <= max_size}
        )
        dim_strategy = st.one_of(
            st.sampled_from(interesting),
            st.integers(min_value=min_size, max_value=max_size),
        )
        return st.tuples(*([dim_strategy] * ndim))


__all__ = [
    "fuzz_shapes",
    "ShapeStrategy",
    "TILE_SIZES",
    "TILE_SIZES_MPS",
    "PRIMES",
    "POWER_OF_2_BOUNDARIES",
    "POWER_OF_2_BOUNDARIES_MPS",
    "LARGE_DIMS",
]
