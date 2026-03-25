"""Shape fuzzer — generates tensor shapes likely to trigger GPU kernel bugs."""

from __future__ import annotations

import random
from itertools import product
from typing import Any

# Common GPU tile sizes used by CUDA/Triton kernels.
TILE_SIZES: tuple[int, ...] = (32, 64, 128)

PRIMES: tuple[int, ...] = (7, 13, 31, 127, 257)

POWER_OF_2_BOUNDARIES: tuple[int, ...] = (127, 128, 129, 255, 256, 257, 511, 512, 513)

LARGE_DIMS: tuple[int, ...] = (2048, 4096, 8192)


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


def _non_tile_aligned_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
    """Shapes not divisible by common tile sizes."""
    candidates = []
    for tile in TILE_SIZES:
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


def _power_of_2_boundary_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
    """Shapes near powers of 2 — expose fencepost errors in tiling logic."""
    vals = [v for v in POWER_OF_2_BOUNDARIES if v <= max_size]
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
) -> list[tuple[int, ...]]:
    """Generate *n* shape tuples designed to find GPU kernel bugs.

    Categories (ranked by bug-finding probability):
      1. Degenerate — zeros, ones
      2. Non-tile-aligned — not divisible by 32/64/128
      3. Prime dimensions — 7, 13, 31, 127, 257
      4. Power-of-2 boundaries — 127..129, 255..257
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
    """
    if min_size > max_size:
        raise ValueError(f"min_size ({min_size}) must be <= max_size ({max_size})")
    if ndim < 0:
        raise ValueError(f"ndim must be >= 0, got {ndim}")

    pool: list[tuple[int, ...]] = []

    pool.extend(_degenerate_shapes(ndim))
    pool.extend(_non_tile_aligned_shapes(ndim, max_size))
    pool.extend(_prime_shapes(ndim, max_size))
    pool.extend(_power_of_2_boundary_shapes(ndim, max_size))
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
    ) -> Any:
        return cls._build(ndim, min_size, max_size)

    @staticmethod
    def _build(ndim: int, min_size: int, max_size: int) -> Any:
        try:
            from hypothesis import strategies as st
        except ImportError as exc:
            raise RuntimeError(
                "ShapeStrategy requires hypothesis: pip install gpucheck[hypothesis]"
            ) from exc

        # Bias towards bug-triggering values.
        interesting = sorted(
            {v for v in (
                0, 1,
                *PRIMES,
                *POWER_OF_2_BOUNDARIES,
                *[t - 1 for t in TILE_SIZES],
                *[t + 1 for t in TILE_SIZES],
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
    "PRIMES",
    "POWER_OF_2_BOUNDARIES",
    "LARGE_DIMS",
]
