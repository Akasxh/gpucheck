"""Determinism sanitizer — assert byte-identical outputs across runs.

Unlike CUDA, MPS is best-effort deterministic (research SYNTHESIS §4):
PyTorch documentation is silent on Apple Silicon determinism guarantees,
and the empirical record (pytorch#181936, #170837, #177116) shows real
run-to-run divergence. This module provides:

- :func:`assert_deterministic` — runs ``fn`` ``n`` times under fixed seeds
  and asserts every output tensor is byte-identical to the first run.
  On MPS, structured failure surfaces ``DeterminismError`` so callers
  know whether the divergence is at the precision floor (acceptable in
  some pipelines) or a literal inconsistency.
- :func:`requires_determinism` — function decorator: wraps the test body
  so that calling it n times is the test (instead of the test author
  having to write the loop themselves).

The seeded run sets:

    torch.manual_seed(seed)
    if mps available: torch.mps.manual_seed(seed)
    if cuda available: torch.cuda.manual_seed_all(seed)
    random.seed(seed); numpy.random.seed(seed)
"""

from __future__ import annotations

import functools
import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable


class DeterminismError(AssertionError):
    """Raised when ``assert_deterministic`` observes diverging outputs."""


def _seed_all(seed: int) -> None:
    """Seed every RNG we know about. Best-effort — silent on missing modules."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            seed_fn = getattr(torch.mps, "manual_seed", None)
            if callable(seed_fn):
                seed_fn(seed)
    except ImportError:
        pass


def _equal(a: Any, b: Any) -> bool:
    """Compare two outputs for byte-identical equality.

    Handles torch.Tensor (same device, same dtype), tuples / lists
    elementwise, and falls back to ``==``.
    """
    try:
        import torch

        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
                return False
            return bool(torch.equal(a, b))
    except ImportError:
        pass
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b, strict=False))
    return bool(a == b)


def assert_deterministic(
    fn: Callable[..., Any],
    *args: Any,
    n: int = 3,
    seed: int = 0,
    **kwargs: Any,
) -> Any:
    """Run *fn* ``n`` times under fixed seeds; assert outputs match exactly.

    Parameters
    ----------
    fn:
        Callable producing the output to compare. May return a tensor, a
        tuple of tensors, or any equality-comparable value.
    *args / **kwargs:
        Forwarded to *fn*.
    n:
        Number of repetitions. Must be ``>= 2``.
    seed:
        Seed applied to ``random``, ``numpy.random``, ``torch.manual_seed``,
        ``torch.cuda.manual_seed_all``, and ``torch.mps.manual_seed`` (if
        available) before each call.

    Returns
    -------
    The output of the first run, so callers can pass through values
    that they want to use after asserting determinism.

    Raises
    ------
    DeterminismError:
        If any run's output differs from the first run.
    """
    if n < 2:
        raise ValueError(f"assert_deterministic requires n >= 2, got {n}")

    _seed_all(seed)
    first = fn(*args, **kwargs)
    for i in range(1, n):
        _seed_all(seed)
        candidate = fn(*args, **kwargs)
        if not _equal(first, candidate):
            raise DeterminismError(
                f"assert_deterministic: run {i} produced output that differs "
                f"from run 0 (n={n}, seed={seed}). On MPS this can happen "
                f"legitimately (best-effort determinism per SYNTHESIS §4); "
                f"consider widening tolerances via tolerance_context, or "
                f"adding the op to the [tool.gpucheck.mps.xfail] block."
            )
    return first


def requires_determinism(
    *,
    n: int = 3,
    seed: int = 0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: invoke the test function ``n`` times under fixed seeds.

    Equivalent to wrapping the test body in
    :func:`assert_deterministic`. Useful when the test's return value is
    the artifact under test::

        @requires_determinism(n=5, seed=42)
        def test_my_kernel():
            x = torch.randn(64, 64, device="mps")
            return my_kernel(x)
    """
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return assert_deterministic(fn, *args, n=n, seed=seed, **kwargs)
        return wrapper
    return decorator


__all__ = ["assert_deterministic", "requires_determinism", "DeterminismError"]
