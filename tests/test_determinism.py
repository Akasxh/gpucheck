"""Determinism sanitizer tests (Track D — D.3)."""

from __future__ import annotations

import random

import pytest

from gpucheck.sanitizers.determinism import (
    DeterminismError,
    assert_deterministic,
    requires_determinism,
)


def test_assert_deterministic_passes_when_output_is_deterministic() -> None:
    def deterministic_fn() -> int:
        return random.randint(0, 100)  # noqa: S311 (test, not crypto)

    # Seeded properly, this returns the same int each call.
    out = assert_deterministic(deterministic_fn, n=4, seed=7)
    assert isinstance(out, int)


def test_assert_deterministic_raises_when_output_diverges() -> None:
    counter = {"i": 0}

    def diverging_fn() -> int:
        counter["i"] += 1
        return counter["i"]  # 1, 2, 3, ... — not seeded by random

    with pytest.raises(DeterminismError, match="differs"):
        assert_deterministic(diverging_fn, n=3, seed=0)


def test_assert_deterministic_rejects_n_lt_2() -> None:
    with pytest.raises(ValueError, match="n >= 2"):
        assert_deterministic(lambda: 1, n=1)


def test_assert_deterministic_returns_first_output() -> None:
    def fn() -> str:
        return "stable"

    out = assert_deterministic(fn, n=3)
    assert out == "stable"


def test_requires_determinism_decorator_calls_fn_n_times() -> None:
    counter = {"i": 0}

    @requires_determinism(n=4, seed=99)
    def fn() -> int:
        counter["i"] += 1
        # Reset randomness here makes this deterministic across calls
        # because the decorator calls _seed_all between invocations.
        return random.randint(0, 1_000_000)  # noqa: S311

    fn()
    assert counter["i"] == 4


def test_requires_determinism_propagates_determinism_error_from_unstable_fn() -> None:
    counter = {"i": 0}

    @requires_determinism(n=2, seed=0)
    def fn() -> int:
        counter["i"] += 1
        return counter["i"]

    with pytest.raises(DeterminismError):
        fn()


def test_assert_deterministic_handles_tuple_outputs() -> None:
    def fn() -> tuple[int, int]:
        return (random.randint(0, 100), random.randint(0, 100))  # noqa: S311

    out = assert_deterministic(fn, n=3, seed=42)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_assert_deterministic_handles_list_outputs_diverging() -> None:
    counter = {"i": 0}

    def fn() -> list[int]:
        counter["i"] += 1
        return [counter["i"]]

    with pytest.raises(DeterminismError):
        assert_deterministic(fn, n=2, seed=0)


def test_assert_deterministic_with_torch_tensor_outputs() -> None:
    """When torch is available, tensor equality goes through torch.equal."""
    torch = pytest.importorskip("torch")

    def fn() -> torch.Tensor:
        return torch.randn(3, 3)  # seeded => same tensor

    out = assert_deterministic(fn, n=3, seed=0)
    assert out.shape == (3, 3)
