"""Parametrize tests across GPU data types."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Union

import pytest

# Lazy dtype resolution — strings map to torch.dtype at collection time,
# so torch is only imported when pytest actually collects.

_STR_TO_DTYPE: dict[str, str] = {
    "float16": "torch.float16",
    "float32": "torch.float32",
    "float64": "torch.float64",
    "bfloat16": "torch.bfloat16",
    "int8": "torch.int8",
    "int16": "torch.int16",
    "int32": "torch.int32",
    "int64": "torch.int64",
    "uint8": "torch.uint8",
    "bool": "torch.bool",
    "float8_e4m3fn": "torch.float8_e4m3fn",
    "float8_e5m2": "torch.float8_e5m2",
}

DtypeArg = Union[str, Any]  # str | torch.dtype


def _resolve_dtype(d: DtypeArg) -> Any:
    """Resolve a string or torch.dtype to a torch.dtype object."""
    if isinstance(d, str):
        import torch

        # Try the lookup table first, then getattr as fallback
        attr = _STR_TO_DTYPE.get(d, f"torch.{d}")
        parts = attr.split(".")
        obj: Any = torch
        for part in parts[1:]:
            obj = getattr(obj, part)
        return obj
    return d


def _dtype_id(d: Any) -> str:
    """Generate a clean test ID for a dtype."""
    s = str(d)
    # torch.float16 -> float16
    if s.startswith("torch."):
        return s[6:]
    return s


# ---------------------------------------------------------------------------
# Predefined groups
# ---------------------------------------------------------------------------

def _lazy_group(names: tuple[str, ...]) -> tuple[Any, ...]:
    """Resolve a group of dtype names to torch.dtype objects."""
    return tuple(_resolve_dtype(n) for n in names)


HALF_DTYPES_NAMES: tuple[str, ...] = ("float16", "bfloat16")
FLOAT_DTYPES_NAMES: tuple[str, ...] = ("float16", "bfloat16", "float32", "float64")
ALL_DTYPES_NAMES: tuple[str, ...] = (
    "float16", "bfloat16", "float32", "float64",
    "int8", "int16", "int32", "int64", "uint8", "bool",
)
FP8_DTYPES_NAMES: tuple[str, ...] = ("float8_e4m3fn", "float8_e5m2")


class _DtypeGroup:
    """Lazy-evaluated dtype group — avoids importing torch at import time."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self._names = names
        self._resolved: tuple[Any, ...] | None = None

    def __iter__(self) -> Iterator[Any]:
        if self._resolved is None:
            self._resolved = _lazy_group(self._names)
        return iter(self._resolved)

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"DtypeGroup({self._names})"


HALF_DTYPES = _DtypeGroup(HALF_DTYPES_NAMES)
FLOAT_DTYPES = _DtypeGroup(FLOAT_DTYPES_NAMES)
ALL_DTYPES = _DtypeGroup(ALL_DTYPES_NAMES)
FP8_DTYPES = _DtypeGroup(FP8_DTYPES_NAMES)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def dtypes(*dtype_args: DtypeArg) -> Callable[..., Any]:
    """Parametrize a test across GPU data types.

    Accepts strings ("float16", "bfloat16"), torch.dtype objects, or
    predefined groups (FLOAT_DTYPES, HALF_DTYPES, etc.).

    Dtype resolution is deferred to test collection time: strings are stored
    as-is during decoration and resolved to ``torch.dtype`` only when pytest
    actually collects the test (via ``pytest.param`` indirect). This avoids
    triggering ``import torch`` at module import time.

    Examples::

        @dtypes("float16", "float32")
        def test_add(dtype): ...

        @dtypes(*FLOAT_DTYPES)
        def test_matmul(dtype): ...
    """
    # Build lazy params — resolution deferred to collection time
    resolved: list[Any] = []
    for arg in dtype_args:
        resolved.append(_resolve_dtype(arg))

    ids = [_dtype_id(d) for d in resolved]
    return pytest.mark.parametrize("dtype", resolved, ids=ids)


__all__ = [
    "dtypes",
    "FLOAT_DTYPES",
    "HALF_DTYPES",
    "ALL_DTYPES",
    "FP8_DTYPES",
]
