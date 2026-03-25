"""All-in-one GPU test parametrization."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from gpucheck.decorators.devices import _detect_cuda_devices, _is_device_available
from gpucheck.decorators.dtypes import DtypeArg, _dtype_id, _resolve_dtype
from gpucheck.decorators.shapes import Shape, _shape_id

# Type for the skip-filter callback
SkipFilter = Callable[..., bool] | None


def _combo_id(dtype: Any, shape: Shape, device: str) -> str:
    """Build a human-readable test ID: 'float16-128x128-cuda0'."""
    parts: list[str] = []
    parts.append(_dtype_id(dtype))
    parts.append(_shape_id(shape))
    parts.append(device.replace(":", ""))
    return "-".join(parts)


def parametrize_gpu(
    *,
    dtypes: Sequence[DtypeArg] = ("float16", "float32"),
    shapes: Sequence[Shape] = ((128, 128),),
    devices: Sequence[str] | None = None,
    skip: SkipFilter = None,
) -> Callable[..., Any]:
    """Parametrize a test over the cartesian product of dtypes x shapes x devices.

    Args:
        dtypes: Dtype strings or torch.dtype objects.
        shapes: Tensor shape tuples.
        devices: Device strings. ``None`` auto-detects CUDA devices.
        skip: Optional callable ``(dtype, shape, device) -> bool``.
              Return ``True`` to skip that combination.

    Example::

        @parametrize_gpu(
            dtypes=("float16", "bfloat16"),
            shapes=((128, 128), (256, 256)),
            devices=("cuda:0",),
        )
        def test_kernel(dtype, shape, device):
            ...
    """
    # Resolve dtypes
    resolved_dtypes = [_resolve_dtype(d) for d in dtypes]

    # Resolve devices
    if devices is None:
        detected = _detect_cuda_devices()
        resolved_devices = detected if detected else ["cuda:0"]
    else:
        resolved_devices = list(devices)

    # Build cartesian product as pytest.param entries
    params: list[Any] = []
    for dtype_val, shape_val, dev_val in itertools.product(
        resolved_dtypes, shapes, resolved_devices
    ):
        test_id = _combo_id(dtype_val, shape_val, dev_val)
        marks: list[Any] = []

        if skip is not None and skip(dtype_val, shape_val, dev_val):
            marks.append(pytest.mark.skip(reason="filtered by skip predicate"))

        if not _is_device_available(dev_val):
            marks.append(
                pytest.mark.skip(reason=f"device {dev_val} not available")
            )

        params.append(
            pytest.param(dtype_val, shape_val, dev_val, id=test_id, marks=marks)
        )

    return pytest.mark.parametrize("dtype,shape,device", params)


__all__ = ["parametrize_gpu"]
