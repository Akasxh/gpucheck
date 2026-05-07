"""All-in-one GPU test parametrization."""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from gpucheck.decorators.devices import _detect_devices, _is_device_available
from gpucheck.decorators.dtypes import DtypeArg, _dtype_id, _resolve_dtype
from gpucheck.decorators.shapes import Shape, _shape_id

# Type for the skip-filter callback
SkipFilter = Callable[..., bool] | None


def _combo_id(
    dtype: Any,
    shape: Shape,
    device: str,
    stride_category: str | None = None,
) -> str:
    """Build a human-readable test ID: 'float16-128x128-cuda0[-broadcast]'."""
    parts: list[str] = []
    parts.append(_dtype_id(dtype))
    parts.append(_shape_id(shape))
    parts.append(device.replace(":", ""))
    if stride_category is not None:
        parts.append(stride_category)
    return "-".join(parts)


def parametrize_gpu(
    *,
    dtypes: Sequence[DtypeArg] = ("float16", "float32"),
    shapes: Sequence[Shape] = ((128, 128),),
    devices: Sequence[str] | None = None,
    skip: SkipFilter = None,
    stride_categories: Sequence[str] | None = None,
) -> Callable[..., Any]:
    """Parametrize a test over the cartesian product of dtypes x shapes x devices.

    Args:
        dtypes: Dtype strings or torch.dtype objects.
        shapes: Tensor shape tuples.
        devices: Device strings. ``None`` auto-detects CUDA devices.
        skip: Optional callable ``(dtype, shape, device) -> bool``
              (or ``(dtype, shape, device, stride_category) -> bool`` when
              ``stride_categories`` is set). Return ``True`` to skip that
              combination.
        stride_categories: Optional sequence of stride-fuzzing categories
              (see :data:`gpucheck.fuzzing.STRIDE_CATEGORIES`). When
              provided, the test signature gains a ``stride_category: str``
              parameter and the cartesian product expands accordingly. Use
              :func:`gpucheck.fuzzing.fuzz_strides_for_category` inside the
              test body to materialize the perturbed tensor.

    Examples::

        @parametrize_gpu(
            dtypes=("float16", "bfloat16"),
            shapes=((128, 128), (256, 256)),
            devices=("cuda:0",),
        )
        def test_kernel(dtype, shape, device): ...

        @parametrize_gpu(
            dtypes=("float32",),
            shapes=((64, 64),),
            stride_categories=("row_major", "transpose", "broadcast"),
        )
        def test_layout_invariant(dtype, shape, device, stride_category):
            from gpucheck.fuzzing import fuzz_strides_for_category
            t = fuzz_strides_for_category(shape, dtype, stride_category, device=device)
            ...
    """
    # Resolve dtypes
    resolved_dtypes = [_resolve_dtype(d) for d in dtypes]

    # Resolve devices: auto-detect CUDA + MPS when caller passes ``None``.
    if devices is None:
        detected = _detect_devices()
        resolved_devices = detected if detected else ["cuda:0"]
    else:
        resolved_devices = list(devices)

    # Resolve stride categories
    use_strides = stride_categories is not None
    resolved_strides: list[str] = list(stride_categories) if stride_categories else []
    if use_strides:
        # Validate eagerly; bad input here is a test-author bug.
        from gpucheck.fuzzing.strides import CATEGORIES as _ALLOWED

        invalid = [c for c in resolved_strides if c not in _ALLOWED]
        if invalid:
            raise ValueError(
                f"Unknown stride categories: {invalid}; "
                f"expected from {sorted(_ALLOWED)}"
            )

    # Build cartesian product as pytest.param entries
    params: list[Any] = []

    if not use_strides:
        for dtype_val, shape_val, dev_val in itertools.product(
            resolved_dtypes, shapes, resolved_devices,
        ):
            test_id = _combo_id(dtype_val, shape_val, dev_val)
            marks: list[Any] = []

            if skip is not None and skip(dtype_val, shape_val, dev_val):
                marks.append(pytest.mark.skip(reason="filtered by skip predicate"))

            if not _is_device_available(dev_val):
                marks.append(
                    pytest.mark.skip(reason=f"device {dev_val} not available"),
                )

            params.append(
                pytest.param(dtype_val, shape_val, dev_val, id=test_id, marks=marks),
            )

        return pytest.mark.parametrize("dtype,shape,device", params)

    # Stride-fuzzing branch: cartesian also includes stride_category.
    for dtype_val, shape_val, dev_val, stride_cat in itertools.product(
        resolved_dtypes, shapes, resolved_devices, resolved_strides,
    ):
        test_id = _combo_id(dtype_val, shape_val, dev_val, stride_cat)
        marks = []

        if skip is not None:
            # 4-arg skip; tolerate 3-arg by checking signature length.
            try:
                hit = skip(dtype_val, shape_val, dev_val, stride_cat)
            except TypeError:
                hit = skip(dtype_val, shape_val, dev_val)
            if hit:
                marks.append(pytest.mark.skip(reason="filtered by skip predicate"))

        if not _is_device_available(dev_val):
            marks.append(
                pytest.mark.skip(reason=f"device {dev_val} not available"),
            )

        params.append(
            pytest.param(
                dtype_val, shape_val, dev_val, stride_cat,
                id=test_id, marks=marks,
            ),
        )

    return pytest.mark.parametrize("dtype,shape,device,stride_category", params)


__all__ = ["parametrize_gpu"]
