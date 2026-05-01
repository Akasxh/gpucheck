"""parametrize_gpu(stride_categories=...) wiring (Track B)."""

from __future__ import annotations

import pytest

from gpucheck.decorators.parametrize import parametrize_gpu

torch = pytest.importorskip("torch")


@parametrize_gpu(
    dtypes=("float32",),
    shapes=((4, 4),),
    devices=("cpu",),
    stride_categories=("row_major", "transpose"),
)
def test_stride_categories_appear_in_signature(dtype, shape, device, stride_category) -> None:
    assert stride_category in {"row_major", "transpose"}
    assert shape == (4, 4)
    assert device == "cpu"


def test_parametrize_gpu_rejects_unknown_stride_category() -> None:
    with pytest.raises(ValueError, match="Unknown stride categories"):
        parametrize_gpu(
            dtypes=("float32",),
            shapes=((4, 4),),
            devices=("cpu",),
            stride_categories=("row_major", "wat"),
        )


def test_parametrize_gpu_without_stride_categories_keeps_old_signature() -> None:
    decorator = parametrize_gpu(
        dtypes=("float32",),
        shapes=((4, 4),),
        devices=("cpu",),
    )

    # The decorator marker name should not contain stride_category.
    @decorator
    def _fake_test(dtype, shape, device) -> None:  # noqa: ARG001
        pass

    # Inspect the param names attached by pytest.mark.parametrize:
    marks = list(_fake_test.pytestmark)
    assert any("stride_category" not in m.args[0] for m in marks)
