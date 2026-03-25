"""Exhaustive tests for gpucheck's decorator system.

Exercises every decorator alone, every pairwise combination, the triple
product, parametrize_gpu, predefined groups, string dtypes, unavailable
device skip markers, and test-ID hygiene.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

from gpucheck.decorators.devices import devices
from gpucheck.decorators.dtypes import (
    ALL_DTYPES,
    ALL_DTYPES_NAMES,
    FLOAT_DTYPES,
    FP8_DTYPES,
    HALF_DTYPES,
    _dtype_id,
    _DtypeGroup,
    _resolve_dtype,
    dtypes,
)
from gpucheck.decorators.parametrize import parametrize_gpu
from gpucheck.decorators.shapes import (
    EDGE_SHAPES,
    SMALL_SHAPES,
    _shape_id,
    shapes,
)

# ===================================================================
# Helpers
# ===================================================================

def _get_parametrize_marks(func: Any) -> list[Any]:
    """Extract all pytest.mark.parametrize marks from a decorated function."""
    marks = getattr(func, "pytestmark", [])
    return [m for m in marks if m.name == "parametrize"]


def _param_count(mark: Any) -> int:
    """Number of parameter values in a parametrize mark."""
    return len(mark.args[1])


# ===================================================================
# 1. @dtypes alone
# ===================================================================

class TestDtypesAlone:
    """@dtypes alone must parametrize exactly over 'dtype'."""

    def test_single_dtype(self) -> None:
        @dtypes("float32")
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 1
        assert marks[0].args[0] == "dtype"
        assert _param_count(marks[0]) == 1

    def test_multiple_dtypes(self) -> None:
        @dtypes("float16", "float32", "float64")
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 3

    def test_torch_dtype_objects(self) -> None:
        @dtypes(torch.float16, torch.bfloat16)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2
        # Values should be actual torch.dtype objects
        values = [p.values[0] for p in marks[0].args[1]]
        assert values == [torch.float16, torch.bfloat16]

    def test_string_dtypes_stay_as_strings_at_decoration(self) -> None:
        """String dtypes should NOT be resolved at decoration time."""
        @dtypes("float16", "float32")
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        values = [p.values[0] for p in marks[0].args[1]]
        # They remain strings — resolution is deferred
        assert all(isinstance(v, str) for v in values)

    def test_ids_are_clean_strings(self) -> None:
        @dtypes("float16", "bfloat16", "float32")
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        ids = [p.id for p in marks[0].args[1]]
        assert ids == ["float16", "bfloat16", "float32"]

    def test_mixed_str_and_torch_dtype(self) -> None:
        @dtypes("float16", torch.float32)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2
        ids = [p.id for p in marks[0].args[1]]
        assert ids == ["float16", "float32"]


# ===================================================================
# 2. @shapes alone
# ===================================================================

class TestShapesAlone:
    """@shapes alone must parametrize exactly over 'shape'."""

    def test_single_shape(self) -> None:
        @shapes((128, 128))
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 1
        assert marks[0].args[0] == "shape"
        assert _param_count(marks[0]) == 1

    def test_multiple_shapes(self) -> None:
        @shapes((32,), (64, 64), (128, 256), (1, 1, 1))
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 4

    def test_shape_values_are_tuples(self) -> None:
        @shapes((32, 32), (7, 13))
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        values = marks[0].args[1]
        assert values[0] == (32, 32)
        assert values[1] == (7, 13)

    def test_shape_ids_use_x_separator(self) -> None:
        assert _shape_id((128, 256)) == "128x256"
        assert _shape_id((32,)) == "32"
        assert _shape_id((1, 1, 1)) == "1x1x1"
        assert _shape_id((0, 128)) == "0x128"

    def test_1d_shapes(self) -> None:
        @shapes((128,), (256,), (8192,))
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 3

    def test_3d_shapes(self) -> None:
        @shapes((2, 3, 4), (1, 1024, 1024))
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2


# ===================================================================
# 3. @devices alone
# ===================================================================

class TestDevicesAlone:
    """@devices alone must parametrize exactly over 'device'."""

    def test_explicit_cpu(self) -> None:
        @devices("cpu")
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 1
        assert marks[0].args[0] == "device"
        assert _param_count(marks[0]) == 1

    def test_explicit_cuda(self) -> None:
        @devices("cuda:0")
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 1

    def test_multiple_devices(self) -> None:
        @devices("cpu", "cuda:0")
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2

    def test_auto_detect_produces_at_least_one(self) -> None:
        """@devices() with no args auto-detects; should yield >= 1 device."""
        @devices()
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) >= 1

    def test_device_id_removes_colon(self) -> None:
        from gpucheck.decorators.devices import _device_id
        assert _device_id("cuda:0") == "cuda0"
        assert _device_id("cuda:1") == "cuda1"
        assert _device_id("cpu") == "cpu"

    def test_unavailable_device_gets_skip_marker(self) -> None:
        """A device that doesn't exist must get pytest.mark.skip."""
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            side_effect=lambda d: d == "cpu",
        ):
            dec = devices("cuda:7", "cpu")

        @dec
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        params = marks[0].args[1]
        # cuda:7 param
        cuda_param = params[0]
        assert any(m.name == "skip" for m in cuda_param.marks)
        # cpu param — no skip
        cpu_param = params[1]
        assert not any(m.name == "skip" for m in cpu_param.marks)

    def test_all_keyword_expands(self) -> None:
        """'all' should expand to detected CUDA devices."""
        @devices("all")
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) >= 1


# ===================================================================
# 4. @dtypes x @shapes — cartesian product
# ===================================================================

class TestDtypesTimesShapes:
    """Stacking @dtypes and @shapes produces their cartesian product."""

    def test_two_marks_exist(self) -> None:
        @dtypes("float16", "float32", "float64")
        @shapes((32, 32), (64, 64), (128, 128), (256, 256))
        def fn(dtype: Any, shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 2
        param_names = {m.args[0] for m in marks}
        assert param_names == {"dtype", "shape"}

    def test_cartesian_product_count_3x4(self) -> None:
        """3 dtypes x 4 shapes = 12 tests."""

        @dtypes("float16", "float32", "float64")
        @shapes((32, 32), (64, 64), (128, 128), (256, 256))
        def fn(dtype: Any, shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        dtype_mark = [m for m in marks if m.args[0] == "dtype"][0]
        shape_mark = [m for m in marks if m.args[0] == "shape"][0]
        dtype_count = _param_count(dtype_mark)
        shape_count = _param_count(shape_mark)
        assert dtype_count == 3
        assert shape_count == 4
        # pytest creates cartesian product: 3 * 4 = 12
        assert dtype_count * shape_count == 12

    def test_order_independent(self) -> None:
        """Order of decorator stacking shouldn't affect the product size."""

        @shapes((32, 32), (64, 64))
        @dtypes("float16", "float32")
        def fn(dtype: Any, shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 2


# ===================================================================
# 5. @dtypes x @shapes x @devices — triple product
# ===================================================================

class TestTripleProduct:
    """Stacking all three decorators produces dtype x shape x device."""

    def test_three_marks_exist(self) -> None:
        @dtypes("float16", "float32")
        @shapes((32, 32), (64, 64))
        @devices("cpu")
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 3
        param_names = {m.args[0] for m in marks}
        assert param_names == {"dtype", "shape", "device"}

    def test_triple_product_count(self) -> None:
        """2 dtypes x 2 shapes x 2 devices = 8 tests."""

        @dtypes("float16", "float32")
        @shapes((32, 32), (64, 64))
        @devices("cpu", "cuda:0")
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        counts = {m.args[0]: _param_count(m) for m in marks}
        assert counts["dtype"] == 2
        assert counts["shape"] == 2
        assert counts["device"] == 2
        assert counts["dtype"] * counts["shape"] * counts["device"] == 8

    def test_triple_product_with_skip(self) -> None:
        """Unavailable device in triple stack should still get skip."""
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            side_effect=lambda d: d == "cpu",
        ):
            dec = devices("cuda:7", "cpu")

        @dtypes("float32")
        @shapes((32, 32))
        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        dev_mark = [m for m in marks if m.args[0] == "device"][0]
        cuda_param = dev_mark.args[1][0]
        assert any(m.name == "skip" for m in cuda_param.marks)


# ===================================================================
# 6. @parametrize_gpu — single-mark cartesian product
# ===================================================================

class TestParametrizeGpu:
    """@parametrize_gpu produces a SINGLE parametrize mark with the full product."""

    def test_single_mark(self) -> None:
        dec = parametrize_gpu(
            dtypes=("float16",),
            shapes=((64, 64),),
            devices=("cuda:0",),
        )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert len(marks) == 1
        assert "dtype" in marks[0].args[0]
        assert "shape" in marks[0].args[0]
        assert "device" in marks[0].args[0]

    def test_product_2x2x1(self) -> None:
        dec = parametrize_gpu(
            dtypes=("float16", "float32"),
            shapes=((128, 128), (256, 256)),
            devices=("cuda:0",),
        )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 4

    def test_product_3x3x2(self) -> None:
        dec = parametrize_gpu(
            dtypes=("float16", "float32", "float64"),
            shapes=((32,), (64, 64), (128, 128)),
            devices=("cpu", "cuda:0"),
        )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 18  # 3*3*2

    def test_combo_ids_format(self) -> None:
        """IDs should be 'dtype-shape-device' format."""
        from gpucheck.decorators.parametrize import _combo_id
        assert _combo_id(torch.float16, (128, 128), "cuda:0") == "float16-128x128-cuda0"
        assert _combo_id(torch.bfloat16, (32,), "cpu") == "bfloat16-32-cpu"

    def test_skip_filter(self) -> None:
        """skip predicate should mark filtered combos."""
        dec = parametrize_gpu(
            dtypes=("float16", "float32"),
            shapes=((64, 64),),
            devices=("cpu",),
            skip=lambda d, s, dev: str(d) == "torch.float16",
        )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        params = marks[0].args[1]
        # One of the two should have skip mark
        skip_count = sum(1 for p in params if any(m.name == "skip" for m in p.marks))
        assert skip_count == 1

    def test_unavailable_device_in_parametrize_gpu(self) -> None:
        """parametrize_gpu should also skip unavailable devices."""
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            side_effect=lambda d: d == "cpu",
        ):
            dec = parametrize_gpu(
                dtypes=("float32",),
                shapes=((64, 64),),
                devices=("cuda:7", "cpu"),
            )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        params = marks[0].args[1]
        # cuda:7 combo should have skip
        cuda_params = [p for p in params if "cuda7" in (p.id or "")]
        assert len(cuda_params) == 1
        assert any(m.name == "skip" for m in cuda_params[0].marks)

    def test_defaults(self) -> None:
        """Default args: dtypes=('float16','float32'), shapes=((128,128),), auto devices."""
        dec = parametrize_gpu()

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        # 2 dtypes * 1 shape * (auto devices >= 1)
        assert _param_count(marks[0]) >= 2

    def test_dtypes_resolved_to_torch_dtype(self) -> None:
        """parametrize_gpu eagerly resolves string dtypes to torch.dtype."""
        dec = parametrize_gpu(
            dtypes=("float32",),
            shapes=((32,),),
            devices=("cpu",),
        )

        @dec
        def fn(dtype: Any, shape: tuple[int, ...], device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        param = marks[0].args[1][0]
        dtype_val = param.values[0]
        assert dtype_val is torch.float32


# ===================================================================
# 7. Predefined groups: FLOAT_DTYPES, EDGE_SHAPES
# ===================================================================

class TestPredefinedGroups:
    """Predefined groups should work as spread args to decorators."""

    def test_float_dtypes_with_decorator(self) -> None:
        @dtypes(*FLOAT_DTYPES)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == len(FLOAT_DTYPES)
        assert _param_count(marks[0]) == 4  # float16, bfloat16, float32, float64

    def test_half_dtypes_with_decorator(self) -> None:
        @dtypes(*HALF_DTYPES)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2

    def test_all_dtypes_with_decorator(self) -> None:
        @dtypes(*ALL_DTYPES)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == len(ALL_DTYPES_NAMES)
        assert _param_count(marks[0]) == 10

    def test_fp8_dtypes_with_decorator(self) -> None:
        @dtypes(*FP8_DTYPES)
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 2

    def test_edge_shapes_with_decorator(self) -> None:
        @shapes(*EDGE_SHAPES)
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == len(EDGE_SHAPES)
        assert _param_count(marks[0]) == 7

    def test_small_shapes_with_decorator(self) -> None:
        @shapes(*SMALL_SHAPES)
        def fn(shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        assert _param_count(marks[0]) == 4

    def test_dtype_group_lazy_resolution(self) -> None:
        """_DtypeGroup should not resolve until iterated."""
        group = _DtypeGroup(("float32", "float16"))
        assert group._resolved is None
        items = list(group)
        assert group._resolved is not None
        assert len(items) == 2
        assert items[0] is torch.float32
        assert items[1] is torch.float16

    def test_dtype_group_repr(self) -> None:
        group = _DtypeGroup(("float32",))
        assert "DtypeGroup" in repr(group)

    def test_float_dtypes_x_edge_shapes_product(self) -> None:
        """FLOAT_DTYPES(4) x EDGE_SHAPES(7) = 28 combos."""
        @dtypes(*FLOAT_DTYPES)
        @shapes(*EDGE_SHAPES)
        def fn(dtype: Any, shape: tuple[int, ...]) -> None: ...

        marks = _get_parametrize_marks(fn)
        dtype_mark = [m for m in marks if m.args[0] == "dtype"][0]
        shape_mark = [m for m in marks if m.args[0] == "shape"][0]
        assert _param_count(dtype_mark) * _param_count(shape_mark) == 28


# ===================================================================
# 8. String dtypes work without torch import at decoration time
# ===================================================================

class TestStringDtypesLazy:
    """Decoration-time should NOT require torch to be imported."""

    def test_string_dtype_values_are_strings(self) -> None:
        """The parametrize values should be raw strings, not torch.dtype."""
        @dtypes("float16", "float32")
        def fn(dtype: Any) -> None: ...

        marks = _get_parametrize_marks(fn)
        for param in marks[0].args[1]:
            assert isinstance(param.values[0], str)

    def test_resolve_dtype_converts_string(self) -> None:
        assert _resolve_dtype("float32") is torch.float32
        assert _resolve_dtype("bfloat16") is torch.bfloat16
        assert _resolve_dtype("int8") is torch.int8

    def test_resolve_dtype_passthrough_torch_dtype(self) -> None:
        assert _resolve_dtype(torch.float16) is torch.float16

    def test_dtype_id_from_string(self) -> None:
        assert _dtype_id("float32") == "float32"
        assert _dtype_id("bfloat16") == "bfloat16"

    def test_dtype_id_from_torch(self) -> None:
        assert _dtype_id(torch.float32) == "float32"
        assert _dtype_id(torch.int64) == "int64"


# ===================================================================
# 9. Unavailable devices get skip markers
# ===================================================================

class TestUnavailableDeviceSkip:
    """Non-existent devices must be marked with pytest.mark.skip."""

    def test_nonexistent_cuda_device(self) -> None:
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            return_value=False,
        ):
            dec = devices("cuda:99")

        @dec
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        param = marks[0].args[1][0]
        assert any(m.name == "skip" for m in param.marks)

    def test_skip_reason_mentions_device(self) -> None:
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            return_value=False,
        ):
            dec = devices("cuda:99")

        @dec
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        param = marks[0].args[1][0]
        skip_marks = [m for m in param.marks if m.name == "skip"]
        assert "cuda:99" in skip_marks[0].kwargs.get("reason", "")

    def test_available_device_no_skip(self) -> None:
        """cpu should always be available and have no skip mark."""
        @devices("cpu")
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        param = marks[0].args[1][0]
        assert not any(m.name == "skip" for m in param.marks)

    def test_mixed_available_unavailable(self) -> None:
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(
            _mod, "_is_device_available",
            side_effect=lambda d: d in ("cpu", "cuda:0"),
        ):
            dec = devices("cpu", "cuda:0", "cuda:5", "cuda:6")

        @dec
        def fn(device: str) -> None: ...

        marks = _get_parametrize_marks(fn)
        params = marks[0].args[1]
        assert len(params) == 4
        skipped = [p for p in params if any(m.name == "skip" for m in p.marks)]
        not_skipped = [p for p in params if not any(m.name == "skip" for m in p.marks)]
        assert len(skipped) == 2   # cuda:5, cuda:6
        assert len(not_skipped) == 2  # cpu, cuda:0


# ===================================================================
# 10. Live pytest run: verify test IDs are clean
# ===================================================================

# These are actual parametrized tests that pytest collects and runs.
# The test names in `pytest -v` output verify ID cleanliness.

@dtypes("float16", "float32")
def test_live_dtypes_only(dtype: Any) -> None:
    """Collected as test_live_dtypes_only[float16] and [float32]."""
    assert isinstance(dtype, str)


@shapes((32, 32), (7, 13))
def test_live_shapes_only(shape: tuple[int, ...]) -> None:
    """Collected as test_live_shapes_only[32x32] and [7x13]."""
    assert isinstance(shape, tuple)
    assert all(isinstance(d, int) for d in shape)


@devices("cpu")
def test_live_devices_cpu(device: str) -> None:
    """Collected as test_live_devices_cpu[cpu]."""
    assert device == "cpu"


@dtypes("float16", "float32")
@shapes((32, 32), (64, 64))
def test_live_dtypes_x_shapes(dtype: Any, shape: tuple[int, ...]) -> None:
    """4 combos: [float16-32x32], [float16-64x64], [float32-32x32], [float32-64x64]."""
    assert isinstance(dtype, str)
    assert isinstance(shape, tuple)


@dtypes("float16", "float32")
@shapes((32, 32), (64, 64))
@devices("cpu")
def test_live_triple_product(dtype: Any, shape: tuple[int, ...], device: str) -> None:
    """4 combos (1 device): verify all three params arrive."""
    assert isinstance(dtype, str)
    assert isinstance(shape, tuple)
    assert device == "cpu"


@pytest.mark.parametrize("dtype,shape,device", [
    pytest.param(torch.float16, (32, 32), "cpu", id="float16-32x32-cpu"),
    pytest.param(torch.float32, (64, 64), "cpu", id="float32-64x64-cpu"),
])
def test_live_parametrize_gpu_equivalent(
    dtype: Any, shape: tuple[int, ...], device: str
) -> None:
    """Manual parametrize_gpu-style test to verify ID format."""
    assert hasattr(dtype, "is_floating_point")
    assert isinstance(shape, tuple)
    assert device == "cpu"
