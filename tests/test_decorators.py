"""Tests for the decorator module: dtypes, shapes, devices, parametrize_gpu."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from gpucheck.decorators.dtypes import (
    FLOAT_DTYPES_NAMES,
    HALF_DTYPES_NAMES,
    _DtypeGroup,
    _dtype_id,
    dtypes,
)
from gpucheck.decorators.shapes import (
    EDGE_SHAPES,
    LARGE_SHAPES,
    MEDIUM_SHAPES,
    SMALL_SHAPES,
    shapes,
)
from gpucheck.decorators.devices import devices
from gpucheck.decorators.parametrize import parametrize_gpu


# ---------------------------------------------------------------------------
# dtypes decorator
# ---------------------------------------------------------------------------


class TestDtypesDecoratorParametrizes:
    """@dtypes should produce pytest.mark.parametrize over dtype param."""

    def test_creates_parametrize_marker(self) -> None:
        import torch

        decorator = dtypes("float32", "float16")
        # The decorator is a pytest.mark.parametrize instance
        assert hasattr(decorator, "args") or hasattr(decorator, "mark")

    def test_parametrize_applied_to_function(self) -> None:
        import torch

        @dtypes("float32", "float16")
        def dummy(dtype: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        assert len(marks) >= 1
        # Find the parametrize mark
        param_mark = [m for m in marks if m.name == "parametrize"]
        assert len(param_mark) == 1
        assert param_mark[0].args[0] == "dtype"
        # Should have 2 parameter values
        assert len(param_mark[0].args[1]) == 2

    def test_dtype_ids_are_clean(self) -> None:
        import torch

        assert _dtype_id(torch.float32) == "float32"
        assert _dtype_id(torch.float16) == "float16"
        assert _dtype_id("float32") == "float32"


# ---------------------------------------------------------------------------
# shapes decorator
# ---------------------------------------------------------------------------


class TestShapesDecoratorParametrizes:
    """@shapes should produce pytest.mark.parametrize over shape param."""

    def test_creates_parametrize_marker(self) -> None:
        decorator = shapes((128, 128), (256, 256))
        assert hasattr(decorator, "args") or hasattr(decorator, "mark")

    def test_parametrize_applied_to_function(self) -> None:
        @shapes((32, 32), (64, 64), (7, 13))
        def dummy(shape: tuple[int, ...]) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"]
        assert len(param_mark) == 1
        assert param_mark[0].args[0] == "shape"
        assert len(param_mark[0].args[1]) == 3


# ---------------------------------------------------------------------------
# devices decorator
# ---------------------------------------------------------------------------


class TestDevicesDecoratorParametrizes:
    """@devices should produce pytest.mark.parametrize over device param."""

    def test_explicit_devices(self) -> None:
        decorator = devices("cuda:0", "cpu")
        assert hasattr(decorator, "args") or hasattr(decorator, "mark")

    def test_parametrize_applied_to_function(self) -> None:
        @devices("cuda:0", "cpu")
        def dummy(device: str) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"]
        assert len(param_mark) == 1
        assert param_mark[0].args[0] == "device"
        # Should have 2 params (cuda:0 and cpu)
        assert len(param_mark[0].args[1]) == 2

    def test_unavailable_device_gets_skip_mark(self) -> None:
        """Devices not available should get pytest.mark.skip."""
        with patch(
            "gpucheck.decorators.devices._is_device_available",
            side_effect=lambda d: d == "cpu",
        ):
            decorator = devices("cuda:0", "cpu")

        @decorator
        def dummy(device: str) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        params = param_mark.args[1]
        # cuda:0 should have a skip mark
        cuda_param = params[0]
        assert any(m.name == "skip" for m in cuda_param.marks)


# ---------------------------------------------------------------------------
# Stacked decorators → cartesian product
# ---------------------------------------------------------------------------


class TestStackedDecoratorsCartesianProduct:
    """Stacking @dtypes and @shapes should produce their cartesian product."""

    def test_stacked_marks(self) -> None:
        import torch

        @dtypes("float32", "float16")
        @shapes((32, 32), (64, 64))
        def dummy(dtype: Any, shape: tuple[int, ...]) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_marks = [m for m in marks if m.name == "parametrize"]
        # Should have 2 separate parametrize marks
        assert len(param_marks) == 2
        param_names = {m.args[0] for m in param_marks}
        assert param_names == {"dtype", "shape"}


# ---------------------------------------------------------------------------
# parametrize_gpu
# ---------------------------------------------------------------------------


class TestParametrizeGpuAllInOne:
    """@parametrize_gpu should produce a single parametrize with cartesian product."""

    def test_creates_combined_parametrize(self) -> None:
        import torch

        decorator = parametrize_gpu(
            dtypes=("float32",),
            shapes=((128, 128),),
            devices=("cuda:0",),
        )

        @decorator
        def dummy(dtype: Any, shape: tuple[int, ...], device: str) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"]
        assert len(param_mark) == 1
        assert "dtype" in param_mark[0].args[0]
        assert "shape" in param_mark[0].args[0]
        assert "device" in param_mark[0].args[0]

    def test_cartesian_product_count(self) -> None:
        import torch

        decorator = parametrize_gpu(
            dtypes=("float32", "float16"),
            shapes=((128, 128), (256, 256)),
            devices=("cuda:0",),
        )

        @decorator
        def dummy(dtype: Any, shape: tuple[int, ...], device: str) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        # 2 dtypes x 2 shapes x 1 device = 4 combos
        assert len(param_mark.args[1]) == 4


# ---------------------------------------------------------------------------
# Predefined groups
# ---------------------------------------------------------------------------


class TestPredefinedDtypeGroups:
    """Predefined dtype groups should have expected members."""

    def test_half_dtypes_names(self) -> None:
        assert "float16" in HALF_DTYPES_NAMES
        assert "bfloat16" in HALF_DTYPES_NAMES
        assert len(HALF_DTYPES_NAMES) == 2

    def test_float_dtypes_names(self) -> None:
        assert "float32" in FLOAT_DTYPES_NAMES
        assert "float64" in FLOAT_DTYPES_NAMES
        assert len(FLOAT_DTYPES_NAMES) == 4

    def test_dtype_group_is_lazy(self) -> None:
        group = _DtypeGroup(("float32",))
        # _resolved should be None before iteration
        assert group._resolved is None
        assert len(group) == 1


class TestPredefinedShapeGroups:
    """Predefined shape groups should contain valid tuples."""

    def test_small_shapes(self) -> None:
        assert len(SMALL_SHAPES) > 0
        for s in SMALL_SHAPES:
            assert isinstance(s, tuple)
            assert all(isinstance(d, int) and d >= 0 for d in s)

    def test_edge_shapes_contain_adversarial(self) -> None:
        # Should include (1,1), primes, zero-dims
        dims_present = {s for s in EDGE_SHAPES}
        assert (1, 1) in dims_present
        assert (7, 13) in dims_present
        assert (0, 128) in dims_present

    def test_medium_shapes(self) -> None:
        assert len(MEDIUM_SHAPES) > 0
        for s in MEDIUM_SHAPES:
            assert all(d > 0 for d in s)

    def test_large_shapes(self) -> None:
        assert len(LARGE_SHAPES) > 0
        # At least one should have dim >= 2048
        assert any(any(d >= 2048 for d in s) for s in LARGE_SHAPES)
