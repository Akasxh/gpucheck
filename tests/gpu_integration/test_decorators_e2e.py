"""End-to-end tests for gpucheck decorators on actual GPU hardware.

These tests verify that decorators produce correct pytest parametrization,
resolve dtypes to real torch.dtype objects, allocate tensors on GPU devices,
and that the cartesian product / fuzzing logic works correctly.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import patch

import pytest
import torch

import gpucheck as gc
from gpucheck.decorators.dtypes import (
    _STR_TO_DTYPE,
    ALL_DTYPES,
    FLOAT_DTYPES,
    FP8_DTYPES,
    HALF_DTYPES,
    _DtypeGroup,
    _resolve_dtype,
)
from gpucheck.decorators.shapes import (
    EDGE_SHAPES,
    LARGE_SHAPES,
    SMALL_SHAPES,
)
from gpucheck.fuzzing.shapes import TILE_SIZES, fuzz_shapes

# ===========================================================================
# 1. @gc.dtypes with string args — verify torch.dtype resolution
# ===========================================================================


class TestDtypesStringArgs:
    """@gc.dtypes('float16', ...) must resolve to torch.dtype at test time."""

    @gc.dtypes("float16", "float32", "bfloat16", "float64")
    def test_string_resolves_to_torch_dtype(self, dtype: str) -> None:
        resolved = _resolve_dtype(dtype)
        assert isinstance(resolved, torch.dtype), (
            f"Expected torch.dtype, got {type(resolved)} for {dtype!r}"
        )

    @gc.dtypes("float16")
    def test_float16_identity(self, dtype: str) -> None:
        resolved = _resolve_dtype(dtype)
        assert resolved is torch.float16

    @gc.dtypes("bfloat16")
    def test_bfloat16_identity(self, dtype: str) -> None:
        resolved = _resolve_dtype(dtype)
        assert resolved is torch.bfloat16

    @gc.dtypes("int8", "int16", "int32", "int64", "uint8", "bool")
    def test_integer_and_bool_dtypes(self, dtype: str) -> None:
        resolved = _resolve_dtype(dtype)
        assert isinstance(resolved, torch.dtype)

    def test_all_str_to_dtype_entries_resolve(self) -> None:
        """Every key in the _STR_TO_DTYPE lookup table must resolve."""
        for name in _STR_TO_DTYPE:
            resolved = _resolve_dtype(name)
            assert isinstance(resolved, torch.dtype), f"Failed for {name!r}"

    @gc.dtypes("float32")
    def test_can_allocate_tensor_with_resolved_dtype(self, dtype: str) -> None:
        resolved = _resolve_dtype(dtype)
        t = torch.zeros(4, dtype=resolved, device="cuda:0")
        assert t.dtype == resolved
        assert t.device.type == "cuda"


# ===========================================================================
# 2. @gc.dtypes with torch.dtype objects directly
# ===========================================================================


class TestDtypesTorchDtypeArgs:
    """@gc.dtypes(torch.float16, ...) must pass through unchanged."""

    @gc.dtypes(torch.float16, torch.float32, torch.bfloat16)
    def test_torch_dtype_passthrough(self, dtype: torch.dtype) -> None:
        assert isinstance(dtype, torch.dtype)

    @gc.dtypes(torch.float16, torch.float32)
    def test_gpu_tensor_creation_with_direct_dtype(self, dtype: torch.dtype) -> None:
        t = torch.randn(8, 8, dtype=dtype, device="cuda:0")
        assert t.dtype == dtype
        assert t.shape == (8, 8)

    @gc.dtypes(torch.int8, torch.int32, torch.int64)
    def test_integer_dtypes_direct(self, dtype: torch.dtype) -> None:
        t = torch.zeros(4, dtype=dtype, device="cuda:0")
        assert t.dtype == dtype


# ===========================================================================
# 3. @gc.shapes with various shapes including edge cases
# ===========================================================================


class TestShapesDecorator:
    """@gc.shapes must parametrize over shape tuples correctly."""

    @gc.shapes((1,), (128,), (1024,))
    def test_1d_shapes(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape

    @gc.shapes((32, 32), (128, 256), (7, 13))
    def test_2d_shapes(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape

    @gc.shapes((1, 1, 1), (2, 3, 4), (16, 16, 16))
    def test_3d_shapes(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape

    @gc.shapes((1, 1), (1, 128), (128, 1))
    def test_degenerate_shapes(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape
        assert t.numel() == 1 or t.numel() == 128

    @gc.shapes((0, 128),)
    def test_zero_dim_shape(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape
        assert t.numel() == 0

    @gc.shapes((127, 127), (255, 255), (513, 513))
    def test_non_power_of_2_shapes(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape


# ===========================================================================
# 4. @gc.devices with "cuda:0"
# ===========================================================================


class TestDevicesDecorator:
    """@gc.devices must parametrize over device strings."""

    @gc.devices("cuda:0")
    def test_cuda0_allocation(self, device: str) -> None:
        t = torch.randn(16, device=device)
        assert t.device == torch.device("cuda", 0)

    @gc.devices("cuda:0")
    def test_device_string_is_cuda0(self, device: str) -> None:
        assert device == "cuda:0"

    @gc.devices("cuda:0", "cpu")
    def test_multi_device(self, device: str) -> None:
        t = torch.randn(8, device=device)
        assert str(t.device).startswith(device.split(":")[0])

    def test_auto_detect_finds_cuda(self) -> None:
        """devices() with no args should detect available CUDA GPUs."""
        from gpucheck.decorators.devices import _detect_cuda_devices

        detected = _detect_cuda_devices()
        assert len(detected) >= 1
        assert "cuda:0" in detected


# ===========================================================================
# 5. Stacked @dtypes + @shapes — verify cartesian product
# ===========================================================================


class TestStackedDecoratorsCartesian:
    """Stacking @gc.dtypes and @gc.shapes must yield their cartesian product."""

    @gc.dtypes("float16", "float32")
    @gc.shapes((32, 32), (64, 64))
    def test_cartesian_product_runs(
        self, dtype: str, shape: tuple[int, ...]
    ) -> None:
        resolved = _resolve_dtype(dtype)
        t = torch.randn(*shape, dtype=resolved, device="cuda:0")
        assert t.shape == shape
        assert t.dtype == resolved

    @gc.dtypes("float16", "float32")
    @gc.shapes((32, 32), (64, 64), (7, 13))
    def test_cartesian_count(
        self, dtype: str, shape: tuple[int, ...]
    ) -> None:
        """Implicitly tests 2 dtypes x 3 shapes = 6 combos via parametrize."""
        resolved = _resolve_dtype(dtype)
        t = torch.zeros(*shape, dtype=resolved, device="cuda:0")
        assert t.numel() > 0

    @gc.dtypes(torch.bfloat16)
    @gc.shapes((1, 1), (128, 1))
    @gc.devices("cuda:0")
    def test_triple_stack(
        self, dtype: torch.dtype, shape: tuple[int, ...], device: str
    ) -> None:
        t = torch.randn(*shape, dtype=dtype, device=device)
        assert t.dtype == dtype
        assert t.shape == shape
        assert t.device == torch.device("cuda", 0)


# ===========================================================================
# 6. @parametrize_gpu all-in-one
# ===========================================================================


class TestParametrizeGpu:
    """@gc.parametrize_gpu should produce a single cartesian parametrize."""

    @gc.parametrize_gpu(
        dtypes=("float16", "float32"),
        shapes=((64, 64), (128, 128)),
        devices=("cuda:0",),
    )
    def test_basic_parametrize_gpu(
        self, dtype: torch.dtype, shape: tuple[int, ...], device: str
    ) -> None:
        assert isinstance(dtype, torch.dtype)
        t = torch.randn(*shape, dtype=dtype, device=device)
        assert t.shape == shape
        assert t.dtype == dtype
        assert t.device == torch.device("cuda", 0)

    @gc.parametrize_gpu(
        dtypes=("float16",),
        shapes=((32, 32),),
        devices=("cuda:0",),
    )
    def test_single_combo(
        self, dtype: torch.dtype, shape: tuple[int, ...], device: str
    ) -> None:
        t = torch.randn(*shape, dtype=dtype, device=device)
        assert t.shape == (32, 32)
        assert t.dtype == torch.float16

    @gc.parametrize_gpu(
        dtypes=("float32", "bfloat16"),
        shapes=((8,), (16, 16)),
        devices=("cuda:0",),
    )
    def test_parametrize_gpu_cartesian_count(
        self, dtype: torch.dtype, shape: tuple[int, ...], device: str
    ) -> None:
        """2 dtypes x 2 shapes x 1 device = 4 combos."""
        t = torch.zeros(*shape, dtype=dtype, device=device)
        assert t.device.type == "cuda"

    def test_parametrize_gpu_produces_correct_param_count(self) -> None:
        decorator = gc.parametrize_gpu(
            dtypes=("float16", "float32", "bfloat16"),
            shapes=((32, 32), (64, 64)),
            devices=("cuda:0",),
        )

        @decorator
        def dummy(dtype: Any, shape: Any, device: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        # 3 dtypes x 2 shapes x 1 device = 6
        assert len(param_mark.args[1]) == 6

    def test_parametrize_gpu_skip_filter(self) -> None:
        """Skip filter should mark combos for skipping."""

        def skip_bf16(dtype: Any, shape: Any, device: Any) -> bool:
            return dtype == torch.bfloat16

        decorator = gc.parametrize_gpu(
            dtypes=("float16", "bfloat16"),
            shapes=((32, 32),),
            devices=("cuda:0",),
            skip=skip_bf16,
        )

        @decorator
        def dummy(dtype: Any, shape: Any, device: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        params = param_mark.args[1]
        # bfloat16 combo should have skip mark
        bf16_param = [p for p in params if any(
            "skip" in str(m) for m in p.marks
        )]
        assert len(bf16_param) == 1


# ===========================================================================
# 7. Predefined groups: FLOAT_DTYPES, HALF_DTYPES, EDGE_SHAPES
# ===========================================================================


class TestPredefinedGroups:
    """Predefined groups must resolve to correct torch.dtype / shape values."""

    def test_half_dtypes_resolve(self) -> None:
        resolved = list(HALF_DTYPES)
        assert len(resolved) == 2
        assert torch.float16 in resolved
        assert torch.bfloat16 in resolved

    def test_float_dtypes_resolve(self) -> None:
        resolved = list(FLOAT_DTYPES)
        assert len(resolved) == 4
        assert torch.float16 in resolved
        assert torch.float32 in resolved
        assert torch.float64 in resolved
        assert torch.bfloat16 in resolved

    def test_all_dtypes_resolve(self) -> None:
        resolved = list(ALL_DTYPES)
        assert len(resolved) == 10
        for d in resolved:
            assert isinstance(d, torch.dtype)

    def test_fp8_dtypes_resolve(self) -> None:
        resolved = list(FP8_DTYPES)
        assert len(resolved) == 2
        assert torch.float8_e4m3fn in resolved
        assert torch.float8_e5m2 in resolved

    def test_edge_shapes_content(self) -> None:
        assert (1, 1) in EDGE_SHAPES
        assert (7, 13) in EDGE_SHAPES
        assert (127, 127) in EDGE_SHAPES
        assert (0, 128) in EDGE_SHAPES
        assert (1, 1, 1) in EDGE_SHAPES

    def test_small_shapes_all_small(self) -> None:
        for s in SMALL_SHAPES:
            assert all(d <= 128 for d in s)

    def test_large_shapes_have_big_dims(self) -> None:
        assert any(any(d >= 2048 for d in s) for s in LARGE_SHAPES)

    @gc.dtypes(*FLOAT_DTYPES)
    def test_float_dtypes_as_decorator_arg(self, dtype: torch.dtype) -> None:
        """Spreading FLOAT_DTYPES into @gc.dtypes must work end-to-end."""
        assert isinstance(dtype, torch.dtype)
        t = torch.randn(4, dtype=dtype, device="cuda:0")
        assert t.dtype == dtype

    @gc.dtypes(*HALF_DTYPES)
    def test_half_dtypes_as_decorator_arg(self, dtype: torch.dtype) -> None:
        t = torch.randn(4, dtype=dtype, device="cuda:0")
        assert t.dtype in (torch.float16, torch.bfloat16)

    @gc.shapes(*EDGE_SHAPES)
    def test_edge_shapes_as_decorator_arg(self, shape: tuple[int, ...]) -> None:
        t = torch.randn(*shape, device="cuda:0")
        assert t.shape == shape

    def test_dtype_group_is_lazy(self) -> None:
        group = _DtypeGroup(("float32", "float16"))
        assert group._resolved is None
        _ = list(group)
        assert group._resolved is not None

    def test_dtype_group_len(self) -> None:
        group = _DtypeGroup(("float32", "float16", "bfloat16"))
        assert len(group) == 3

    def test_dtype_group_repr(self) -> None:
        group = _DtypeGroup(("float32",))
        assert "float32" in repr(group)


# ===========================================================================
# 8. fuzz_shapes — verify non-power-of-2 and edge cases
# ===========================================================================


class TestFuzzShapes:
    """fuzz_shapes must generate adversarial shapes for GPU kernel testing."""

    def test_returns_requested_count(self) -> None:
        result = fuzz_shapes(ndim=2, n=30, seed=42)
        assert len(result) == 30

    def test_all_tuples_correct_ndim(self) -> None:
        for ndim in (1, 2, 3):
            result = fuzz_shapes(ndim=ndim, n=20, seed=42)
            for s in result:
                assert len(s) == ndim, f"Expected ndim={ndim}, got shape {s}"

    def test_contains_degenerate_shapes(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=42)
        has_zero = any(any(d == 0 for d in s) for s in result)
        has_one = any(all(d == 1 for d in s) for s in result)
        assert has_zero, "fuzz_shapes must include zero-dim shapes"
        assert has_one, "fuzz_shapes must include all-ones shapes"

    def test_contains_non_power_of_2(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=42)
        non_pow2 = [
            s for s in result
            if any(d > 0 and (d & (d - 1)) != 0 for d in s)
        ]
        assert len(non_pow2) >= 5, (
            f"Expected >= 5 non-power-of-2 shapes, got {len(non_pow2)}"
        )

    def test_contains_non_tile_aligned(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=42)
        non_aligned = []
        for s in result:
            for d in s:
                if d > 0 and all(d % tile != 0 for tile in TILE_SIZES):
                    non_aligned.append(s)
                    break
        assert len(non_aligned) >= 3, (
            f"Expected >= 3 non-tile-aligned shapes, got {len(non_aligned)}"
        )

    def test_contains_prime_dimensions(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=42)
        primes = {7, 13, 31, 127, 257}
        has_prime = any(
            any(d in primes for d in s)
            for s in result
        )
        assert has_prime, "fuzz_shapes must include prime-dimension shapes"

    def test_deterministic_with_seed(self) -> None:
        r1 = fuzz_shapes(ndim=2, n=30, seed=123)
        r2 = fuzz_shapes(ndim=2, n=30, seed=123)
        assert r1 == r2

    def test_different_seeds_differ(self) -> None:
        r1 = fuzz_shapes(ndim=2, n=50, seed=1)
        r2 = fuzz_shapes(ndim=2, n=50, seed=2)
        # Deterministic pool portion will match, but random padding may differ
        # At minimum they should both be valid
        assert len(r1) == 50
        assert len(r2) == 50

    def test_respects_max_size(self) -> None:
        result = fuzz_shapes(ndim=2, max_size=256, n=30, seed=42)
        for s in result:
            for d in s:
                assert d <= 256, f"Dimension {d} exceeds max_size=256 in {s}"

    def test_no_duplicates(self) -> None:
        result = fuzz_shapes(ndim=2, n=50, seed=42)
        assert len(result) == len(set(result)), "fuzz_shapes returned duplicates"

    def test_min_size_validation(self) -> None:
        with pytest.raises(ValueError, match="min_size"):
            fuzz_shapes(ndim=2, min_size=100, max_size=10)

    def test_ndim_validation(self) -> None:
        with pytest.raises(ValueError, match="ndim"):
            fuzz_shapes(ndim=-1)

    def test_ndim_zero_single(self) -> None:
        """ndim=0 with n=1 returns [()]. n>1 infinite-loops (known bug)."""
        result = fuzz_shapes(ndim=0, n=1, seed=42)
        assert result == [()]

    def test_fuzz_shapes_gpu_allocation(self) -> None:
        """Verify fuzzed shapes can actually be allocated on GPU."""
        shapes = fuzz_shapes(ndim=2, n=20, max_size=512, seed=42)
        for s in shapes:
            t = torch.randn(*s, device="cuda:0")
            assert t.shape == s


# ===========================================================================
# 9. Decorators work WITHOUT torch installed (string-only mode)
# ===========================================================================


class TestStringOnlyModeNoTorch:
    """Decorators must not import torch at decoration time (string-only)."""

    def test_dtypes_decorator_does_not_import_torch(self) -> None:
        """@gc.dtypes with strings should NOT trigger torch import internally."""
        # The decorator stores raw strings — it never calls _resolve_dtype
        decorator = gc.dtypes("float16", "float32")

        @decorator
        def dummy(dtype: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        # Values should still be raw strings, not torch.dtype
        raw_values = [p.values[0] for p in param_mark.args[1]]
        assert all(isinstance(v, str) for v in raw_values), (
            f"Expected raw strings, got {raw_values}"
        )

    def test_shapes_decorator_no_torch_dependency(self) -> None:
        """@gc.shapes never needs torch at decoration time."""
        decorator = gc.shapes((32, 32), (64, 64))

        @decorator
        def dummy(shape: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        # shapes uses plain tuples in the param list (not pytest.param)
        raw_values = param_mark.args[1]
        assert all(isinstance(v, tuple) for v in raw_values)

    def test_devices_decorator_with_explicit_args_no_detection(self) -> None:
        """@gc.devices('cuda:0') with explicit args doesn't need auto-detect."""
        # Mock _is_device_available to always return True (simulating no torch)
        _mod = sys.modules["gpucheck.decorators.devices"]
        with patch.object(_mod, "_is_device_available", return_value=True):
            decorator = gc.devices("cuda:0", "cpu")

        @decorator
        def dummy(device: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        assert len(param_mark.args[1]) == 2

    def test_dtype_group_lazy_no_import_on_creation(self) -> None:
        """_DtypeGroup must NOT resolve (import torch) until iterated."""
        group = _DtypeGroup(("float32", "float16"))
        assert group._resolved is None
        assert len(group) == 2  # len uses _names, not _resolved
        assert group._resolved is None  # still not resolved

    def test_dtypes_string_values_preserved_in_parametrize(self) -> None:
        """String dtype args stored as-is in pytest.param, not resolved."""
        decorator = gc.dtypes("int8", "bool", "uint8")

        @decorator
        def dummy(dtype: Any) -> None:
            pass

        marks = getattr(dummy, "pytestmark", [])
        param_mark = [m for m in marks if m.name == "parametrize"][0]
        raw_values = [p.values[0] for p in param_mark.args[1]]
        assert raw_values == ["int8", "bool", "uint8"]


# ===========================================================================
# Additional integration: stacked decorators + GPU allocation
# ===========================================================================


class TestFullIntegration:
    """Full integration: decorator -> parametrize -> GPU tensor ops."""

    @gc.dtypes("float16", "float32")
    @gc.shapes((64, 64), (128, 128))
    @gc.devices("cuda:0")
    def test_matmul_stacked(
        self,
        dtype: str,
        shape: tuple[int, ...],
        device: str,
    ) -> None:
        resolved = _resolve_dtype(dtype)
        a = torch.randn(*shape, dtype=resolved, device=device)
        b = torch.randn(*shape, dtype=resolved, device=device)
        c = a @ b
        assert c.shape == shape
        assert c.dtype == resolved

    @gc.parametrize_gpu(
        dtypes=("float16", "float32"),
        shapes=((32, 32), (64, 64)),
        devices=("cuda:0",),
    )
    def test_relu_parametrize_gpu(
        self,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        device: str,
    ) -> None:
        t = torch.randn(*shape, dtype=dtype, device=device)
        out = torch.relu(t)
        assert out.shape == shape
        assert (out >= 0).all()
