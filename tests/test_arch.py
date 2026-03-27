"""Tests for architecture detection and capability decorators."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from gpucheck.arch.compatibility import (
    SM_ARCH_MAP,
    _cc_to_sm_tag,
    check_compatibility,
    require_arch,
    require_capability,
)
from gpucheck.arch.detection import (
    _TENSOR_CORE_GEN,
    SM_TO_ARCH,
    GPUInfo,
    _resolve_arch,
    _tensor_core_gen,
)

# ---------------------------------------------------------------------------
# GPUInfo dataclass
# ---------------------------------------------------------------------------


class TestGPUInfoDataclass:
    """GPUInfo should be a frozen dataclass with expected fields."""

    def test_create_instance(self) -> None:
        info = GPUInfo(
            device_id=0,
            name="Test GPU",
            compute_capability=(8, 0),
            architecture="Ampere",
            memory_total_mb=40960,
            memory_free_mb=38000,
            driver_version="535.0",
            cuda_version="12.2",
            supports_fp16=True,
            supports_bf16=True,
            supports_fp8=False,
            supports_tf32=True,
            tensor_core_generation=3,
            max_shared_memory_per_block=164 * 1024,
        )
        assert info.device_id == 0
        assert info.name == "Test GPU"
        assert info.compute_capability == (8, 0)
        assert info.architecture == "Ampere"

    def test_is_frozen(self) -> None:
        info = GPUInfo(
            device_id=0, name="X", compute_capability=(8, 0),
            architecture="Ampere", memory_total_mb=0, memory_free_mb=0,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=True, supports_fp8=False, supports_tf32=True,
            tensor_core_generation=3, max_shared_memory_per_block=0,
        )
        with pytest.raises(AttributeError):
            info.name = "Y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SM → architecture mapping
# ---------------------------------------------------------------------------


class TestSmToArchitectureMapping:
    """SM_TO_ARCH should map known compute capabilities correctly."""

    @pytest.mark.parametrize(
        "cc,expected_arch",
        [
            ((7, 0), "Volta"),
            ((7, 5), "Turing"),
            ((8, 0), "Ampere"),
            ((8, 6), "Ampere"),
            ((8, 9), "Ada"),
            ((9, 0), "Hopper"),
        ],
    )
    def test_known_mappings(self, cc: tuple[int, int], expected_arch: str) -> None:
        assert SM_TO_ARCH[cc] == expected_arch

    def test_resolve_arch_fallback(self) -> None:
        # Unknown minor version but known major should still resolve
        arch = _resolve_arch((8, 5))
        # Should resolve to something in the sm8x family
        assert arch in ("Ampere", "Ada", "Unknown")

    def test_resolve_arch_pre_volta(self) -> None:
        arch = _resolve_arch((5, 0))
        assert arch == "Unknown"


# ---------------------------------------------------------------------------
# require_arch decorator
# ---------------------------------------------------------------------------


class TestRequireArchDecorator:
    """@require_arch should skip if GPU architecture doesn't match."""

    def test_matching_arch_runs(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="A100", compute_capability=(8, 0),
            architecture="Ampere", memory_total_mb=40960, memory_free_mb=38000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=True, supports_fp8=False, supports_tf32=True,
            tensor_core_generation=3, max_shared_memory_per_block=164 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Ampere")
            def my_test() -> str:
                return "ran"

            assert my_test() == "ran"

    def test_mismatched_arch_skips(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="V100", compute_capability=(7, 0),
            architecture="Volta", memory_total_mb=16384, memory_free_mb=15000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=1, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Hopper")
            def my_test() -> str:
                return "ran"

            with pytest.raises(pytest.skip.Exception):
                my_test()

    def test_no_gpu_skips(self) -> None:
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=None,
        ):
            @require_arch("Ampere")
            def my_test() -> str:
                return "ran"

            with pytest.raises(pytest.skip.Exception):
                my_test()


# ---------------------------------------------------------------------------
# require_capability decorator
# ---------------------------------------------------------------------------


class TestRequireCapabilityDecorator:
    """@require_capability should skip if CC is below threshold."""

    def test_sufficient_capability_runs(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="A100", compute_capability=(8, 0),
            architecture="Ampere", memory_total_mb=40960, memory_free_mb=38000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=True, supports_fp8=False, supports_tf32=True,
            tensor_core_generation=3, max_shared_memory_per_block=164 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_capability(7, 0)
            def my_test() -> str:
                return "ran"

            assert my_test() == "ran"

    def test_insufficient_capability_skips(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="V100", compute_capability=(7, 0),
            architecture="Volta", memory_total_mb=16384, memory_free_mb=15000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=1, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_capability(9, 0)
            def my_test() -> str:
                return "ran"

            with pytest.raises(pytest.skip.Exception):
                my_test()


# ---------------------------------------------------------------------------
# Tensor core support mapping
# ---------------------------------------------------------------------------


class TestTensorCoreSupportMapping:
    """Tensor core generation should be correct for known architectures."""

    @pytest.mark.parametrize(
        "cc,expected_gen",
        [
            ((7, 0), 1),  # Volta
            ((7, 5), 2),  # Turing
            ((8, 0), 3),  # Ampere
            ((8, 9), 4),  # Ada
            ((9, 0), 4),  # Hopper
        ],
    )
    def test_known_tensor_core_generations(
        self, cc: tuple[int, int], expected_gen: int
    ) -> None:
        assert _TENSOR_CORE_GEN[cc] == expected_gen

    def test_pre_volta_no_tensor_cores(self) -> None:
        assert _tensor_core_gen((6, 0)) is None
        assert _tensor_core_gen((5, 0)) is None

    def test_sm_to_sm_tag(self) -> None:
        assert _cc_to_sm_tag((8, 0)) == "SM80"
        assert _cc_to_sm_tag((9, 0)) == "SM90"


# ---------------------------------------------------------------------------
# Blackwell naming consistency
# ---------------------------------------------------------------------------


class TestBlackwellNamingConsistency:
    """Detection and compatibility must agree on Blackwell naming."""

    def test_detection_reports_blackwell(self) -> None:
        assert _resolve_arch((10, 0)) == "Blackwell-DC"
        assert _resolve_arch((12, 0)) == "Blackwell-Consumer"

    def test_sm_arch_map_uses_blackwell(self) -> None:
        assert SM_ARCH_MAP["SM100"] == "Blackwell-DC"
        assert SM_ARCH_MAP["SM120"] == "Blackwell-Consumer"

    def test_require_arch_blackwell_matches_dc(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="B200", compute_capability=(10, 0),
            architecture="Blackwell-DC", memory_total_mb=196608,
            memory_free_mb=190000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=5,
            max_shared_memory_per_block=228 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Blackwell")
            def my_test() -> str:
                return "ran"

            assert my_test() == "ran"

    def test_require_arch_blackwell_matches_consumer(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="RTX 5090", compute_capability=(12, 0),
            architecture="Blackwell-Consumer", memory_total_mb=32768,
            memory_free_mb=30000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=5,
            max_shared_memory_per_block=228 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Blackwell")
            def my_test() -> str:
                return "ran"

            assert my_test() == "ran"

    def test_require_arch_blackwell_dc_runs_on_dc(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="B200", compute_capability=(10, 0),
            architecture="Blackwell-DC", memory_total_mb=196608,
            memory_free_mb=190000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=5,
            max_shared_memory_per_block=228 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Blackwell-DC")
            def my_test() -> str:
                return "ran"

            assert my_test() == "ran"

    def test_require_arch_blackwell_dc_skips_consumer(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="RTX 5090", compute_capability=(12, 0),
            architecture="Blackwell-Consumer", memory_total_mb=32768,
            memory_free_mb=30000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=5,
            max_shared_memory_per_block=228 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Blackwell-DC")
            def my_test() -> str:
                return "ran"

            with pytest.raises(pytest.skip.Exception):
                my_test()

    def test_check_compatibility_blackwell_resolves(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="H100", compute_capability=(9, 0),
            architecture="Hopper", memory_total_mb=81920,
            memory_free_mb=80000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=4,
            max_shared_memory_per_block=228 * 1024,
        )
        # "Blackwell" should resolve to an SM tag and produce a forward-compat warning
        issues = check_compatibility("Blackwell", mock_gpu)
        assert len(issues) > 0

    def test_check_compatibility_blackwell_dc_resolves(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="H100", compute_capability=(9, 0),
            architecture="Hopper", memory_total_mb=81920,
            memory_free_mb=80000, driver_version="", cuda_version="",
            supports_fp16=True, supports_bf16=True, supports_fp8=True,
            supports_tf32=True, tensor_core_generation=4,
            max_shared_memory_per_block=228 * 1024,
        )
        # "Blackwell-DC" should also resolve to SM100 and produce warnings
        issues = check_compatibility("Blackwell-DC", mock_gpu)
        assert len(issues) > 0
