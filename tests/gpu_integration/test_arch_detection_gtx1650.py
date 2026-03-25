"""Hardware-validated GPU architecture detection tests for GTX 1650 (Turing).

Verifies detect_gpus() output against nvidia-smi ground truth and checks
all capability flags, decorator behavior, and cross-validation.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from gpucheck.arch import (
    detect_gpu,
    detect_gpus,
    gpu_available,
    gpu_count,
    require_arch,
    require_capability,
)
from gpucheck.arch.detection import GPUInfo, _tensor_core_gen
from gpucheck.arch.detection import detect_gpus as _raw_detect

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nvidia_smi_query() -> dict[str, str]:
    """Query nvidia-smi for ground truth GPU info."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    return {
        "name": parts[0],
        "compute_cap": parts[1],
        "memory_total_mib": parts[2],
        "driver_version": parts[3],
    }


@pytest.fixture(scope="module")
def gpu_info() -> GPUInfo:
    """Return the first detected GPU; skip module if no GPU."""
    _raw_detect.cache_clear()
    gpus = detect_gpus()
    if not gpus:
        pytest.skip("No GPU available")
    return gpus[0]


@pytest.fixture(scope="module")
def smi_info() -> dict[str, str]:
    """nvidia-smi ground truth; skip if nvidia-smi is absent."""
    try:
        return _nvidia_smi_query()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("nvidia-smi not available")
        return {}  # unreachable, keeps type checker happy


# ---------------------------------------------------------------------------
# 1. detect_gpus() returns correct info for GTX 1650
# ---------------------------------------------------------------------------

class TestDetectGpusBasic:
    def test_returns_non_empty_list(self, gpu_info: GPUInfo) -> None:
        assert gpu_info is not None

    def test_device_id_is_zero(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.device_id == 0


# ---------------------------------------------------------------------------
# 2. Name matches nvidia-smi output
# ---------------------------------------------------------------------------

class TestNameMatchesNvidiaSmi:
    def test_name_matches(self, gpu_info: GPUInfo, smi_info: dict[str, str]) -> None:
        # nvidia-smi: "NVIDIA GeForce GTX 1650", torch may return same or similar
        smi_name = smi_info["name"]
        assert gpu_info.name == smi_name, (
            f"detect_gpus() name={gpu_info.name!r} != nvidia-smi name={smi_name!r}"
        )

    def test_name_contains_gtx_1650(self, gpu_info: GPUInfo) -> None:
        assert "GTX 1650" in gpu_info.name


# ---------------------------------------------------------------------------
# 3. compute_capability = (7, 5) for Turing
# ---------------------------------------------------------------------------

class TestComputeCapability:
    def test_cc_tuple(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.compute_capability == (7, 5)

    def test_cc_matches_nvidia_smi(
        self, gpu_info: GPUInfo, smi_info: dict[str, str]
    ) -> None:
        smi_cc = smi_info["compute_cap"]  # e.g. "7.5"
        major, minor = (int(x) for x in smi_cc.split("."))
        assert gpu_info.compute_capability == (major, minor)


# ---------------------------------------------------------------------------
# 4. architecture = "Turing"
# ---------------------------------------------------------------------------

class TestArchitecture:
    def test_architecture_is_turing(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.architecture == "Turing"


# ---------------------------------------------------------------------------
# 5. Memory matches actual GPU memory
# ---------------------------------------------------------------------------

class TestMemory:
    def test_memory_total_within_range(
        self, gpu_info: GPUInfo, smi_info: dict[str, str]
    ) -> None:
        """torch reports usable VRAM which is less than physical.
        nvidia-smi reports physical (4096 MiB for GTX 1650).
        Allow up to 15% difference for OS/driver reserved memory.
        """
        smi_total_mib = int(smi_info["memory_total_mib"])
        detected_mb = gpu_info.memory_total_mb
        # detected should be <= smi (physical) and within 15%
        assert detected_mb <= smi_total_mib, (
            f"Detected {detected_mb} MB > nvidia-smi {smi_total_mib} MiB"
        )
        ratio = detected_mb / smi_total_mib
        assert ratio > 0.85, (
            f"Detected memory {detected_mb} MB is only {ratio:.1%} of "
            f"nvidia-smi {smi_total_mib} MiB — too large a gap"
        )

    def test_memory_total_positive(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.memory_total_mb > 0

    def test_memory_free_leq_total(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.memory_free_mb <= gpu_info.memory_total_mb


# ---------------------------------------------------------------------------
# 6. Dtype support flags (Turing: FP16=True, BF16=False)
# ---------------------------------------------------------------------------

class TestDtypeSupport:
    def test_supports_fp16(self, gpu_info: GPUInfo) -> None:
        assert gpu_info.supports_fp16 is True

    def test_supports_bf16_false(self, gpu_info: GPUInfo) -> None:
        # BF16 requires CC >= 8.0; Turing is 7.5
        assert gpu_info.supports_bf16 is False

    def test_supports_fp8_false(self, gpu_info: GPUInfo) -> None:
        # FP8 requires CC >= 8.9
        assert gpu_info.supports_fp8 is False

    def test_supports_tf32_false(self, gpu_info: GPUInfo) -> None:
        # TF32 requires CC >= 8.0
        assert gpu_info.supports_tf32 is False


# ---------------------------------------------------------------------------
# 7. tensor_core_generation matches expected for Turing
# ---------------------------------------------------------------------------

class TestTensorCoreGeneration:
    def test_tensor_core_gen_is_2(self, gpu_info: GPUInfo) -> None:
        # Turing = 2nd generation tensor cores
        assert gpu_info.tensor_core_generation == 2

    def test_tensor_core_gen_function(self) -> None:
        assert _tensor_core_gen((7, 5)) == 2


# ---------------------------------------------------------------------------
# 8. gpu_available() and gpu_count()
# ---------------------------------------------------------------------------

class TestGpuAvailableAndCount:
    def test_gpu_available_true(self) -> None:
        _raw_detect.cache_clear()
        assert gpu_available() is True

    def test_gpu_count_at_least_one(self) -> None:
        _raw_detect.cache_clear()
        assert gpu_count() >= 1


# ---------------------------------------------------------------------------
# 9. detect_gpu() (singular) returns first GPU
# ---------------------------------------------------------------------------

class TestDetectGpuSingular:
    def test_returns_first_gpu(self) -> None:
        _raw_detect.cache_clear()
        gpu = detect_gpu()
        assert gpu is not None
        assert gpu.device_id == 0

    def test_matches_detect_gpus_first(self) -> None:
        _raw_detect.cache_clear()
        single = detect_gpu()
        _raw_detect.cache_clear()
        multi = detect_gpus()
        assert single == multi[0]


# ---------------------------------------------------------------------------
# 10. require_arch decorator — skip when arch doesn't match
# ---------------------------------------------------------------------------

class TestRequireArchDecorator:
    def test_turing_matches(self) -> None:
        """Should run because we ARE on Turing."""
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Turing")
            def my_test() -> str:
                return "executed"

            assert my_test() == "executed"

    def test_hopper_skips_on_turing(self) -> None:
        """Should skip because we require Hopper but have Turing."""
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Hopper")
            def my_test() -> str:
                return "executed"

            with pytest.raises(pytest.skip.Exception):
                my_test()

    def test_multi_arch_match(self) -> None:
        """Should run when one of multiple archs matches."""
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_arch("Ampere", "Turing")
            def my_test() -> str:
                return "executed"

            assert my_test() == "executed"


# ---------------------------------------------------------------------------
# 11. require_capability decorator
# ---------------------------------------------------------------------------

class TestRequireCapabilityDecorator:
    def test_cc_75_passes_on_turing(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_capability(7, 5)
            def my_test() -> str:
                return "executed"

            assert my_test() == "executed"

    def test_cc_80_skips_on_turing(self) -> None:
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_capability(8, 0)
            def my_test() -> str:
                return "executed"

            with pytest.raises(pytest.skip.Exception):
                my_test()

    def test_cc_70_passes_on_turing(self) -> None:
        """Lower CC requirement should pass."""
        mock_gpu = GPUInfo(
            device_id=0, name="GTX 1650", compute_capability=(7, 5),
            architecture="Turing", memory_total_mb=4096, memory_free_mb=3000,
            driver_version="", cuda_version="", supports_fp16=True,
            supports_bf16=False, supports_fp8=False, supports_tf32=False,
            tensor_core_generation=2, max_shared_memory_per_block=96 * 1024,
        )
        with patch(
            "gpucheck.arch.compatibility._get_primary_gpu",
            return_value=mock_gpu,
        ):
            @require_capability(7, 0)
            def my_test() -> str:
                return "executed"

            assert my_test() == "executed"


# ---------------------------------------------------------------------------
# 12. Cross-validate detect_gpus() vs nvidia-smi
# ---------------------------------------------------------------------------

class TestCrossValidateWithNvidiaSmi:
    def test_full_cross_validation(
        self, gpu_info: GPUInfo, smi_info: dict[str, str]
    ) -> None:
        """Cross-validate all queryable fields against nvidia-smi."""
        # Name
        assert gpu_info.name == smi_info["name"]

        # Compute capability
        smi_major, smi_minor = (int(x) for x in smi_info["compute_cap"].split("."))
        assert gpu_info.compute_capability == (smi_major, smi_minor)

        # Driver version: torch backend doesn't expose driver_version
        # (returns ""), so we only validate if it's non-empty
        if gpu_info.driver_version:
            assert gpu_info.driver_version == smi_info["driver_version"]

        # Memory: within 15% of nvidia-smi physical
        smi_mem = int(smi_info["memory_total_mib"])
        assert gpu_info.memory_total_mb <= smi_mem
        assert gpu_info.memory_total_mb / smi_mem > 0.85

    def test_derived_fields_consistency(self, gpu_info: GPUInfo) -> None:
        """Verify that derived fields (arch, dtype support, TC gen) are
        internally consistent with compute_capability."""
        cc = gpu_info.compute_capability

        # Architecture must match SM_TO_ARCH
        from gpucheck.arch.detection import SM_TO_ARCH
        if cc in SM_TO_ARCH:
            assert gpu_info.architecture == SM_TO_ARCH[cc]

        # FP16: cc >= (5, 3)
        assert gpu_info.supports_fp16 == (cc >= (5, 3))
        # BF16: cc >= (8, 0)
        assert gpu_info.supports_bf16 == (cc >= (8, 0))
        # FP8: cc >= (8, 9)
        assert gpu_info.supports_fp8 == (cc >= (8, 9))
        # TF32: cc >= (8, 0)
        assert gpu_info.supports_tf32 == (cc >= (8, 0))

        # Tensor core generation
        assert gpu_info.tensor_core_generation == _tensor_core_gen(cc)
