"""GPU detection and capability mapping."""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# SM version → architecture name
SM_TO_ARCH: dict[tuple[int, int], str] = {
    (7, 0): "Volta",
    (7, 2): "Volta",
    (7, 5): "Turing",
    (8, 0): "Ampere",
    (8, 6): "Ampere",
    (8, 7): "Ampere",
    (8, 9): "Ada",
    (9, 0): "Hopper",
    (10, 0): "Blackwell",
    (12, 0): "Blackwell",
}

# Minimum compute capability for dtype support
_FP16_MIN_CC = (5, 3)
_BF16_MIN_CC = (8, 0)
_FP8_MIN_CC = (8, 9)
_TF32_MIN_CC = (8, 0)

# Tensor core generations by SM major version
_TENSOR_CORE_GEN: dict[tuple[int, int], int] = {
    (7, 0): 1,  # Volta — 1st gen
    (7, 2): 1,
    (7, 5): 2,  # Turing — 2nd gen
    (8, 0): 3,  # Ampere — 3rd gen
    (8, 6): 3,
    (8, 7): 3,
    (8, 9): 4,  # Ada — 4th gen
    (9, 0): 4,  # Hopper — 4th gen
    (10, 0): 5,  # Blackwell — 5th gen
    (12, 0): 5,
}


def _resolve_arch(cc: tuple[int, int]) -> str:
    """Map compute capability to architecture name, with best-effort fallback."""
    if cc in SM_TO_ARCH:
        return SM_TO_ARCH[cc]
    # Fallback: match by major version
    for (major, _minor), name in sorted(SM_TO_ARCH.items(), reverse=True):
        if cc[0] == major:
            return name
    if cc >= (12, 0):
        return "Blackwell"
    if cc >= (10, 0):
        return "Blackwell"
    if cc >= (9, 0):
        return "Hopper"
    if cc >= (8, 9):
        return "Ada"
    if cc >= (8, 0):
        return "Ampere"
    if cc >= (7, 5):
        return "Turing"
    if cc >= (7, 0):
        return "Volta"
    return "Unknown"


def _tensor_core_gen(cc: tuple[int, int]) -> int | None:
    """Return tensor core generation or None if no tensor cores."""
    if cc in _TENSOR_CORE_GEN:
        return _TENSOR_CORE_GEN[cc]
    # Best-effort by major version
    for (major, _minor), gen in sorted(_TENSOR_CORE_GEN.items(), reverse=True):
        if cc[0] == major and cc >= (major, _minor):
            return gen
    if cc < (7, 0):
        return None
    return None


@dataclass(frozen=True)
class GPUInfo:
    """Detailed information about a single GPU device."""

    device_id: int
    name: str
    compute_capability: tuple[int, int]
    architecture: str
    memory_total_mb: int
    memory_free_mb: int
    driver_version: str
    cuda_version: str
    supports_fp16: bool
    supports_bf16: bool
    supports_fp8: bool
    supports_tf32: bool
    tensor_core_generation: int | None
    max_shared_memory_per_block: int  # bytes


def _detect_via_pynvml() -> list[GPUInfo] | None:
    """Try GPU detection via pynvml. Returns None if unavailable."""
    try:
        import pynvml  # type: ignore[import-untyped]
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return None

    gpus: list[GPUInfo] = []
    try:
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_version, bytes):
            driver_version = driver_version.decode()

        # CUDA version from driver
        try:
            cuda_ver_int = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_major = cuda_ver_int // 1000
            cuda_minor = (cuda_ver_int % 1000) // 10
            cuda_version = f"{cuda_major}.{cuda_minor}"
        except Exception:
            cuda_version = ""

        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()

            cc_major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            if isinstance(cc_major, tuple):
                cc = (cc_major[0], cc_major[1])
            else:
                cc_minor = 0
                # Some pynvml versions return (major, minor) tuple directly
                cc = (cc_major, cc_minor)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mb = mem_info.total // (1024 * 1024)
            free_mb = mem_info.free // (1024 * 1024)

            try:
                max_smem = pynvml.nvmlDeviceGetMaxSharedMemoryPerBlock(handle)
            except (AttributeError, pynvml.NVMLError):
                # Fallback: typical defaults by arch
                max_smem = _default_shared_memory(cc)

            gpus.append(GPUInfo(
                device_id=i,
                name=name,
                compute_capability=cc,
                architecture=_resolve_arch(cc),
                memory_total_mb=total_mb,
                memory_free_mb=free_mb,
                driver_version=driver_version,
                cuda_version=cuda_version,
                supports_fp16=cc >= _FP16_MIN_CC,
                supports_bf16=cc >= _BF16_MIN_CC,
                supports_fp8=cc >= _FP8_MIN_CC,
                supports_tf32=cc >= _TF32_MIN_CC,
                tensor_core_generation=_tensor_core_gen(cc),
                max_shared_memory_per_block=max_smem,
            ))
    finally:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()

    return gpus


def _detect_via_torch() -> list[GPUInfo] | None:
    """Fallback GPU detection via torch.cuda."""
    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None

    gpus: list[GPUInfo] = []
    cuda_version = getattr(torch.version, "cuda", "") or ""

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        cc = (props.major, props.minor)

        total_mb = props.total_mem // (1024 * 1024)
        # torch doesn't expose free memory without allocating context; approximate
        try:
            torch.cuda.set_device(i)
            free_bytes, _total = torch.cuda.mem_get_info(i)
            free_mb = free_bytes // (1024 * 1024)
        except Exception:
            free_mb = total_mb  # best guess

        if hasattr(props, "max_shared_memory_per_block"):
            max_smem = props.max_shared_memory_per_block
        else:
            max_smem = _default_shared_memory(cc)

        gpus.append(GPUInfo(
            device_id=i,
            name=props.name,
            compute_capability=cc,
            architecture=_resolve_arch(cc),
            memory_total_mb=total_mb,
            memory_free_mb=free_mb,
            driver_version="",  # not available via torch
            cuda_version=cuda_version,
            supports_fp16=cc >= _FP16_MIN_CC,
            supports_bf16=cc >= _BF16_MIN_CC,
            supports_fp8=cc >= _FP8_MIN_CC,
            supports_tf32=cc >= _TF32_MIN_CC,
            tensor_core_generation=_tensor_core_gen(cc),
            max_shared_memory_per_block=max_smem,
        ))

    return gpus


def _default_shared_memory(cc: tuple[int, int]) -> int:
    """Return default max shared memory per block for a compute capability."""
    if cc >= (9, 0):
        return 228 * 1024  # Hopper+
    if cc >= (8, 0):
        return 164 * 1024  # Ampere / Ada
    if cc >= (7, 0):
        return 96 * 1024  # Volta / Turing
    return 48 * 1024  # pre-Volta


@lru_cache(maxsize=1)
def detect_gpus() -> list[GPUInfo]:
    """Detect all available GPUs and return their info.

    Uses pynvml as the primary backend (no torch import needed).
    Falls back to torch.cuda if pynvml is unavailable.
    Result is cached for the session lifetime.
    """
    gpus = _detect_via_pynvml()
    if gpus is not None:
        if gpus:
            logger.debug("Detected %d GPU(s) via pynvml", len(gpus))
        return gpus

    gpus = _detect_via_torch()
    if gpus is not None:
        if gpus:
            logger.debug("Detected %d GPU(s) via torch.cuda", len(gpus))
        return gpus

    warnings.warn(
        "No GPU detection backend available. Install pynvml or torch for GPU support.",
        stacklevel=2,
    )
    return []
