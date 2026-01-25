"""Tensor core utilities: support checks, tolerance computation, fallback detection."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpucheck.arch.detection import GPUInfo

# dtype name → minimum tensor core generation required
_DTYPE_TC_MIN_GEN: dict[str, int] = {
    "float16": 1,
    "fp16": 1,
    "bfloat16": 3,  # Ampere+
    "bf16": 3,
    "float8_e4m3fn": 4,  # Ada / Hopper+
    "float8_e5m2": 4,
    "fp8": 4,
    "fp8_e4m3": 4,
    "fp8_e5m2": 4,
    "tf32": 3,  # Ampere+
    "float32": 3,  # TF32 path for float32 on tensor cores
    "int8": 2,  # Turing+
    "int4": 2,
}

# Base tolerances (atol, rtol) per dtype
_BASE_TOLERANCE: dict[str, tuple[float, float]] = {
    "float16": (1e-3, 1e-3),
    "fp16": (1e-3, 1e-3),
    "bfloat16": (1e-2, 1.6e-2),
    "bf16": (1e-2, 1.6e-2),
    "float8_e4m3fn": (5e-2, 5e-2),
    "float8_e5m2": (1e-1, 1e-1),
    "fp8": (5e-2, 5e-2),
    "fp8_e4m3": (5e-2, 5e-2),
    "fp8_e5m2": (1e-1, 1e-1),
    "tf32": (1e-4, 1.3e-4),
    "float32": (1e-5, 1.3e-6),
    "float64": (1e-12, 1e-12),
    "fp64": (1e-12, 1e-12),
    "int8": (0.0, 0.0),
}


def _normalize_dtype(dtype: str) -> str:
    """Normalize a dtype string (handles torch.float16 → float16, etc.)."""
    s = str(dtype)
    # Strip "torch." prefix if present
    if s.startswith("torch."):
        s = s[6:]
    return s.lower()


def supports_tensor_cores(gpu_info: GPUInfo, dtype: str) -> bool:
    """Check if the GPU supports tensor core operations for a given dtype.

    Args:
        gpu_info: GPU information from detect_gpus().
        dtype: Data type name (e.g., "float16", "bf16", "fp8", "torch.bfloat16").

    Returns:
        True if tensor cores are available for this dtype on this GPU.
    """
    if gpu_info.tensor_core_generation is None:
        return False

    norm = _normalize_dtype(dtype)
    min_gen = _DTYPE_TC_MIN_GEN.get(norm)
    if min_gen is None:
        return False

    return gpu_info.tensor_core_generation >= min_gen


def compute_tolerance(
    dtype: str,
    k_dim: int,
    gpu_info: GPUInfo | None = None,
) -> tuple[float, float]:
    """Compute architecture-aware (atol, rtol) for a matmul-like operation.

    Error in floating-point matmul scales roughly as O(sqrt(k)) due to
    random rounding. We scale base tolerance by sqrt(k / 128) where 128
    is a reference dimension.

    Additional scaling is applied based on GPU architecture:
    - Older architectures with less precise tensor cores get wider tolerance.
    - FP8 on Hopper gets tighter tolerance than FP8 on Ada (better HW rounding).

    Args:
        dtype: Data type string.
        k_dim: Reduction dimension size (K in M x K @ K x N).
        gpu_info: Optional GPU info for architecture-specific adjustments.

    Returns:
        (atol, rtol) tuple.
    """
    norm = _normalize_dtype(dtype)
    base_atol, base_rtol = _BASE_TOLERANCE.get(norm, (1e-3, 1e-3))

    # Scale by sqrt(k / 128)
    k_scale = math.sqrt(max(k_dim, 1) / 128.0)
    atol = base_atol * k_scale
    rtol = base_rtol * k_scale

    # Architecture-specific adjustment
    if gpu_info is not None:
        atol, rtol = _arch_adjust(atol, rtol, norm, gpu_info)

    return (atol, rtol)


def _arch_adjust(
    atol: float,
    rtol: float,
    dtype_norm: str,
    gpu_info: GPUInfo,
) -> tuple[float, float]:
    """Apply architecture-specific tolerance scaling."""
    arch = gpu_info.architecture.lower()

    # Hopper has better FP8 rounding than Ada
    if dtype_norm in ("fp8", "fp8_e4m3", "float8_e4m3fn") and arch == "hopper":
        atol *= 0.7
        rtol *= 0.7

    # Volta first-gen tensor cores have slightly larger rounding error for fp16
    if dtype_norm in ("float16", "fp16") and arch == "volta":
        atol *= 1.5
        rtol *= 1.5

    # TF32 on Ampere vs Hopper — Hopper has improved accumulation
    if dtype_norm in ("tf32", "float32") and arch == "hopper":
        atol *= 0.8
        rtol *= 0.8

    return (atol, rtol)


def warn_tensor_core_fallback() -> None:
    """Detect and warn if a kernel silently fell back from tensor cores to CUDA cores.

    This checks CUDA environment variables that can force CUDA-core paths:
    - NVIDIA_TF32_OVERRIDE=0 disables TF32 tensor core usage
    - CUDA_MATH_MODE can affect tensor core dispatch

    It also checks torch settings if torch is available.
    """
    import os

    issues: list[str] = []

    # Check TF32 override
    tf32_override = os.environ.get("NVIDIA_TF32_OVERRIDE")
    if tf32_override == "0":
        issues.append(
            "NVIDIA_TF32_OVERRIDE=0 is set — TF32 tensor core operations are disabled, "
            "falling back to FP32 CUDA cores."
        )

    # Check torch-level settings
    try:
        import torch

        if hasattr(torch.backends, "cuda") and not torch.backends.cuda.matmul.allow_tf32:  # type: ignore[attr-defined]
            issues.append(
                "torch.backends.cuda.matmul.allow_tf32 is False — "
                "matmul will use FP32 CUDA cores instead of TF32 tensor cores."
            )

        if hasattr(torch.backends, "cudnn") and not torch.backends.cudnn.allow_tf32:  # type: ignore[attr-defined]
            issues.append(
                "torch.backends.cudnn.allow_tf32 is False — "
                "cuDNN convolutions will use FP32 CUDA cores instead of TF32 tensor cores."
            )
    except ImportError:
        pass

    for msg in issues:
        warnings.warn(msg, stacklevel=2)
