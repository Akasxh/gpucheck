"""Cross-architecture compatibility checks and decorators."""

from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, Any

import pytest

from gpucheck.arch.detection import detect_gpus

if TYPE_CHECKING:
    from collections.abc import Callable

    from gpucheck.arch.detection import GPUInfo

# Canonical SM tag → architecture mapping (string keys for SM tag lookups).
# See also detection.SM_TO_ARCH for the tuple-keyed canonical mapping.
SM_ARCH_MAP: dict[str, str] = {
    "SM70": "Volta",
    "SM75": "Turing",
    "SM80": "Ampere",
    "SM86": "Ampere",
    "SM89": "Ada",
    "SM90": "Hopper",
    "SM100": "Blackwell-DC",
    "SM120": "Blackwell-Consumer",
}

# Fine-grained Blackwell variants for users who need to distinguish DC vs Consumer.
SM_ARCH_MAP_DETAILED: dict[str, str] = {
    "SM100": "Blackwell-DC",
    "SM120": "Blackwell-Consumer",
}

# Arch family aliases: a parent name expands to itself + all sub-variants.
# This lets require_arch("Blackwell") match GPUs detected as "Blackwell-DC"
# or "Blackwell-Consumer", while require_arch("Blackwell-DC") only matches
# datacenter Blackwell (SM100).
_ARCH_ALIASES: dict[str, set[str]] = {
    "blackwell": {"blackwell", "blackwell-dc", "blackwell-consumer"},
}

# Parent architecture names that map to a representative SM tag for
# compatibility checking (e.g. "Blackwell" → "SM100").
_ARCH_PARENT_SM: dict[str, str] = {
    "blackwell": "SM100",
}


def _get_primary_gpu() -> GPUInfo | None:
    """Return the first available GPU, or None."""
    gpus = detect_gpus()
    return gpus[0] if gpus else None


def require_arch(*archs: str) -> Callable[..., Any]:
    """Decorator: skip test if GPU architecture doesn't match any of the given names.

    Usage::

        @require_arch("Ampere", "Hopper")
        def test_something():
            ...
    """
    # Normalize: accept both "Ampere" and "ampere", and expand aliases
    normalized: set[str] = set()
    for a in archs:
        key = a.lower()
        if key in _ARCH_ALIASES:
            normalized |= _ARCH_ALIASES[key]
        else:
            normalized.add(key)

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            gpu = _get_primary_gpu()
            if gpu is None:
                pytest.skip("No GPU available")
            if gpu.architecture.lower() not in normalized:
                pytest.skip(
                    f"Requires architecture {', '.join(archs)}, "
                    f"but found {gpu.architecture} "
                    f"(SM{gpu.compute_capability[0]}{gpu.compute_capability[1]})"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def require_capability(major: int, minor: int = 0) -> Callable[..., Any]:
    """Decorator: skip test if GPU compute capability is below (major, minor).

    Usage::

        @require_capability(8, 0)  # Ampere+
        def test_bf16_kernel():
            ...
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            gpu = _get_primary_gpu()
            if gpu is None:
                pytest.skip("No GPU available")
            if gpu.compute_capability < (major, minor):
                pytest.skip(
                    f"Requires compute capability >= {major}.{minor}, "
                    f"but found {gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
                )
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# Known incompatibilities: (kernel_target_sm, gpu_sm) → warning message
_KNOWN_INCOMPATIBILITIES: dict[tuple[str, str], str] = {
    ("SM90", "SM89"): (
        "Hopper-targeted kernels may silently fall back to slower code paths on Ada GPUs."
    ),
    ("SM90", "SM80"): (
        "Hopper wgmma instructions are not available on Ampere. "
        "Expect compilation failure or fallback."
    ),
    ("SM100", "SM90"): (
        "Blackwell-targeted kernels using SM100 features will not run on Hopper."
    ),
}


def _cc_to_sm_tag(cc: tuple[int, int]) -> str:
    """Convert compute capability tuple to SM tag like 'SM80'.

    Uses concatenation of major + minor digits, which is the standard NVIDIA
    convention (e.g., SM80, SM89, SM90, SM100, SM120).
    """
    return f"SM{cc[0] * 10 + cc[1]}"


def check_compatibility(kernel_target: str, gpu_info: GPUInfo) -> list[str]:
    """Check for known incompatibilities between a kernel target and GPU.

    Args:
        kernel_target: SM target string (e.g., "SM90") or architecture name (e.g., "Hopper").
        gpu_info: The GPU to check against.

    Returns:
        List of warning messages. Empty if no known issues.
    """
    # Normalize kernel_target to SM tag
    target_sm = kernel_target.upper()
    if not target_sm.startswith("SM"):
        # Try to resolve architecture name to an SM tag.
        # Check detailed variants first (e.g. "Blackwell-DC" → "SM100"),
        # then fall back to the main map (e.g. "Blackwell" → "SM100").
        target_lower = kernel_target.lower()
        resolved = False
        for source in (SM_ARCH_MAP_DETAILED, SM_ARCH_MAP):
            for sm, arch in source.items():
                if arch.lower() == target_lower:
                    target_sm = sm
                    resolved = True
                    break
            if resolved:
                break
        if not resolved and target_lower in _ARCH_PARENT_SM:
            target_sm = _ARCH_PARENT_SM[target_lower]

    gpu_sm = _cc_to_sm_tag(gpu_info.compute_capability)

    issues: list[str] = []

    # Check direct incompatibility table
    key = (target_sm, gpu_sm)
    if key in _KNOWN_INCOMPATIBILITIES:
        msg = _KNOWN_INCOMPATIBILITIES[key]
        issues.append(msg)
        warnings.warn(msg, stacklevel=2)

    # Generic forward-compatibility warning
    target_cc = _sm_tag_to_cc(target_sm)
    if target_cc is not None and target_cc > gpu_info.compute_capability:
        msg = (
            f"Kernel targets {target_sm} but running on {gpu_sm} "
            f"({gpu_info.architecture}). Forward compatibility is not guaranteed."
        )
        if msg not in issues:
            issues.append(msg)
            warnings.warn(msg, stacklevel=2)

    return issues


def _sm_tag_to_cc(sm_tag: str) -> tuple[int, int] | None:
    """Parse SM tag like 'SM80' into (8, 0). Returns None on failure."""
    tag = sm_tag.upper().removeprefix("SM")
    if len(tag) == 2:
        try:
            return (int(tag[0]), int(tag[1]))
        except ValueError:
            return None
    if len(tag) == 3:
        try:
            return (int(tag[:2]), int(tag[2]))
        except ValueError:
            return None
    return None
