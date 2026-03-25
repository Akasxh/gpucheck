"""Shared fixtures for testing gpucheck itself.

All tests run on CPU-only CI — GPU interactions are mocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Mock GPU device
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MockGPUInfo:
    """Lightweight stand-in for gpucheck.arch.detection.GPUInfo."""

    device_id: int = 0
    name: str = "Mock GPU A100"
    compute_capability: tuple[int, int] = (8, 0)
    architecture: str = "Ampere"
    memory_total_mb: int = 40960
    memory_free_mb: int = 38000
    driver_version: str = "535.129.03"
    cuda_version: str = "12.2"
    supports_fp16: bool = True
    supports_bf16: bool = True
    supports_fp8: bool = False
    supports_tf32: bool = True
    tensor_core_generation: int | None = 3
    max_shared_memory_per_block: int = 164 * 1024


@pytest.fixture()
def mock_gpu() -> MockGPUInfo:
    """Provide a mock GPU device for testing without real hardware."""
    return MockGPUInfo()


@pytest.fixture()
def mock_gpu_hopper() -> MockGPUInfo:
    """Mock Hopper (SM90) GPU."""
    return MockGPUInfo(
        name="Mock GPU H100",
        compute_capability=(9, 0),
        architecture="Hopper",
        memory_total_mb=81920,
        memory_free_mb=79000,
        supports_fp8=True,
        tensor_core_generation=4,
        max_shared_memory_per_block=228 * 1024,
    )


# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------

def make_close_tensors(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type = np.float32,
    noise_scale: float = 1e-7,
    seed: int = 42,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generate a pair of numpy arrays that are close but not identical."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(dtype)
    noise = (rng.standard_normal(shape) * noise_scale).astype(dtype)
    return base, base + noise


def make_exact_tensors(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type = np.float32,
    seed: int = 42,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generate a pair of identical numpy arrays."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(dtype)
    return base, base.copy()


def make_divergent_tensors(
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type = np.float32,
    seed: int = 42,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Generate a pair of very different numpy arrays."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(shape).astype(dtype)
    b = rng.standard_normal(shape).astype(dtype) + 100.0
    return a, b
