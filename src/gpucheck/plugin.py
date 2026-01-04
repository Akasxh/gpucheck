"""pytest plugin for gpucheck — registers fixtures, markers, and hooks."""

from __future__ import annotations

from typing import Any

import pytest

from gpucheck.arch import detect_gpu, gpu_available
from gpucheck.fixtures.benchmark import gpu_benchmark as gpu_benchmark  # noqa: F401
from gpucheck.fixtures.gpu import gpu_device as gpu_device  # noqa: F401

try:
    from gpucheck.fixtures.profiler import memory_tracker as memory_tracker  # noqa: F401
except ImportError:
    pass


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("gpucheck", "GPU kernel testing")
    group.addoption(
        "--gpu-device",
        default="cuda:0",
        help="GPU device to use for tests (default: cuda:0)",
    )
    group.addoption(
        "--gpu-benchmark-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations for GPU benchmarks (default: 10)",
    )
    group.addoption(
        "--gpu-benchmark-rounds",
        type=int,
        default=100,
        help="Number of timed iterations for GPU benchmarks (default: 100)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: marks tests requiring a GPU")
    config.addinivalue_line("markers", "slow: marks slow-running tests")
    config.addinivalue_line("markers", "multi_gpu: marks tests requiring multiple GPUs")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    has_gpu = gpu_available()
    info = detect_gpu()

    skip_gpu = pytest.mark.skip(reason="no GPU available")
    skip_multi = pytest.mark.skip(reason="requires multiple GPUs")

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and (not has_gpu or info.device_count < 2):
            item.add_marker(skip_multi)


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    info = detect_gpu()
    writer = terminalreporter
    writer.section("GPU Info")
    if info.available:
        writer.line(f"  Device:  {info.device_name}")
        writer.line(f"  CUDA:    {info.cuda_version}")
        writer.line(f"  Compute: {info.compute_capability[0]}.{info.compute_capability[1]}")
        writer.line(f"  Memory:  {info.memory_total_mb} MB")
        writer.line(f"  GPUs:    {info.device_count}")
    else:
        writer.line("  No GPU detected")
