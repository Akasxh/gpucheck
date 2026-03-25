"""pytest plugin for gpucheck — registers fixtures, markers, and hooks."""

from __future__ import annotations

import contextlib
from typing import Any

import pytest


def _lazy_detect_gpus() -> list[Any]:
    """Lazy wrapper around detect_gpus() to avoid module-level GPU imports."""
    from gpucheck.arch import detect_gpus

    return detect_gpus()


def _gpu_available() -> bool:
    return len(_lazy_detect_gpus()) > 0


def _gpu_count() -> int:
    return len(_lazy_detect_gpus())


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
    has_gpu = _gpu_available()
    gpu_count = _gpu_count()

    skip_gpu = pytest.mark.skip(reason="no GPU available")
    skip_multi = pytest.mark.skip(reason="requires multiple GPUs")

    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "multi_gpu" in item.keywords and (not has_gpu or gpu_count < 2):
            item.add_marker(skip_multi)


def pytest_terminal_summary(
    terminalreporter: Any, exitstatus: int, config: pytest.Config
) -> None:
    gpus = _lazy_detect_gpus()
    writer = terminalreporter
    writer.section("GPU Info")
    if gpus:
        info = gpus[0]
        writer.line(f"  Device:  {info.name}")
        writer.line(f"  CUDA:    {info.cuda_version}")
        writer.line(f"  Compute: {info.compute_capability[0]}.{info.compute_capability[1]}")
        writer.line(f"  Memory:  {info.memory_total_mb} MB")
        writer.line(f"  GPUs:    {len(gpus)}")
    else:
        writer.line("  No GPU detected")


# Deferred fixture imports — only loaded when pytest actually needs them.
def _register_fixtures() -> None:
    """Import fixtures lazily to avoid pulling in torch/pynvml at collection time."""
    pass


# Fixture re-exports: these must be importable from plugin.py for pytest to find them.
# Use lazy imports so the heavy modules are only loaded when the fixture is actually used.

@pytest.fixture()
def gpu_benchmark(request: pytest.FixtureRequest) -> Any:
    """Provide a GPU kernel benchmarker using CUDA event timing."""
    from gpucheck.fixtures.benchmark import _BenchmarkRunner

    warmup = request.config.getoption("--gpu-benchmark-warmup", default=10)
    rounds = request.config.getoption("--gpu-benchmark-rounds", default=100)
    return _BenchmarkRunner(warmup=warmup, rounds=rounds)


@pytest.fixture()
def gpu_device() -> Any:
    """Provide a GPU device for the test, skip if none available."""
    from gpucheck.fixtures.gpu import gpu_device as _gpu_device_impl

    # Delegate to the actual generator-based fixture
    gen = _gpu_device_impl()
    device = next(gen)
    yield device
    with contextlib.suppress(StopIteration):
        next(gen)


@pytest.fixture()
def memory_tracker() -> Any:
    """Track GPU memory usage and detect leaks during a test."""
    try:
        from gpucheck.fixtures.profiler import MemoryTracker
    except ImportError:
        pytest.skip("memory_tracker requires pynvml or torch")

    import warnings

    tracker = MemoryTracker()
    tracker.start()
    yield tracker
    if tracker.report is None:
        report = tracker.stop()
        if report.leak_detected:
            warnings.warn(
                f"GPU memory leak detected: {report.leaked_mb:.1f}MB not freed",
                RuntimeWarning,
                stacklevel=2,
            )
