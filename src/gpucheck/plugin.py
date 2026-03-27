"""pytest plugin for gpucheck — registers fixtures, markers, and hooks."""

from __future__ import annotations

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
def gpu_device(request: pytest.FixtureRequest) -> Any:
    """Provide a GPU device for the test, honoring --gpu-device CLI option.

    When --gpu-device is explicitly set to something other than the default
    ``cuda:0``, a torch.device string is returned directly. Otherwise,
    falls back to auto-detection via pynvml/torch.
    """
    from gpucheck.fixtures.gpu import GPUDevice, _cleanup_gpu, detect_gpu

    cli_device: str = request.config.getoption("--gpu-device", default="cuda:0")

    # If the user overrode the default, build a GPUDevice from the CLI value.
    if cli_device != "cuda:0":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available")

        # Parse "cuda:N" -> N
        try:
            device_id = int(cli_device.split(":")[-1])
        except (ValueError, IndexError):
            pytest.fail(f"Invalid --gpu-device format: {cli_device!r} (expected 'cuda:N')")

        if device_id >= torch.cuda.device_count():
            pytest.fail(
                f"Device {cli_device!r} not available "
                f"(only {torch.cuda.device_count()} GPU(s) detected)"
            )

        props = torch.cuda.get_device_properties(device_id)
        mem_free, mem_total = torch.cuda.mem_get_info(device_id)
        device = GPUDevice(
            device_id=device_id,
            name=props.name,
            compute_capability=(props.major, props.minor),
            memory_total=mem_total,
            memory_free=mem_free,
        )
        yield device
        _cleanup_gpu()
        return

    # Default path: auto-detect via pynvml / torch.
    detected = detect_gpu()
    if detected is None:
        pytest.skip("No GPU available")
    yield detected
    _cleanup_gpu()


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
        if report.has_leak:
            warnings.warn(
                f"GPU memory leak detected: {report.leaked_mb:.1f}MB not freed",
                RuntimeWarning,
                stacklevel=2,
            )
