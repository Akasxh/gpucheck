"""GPU kernel benchmarking with gpucheck.

Demonstrates the gpu_benchmark fixture for timing kernel execution.
Run with: pytest examples/benchmark_example.py -v --gpu-benchmark-rounds=5
"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_matmul_benchmark(gpu_benchmark):
    """Benchmark a matrix multiplication kernel."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    size = 1024
    a = torch.randn(size, size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, size, device="cuda", dtype=torch.float32)

    result = gpu_benchmark(torch.matmul, a, b)

    print(f"\nMatmul {size}x{size}:")
    print(f"  Mean: {result.mean * 1000:.3f} ms")
    print(f"  Min:  {result.min * 1000:.3f} ms")
    print(f"  Max:  {result.max * 1000:.3f} ms")


@pytest.mark.gpu
def test_elementwise_benchmark(gpu_benchmark):
    """Benchmark an element-wise operation."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    size = (4096, 4096)
    x = torch.randn(size, device="cuda", dtype=torch.float32)

    def relu_kernel(t):
        return torch.relu(t)

    result = gpu_benchmark(relu_kernel, x)

    print(f"\nReLU {size[0]}x{size[1]}:")
    print(f"  Mean: {result.mean * 1000:.3f} ms")
    print(f"  Min:  {result.min * 1000:.3f} ms")


@pytest.mark.gpu
def test_memory_tracking(memory_tracker):
    """Demonstrate GPU memory tracking."""
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Allocate a large tensor
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
    y = torch.matmul(x, x)

    del x, y

    report = memory_tracker.stop()
    print(f"\nMemory tracking:")
    print(f"  Before: {report.before.used_mb:.1f} MB")
    print(f"  After:  {report.after.used_mb:.1f} MB")
    print(f"  Peak:   {report.peak_mb:.1f} MB")
    print(f"  Leaked: {report.leaked_mb:.1f} MB")
