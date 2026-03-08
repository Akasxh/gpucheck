"""Basic GPU kernel testing with gpucheck.

Demonstrates dtype/shape parametrization and assert_close.
Run with: pytest examples/basic_kernel_test.py
"""

from __future__ import annotations

import numpy as np
import pytest

import gpucheck as gc


@gc.dtypes("float16", "bfloat16", "float32")
@gc.shapes((128, 128), (256, 512), (7, 13))
def test_my_kernel(dtype, shape):
    """Test a simple element-wise kernel across dtypes and shapes."""
    torch = pytest.importorskip("torch")

    # Reference: CPU computation
    a = torch.randn(shape, dtype=torch.float32)
    b = torch.randn(shape, dtype=torch.float32)
    reference = a + b

    # "Kernel": cast to target dtype, compute, cast back
    a_cast = a.to(dtype)
    b_cast = b.to(dtype)
    output = (a_cast + b_cast).float()

    gc.assert_close(output, reference, baseline_2x=True)


@gc.dtypes("float32")
@gc.shapes((64, 64), (128, 128))
def test_relu_kernel(dtype, shape):
    """Test a ReLU-like operation."""
    torch = pytest.importorskip("torch")

    x = torch.randn(shape, dtype=dtype)
    output = torch.clamp(x, min=0)
    reference = torch.where(x > 0, x, torch.zeros_like(x))

    gc.assert_close(output, reference)


@gc.shapes(*gc.decorators.EDGE_SHAPES)
def test_edge_case_shapes(shape):
    """Test with adversarial shapes (primes, zeros, ones)."""
    # Skip zero-dimension shapes that would produce empty tensors
    if any(d == 0 for d in shape):
        pytest.skip("zero-dim shape")

    a = np.ones(shape, dtype=np.float32)
    b = np.ones(shape, dtype=np.float32)
    gc.assert_close(a, b)
