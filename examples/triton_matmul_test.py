"""Triton matmul kernel testing with gpucheck.

Demonstrates testing a Triton kernel for correctness across dtypes/shapes.
Run with: pytest examples/triton_matmul_test.py -v

Requires: pip install triton torch
"""

from __future__ import annotations

from typing import Any

import pytest

import gpucheck as gc


def triton_available() -> bool:
    """Check if Triton is installed and CUDA is available."""
    try:
        import triton  # noqa: F401
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


skip_no_triton = pytest.mark.skipif(
    not triton_available(),
    reason="Triton or CUDA not available",
)


@skip_no_triton
@gc.dtypes("float16", "float32")
@gc.shapes((128, 128), (256, 256), (512, 512))
def test_triton_matmul_correctness(dtype: Any, shape: tuple[int, ...]) -> None:
    """Test Triton matmul against torch reference."""
    import torch
    import triton
    import triton.language as tl

    dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K))
            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
            offs_k += BLOCK_K

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)

    M, N = shape
    K = M  # square inner dim for simplicity

    a = torch.randn(M, K, device="cuda", dtype=dtype)
    b = torch.randn(K, N, device="cuda", dtype=dtype)
    c = torch.empty(M, N, device="cuda", dtype=dtype)

    BLOCK = 32
    grid = ((M + BLOCK - 1) // BLOCK, (N + BLOCK - 1) // BLOCK)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK,
    )

    reference = torch.matmul(a, b)
    gc.assert_close(c, reference, k_dim=K, baseline_2x=True)


@skip_no_triton
def test_triton_vector_add() -> None:
    """Minimal Triton kernel test: vector addition."""
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x + y, mask=mask)

    n = 1024
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty(n, device="cuda")

    BLOCK = 256
    grid = ((n + BLOCK - 1) // BLOCK,)
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)

    gc.assert_close(out, x + y)
