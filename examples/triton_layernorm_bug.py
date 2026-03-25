"""Bug #2: Triton layer norm variance padding.

The standard Triton tutorial layer norm kernel pads with zeros when
n_cols < BLOCK_SIZE. The zero-padded positions inject (BLOCK - n) * mean^2
into the variance computation, inflating variance and distorting the output.

Expected: relative error > 0.1 (10%) at n_cols=17 (BLOCK_SIZE=32).
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    stride,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton tutorial layer norm — THE BUG is in variance calculation."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load row, zero-pad out-of-bounds
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0)

    # Mean: computed over BLOCK_SIZE (includes padded zeros) — BUG
    mean = tl.sum(x, axis=0) / N

    # Variance: padded zeros contribute (0 - mean)^2 = mean^2 each — BUG
    # The tutorial computes variance over the full block then divides by N,
    # but the (x - mean) for padded positions = (0 - mean) = -mean,
    # adding (BLOCK_SIZE - N) * mean^2 to the numerator.
    var = tl.sum((x - mean) * (x - mean), axis=0) / N

    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    x_hat = (x - mean) * rstd

    # Scale and shift
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    y = x_hat * w + b

    tl.store(y_ptr + row * stride + cols, y, mask=mask)


def triton_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    y = torch.empty_like(x)

    layernorm_kernel[(n_rows,)](
        x, y, weight, bias,
        x.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


def main() -> None:
    torch.manual_seed(42)
    device = "cuda"

    # n_cols=17 -> BLOCK_SIZE=32, 15 padded positions
    n_rows, n_cols = 4, 17
    x = torch.randn((n_rows, n_cols), device=device, dtype=torch.float32)
    # Use non-zero mean input to maximize the bug's impact
    x = x + 5.0  # shift mean away from zero

    weight = torch.ones(n_cols, device=device, dtype=torch.float32)
    bias = torch.zeros(n_cols, device=device, dtype=torch.float32)

    # Reference
    ref = F.layer_norm(x, (n_cols,), weight, bias)

    # Triton kernel
    out = triton_layer_norm(x, weight, bias)

    abs_err = (out - ref).abs()
    rel_err = abs_err / (ref.abs() + 1e-8)

    max_abs = abs_err.max().item()
    max_rel = rel_err.max().item()
    mean_rel = rel_err.mean().item()

    print(f"=== Bug #2: Triton layer norm variance padding ===")
    print(f"Shape: rows={n_rows}, cols={n_cols}, BLOCK_SIZE={triton.next_power_of_2(n_cols)}")
    print(f"Input mean (row 0): {x[0].mean().item():.4f}")
    print(f"Max absolute error: {max_abs}")
    print(f"Max relative error: {max_rel}")
    print(f"Mean relative error: {mean_rel}")

    # Also test n_cols=3 (BLOCK_SIZE=4, 1 padded position but very small N)
    n_cols2 = 3
    x2 = torch.randn((n_rows, n_cols2), device=device, dtype=torch.float32) + 5.0
    w2 = torch.ones(n_cols2, device=device, dtype=torch.float32)
    b2 = torch.zeros(n_cols2, device=device, dtype=torch.float32)
    ref2 = F.layer_norm(x2, (n_cols2,), w2, b2)
    out2 = triton_layer_norm(x2, w2, b2)
    rel_err2 = ((out2 - ref2).abs() / (ref2.abs() + 1e-8)).max().item()
    print(f"\nn_cols=3 max relative error: {rel_err2}")

    threshold = 0.1
    if max_rel > threshold:
        print(f"\nCONFIRMED: max relative error {max_rel:.4f} > {threshold}")
    else:
        print(f"\nNOT_REPRODUCIBLE: max relative error {max_rel:.4f} <= {threshold}")


if __name__ == "__main__":
    main()
