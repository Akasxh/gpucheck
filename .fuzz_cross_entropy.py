"""Fuzz cross_entropy on MPS vs CPU. Time-budgeted; deterministic seeded."""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from typing import Any

import torch  # type: ignore[import-not-found]

from gpucheck.assertions.tolerances import compute_tolerance

DEADLINE = float(os.environ.get("FUZZ_DEADLINE", "0")) or (time.monotonic() + 7 * 60)
ITERATIONS = int(os.environ.get("FUZZ_ITERS", "250"))
SEED = 20260501
random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------

# (label, (N, C)) — drawn from gpucheck shape priorities
SHAPE_BUCKETS: list[tuple[str, tuple[int, int]]] = [
    # degenerate
    ("degenerate", (1, 2)),
    ("degenerate", (1, 1024)),
    ("degenerate", (2, 2)),
    # prime
    ("prime", (7, 13)),
    ("prime", (13, 257)),
    ("prime", (31, 127)),
    ("prime", (127, 251)),
    # power-of-2 boundary
    ("pow2_boundary", (16, 32)),
    ("pow2_boundary", (32, 64)),
    ("pow2_boundary", (64, 128)),
    ("pow2_boundary", (128, 256)),
    ("pow2_boundary", (256, 1024)),
    # non-tile-aligned
    ("non_tile_aligned", (33, 65)),
    ("non_tile_aligned", (65, 129)),
    ("non_tile_aligned", (129, 257)),
    ("non_tile_aligned", (257, 513)),
    # large
    ("large", (1024, 4096)),
    ("large", (2048, 8192)),
    ("large", (4096, 1000)),
]

DTYPES: list[tuple[str, torch.dtype]] = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# Stride pattern categories — only those that preserve (N, C) semantics for
# cross_entropy and that the op can actually consume.
STRIDE_PATTERNS = [
    "contiguous",
    "transpose",     # build (C,N) contig then .t() -> (N,C) view
    "slice_rows",    # pad rows by 2x then [::2]
    "slice_cols",    # pad cols by 2x then [:, ::2]
    "broadcast_row", # build (1,C) then expand((N,C))
    "broadcast_col", # build (N,1) then expand((N,C))  -> uniform softmax
    "non_contig",    # 3D view permuted then sliced back to (N,C)
]


def make_logits(
    N: int, C: int, dtype: torch.dtype, device: str, pattern: str, gen: torch.Generator,
) -> torch.Tensor:
    """Return an (N,C) logits tensor laid out per stride *pattern*."""
    if pattern == "contiguous":
        x = torch.randn((N, C), generator=gen, dtype=torch.float32)
        return x.to(dtype=dtype, device=device).contiguous()
    if pattern == "transpose":
        base = torch.randn((C, N), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base.t()  # view, non-contiguous
    if pattern == "slice_rows":
        base = torch.randn((N * 2, C), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base[::2]
    if pattern == "slice_cols":
        base = torch.randn((N, C * 2), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base[:, ::2]
    if pattern == "broadcast_row":
        base = torch.randn((1, C), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base.expand((N, C))
    if pattern == "broadcast_col":
        base = torch.randn((N, 1), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base.expand((N, C))
    if pattern == "non_contig":
        # Build (2, N, C) contiguous, take view [0] then advance through stride
        base = torch.randn((2, N, C), generator=gen, dtype=torch.float32).to(
            dtype=dtype, device=device,
        ).contiguous()
        return base.permute(1, 0, 2)[:, 0, :]  # (N, C) non-contig view
    raise ValueError(f"unknown pattern {pattern!r}")


def cross_entropy_safe(
    logits: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    """Run F.cross_entropy with reduction='mean' in the tensor's native dtype.

    Cross-entropy on float16/bf16 will internally upcast in PyTorch's reference
    impl on CPU; we mirror that on MPS by passing the tensor through directly
    so any divergence reflects MPS kernel behavior, not our harness.
    """
    return torch.nn.functional.cross_entropy(logits, target, reduction="mean")


def max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max relative error between two scalars (or tensors); fp64 on CPU.

    MPS cannot cast to float64, so always move-to-cpu first, then upcast.
    """
    af = a.detach().to(device="cpu").to(dtype=torch.float64)
    bf = b.detach().to(device="cpu").to(dtype=torch.float64)
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-12)
    return float((diff / denom).max().item())


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    if not torch.backends.mps.is_available():
        print("SKIPPED: torch.mps not available", file=sys.stderr)
        return 0

    device_mps = "mps"
    device_cpu = "cpu"

    completed = 0
    attempted = 0
    unsupported = 0
    errors = 0
    divergences: list[dict[str, Any]] = []
    max_rel_err_seen = 0.0
    err_by_dtype: dict[str, float] = {}

    for it in range(ITERATIONS):
        if time.monotonic() > DEADLINE:
            print(f"BUDGET_EXCEEDED at iter {it}", file=sys.stderr)
            break
        attempted += 1

        shape_label, (N, C) = random.choice(SHAPE_BUCKETS)
        dtype_name, dtype = random.choice(DTYPES)
        pattern = random.choice(STRIDE_PATTERNS)
        seed_i = random.randint(0, 2**31 - 1)

        try:
            gen_cpu = torch.Generator().manual_seed(seed_i)
            logits_cpu_f32 = torch.randn((N, C), generator=gen_cpu, dtype=torch.float32)
            target = torch.randint(
                0, C, (N,), generator=torch.Generator().manual_seed(seed_i ^ 0xA5A5),
                dtype=torch.long,
            )

            # Build CPU and MPS logits independently per pattern but using the
            # same source f32 values so divergence is attributable to MPS.
            def build(device: str) -> torch.Tensor:
                base32 = logits_cpu_f32.to(device=device)
                if pattern == "contiguous":
                    return base32.to(dtype=dtype).contiguous()
                if pattern == "transpose":
                    flipped = base32.t().contiguous().to(dtype=dtype)
                    return flipped.t()
                if pattern == "slice_rows":
                    big = torch.randn(
                        (N * 2, C),
                        generator=torch.Generator().manual_seed(seed_i ^ 0x1111),
                        dtype=torch.float32,
                    ).to(device=device)
                    big[::2] = base32
                    return big.to(dtype=dtype)[::2]
                if pattern == "slice_cols":
                    big = torch.randn(
                        (N, C * 2),
                        generator=torch.Generator().manual_seed(seed_i ^ 0x2222),
                        dtype=torch.float32,
                    ).to(device=device)
                    big[:, ::2] = base32
                    return big.to(dtype=dtype)[:, ::2]
                if pattern == "broadcast_row":
                    row = base32[0:1].to(dtype=dtype).contiguous()
                    return row.expand((N, C))
                if pattern == "broadcast_col":
                    col = base32[:, 0:1].to(dtype=dtype).contiguous()
                    return col.expand((N, C))
                if pattern == "non_contig":
                    pad = torch.zeros((2, N, C), dtype=dtype, device=device)
                    pad[0] = base32.to(dtype=dtype)
                    return pad.permute(1, 0, 2)[:, 0, :]
                raise ValueError(pattern)

            logits_cpu = build(device_cpu)
            logits_mps = build(device_mps)
            target_mps = target.to(device=device_mps)

            try:
                ref = cross_entropy_safe(logits_cpu, target)
            except Exception as e:  # noqa: BLE001
                # CPU reference must work; if not, mark unsupported and continue
                unsupported += 1
                continue

            try:
                got = cross_entropy_safe(logits_mps, target_mps)
                # force completion
                torch.mps.synchronize()
            except (RuntimeError, NotImplementedError) as e:
                msg = str(e)
                # MPS may not implement cross_entropy for some dtypes/strides
                divergences.append({
                    "kind": "UNSUPPORTED",
                    "shape": [N, C],
                    "shape_label": shape_label,
                    "dtype": dtype_name,
                    "stride": pattern,
                    "error": msg[:300],
                    "seed": seed_i,
                })
                unsupported += 1
                continue

            rerr = max_rel_err(ref, got)
            max_rel_err_seen = max(max_rel_err_seen, rerr)
            err_by_dtype[dtype_name] = max(err_by_dtype.get(dtype_name, 0.0), rerr)

            # Tolerance: per-dtype, scaled by sqrt(C/128) (cross_entropy reduces
            # over class dim of size C, so C is the k_dim).
            atol, rtol = compute_tolerance(dtype, k_dim=C, device_type="mps")
            ref_cpu64 = ref.detach().to(device="cpu").to(dtype=torch.float64)
            got_cpu64 = got.detach().to(device="cpu").to(dtype=torch.float64)
            ref_mag = float(ref_cpu64.abs().item())
            tol = atol + rtol * ref_mag
            abs_err = float((got_cpu64 - ref_cpu64).abs().item())

            if abs_err > tol or not math.isfinite(rerr):
                divergences.append({
                    "kind": "DIVERGENCE",
                    "shape": [N, C],
                    "shape_label": shape_label,
                    "dtype": dtype_name,
                    "stride": pattern,
                    "ref": float(ref_cpu64.item()),
                    "got": float(got_cpu64.item()),
                    "abs_err": abs_err,
                    "rel_err": rerr,
                    "tol": tol,
                    "atol": atol,
                    "rtol": rtol,
                    "seed": seed_i,
                })
            completed += 1

        except Exception as e:  # noqa: BLE001
            errors += 1
            print(
                f"ITER_ERROR iter={it} shape={(N, C)} dtype={dtype_name} "
                f"pat={pattern}: {e}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)
            if errors >= 5:
                print("HALT: too many errors", file=sys.stderr)
                break

    summary = {
        "kernel": "cross_entropy",
        "attempted": attempted,
        "completed": completed,
        "unsupported": unsupported,
        "errors": errors,
        "divergences_total": len([d for d in divergences if d["kind"] == "DIVERGENCE"]),
        "unsupported_total": len([d for d in divergences if d["kind"] == "UNSUPPORTED"]),
        "max_rel_err_mps_vs_cpu": max_rel_err_seen,
        "max_rel_err_by_dtype": err_by_dtype,
        "divergences": divergences,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
