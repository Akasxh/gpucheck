"""Stride/shape/dtype fuzzer for torch.nn.functional.group_norm on MPS.

Compares MPS execution to CPU (float32) reference. Uses gpucheck's
compute_tolerance() for per-dtype tolerance (MPS multiplier applied).

Hard runtime budget enforced by an outer wall-clock deadline.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812

# Ensure gpucheck is importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = OUT_DIR / "RESULTS_groupnorm.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "torch.nn.functional.group_norm"
SEED = 0xC0FFEE
N_ITERS = 250
WALL_BUDGET_S = 8 * 60 - 30  # leave ~30s for writeup
START = time.monotonic()

DTYPES = [torch.float32, torch.float16, torch.bfloat16]
SHAPE_CATEGORIES = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
STRIDE_CATEGORIES = ["contiguous", "slice", "transpose", "broadcast"]


def deadline_left() -> float:
    return WALL_BUDGET_S - (time.monotonic() - START)


def sample_shape(rng: random.Random, category: str) -> tuple[tuple[int, int, int, int], int]:
    """Return ((N, C, H, W), num_groups) with C % num_groups == 0."""
    if category == "degenerate":
        # Tiny / minimal
        N = rng.choice([1, 1, 2])
        C = rng.choice([1, 2, 4])
        H = rng.choice([1, 2])
        W = rng.choice([1, 2])
        G = 1 if C == 1 else rng.choice([g for g in (1, 2) if C % g == 0])
        return (N, C, H, W), G
    if category == "prime":
        primes = [3, 5, 7, 11, 13, 17, 19, 23]
        N = rng.choice([1, 2, 3])
        C = rng.choice(primes)
        H = rng.choice(primes)
        W = rng.choice(primes)
        # Only divisor of a prime is 1 (or itself) -> num_groups = 1 or C
        G = rng.choice([1, C])
        return (N, C, H, W), G
    if category == "pow2_boundary":
        # 31, 32, 33 / 63, 64, 65 patterns
        Cs = [16, 32, 64]
        C = rng.choice(Cs)
        N = rng.choice([1, 2])
        H = rng.choice([15, 16, 17, 31, 32, 33])
        W = rng.choice([15, 16, 17, 31, 32, 33])
        divisors = [g for g in (1, 2, 4, 8, 16, 32) if C % g == 0]
        G = rng.choice(divisors)
        return (N, C, H, W), G
    if category == "non_tile_aligned":
        # Sizes that don't align with typical 16/32 tile widths
        Cs_with_gs = [
            (24, [1, 2, 3, 4, 6, 8, 12, 24]),
            (40, [1, 2, 4, 5, 8, 10, 20, 40]),
            (48, [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]),
            (12, [1, 2, 3, 4, 6, 12]),
            (18, [1, 2, 3, 6, 9, 18]),
        ]
        C, groups = rng.choice(Cs_with_gs)
        N = rng.choice([1, 2, 3])
        H = rng.choice([13, 21, 27, 35])
        W = rng.choice([13, 21, 27, 35])
        G = rng.choice(groups)
        return (N, C, H, W), G
    if category == "large":
        N = rng.choice([2, 4])
        C = rng.choice([64, 96, 128])
        H = rng.choice([32, 48, 64])
        W = rng.choice([32, 48, 64])
        divisors = [g for g in (1, 2, 4, 8, 16, 32) if C % g == 0]
        G = rng.choice(divisors)
        return (N, C, H, W), G
    raise ValueError(category)


def make_input(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: str,
    stride_cat: str,
    rng_seed: int,
) -> tuple[torch.Tensor, str]:
    """Build an input tensor in fp32 on `device`, then cast to `dtype`.

    Returns (tensor, applied_layout_label). Some stride choices fall back
    to row-major on shape constraints; we report the *applied* layout.
    """
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    base_fp32 = torch.randn(shape, dtype=torch.float32, generator=g)
    N, C, H, W = shape

    if stride_cat == "contiguous":
        x = base_fp32.to(device=device, dtype=dtype).contiguous()
        return x, "contiguous"
    if stride_cat == "slice":
        # Build doubled-N then slice [::2]; preserves (N,C,H,W).
        big = torch.randn((N * 2, C, H, W), dtype=torch.float32, generator=g)
        x = big.to(device=device, dtype=dtype)[::2]
        return x, "slice_dim0_step2"
    if stride_cat == "transpose":
        # Build (N, C, W, H) contiguous, transpose last two -> (N, C, H, W) non-contiguous.
        flipped = torch.randn((N, C, W, H), dtype=torch.float32, generator=g)
        x = flipped.to(device=device, dtype=dtype).transpose(-1, -2)
        return x, "transpose_HW"
    if stride_cat == "broadcast":
        # Stride-0 along W via expand. Avoid degenerate when W==1 (already trivial).
        if W <= 1:
            x = base_fp32.to(device=device, dtype=dtype).contiguous()
            return x, "contiguous_fallback_from_broadcast"
        small = torch.randn((N, C, H, 1), dtype=torch.float32, generator=g)
        x = small.to(device=device, dtype=dtype).expand(N, C, H, W)
        return x, "broadcast_W"
    raise ValueError(stride_cat)


def run_one(rng: random.Random, idx: int) -> dict:
    shape_cat = rng.choice(SHAPE_CATEGORIES)
    stride_cat = rng.choice(STRIDE_CATEGORIES)
    dtype = rng.choice(DTYPES)
    shape, num_groups = sample_shape(rng, shape_cat)

    rec: dict = {
        "iter": idx,
        "kernel": KERNEL_NAME,
        "shape_category": shape_cat,
        "stride_category": stride_cat,
        "dtype": str(dtype).replace("torch.", ""),
        "shape": list(shape),
        "num_groups": num_groups,
        "status": "ok",
    }

    seed = SEED ^ (idx * 2654435761 & 0xFFFFFFFF)

    try:
        x_mps, applied = make_input(shape, dtype, "mps", stride_cat, seed)
        x_cpu_fp32 = x_mps.detach().to(device="cpu", dtype=torch.float32)
        rec["applied_layout"] = applied
        # group_norm signature: input, num_groups, weight, bias, eps
        y_mps = F.group_norm(x_mps, num_groups)
        y_cpu = F.group_norm(x_cpu_fp32, num_groups)
        # Bring MPS result back to CPU fp32 for comparison
        y_mps_fp32 = y_mps.detach().to(device="cpu", dtype=torch.float32)
    except NotImplementedError as e:
        rec["status"] = "UNSUPPORTED"
        rec["error"] = f"NotImplementedError: {e}"
        return rec
    except RuntimeError as e:
        msg = str(e)
        if "MPS" in msg and ("not implement" in msg or "not supported" in msg):
            rec["status"] = "UNSUPPORTED"
        else:
            rec["status"] = "ERROR"
        rec["error"] = msg[:300]
        return rec
    except Exception as e:  # noqa: BLE001
        rec["status"] = "ERROR"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec

    # Tolerance — use C/num_groups * H * W as the reduction k_dim.
    N, C, H, W = shape
    k_dim = max(1, (C // num_groups) * H * W)
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
    rec["atol"] = atol
    rec["rtol"] = rtol
    rec["k_dim"] = k_dim

    diff = (y_mps_fp32 - y_cpu).abs()
    abs_y = y_cpu.abs()
    max_abs = float(diff.max().item())
    # Relative error guarded for tiny denominators
    denom = abs_y.clamp_min(1e-12)
    rel = diff / denom
    max_rel = float(rel.max().item())

    rec["max_abs_err"] = max_abs
    rec["max_rel_err"] = max_rel

    # Divergence iff BOTH atol and rtol are exceeded (matches torch.allclose)
    diverged = max_abs > atol + rtol * float(abs_y.max().item())
    # Stronger isolation: also flag pure relative blow-ups in non-tiny outputs
    if not diverged and abs_y.max().item() > 1e-4 and max_rel > rtol * 4:
        diverged = True
    rec["diverged"] = bool(diverged)
    return rec


def main() -> int:
    rng = random.Random(SEED)

    if not torch.backends.mps.is_available():
        with RESULTS_MD.open("w") as f:
            f.write("# groupnorm fuzz — SKIPPED\n\nMPS not available.\n")
        sys.stdout.write("SKIPPED: MPS not available\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED",
                                "reason": "mps_unavailable"}) + "\n")
        return 0

    results: list[dict] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if deadline_left() <= 5:
            sys.stdout.write(f"[budget] stopping after {i} iters\n")
            break
        attempted += 1
        try:
            rec = run_one(rng, i)
            results.append(rec)
            if rec["status"] == "ok":
                completed += 1
        except Exception as e:  # noqa: BLE001
            results.append({
                "iter": i, "status": "FATAL",
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}",
            })
            sys.stdout.write(f"[fatal] iter {i}: {e}\n")

    # Build summary
    divergences = [r for r in results if r.get("diverged")]
    unsupported = [r for r in results if r.get("status") == "UNSUPPORTED"]
    errors = [r for r in results if r.get("status") in ("ERROR", "FATAL")]

    # Top 3 minimal repros: prefer smallest tensor element count among divergences
    def size_of(r: dict) -> int:
        s = r.get("shape") or [1, 1, 1, 1]
        prod = 1
        for v in s:
            prod *= max(1, int(v))
        return prod

    divergences_sorted = sorted(divergences, key=size_of)
    top3 = divergences_sorted[:3]

    # Aggregate max-rel-err on ok runs
    ok_runs = [r for r in results if r.get("status") == "ok"]
    max_rel_overall = max((r["max_rel_err"] for r in ok_runs), default=0.0)

    md = []
    md.append(f"# groupnorm fuzz — {KERNEL_NAME}\n")
    md.append("## Setup\n")
    md.append(f"- torch: {torch.__version__}\n")
    md.append(f"- mps available: {torch.backends.mps.is_available()}\n")
    md.append(f"- seed: {hex(SEED)}\n")
    md.append(f"- requested iterations: {N_ITERS}\n")
    md.append(f"- attempted: {attempted}\n")
    md.append(f"- completed (status=ok): {completed}\n")
    md.append(f"- unsupported: {len(unsupported)}\n")
    md.append(f"- errors: {len(errors)}\n")
    md.append(f"- divergences found: {len(divergences)}\n")
    md.append(f"- max rel err (ok runs): {max_rel_overall:.3e}\n\n")

    md.append("## Results table — counts by category\n")
    by_shape: dict[str, int] = {}
    by_stride: dict[str, int] = {}
    by_dtype: dict[str, int] = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        by_shape[r["shape_category"]] = by_shape.get(r["shape_category"], 0) + 1
        by_stride[r["stride_category"]] = by_stride.get(r["stride_category"], 0) + 1
        by_dtype[r["dtype"]] = by_dtype.get(r["dtype"], 0) + 1
    md.append(f"- shape categories: {dict(sorted(by_shape.items()))}\n")
    md.append(f"- stride categories: {dict(sorted(by_stride.items()))}\n")
    md.append(f"- dtypes: {dict(sorted(by_dtype.items()))}\n\n")

    md.append("## Top divergences (minimal repro)\n")
    if not top3:
        md.append("None — no MPS-vs-CPU divergence exceeded gpucheck tolerance.\n\n")
    else:
        for i, r in enumerate(top3, 1):
            md.append(
                f"{i}. shape={tuple(r['shape'])} num_groups={r['num_groups']} "
                f"dtype={r['dtype']} stride={r['stride_category']} "
                f"(applied={r.get('applied_layout', 'n/a')}) "
                f"max_abs_err={r['max_abs_err']:.3e} max_rel_err={r['max_rel_err']:.3e} "
                f"atol={r['atol']:.3e} rtol={r['rtol']:.3e}\n",
            )
        md.append("\n")

    md.append("## Cross-backend max relative error\n")
    md.append(f"- MPS-vs-CPU max rel err (across ok runs): {max_rel_overall:.3e}\n")
    md.append("- MPS-vs-CUDA-mock max rel err: N/A (no NVIDIA GPU; CUDA backend "
              "mocked at detection level only — no kernel execution)\n\n")

    md.append("## Unsupported / errors (sample up to 3)\n")
    for r in (unsupported + errors)[:3]:
        md.append(f"- iter={r.get('iter')} status={r.get('status')} "
                  f"shape={r.get('shape')} dtype={r.get('dtype')} "
                  f"stride={r.get('stride_category')} err={(r.get('error') or '')[:180]}\n")
    if not (unsupported or errors):
        md.append("None.\n")
    md.append("\n")

    md.append("## Recommended upstream filing target\n")
    if divergences:
        md.append("- pytorch/pytorch — divergences are between PyTorch's CPU "
                  "and MPS group_norm implementations; both are PyTorch native.\n")
        target = "pytorch/pytorch"
    else:
        md.append("- none — no divergences exceeded gpucheck tolerance.\n")
        target = "none"

    RESULTS_MD.write_text("".join(md))

    summary = {
        "kernel": KERNEL_NAME,
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "iterations_unsupported": len(unsupported),
        "iterations_errored": len(errors),
        "divergences": len(divergences),
        "top_repros": [
            {
                "shape": list(r["shape"]),
                "dtype": r["dtype"],
                "stride": r["stride_category"],
                "applied_layout": r.get("applied_layout"),
                "num_groups": r["num_groups"],
                "max_rel_err": r["max_rel_err"],
                "max_abs_err": r["max_abs_err"],
                "atol": r["atol"],
                "rtol": r["rtol"],
            }
            for r in top3
        ],
        "mps_vs_cpu_max_rel_err": max_rel_overall,
        "mps_vs_cuda_mock_max_rel_err": None,
        "upstream_target": target,
        "torch_version": torch.__version__,
        "seed": SEED,
        "wall_seconds": round(time.monotonic() - START, 2),
        "status": "ok",
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    sys.stdout.write(
        f"done: attempted={attempted} ok={completed} divergences={len(divergences)} "
        f"unsupported={len(unsupported)} errors={len(errors)} "
        f"wall={summary['wall_seconds']}s\n",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
