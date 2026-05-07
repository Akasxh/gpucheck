"""Fuzzer for matmul-fp16 across MPS (real) and CUDA (mocked detection).

Drives gpucheck's per-dtype tolerance + sqrt(k/128) matmul scaling. CUDA path
is N/A on this Mac (no NVIDIA hardware) — we mark CUDA results UNSUPPORTED
and only validate the MPS-vs-CPU divergence channel.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass

# Make gpucheck importable from this worktree
WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-matmul-fp16"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "matmul-fp16"
N_ITERS = 250
BUDGET_S = 8 * 60 - 30  # leave 30s for write-out

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_matmul-fp16.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]  # zero-K and 0-row would skip — handle explicitly via flag


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    M: int
    K: int
    N: int
    dtype: str
    stride: str
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    max_abs_err: float | None
    max_rel_err: float | None
    atol: float | None
    rtol: float | None
    note: str = ""


def pick_dim(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(DEGEN)
    if bucket == "prime":
        return rng.choice(PRIMES)
    if bucket == "pow2_boundary":
        return rng.choice(POW2_BOUNDARY)
    if bucket == "non_tile_aligned":
        return rng.choice(NON_TILE)
    if bucket == "large":
        return rng.choice(LARGE)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_inputs(M: int, K: int, N: int, dtype_name: str, stride: str, rng: random.Random,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct A (M,K) and B (K,N) on `device` with the requested stride pattern."""
    dt = torch_dtype(dtype_name)
    # Use float32 backing storage for stable random init, then cast — avoids
    # NaNs from extreme bf16/fp16 random draws on some torch versions.
    if stride == "contiguous":
        A = torch.randn(M, K, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)), dtype=torch.float32).to(device=device, dtype=dt)
        B = torch.randn(K, N, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)), dtype=torch.float32).to(device=device, dtype=dt)
    elif stride == "slice":
        # over-allocate, then slice with stride 2 along the K axis on both sides
        A_full = torch.randn(M, K * 2, dtype=torch.float32).to(device=device, dtype=dt)
        B_full = torch.randn(K * 2, N, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[:, ::2]
        B = B_full[::2, :]
        assert A.shape == (M, K) and B.shape == (K, N)
    elif stride == "transpose":
        A_t = torch.randn(K, M, dtype=torch.float32).to(device=device, dtype=dt)
        B_t = torch.randn(N, K, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.t()
        B = B_t.t()
        assert A.shape == (M, K) and B.shape == (K, N)
    elif stride == "broadcast":
        # broadcast along K on A: 1xK expanded to MxK; broadcast along N on B: Kx1 -> KxN
        A_row = torch.randn(1, K, dtype=torch.float32).to(device=device, dtype=dt).expand(M, K)
        B_col = torch.randn(K, 1, dtype=torch.float32).to(device=device, dtype=dt).expand(K, N)
        A = A_row
        B = B_col
    else:
        raise ValueError(stride)
    return A, B


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)
    M = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)
    N = pick_dim(bucket, rng)

    base = IterResult(
        idx=idx, shape_bucket=bucket, M=M, K=K, N=N, dtype=dtype, stride=stride,
        status="ERROR", max_abs_err=None, max_rel_err=None, atol=None, rtol=None,
    )

    if M == 0 or K == 0 or N == 0:
        base.status = "SKIP"
        base.note = "degenerate zero-dim"
        return base

    try:
        cpu_dev = torch.device("cpu")
        mps_dev = torch.device("mps")

        # Build inputs on CPU, then transfer (so MPS and CPU see numerically identical
        # source data — avoids RNG drift between devices.)
        A_cpu, B_cpu = make_inputs(M, K, N, dtype, stride, rng, cpu_dev)
        A_mps = A_cpu.detach().clone().to(mps_dev)
        B_mps = B_cpu.detach().clone().to(mps_dev)

        # Reference in float64 for fp32; for fp16/bf16 promote inputs to float32 on CPU
        # to model "infinite precision" baseline. This is a defensive comparison —
        # the divergence between CPU and MPS at the same low precision is what we
        # care about for kernel-level fuzzing.
        ref_dtype = torch.float64 if dtype == "float32" else torch.float32
        Y_ref = (A_cpu.to(ref_dtype) @ B_cpu.to(ref_dtype))

        Y_cpu = (A_cpu @ B_cpu).to(ref_dtype)
        Y_mps = (A_mps @ B_mps).to(cpu_dev).to(ref_dtype)
        # MPS returns bf16 on CPU only after transfer — the .to(cpu).to(ref_dtype) chain
        # handles bf16 (since bf16 is CPU-supported in 2.x).

        # gpucheck tolerance with k-scaling for matmul; device_type="mps" overlay
        atol_mps, rtol_mps = compute_tolerance(dtype, k_dim=K, device_type="mps")
        atol_cpu, rtol_cpu = compute_tolerance(dtype, k_dim=K, device_type="cpu")
        # Use the MPS overlay as the test gate (we are validating MPS).
        atol, rtol = atol_mps, rtol_mps

        diff = (Y_mps - Y_cpu).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        # Relative error vs CPU magnitude (avoid div-by-zero).
        denom = Y_cpu.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0

        base.atol, base.rtol = atol, rtol
        base.max_abs_err, base.max_rel_err = max_abs, max_rel

        # gpucheck combined check: |a-b| <= atol + rtol * |b|
        ok = bool(((diff - (atol + rtol * Y_cpu.abs())) <= 0).all().item())
        base.status = "OK" if ok else "DIVERGENCE"
        if not ok:
            # Note also vs ref to help triage (kernel error vs accumulator error)
            ref_diff = float((Y_mps - Y_ref).abs().max().item())
            cpu_ref_diff = float((Y_cpu - Y_ref).abs().max().item())
            base.note = (
                f"mps-vs-cpu max_abs={max_abs:.3e}, "
                f"mps-vs-ref={ref_diff:.3e}, cpu-vs-ref={cpu_ref_diff:.3e}, "
                f"atol_cpu={atol_cpu:.3e}"
            )
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {e}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001 — last-resort capture
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def main() -> int:
    rng = random.Random(0xF00DBABE)
    # Sanity check: torch.mps.is_available()
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        # Still write a short SKIPPED report so the swarm has a record.
        with open(RESULTS_MD, "w") as f:
            f.write(f"# matmul-fp16 fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters", flush=True)
            break
        attempted = i + 1
        r = run_one(i, rng)
        results.append(r)
        completed += 1
        if r.status == "DIVERGENCE":
            print(f"[{i:03d}] DIVERGENCE {r.dtype} {r.stride} M={r.M} K={r.K} N={r.N} "
                  f"max_rel={r.max_rel_err:.3e} atol={r.atol:.3e}", flush=True)
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.note}", flush=True)
        elif r.status == "UNSUPPORTED":
            print(f"[{i:03d}] UNSUPPORTED {r.dtype} {r.stride} {r.note}", flush=True)

    # Aggregate
    divergences = [r for r in results if r.status == "DIVERGENCE"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    ok = [r for r in results if r.status == "OK"]
    skipped = [r for r in results if r.status == "SKIP"]

    # Top divergences = highest max_rel_err
    divergences.sort(key=lambda r: (r.max_rel_err or 0.0), reverse=True)
    top3 = divergences[:3]

    overall_max_rel = max((r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0)

    # Recommend filing target — for a pure linalg op, pytorch/pytorch is the right home.
    if divergences:
        # Heuristic: if all divergences come from `broadcast` stride only, often a stride bug.
        target = "pytorch/pytorch"
    else:
        target = "none"

    # Markdown report
    md = []
    md.append(f"# matmul-fp16 fuzz results\n")
    md.append(f"- **kernel:** {KERNEL_NAME}")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **OK:** {len(ok)}")
    md.append(f"- **DIVERGENCE:** {len(divergences)}")
    md.append(f"- **UNSUPPORTED:** {len(unsupported)}")
    md.append(f"- **ERROR:** {len(errors)}")
    md.append(f"- **SKIP:** {len(skipped)}")
    md.append(f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}")
    md.append(f"- **MPS-vs-CUDA-mock max relative error:** N/A (no NVIDIA hardware; CUDA detection is mocked, kernel cannot run)")
    md.append(f"- **recommended upstream target:** {target}\n")
    md.append("## Top 3 minimal repros (by max relative error)\n")
    if top3:
        md.append("| # | dtype | stride | M | K | N | shape_bucket | max_rel_err | atol_used | note |")
        md.append("|---|-------|--------|---|---|---|--------------|-------------|-----------|------|")
        for i, r in enumerate(top3, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.M} | {r.K} | {r.N} | {r.shape_bucket} | "
                f"{r.max_rel_err:.3e} | {r.atol:.3e} | {r.note or ''} |"
            )
    else:
        md.append("_No divergences observed at gpucheck MPS-overlay tolerances._")
    md.append("")
    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})")
        seen = {}
        for r in unsupported:
            key = (r.dtype, r.stride)
            seen.setdefault(key, []).append(r)
        for (dt, st), rs in seen.items():
            md.append(f"- {dt} / {st}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")
    if errors:
        md.append(f"## ERROR summary ({len(errors)})")
        for r in errors[:10]:
            md.append(f"- iter {r.idx} {r.dtype}/{r.stride}/{r.shape_bucket} M={r.M} K={r.K} N={r.N}: {r.note}")
        md.append("")
    md.append("## Methodology notes")
    md.append("- Source RNG seeded fp32 on CPU, then cast/transferred so MPS and CPU see identical inputs.")
    md.append("- Reference Y is computed in fp64 (for fp32 ops) or fp32 (for fp16/bf16 ops) on CPU.")
    md.append("- Pass condition: `|Y_mps - Y_cpu| <= atol + rtol*|Y_cpu|` element-wise.")
    md.append("- Tolerance: `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='mps')` — base + sqrt(K/128) scaling + MPS 2× overlay.")
    md.append("- CUDA channel: gpucheck arch detection is mockable (pynvml/torch dual backend), but the kernel itself cannot execute without an NVIDIA device — so the CUDA-vs-MPS comparison is N/A on this host.")
    md.append("- Stride patterns: contiguous, slice (stride-2 over K), transpose (`.t()` view), broadcast (1×K and K×1 expanded).")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    # JSON line
    record = {
        "kernel": KERNEL_NAME,
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "divergences": len(divergences),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_mocked_no_nvidia_hardware",
        "top3": [
            {
                "rank": i + 1,
                "dtype": r.dtype,
                "stride": r.stride,
                "M": r.M, "K": r.K, "N": r.N,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err,
                "max_abs_err": r.max_abs_err,
                "atol": r.atol, "rtol": r.rtol,
                "note": r.note,
            }
            for i, r in enumerate(top3)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": time.monotonic() - t0,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
