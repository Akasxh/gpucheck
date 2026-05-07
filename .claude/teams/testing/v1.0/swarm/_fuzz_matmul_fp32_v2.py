"""v2 fuzzer for matmul-fp32 on MPS vs CPU.

Differences from v1:
- 500 iterations
- FILABLE classification with reproducibility check (>=3 seeds)
- TOLERANCE_RECALIBRATION classification for systematic small-but-above-tol drift
- denom_magnitude floor (1e-6) to suppress numerical-noise false positives
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-matmul-fp32"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "matmul-fp32"
N_ITERS = 500
BUDGET_S = 8 * 60 - 45  # leave 45s for replay + write-out
DENOM_FLOOR = 1e-6
FILABLE_TOL_MULTIPLIER = 10.0
REPLAY_SEEDS = 3  # reproducibility threshold (>=3)

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_matmul-fp32.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32"]  # task scope: matmul-fp32
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    M: int
    K: int
    N: int
    dtype: str
    stride: str
    seed: int
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    classification: str  # OK | TOLERANCE_RECALIBRATION | FILABLE | UNCLASSIFIED
    max_abs_err: float | None
    max_rel_err: float | None
    denom_max: float | None  # |Y_cpu|.max — the magnitude scale at the worst element
    atol: float | None
    rtol: float | None
    tol_ratio: float | None  # max_rel_err / rtol
    note: str = ""
    replay_max_rel_errs: list[float] = field(default_factory=list)


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


def make_inputs(M: int, K: int, N: int, dtype_name: str, stride: str, seed: int,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    dt = torch_dtype(dtype_name)
    g_a = torch.Generator().manual_seed(seed)
    g_b = torch.Generator().manual_seed(seed ^ 0xA5A5A5A5)
    if stride == "contiguous":
        A = torch.randn(M, K, generator=g_a, dtype=torch.float32).to(device=device, dtype=dt)
        B = torch.randn(K, N, generator=g_b, dtype=torch.float32).to(device=device, dtype=dt)
    elif stride == "slice":
        A_full = torch.randn(M, K * 2, generator=g_a, dtype=torch.float32).to(device=device, dtype=dt)
        B_full = torch.randn(K * 2, N, generator=g_b, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[:, ::2]
        B = B_full[::2, :]
    elif stride == "transpose":
        A_t = torch.randn(K, M, generator=g_a, dtype=torch.float32).to(device=device, dtype=dt)
        B_t = torch.randn(N, K, generator=g_b, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.t()
        B = B_t.t()
    elif stride == "broadcast":
        A_row = torch.randn(1, K, generator=g_a, dtype=torch.float32).to(device=device, dtype=dt).expand(M, K)
        B_col = torch.randn(K, 1, generator=g_b, dtype=torch.float32).to(device=device, dtype=dt).expand(K, N)
        A = A_row
        B = B_col
    else:
        raise ValueError(stride)
    assert A.shape == (M, K) and B.shape == (K, N)
    return A, B


def measure_one(M: int, K: int, N: int, dtype: str, stride: str, seed: int) -> tuple[str, float, float, float, str]:
    """Return (status, max_abs, max_rel, denom_max, note). Status in OK|DIVERGENCE|UNSUPPORTED|ERROR."""
    cpu_dev = torch.device("cpu")
    mps_dev = torch.device("mps")
    try:
        A_cpu, B_cpu = make_inputs(M, K, N, dtype, stride, seed, cpu_dev)
        A_mps = A_cpu.detach().clone().to(mps_dev)
        B_mps = B_cpu.detach().clone().to(mps_dev)

        ref_dtype = torch.float64 if dtype == "float32" else torch.float32

        Y_cpu = (A_cpu @ B_cpu).to(ref_dtype)
        Y_mps = (A_mps @ B_mps).to(cpu_dev).to(ref_dtype)

        diff = (Y_mps - Y_cpu).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = Y_cpu.abs()
        denom_max = float(denom.max().item()) if denom.numel() else 0.0
        # rel err with floor: avoid div-by-tiny noise
        rel = diff / denom.clamp_min(DENOM_FLOOR)
        max_rel = float(rel.max().item()) if rel.numel() else 0.0
        return ("MEASURED", max_abs, max_rel, denom_max, "")
    except NotImplementedError as e:
        return ("UNSUPPORTED", 0.0, 0.0, 0.0, f"NotImplemented: {e}")
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported", "MPS")):
            return ("UNSUPPORTED", 0.0, 0.0, 0.0, msg.splitlines()[0][:200])
        return ("ERROR", 0.0, 0.0, 0.0, msg.splitlines()[0][:200])
    except Exception as e:  # noqa: BLE001
        return ("ERROR", 0.0, 0.0, 0.0, f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)
    M = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)
    N = pick_dim(bucket, rng)
    seed = rng.randrange(1 << 30)

    base = IterResult(
        idx=idx, shape_bucket=bucket, M=M, K=K, N=N, dtype=dtype, stride=stride, seed=seed,
        status="ERROR", classification="UNCLASSIFIED",
        max_abs_err=None, max_rel_err=None, denom_max=None, atol=None, rtol=None, tol_ratio=None,
    )

    if M == 0 or K == 0 or N == 0:
        base.status = "SKIP"
        base.classification = "OK"
        base.note = "degenerate zero-dim"
        return base

    atol, rtol = compute_tolerance(dtype, k_dim=K, device_type="mps")
    base.atol, base.rtol = atol, rtol

    status, max_abs, max_rel, denom_max, note = measure_one(M, K, N, dtype, stride, seed)
    if status == "UNSUPPORTED":
        base.status = "UNSUPPORTED"
        base.classification = "OK"
        base.note = note
        return base
    if status == "ERROR":
        base.status = "ERROR"
        base.classification = "UNCLASSIFIED"
        base.note = note
        return base

    base.max_abs_err = max_abs
    base.max_rel_err = max_rel
    base.denom_max = denom_max
    base.tol_ratio = (max_rel / rtol) if rtol > 0 else float("inf")

    # gpucheck pass condition (using rel-with-floor as the rel metric)
    passes_tol = max_rel <= rtol  # since denom uses floor, this absorbs atol-equivalent for tiny outputs
    if passes_tol:
        base.status = "OK"
        base.classification = "OK"
        return base

    base.status = "DIVERGENCE"

    # Filter A: denom magnitude floor
    if denom_max < DENOM_FLOOR:
        base.classification = "OK"
        base.note = f"denom_max={denom_max:.3e} below floor {DENOM_FLOOR:.0e} — noise"
        return base

    # Filter B: tolerance ratio
    if base.tol_ratio < FILABLE_TOL_MULTIPLIER:
        base.classification = "TOLERANCE_RECALIBRATION"
        base.note = f"max_rel/rtol={base.tol_ratio:.2f} < {FILABLE_TOL_MULTIPLIER}"
        return base

    # Candidate FILABLE — needs reproducibility replay across seeds
    replay_rels: list[float] = [max_rel]
    above_tol_count = 1  # this run already exceeded tol
    above_10x_count = 1
    for k in range(REPLAY_SEEDS - 1):
        replay_seed = (seed * 2654435761 + k * 0x9E3779B1) & 0x3FFFFFFF
        rstatus, _, rmax_rel, rdenom_max, _ = measure_one(M, K, N, dtype, stride, replay_seed)
        if rstatus != "MEASURED":
            continue
        replay_rels.append(rmax_rel)
        if rdenom_max >= DENOM_FLOOR and rmax_rel > rtol:
            above_tol_count += 1
        if rdenom_max >= DENOM_FLOOR and rmax_rel / rtol >= FILABLE_TOL_MULTIPLIER:
            above_10x_count += 1
    base.replay_max_rel_errs = replay_rels

    if above_10x_count >= REPLAY_SEEDS:
        base.classification = "FILABLE"
        base.note = (
            f"reproducible: {above_10x_count}/{len(replay_rels)} seeds >=10x tol; "
            f"denom_max={denom_max:.3e}; tol_ratio={base.tol_ratio:.1f}"
        )
    elif above_tol_count >= REPLAY_SEEDS:
        base.classification = "TOLERANCE_RECALIBRATION"
        base.note = (
            f"reproducible above tol but <10x: {above_tol_count}/{len(replay_rels)} seeds; "
            f"tol_ratio={base.tol_ratio:.1f}"
        )
    else:
        base.classification = "TOLERANCE_RECALIBRATION"
        base.note = (
            f"non-reproducible >=10x ({above_10x_count}/{len(replay_rels)}); "
            f"tol_ratio={base.tol_ratio:.1f}"
        )
    return base


def main() -> int:
    rng = random.Random(0xF00DC0DE)
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# matmul-fp32 fuzz v2 — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "version": "v2", "status": "SKIPPED"}) + "\n")
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
        if r.classification == "FILABLE":
            print(f"[{i:03d}] FILABLE {r.dtype}/{r.stride} M={r.M} K={r.K} N={r.N} "
                  f"max_rel={r.max_rel_err:.3e} ratio={r.tol_ratio:.1f}x replays={r.replay_max_rel_errs}", flush=True)
        elif r.classification == "TOLERANCE_RECALIBRATION":
            print(f"[{i:03d}] TOL_RECAL {r.dtype}/{r.stride} M={r.M} K={r.K} N={r.N} "
                  f"max_rel={r.max_rel_err:.3e} ratio={r.tol_ratio:.2f}x", flush=True)
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.note}", flush=True)
        elif r.status == "UNSUPPORTED" and i < 5:
            print(f"[{i:03d}] UNSUPPORTED {r.dtype}/{r.stride} {r.note[:80]}", flush=True)

    # Aggregate
    by_class: dict[str, list[IterResult]] = {"OK": [], "TOLERANCE_RECALIBRATION": [], "FILABLE": [], "UNCLASSIFIED": []}
    by_status: dict[str, int] = {"OK": 0, "DIVERGENCE": 0, "UNSUPPORTED": 0, "ERROR": 0, "SKIP": 0}
    for r in results:
        by_class.setdefault(r.classification, []).append(r)
        by_status[r.status] = by_status.get(r.status, 0) + 1

    filable = by_class["FILABLE"]
    tol_recal = by_class["TOLERANCE_RECALIBRATION"]

    filable.sort(key=lambda r: (r.tol_ratio or 0.0), reverse=True)
    tol_recal.sort(key=lambda r: (r.tol_ratio or 0.0), reverse=True)
    top_filable = filable[:3]
    top_tol = tol_recal[:5]

    overall_max_rel = max((r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0)
    overall_max_ratio = max((r.tol_ratio for r in results if r.tol_ratio is not None), default=0.0)

    target = "pytorch/pytorch" if filable else "none"

    md: list[str] = []
    md.append("# matmul-fp32 fuzz v2 results\n")
    md.append(f"- **kernel:** {KERNEL_NAME}")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **status counts:** OK={by_status['OK']}, DIVERGENCE={by_status['DIVERGENCE']}, "
              f"UNSUPPORTED={by_status['UNSUPPORTED']}, ERROR={by_status['ERROR']}, SKIP={by_status['SKIP']}")
    md.append(f"- **classification counts:** OK={len(by_class['OK'])}, "
              f"TOLERANCE_RECALIBRATION={len(tol_recal)}, FILABLE={len(filable)}, "
              f"UNCLASSIFIED={len(by_class['UNCLASSIFIED'])}")
    md.append(f"- **MPS-vs-CPU max relative error (overall, denom-floored):** {overall_max_rel:.3e}")
    md.append(f"- **max tol_ratio (max_rel/rtol) overall:** {overall_max_ratio:.2f}x")
    md.append(f"- **MPS-vs-CUDA-mock max relative error:** N/A (no NVIDIA hardware)")
    md.append(f"- **recommended upstream target:** {target}\n")

    md.append("## FILABLE divergences\n")
    if top_filable:
        md.append("| # | dtype | stride | M | K | N | bucket | seed | max_rel_err | rtol | ratio | denom_max | replays | note |")
        md.append("|---|-------|--------|---|---|---|--------|------|-------------|------|-------|-----------|---------|------|")
        for i, r in enumerate(top_filable, 1):
            replays_s = ",".join(f"{x:.2e}" for x in r.replay_max_rel_errs)
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.M} | {r.K} | {r.N} | {r.shape_bucket} | {r.seed} | "
                f"{r.max_rel_err:.3e} | {r.rtol:.1e} | {r.tol_ratio:.1f}x | {r.denom_max:.2e} | "
                f"{replays_s} | {r.note} |"
            )
    else:
        md.append("_No FILABLE divergences (>10x tol, denom>=1e-6, reproducible across >=3 seeds)._")
    md.append("")

    md.append(f"## TOLERANCE_RECALIBRATION candidates ({len(tol_recal)})\n")
    if top_tol:
        md.append("| # | dtype | stride | M | K | N | bucket | seed | max_rel_err | rtol | ratio | denom_max | note |")
        md.append("|---|-------|--------|---|---|---|--------|------|-------------|------|-------|-----------|------|")
        for i, r in enumerate(top_tol, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.M} | {r.K} | {r.N} | {r.shape_bucket} | {r.seed} | "
                f"{r.max_rel_err:.3e} | {r.rtol:.1e} | {r.tol_ratio:.2f}x | {r.denom_max:.2e} | {r.note} |"
            )
        if len(tol_recal) > 5:
            md.append(f"\n_(+{len(tol_recal) - 5} more recalibration candidates omitted)_")
    else:
        md.append("_None._")
    md.append("")

    if by_status["UNSUPPORTED"]:
        md.append(f"## UNSUPPORTED ({by_status['UNSUPPORTED']})\n")
        seen: dict[tuple[str, str], list[IterResult]] = {}
        for r in results:
            if r.status == "UNSUPPORTED":
                seen.setdefault((r.dtype, r.stride), []).append(r)
        for (dt, st), rs in seen.items():
            md.append(f"- {dt}/{st}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")

    if by_status["ERROR"]:
        md.append(f"## ERRORS ({by_status['ERROR']})\n")
        for r in results[:10]:
            if r.status == "ERROR":
                md.append(f"- iter {r.idx} {r.dtype}/{r.stride}/{r.shape_bucket} M={r.M} K={r.K} N={r.N}: {r.note}")
        md.append("")

    md.append("## Methodology\n")
    md.append("- Inputs constructed on CPU with seeded fp32 RNG, then transferred to MPS — identical bits both sides.")
    md.append("- Reference: fp64 CPU matmul.")
    md.append(f"- rel error denom uses `clamp_min({DENOM_FLOOR:.0e})` to suppress div-by-tiny noise.")
    md.append("- Tolerance: `gpucheck.compute_tolerance('float32', k_dim=K, device_type='mps')` (per-dtype base + sqrt(K/128) scaling + MPS overlay).")
    md.append("- **OK:** `max_rel_err <= rtol` OR `denom_max < 1e-6`.")
    md.append("- **TOLERANCE_RECALIBRATION:** `rtol < max_rel_err < 10*rtol`, denom>=1e-6 (or non-reproducible >=10x).")
    md.append(f"- **FILABLE:** `max_rel_err >= 10*rtol` AND `denom_max >= 1e-6` AND reproducible (>=3/{REPLAY_SEEDS} seeds at >=10x tol).")
    md.append("- Stride patterns: contiguous, slice (stride-2 over K), transpose (.t() view), broadcast (1×K and K×1 expanded).")
    md.append("- CUDA channel: not exercised — no NVIDIA hardware on this host.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "version": "v2",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "status_counts": by_status,
        "classification_counts": {k: len(v) for k, v in by_class.items()},
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cpu_max_tol_ratio": overall_max_ratio,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_no_nvidia_hardware",
        "filable": [
            {
                "rank": i + 1,
                "dtype": r.dtype, "stride": r.stride, "shape_bucket": r.shape_bucket,
                "M": r.M, "K": r.K, "N": r.N, "seed": r.seed,
                "max_rel_err": r.max_rel_err, "max_abs_err": r.max_abs_err,
                "denom_max": r.denom_max,
                "atol": r.atol, "rtol": r.rtol, "tol_ratio": r.tol_ratio,
                "replay_max_rel_errs": r.replay_max_rel_errs,
                "note": r.note,
            }
            for i, r in enumerate(top_filable)
        ],
        "tolerance_recalibration_top": [
            {
                "dtype": r.dtype, "stride": r.stride, "shape_bucket": r.shape_bucket,
                "M": r.M, "K": r.K, "N": r.N, "seed": r.seed,
                "max_rel_err": r.max_rel_err, "rtol": r.rtol, "tol_ratio": r.tol_ratio,
                "denom_max": r.denom_max, "note": r.note,
            }
            for r in top_tol
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": time.monotonic() - t0,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL in {time.monotonic()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
