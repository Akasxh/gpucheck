"""Fuzzer for bmm-fp16 across MPS (real) and CPU (reference).

Drives gpucheck's per-dtype tolerance + sqrt(k/128) matmul scaling. CUDA path
is N/A on this Mac (no NVIDIA hardware) — we only validate MPS-vs-CPU divergence.

torch.bmm: (B,M,K) @ (B,K,N) -> (B,M,N). Same accumulation depth as matmul,
so we drive the K-scaling tolerance via gpucheck.compute_tolerance(k_dim=K).

Classification (per agent spec):
- FILABLE: max_rel_err > 10x tol AND denom_magnitude >= 1e-6 AND reproduces on >=3 of 5 seeds
- TOLERANCE_RECALIBRATION: any divergence not meeting the FILABLE bar
- OK: passes the gpucheck tolerance gate
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-bmm-fp16"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "bmm-fp16"
AGENT_NAME = "kernel-fuzzer-bmm-fp16-v2"
N_CONFIGS = 100
N_SEEDS = 5  # 100 * 5 = 500 iterations
SEEDS = [0, 1, 2, 3, 4]
BUDGET_S = 8 * 60 - 30
DENOM_FLOOR = 1e-6

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_bmm-fp16.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large", "mixed"]
DTYPES = ["float32", "float16", "bfloat16"]
# 7 canonical gpucheck stride categories
STRIDE_CATS = [
    "row_major",
    "column_major",
    "broadcast",
    "transpose",
    "slice",
    "non_contig",
    "gather",
]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]
BATCH_BUCKETS = {
    "degenerate": [1, 2],
    "prime": [3, 5, 7],
    "pow2_boundary": [1, 2, 4, 8, 16, 32],
    "non_tile_aligned": [3, 5, 6, 9],
    "large": [16, 32, 64],
    "mixed": [1, 2, 4, 8],
}


@dataclass
class ConfigKey:
    B: int
    M: int
    K: int
    N: int
    dtype: str
    stride_cat: str
    bucket: str

    def key(self) -> str:
        return f"({self.B},{self.M},{self.K},{self.N})|{self.dtype}|{self.stride_cat}"


@dataclass
class SeedRun:
    seed: int
    max_abs_err: float
    max_rel_err: float
    denom_at_max_rel: float
    atol: float
    rtol: float
    factor: float  # max_rel_err / max(atol/denom + rtol, eps)
    status: str  # OK | FILABLE_HIT | RECAL | UNSUPPORTED | ERROR | SKIP
    note: str = ""


@dataclass
class ConfigResult:
    cfg: ConfigKey
    seed_runs: list[SeedRun] = field(default_factory=list)


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
    if bucket == "mixed":
        # mix of buckets
        sub = rng.choice(["prime", "pow2_boundary", "non_tile_aligned"])
        return pick_dim(sub, rng)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_bmm_inputs(
    cfg: ConfigKey, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct A (B,M,K) and B (B,K,N) on `device` with the requested stride pattern.

    Backed by a single deterministic generator → identical numbers across CPU and MPS.
    """
    dt = torch_dtype(cfg.dtype)
    B, M, K, N = cfg.B, cfg.M, cfg.K, cfg.N
    g = torch.Generator().manual_seed(seed)
    cat = cfg.stride_cat

    # Default — contiguous row-major.
    if cat == "row_major":
        A = torch.randn(B, M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm = torch.randn(B, K, N, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
    elif cat == "column_major":
        # column-major within the trailing 2D slabs
        A_t = torch.randn(B, K, M, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_t = torch.randn(B, N, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.transpose(-1, -2).contiguous().transpose(-1, -2)  # column-major view
        Bm = Bm_t.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif cat == "transpose":
        A_t = torch.randn(B, K, M, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_t = torch.randn(B, N, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.transpose(-1, -2)  # non-contig view, shape (B,M,K)
        Bm = Bm_t.transpose(-1, -2)  # shape (B,K,N)
    elif cat == "slice":
        A_full = torch.randn(B, M, K * 2, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_full = torch.randn(B, K * 2, N, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[:, :, ::2]
        Bm = Bm_full[:, ::2, :]
    elif cat == "broadcast":
        # broadcast batch dim: A (1,M,K) → expanded (B,M,K); B (1,K,N) → expanded (B,K,N)
        A_one = torch.randn(1, M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_one = torch.randn(1, K, N, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_one.expand(B, M, K)
        Bm = Bm_one.expand(B, K, N)
    elif cat == "non_contig":
        # over-allocate then take a non-contig sub-view via permute
        A_p = torch.randn(M, B, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_p = torch.randn(K, B, N, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_p.permute(1, 0, 2)  # (B,M,K), strides non-canonical
        Bm = Bm_p.permute(1, 0, 2)
    elif cat == "gather":
        # gather-induced layout via index_select on K axis
        A_pad = torch.randn(B, M, K + 4, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        Bm_pad = torch.randn(B, K + 4, N, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        # use a contiguous index range so semantics match a slice but the op path differs
        idx = torch.arange(K, device=device)
        A = A_pad.index_select(-1, idx)
        Bm = Bm_pad.index_select(-2, idx)
    else:
        raise ValueError(cat)

    assert A.shape == (B, M, K), f"A shape {A.shape} != ({B},{M},{K})"
    assert Bm.shape == (B, K, N), f"Bm shape {Bm.shape} != ({B},{K},{N})"
    return A, Bm


def run_one(cfg: ConfigKey, seed: int) -> SeedRun:
    base = SeedRun(
        seed=seed, max_abs_err=0.0, max_rel_err=0.0, denom_at_max_rel=0.0,
        atol=0.0, rtol=0.0, factor=0.0, status="ERROR", note="",
    )
    if 0 in (cfg.B, cfg.M, cfg.K, cfg.N):
        base.status = "SKIP"
        base.note = "zero-dim"
        return base
    try:
        cpu_dev = torch.device("cpu")
        mps_dev = torch.device("mps")

        A_cpu, B_cpu = make_bmm_inputs(cfg, seed, cpu_dev)
        A_mps, B_mps = make_bmm_inputs(cfg, seed, mps_dev)

        ref_dtype = torch.float64 if cfg.dtype == "float32" else torch.float32
        Y_cpu = torch.bmm(A_cpu, B_cpu).to(ref_dtype)
        Y_mps = torch.bmm(A_mps, B_mps).to(cpu_dev).to(ref_dtype)

        atol, rtol = compute_tolerance(cfg.dtype, k_dim=cfg.K, device_type="mps")
        base.atol, base.rtol = atol, rtol

        diff = (Y_mps - Y_cpu).abs()
        denom = Y_cpu.abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        # locate the element with the worst relative error using a denom floor
        denom_clamped = denom.clamp_min(DENOM_FLOOR)
        rel = diff / denom_clamped
        rel_max_idx = int(rel.argmax().item())
        rel_flat = rel.reshape(-1)
        denom_flat = denom.reshape(-1)
        max_rel = float(rel_flat[rel_max_idx].item()) if rel_flat.numel() else 0.0
        denom_at = float(denom_flat[rel_max_idx].item()) if denom_flat.numel() else 0.0

        base.max_abs_err = max_abs
        base.max_rel_err = max_rel
        base.denom_at_max_rel = denom_at

        # Effective gate at this element: atol + rtol*|y_cpu|. The "factor" is
        # how many times the gate the observed |diff| exceeds.
        gate = atol + rtol * denom_at
        observed = float(diff.reshape(-1)[rel_max_idx].item()) if diff.numel() else 0.0
        factor = (observed / gate) if gate > 0 else 0.0
        base.factor = factor

        # Pass condition: |Y_mps - Y_cpu| <= atol + rtol * |Y_cpu|
        ok = bool(((diff - (atol + rtol * denom)) <= 0).all().item())
        if ok:
            base.status = "OK"
            return base

        # Apply FILABLE-per-seed criterion: factor > 10 AND denom_at >= 1e-6
        if factor > 10.0 and denom_at >= DENOM_FLOOR:
            base.status = "FILABLE_HIT"
        else:
            base.status = "RECAL"
        return base
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e)[:160]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:160]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:160]
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
    return base


def sample_configs(rng: random.Random, n: int) -> list[ConfigKey]:
    cfgs: list[ConfigKey] = []
    seen: set[str] = set()
    attempts = 0
    while len(cfgs) < n and attempts < n * 5:
        attempts += 1
        bucket = rng.choice(SHAPE_BUCKETS)
        dtype = rng.choice(DTYPES)
        stride_cat = rng.choice(STRIDE_CATS)
        B = rng.choice(BATCH_BUCKETS[bucket])
        M = pick_dim(bucket, rng)
        K = pick_dim(bucket, rng)
        N = pick_dim(bucket, rng)
        # cap memory: B*M*K + B*K*N + B*M*N <= ~80M elements
        elements = B * (M * K + K * N + M * N)
        if elements > 80_000_000:
            continue
        cfg = ConfigKey(B=B, M=M, K=K, N=N, dtype=dtype, stride_cat=stride_cat, bucket=bucket)
        k = cfg.key()
        if k in seen:
            continue
        seen.add(k)
        cfgs.append(cfg)
    return cfgs


def classify_config(cr: ConfigResult) -> str:
    """Per agent spec: FILABLE if max_rel_err > 10x tol AND denom>=1e-6 AND repro on >=3 seeds."""
    hits = [s for s in cr.seed_runs if s.status == "FILABLE_HIT"]
    if len(hits) >= 3:
        return "FILABLE"
    any_div = [s for s in cr.seed_runs if s.status in ("FILABLE_HIT", "RECAL")]
    if any_div:
        return "TOLERANCE_RECALIBRATION"
    if all(s.status in ("UNSUPPORTED",) for s in cr.seed_runs):
        return "UNSUPPORTED"
    if all(s.status in ("ERROR",) for s in cr.seed_runs):
        return "ERROR"
    if all(s.status in ("SKIP",) for s in cr.seed_runs):
        return "SKIP"
    return "OK"


def main() -> int:
    if not torch.mps.is_available():
        with open(RESULTS_MD, "w") as f:
            f.write("# bmm-fp16 fuzz — SKIPPED (MPS unavailable)\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"agent": AGENT_NAME, "kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    rng = random.Random(0xBADBEEF1)
    cfgs = sample_configs(rng, N_CONFIGS)

    t0 = time.monotonic()
    results: list[ConfigResult] = []
    iters_completed = 0
    iters_attempted = 0
    aborted = ""

    for ci, cfg in enumerate(cfgs):
        if time.monotonic() - t0 > BUDGET_S:
            aborted = f"budget at config {ci}"
            break
        cr = ConfigResult(cfg=cfg)
        for seed in SEEDS:
            iters_attempted += 1
            if time.monotonic() - t0 > BUDGET_S:
                aborted = f"budget at config {ci} seed {seed}"
                break
            sr = run_one(cfg, seed)
            cr.seed_runs.append(sr)
            iters_completed += 1
            if sr.status == "FILABLE_HIT":
                print(f"[c{ci:03d} s{seed}] FILABLE_HIT {cfg.dtype} {cfg.stride_cat} "
                      f"B={cfg.B} M={cfg.M} K={cfg.K} N={cfg.N} factor={sr.factor:.2f}", flush=True)
            elif sr.status == "ERROR":
                print(f"[c{ci:03d} s{seed}] ERROR {sr.note}", flush=True)
        results.append(cr)
        if aborted:
            break

    elapsed = time.monotonic() - t0

    # Aggregate
    per_cfg_verdict = {classify_config(cr): 0 for cr in results}
    per_cfg_verdict.clear()
    for cr in results:
        v = classify_config(cr)
        per_cfg_verdict[v] = per_cfg_verdict.get(v, 0) + 1

    filable_cfgs = [cr for cr in results if classify_config(cr) == "FILABLE"]
    recal_cfgs = [cr for cr in results if classify_config(cr) == "TOLERANCE_RECALIBRATION"]
    ok_cfgs = [cr for cr in results if classify_config(cr) == "OK"]

    # Per-seed roll-up counters
    seed_status_counts: dict[str, int] = {}
    for cr in results:
        for sr in cr.seed_runs:
            seed_status_counts[sr.status] = seed_status_counts.get(sr.status, 0) + 1

    # Top 3 repros = highest max factor among configs with any divergence,
    # ordered by FILABLE first then by max factor.
    def cfg_max_factor(cr: ConfigResult) -> float:
        return max((s.factor for s in cr.seed_runs), default=0.0)

    div_cfgs = filable_cfgs + recal_cfgs
    div_cfgs.sort(key=lambda cr: (classify_config(cr) != "FILABLE", -cfg_max_factor(cr)))
    top3 = div_cfgs[:3]

    overall_max_abs = max(
        (s.max_abs_err for cr in results for s in cr.seed_runs if s.max_abs_err is not None),
        default=0.0,
    )
    overall_max_rel = max(
        (s.max_rel_err for cr in results for s in cr.seed_runs if s.max_rel_err is not None),
        default=0.0,
    )

    # Histograms
    per_bucket: dict[str, int] = {}
    per_dtype: dict[str, int] = {}
    per_stride: dict[str, int] = {}
    for cr in results:
        per_bucket[cr.cfg.bucket] = per_bucket.get(cr.cfg.bucket, 0) + len(cr.seed_runs)
        per_dtype[cr.cfg.dtype] = per_dtype.get(cr.cfg.dtype, 0) + len(cr.seed_runs)
        per_stride[cr.cfg.stride_cat] = per_stride.get(cr.cfg.stride_cat, 0) + len(cr.seed_runs)

    # Markdown report
    md: list[str] = []
    md.append("# bmm-fp16 fuzz results\n")
    md.append(f"- **kernel:** {KERNEL_NAME}")
    md.append(f"- **agent:** {AGENT_NAME}")
    md.append(f"- **op:** torch.bmm (B,M,K) @ (B,K,N) → (B,M,N)")
    md.append(f"- **device under test:** mps")
    md.append(f"- **reference:** cpu (fp32 ref dtype for fp16/bf16; fp64 for fp32)")
    md.append(f"- **CUDA backend:** N/A (mocked — no NVIDIA GPU on host)")
    md.append(f"- **iterations attempted:** {iters_attempted}")
    md.append(f"- **iterations completed:** {iters_completed}")
    md.append(f"- **configs:** {len(results)} (each × {N_SEEDS} seeds = {iters_completed} runs)")
    md.append(f"- **seeds:** {SEEDS}")
    md.append(f"- **denom floor:** {DENOM_FLOOR}")
    md.append(f"- **FILABLE bar:** factor > 10x tol AND denom >= {DENOM_FLOOR} AND >=3 seeds")
    md.append(f"- **divergences_filable:** {len(filable_cfgs)}")
    md.append(f"- **divergences_recalibration:** {len(recal_cfgs)}")
    md.append(f"- **OK configs:** {len(ok_cfgs)}")
    md.append(f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}")
    md.append(f"- **MPS-vs-CPU max absolute error (overall):** {overall_max_abs:.3e}")
    md.append(f"- **elapsed:** {elapsed:.2f}s (budget {BUDGET_S}s, aborted={aborted!r})")
    md.append(f"- **torch:** {torch.__version__}\n")

    md.append("## Top 3 repros (by FILABLE first, then factor)\n")
    if top3:
        md.append("| # | verdict | dtype | stride | B | M | K | N | bucket | max_rel | factor | atol | seeds_with_div |")
        md.append("|---|---------|-------|--------|---|---|---|---|--------|---------|--------|------|---------------|")
        for i, cr in enumerate(top3, 1):
            verdict = classify_config(cr)
            top_seed = max(cr.seed_runs, key=lambda s: s.factor)
            seeds_div = [s.seed for s in cr.seed_runs if s.status in ("FILABLE_HIT", "RECAL")]
            md.append(
                f"| {i} | {verdict} | {cr.cfg.dtype} | {cr.cfg.stride_cat} | {cr.cfg.B} | "
                f"{cr.cfg.M} | {cr.cfg.K} | {cr.cfg.N} | {cr.cfg.bucket} | "
                f"{top_seed.max_rel_err:.3e} | {top_seed.factor:.2f} | {top_seed.atol:.3e} | "
                f"{seeds_div} |"
            )
    else:
        md.append("_No divergences observed at gpucheck MPS-overlay tolerances._")
    md.append("")

    if filable_cfgs:
        md.append(f"## FILABLE configs ({len(filable_cfgs)})")
        for cr in filable_cfgs:
            top_seed = max(cr.seed_runs, key=lambda s: s.factor)
            seeds_hit = [s.seed for s in cr.seed_runs if s.status == "FILABLE_HIT"]
            md.append(
                f"- {cr.cfg.key()} bucket={cr.cfg.bucket} factor={top_seed.factor:.2f} "
                f"max_rel={top_seed.max_rel_err:.3e} seeds_hit={seeds_hit}"
            )
        md.append("")

    md.append("## Per-bucket / per-dtype / per-stride iteration counts")
    md.append(f"- buckets: {per_bucket}")
    md.append(f"- dtypes:  {per_dtype}")
    md.append(f"- strides: {per_stride}")
    md.append("")

    md.append("## Methodology")
    md.append("- 100 unique configs × 5 seeds = 500 iterations.")
    md.append("- Inputs constructed deterministically from the same seed on CPU and MPS so both devices see identical bits.")
    md.append("- Reference computed on CPU at the same dtype; truth tensor cast to fp32 (or fp64 for fp32 ops) before subtraction.")
    md.append("- Tolerance: `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='mps')` (base + sqrt(K/128) scaling + MPS 2× overlay).")
    md.append("- Pass gate: `|Y_mps - Y_cpu| <= atol + rtol*|Y_cpu|` element-wise.")
    md.append("- `factor` is the worst-element ratio of `|diff|` over the local gate `(atol + rtol*|y_cpu|)`.")
    md.append("- FILABLE = factor > 10 AND denom >= 1e-6 AND reproduces on >=3 of 5 seeds; otherwise TOLERANCE_RECALIBRATION; OK if no divergence.")
    md.append("- Stride categories: 7 canonical gpucheck classes (row_major, column_major, broadcast, transpose, slice, non_contig, gather) applied to (B,M,K) and (B,K,N).")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    # JSONL record
    record = {
        "agent": AGENT_NAME,
        "kernel": KERNEL_NAME,
        "op_path": "torch.bmm",
        "device_under_test": "mps",
        "reference": "cpu (fp32 ref for fp16/bf16, fp64 for fp32)",
        "cuda_backend": "mocked (no NVIDIA GPU on host)",
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "configs_run": len(results),
        "configs_planned": N_CONFIGS,
        "seeds": SEEDS,
        "denom_floor": DENOM_FLOOR,
        "filable_threshold": "factor > 10x AND denom >= 1e-6 AND seeds_hit >= 3",
        "recal_threshold": "any divergence not meeting filable bar",
        "divergences_filable": len(filable_cfgs),
        "divergences_recalibration": len(recal_cfgs),
        "ok_configs": len(ok_cfgs),
        "per_seed_status": seed_status_counts,
        "max_abs_err": overall_max_abs,
        "max_rel_err": overall_max_rel,
        "top_3_repros": [
            {
                "rank": i + 1,
                "verdict": classify_config(cr),
                "shape": [cr.cfg.B, cr.cfg.M, cr.cfg.K, cr.cfg.N],
                "dtype": cr.cfg.dtype,
                "stride_category": cr.cfg.stride_cat,
                "shape_bucket": cr.cfg.bucket,
                "max_rel_err": max(s.max_rel_err for s in cr.seed_runs),
                "max_abs_err": max(s.max_abs_err for s in cr.seed_runs),
                "max_factor": max(s.factor for s in cr.seed_runs),
                "denom_at_max_rel": max(cr.seed_runs, key=lambda s: s.factor).denom_at_max_rel,
                "atol": cr.seed_runs[0].atol,
                "rtol": cr.seed_runs[0].rtol,
                "seeds_with_divergence": [s.seed for s in cr.seed_runs if s.status in ("FILABLE_HIT", "RECAL")],
                "n_filable_seeds": sum(1 for s in cr.seed_runs if s.status == "FILABLE_HIT"),
            }
            for i, cr in enumerate(top3)
        ],
        "filable_configs": [
            {
                "config_key": cr.cfg.key(),
                "shape": [cr.cfg.B, cr.cfg.M, cr.cfg.K, cr.cfg.N],
                "dtype": cr.cfg.dtype,
                "stride_category": cr.cfg.stride_cat,
                "seeds_hit": [s.seed for s in cr.seed_runs if s.status == "FILABLE_HIT"],
                "max_factor": max(s.factor for s in cr.seed_runs),
            }
            for cr in filable_cfgs
        ],
        "per_shape_bucket": per_bucket,
        "per_dtype": per_dtype,
        "per_stride_category": per_stride,
        "elapsed_s": round(elapsed, 2),
        "wall_budget_s": BUDGET_S,
        "aborted": aborted,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "recommended_filing_target": "pytorch/pytorch" if filable_cfgs else "none",
        "results_md": RESULTS_MD,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] {iters_completed}/{iters_attempted} iters in {elapsed:.2f}s; "
          f"FILABLE={len(filable_cfgs)} RECAL={len(recal_cfgs)} OK={len(ok_cfgs)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
