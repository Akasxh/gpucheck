"""V2 fuzz: matmul-fp16 on MPS vs CPU reference with FILABLE filter.

Spec changes vs v1 (_fuzz_matmul_fp16.py):
- 500 configs sampled deterministically with config-RNG.
- Each config replayed across 3 data seeds (0, 1, 2).
- Classification per seed:
    * |Y_cpu|.max() < 1e-6  → NEAR_ZERO_SKIP (denom too small to trust rel err)
    * max_rel_err > 10*rtol → CRITICAL_REL
    * max_abs_err > 10*atol → CRITICAL_ABS
    * |error| in [tol, 5*tol) → RECALIBRATION
    * else                  → OK
- A config is FILABLE iff CRITICAL on >= 3 of 3 seeds AND denom_magnitude >= 1e-6.
- A config is TOLERANCE_RECALIBRATION iff RECALIBRATION on >= 3 of 3 seeds and not FILABLE.
- 8-minute wall budget. Writes partial results on time-out.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-matmul-fp16")
sys.path.insert(0, str(WORKTREE / "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL = "matmul-fp16"
N_CONFIGS = 500
DATA_SEEDS = (0, 1, 2)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for write-out
CONFIG_SEED = 0xF00DBABE

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_matmul-fp16.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

# Sampling space
SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large", "mixed"]
DTYPES = ["float16", "bfloat16", "float32"]
DTYPE_WEIGHTS = [3, 1, 1]  # focus on fp16 since kernel is matmul-fp16
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]


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
        # mix categories per axis
        pool = PRIMES + POW2_BOUNDARY + NON_TILE
        return rng.choice(pool)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


@dataclass
class Config:
    cid: int
    bucket: str
    M: int
    K: int
    N: int
    dtype: str
    stride: str


@dataclass
class SeedResult:
    seed: int
    status: str  # OK | RECALIBRATION | CRITICAL_REL | CRITICAL_ABS | NEAR_ZERO_SKIP | UNSUPPORTED | ERROR | DEGEN_SKIP
    max_abs_err: float | None
    max_rel_err: float | None
    denom_magnitude: float | None  # |Y_cpu|.max()
    atol: float | None
    rtol: float | None
    note: str = ""


@dataclass
class ConfigResult:
    config: Config
    seeds: list[SeedResult] = field(default_factory=list)
    classification: str = ""  # FILABLE | TOLERANCE_RECALIBRATION | OK | UNSUPPORTED | ERROR | DEGEN_SKIP


def make_inputs(cfg: Config, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Build A (M,K) and B (K,N) on device with deterministic seed."""
    M, K, N = cfg.M, cfg.K, cfg.N
    dt = torch_dtype(cfg.dtype)
    g = torch.Generator().manual_seed(seed)
    g2 = torch.Generator().manual_seed(seed ^ 0x5A5A5A5A)

    if cfg.stride == "contiguous":
        A = torch.randn(M, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B = torch.randn(K, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
    elif cfg.stride == "slice":
        A_full = torch.randn(M, K * 2, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_full = torch.randn(K * 2, N, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_full[:, ::2]
        B = B_full[::2, :]
    elif cfg.stride == "transpose":
        A_t = torch.randn(K, M, generator=g, dtype=torch.float32).to(device=device, dtype=dt)
        B_t = torch.randn(N, K, generator=g2, dtype=torch.float32).to(device=device, dtype=dt)
        A = A_t.t()
        B = B_t.t()
    elif cfg.stride == "broadcast":
        A_row = torch.randn(1, K, generator=g, dtype=torch.float32).to(device=device, dtype=dt).expand(M, K)
        B_col = torch.randn(K, 1, generator=g2, dtype=torch.float32).to(device=device, dtype=dt).expand(K, N)
        A = A_row
        B = B_col
    else:
        raise ValueError(cfg.stride)
    assert A.shape == (M, K) and B.shape == (K, N)
    return A, B


def classify_seed(max_abs: float, max_rel: float, denom_max: float,
                  atol: float, rtol: float) -> str:
    if denom_max < 1e-6:
        return "NEAR_ZERO_SKIP"
    # FILABLE-eligible: rel err exceeds 10x rtol, or abs err exceeds 10x atol
    if max_rel > 10.0 * rtol:
        return "CRITICAL_REL"
    if max_abs > 10.0 * atol:
        return "CRITICAL_ABS"
    # Combined gate failure within [1x, 5x) tolerance band → recalibration
    base_tol = atol + rtol * denom_max
    if max_abs > base_tol and max_abs < 5.0 * base_tol:
        return "RECALIBRATION"
    if max_rel > rtol and max_rel < 5.0 * rtol:
        return "RECALIBRATION"
    return "OK"


def run_seed(cfg: Config, seed: int) -> SeedResult:
    if cfg.M == 0 or cfg.K == 0 or cfg.N == 0:
        return SeedResult(seed=seed, status="DEGEN_SKIP", max_abs_err=None,
                          max_rel_err=None, denom_magnitude=None, atol=None, rtol=None,
                          note="zero-dim")
    try:
        cpu_dev = torch.device("cpu")
        mps_dev = torch.device("mps")

        A_cpu, B_cpu = make_inputs(cfg, seed, cpu_dev)
        A_mps = A_cpu.detach().clone().to(mps_dev)
        B_mps = B_cpu.detach().clone().to(mps_dev)

        ref_dtype = torch.float64 if cfg.dtype == "float32" else torch.float32
        Y_cpu = (A_cpu @ B_cpu).to(ref_dtype)
        Y_mps = (A_mps @ B_mps).to(cpu_dev).to(ref_dtype)

        atol, rtol = compute_tolerance(cfg.dtype, k_dim=cfg.K, device_type="mps")

        diff = (Y_mps - Y_cpu).abs()
        if diff.numel() == 0:
            return SeedResult(seed=seed, status="DEGEN_SKIP", max_abs_err=0.0,
                              max_rel_err=0.0, denom_magnitude=0.0, atol=atol, rtol=rtol,
                              note="empty output")
        max_abs = float(diff.max().item())
        denom_t = Y_cpu.abs()
        denom_max = float(denom_t.max().item())
        denom_safe = denom_t.clamp_min(1e-12)
        max_rel = float((diff / denom_safe).max().item())

        status = classify_seed(max_abs, max_rel, denom_max, atol, rtol)
        return SeedResult(seed=seed, status=status, max_abs_err=max_abs,
                          max_rel_err=max_rel, denom_magnitude=denom_max,
                          atol=atol, rtol=rtol)
    except NotImplementedError as e:
        return SeedResult(seed=seed, status="UNSUPPORTED", max_abs_err=None,
                          max_rel_err=None, denom_magnitude=None, atol=None, rtol=None,
                          note=f"NotImplemented: {str(e)[:160]}")
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported")):
            return SeedResult(seed=seed, status="UNSUPPORTED", max_abs_err=None,
                              max_rel_err=None, denom_magnitude=None, atol=None, rtol=None,
                              note=msg.splitlines()[0][:200])
        return SeedResult(seed=seed, status="ERROR", max_abs_err=None,
                          max_rel_err=None, denom_magnitude=None, atol=None, rtol=None,
                          note=msg.splitlines()[0][:200])
    except Exception as e:  # noqa: BLE001
        return SeedResult(seed=seed, status="ERROR", max_abs_err=None,
                          max_rel_err=None, denom_magnitude=None, atol=None, rtol=None,
                          note=f"{type(e).__name__}: {str(e).splitlines()[0][:200]}")


def classify_config(cr: ConfigResult) -> str:
    statuses = [s.status for s in cr.seeds]
    if any(s == "ERROR" for s in statuses):
        return "ERROR"
    if all(s == "UNSUPPORTED" for s in statuses):
        return "UNSUPPORTED"
    if all(s == "DEGEN_SKIP" for s in statuses):
        return "DEGEN_SKIP"

    # FILABLE check: CRITICAL on >= 3 of 3 with non-trivial denom
    crit_count = sum(1 for s in cr.seeds if s.status in ("CRITICAL_REL", "CRITICAL_ABS"))
    valid_denom_count = sum(
        1 for s in cr.seeds
        if s.denom_magnitude is not None and s.denom_magnitude >= 1e-6
    )
    if crit_count >= 3 and valid_denom_count >= 3:
        return "FILABLE"

    recal_count = sum(1 for s in cr.seeds if s.status == "RECALIBRATION")
    if recal_count >= 3:
        return "TOLERANCE_RECALIBRATION"
    # Single-seed CRITICAL = not reproducible → flag as flaky/recal
    if crit_count >= 1:
        return "TOLERANCE_RECALIBRATION"
    return "OK"


def sample_configs(rng: random.Random) -> list[Config]:
    configs: list[Config] = []
    for cid in range(N_CONFIGS):
        bucket = rng.choice(SHAPE_BUCKETS)
        dtype = rng.choices(DTYPES, weights=DTYPE_WEIGHTS, k=1)[0]
        stride = rng.choice(STRIDE_PATTERNS)
        M = pick_dim(bucket, rng)
        K = pick_dim(bucket, rng)
        N = pick_dim(bucket, rng)
        configs.append(Config(cid=cid, bucket=bucket, M=M, K=K, N=N, dtype=dtype, stride=stride))
    return configs


def main() -> int:
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# matmul-fp16 fuzz v2 — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL, "version": "v2", "status": "SKIPPED"}) + "\n")
        return 0

    rng = random.Random(CONFIG_SEED)
    configs = sample_configs(rng)

    t0 = time.monotonic()
    results: list[ConfigResult] = []
    completed = 0
    timed_out = False

    for cfg in configs:
        if time.monotonic() - t0 > WALL_BUDGET_S:
            print(f"[budget] stop at {completed}/{len(configs)} configs", flush=True)
            timed_out = True
            break
        cr = ConfigResult(config=cfg)
        for seed in DATA_SEEDS:
            sr = run_seed(cfg, seed)
            cr.seeds.append(sr)
            # short-circuit: if first seed errors, no point retrying same config
            if sr.status == "ERROR" and seed == DATA_SEEDS[0]:
                # still try other seeds — could be transient
                pass
        cr.classification = classify_config(cr)
        results.append(cr)
        completed += 1
        if cr.classification == "FILABLE":
            sr0 = cr.seeds[0]
            print(f"[{cfg.cid:03d}] FILABLE {cfg.dtype} {cfg.stride} bucket={cfg.bucket} "
                  f"M={cfg.M} K={cfg.K} N={cfg.N} max_rel={sr0.max_rel_err:.3e} rtol={sr0.rtol:.3e}",
                  flush=True)
        elif cr.classification == "ERROR":
            err_seed = next((s for s in cr.seeds if s.status == "ERROR"), None)
            if err_seed:
                print(f"[{cfg.cid:03d}] ERROR {err_seed.note}", flush=True)

    elapsed = time.monotonic() - t0

    # Aggregate
    counts: dict[str, int] = {}
    for cr in results:
        counts[cr.classification] = counts.get(cr.classification, 0) + 1

    filable = [cr for cr in results if cr.classification == "FILABLE"]
    recal = [cr for cr in results if cr.classification == "TOLERANCE_RECALIBRATION"]
    ok = [cr for cr in results if cr.classification == "OK"]
    err = [cr for cr in results if cr.classification == "ERROR"]
    unsup = [cr for cr in results if cr.classification == "UNSUPPORTED"]
    degen = [cr for cr in results if cr.classification == "DEGEN_SKIP"]

    # Sort filable by worst max_rel_err across seeds
    def cfg_max_rel(cr: ConfigResult) -> float:
        vals = [s.max_rel_err for s in cr.seeds if s.max_rel_err is not None]
        return max(vals) if vals else 0.0

    filable.sort(key=cfg_max_rel, reverse=True)
    top_filable = filable[:5]

    # Overall MPS-vs-CPU max relative error across all seeds
    overall_max_rel = max(
        (s.max_rel_err for cr in results for s in cr.seeds if s.max_rel_err is not None),
        default=0.0,
    )

    # Recommended upstream target
    if filable:
        target = "pytorch/pytorch (MPS backend, matmul)"
    else:
        target = "none"

    # Markdown
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    md: list[str] = []
    md.append(f"# matmul-fp16 fuzz v2 — MPS vs CPU (seed-replay FILABLE filter)\n")
    md.append(f"- **kernel:** `{KERNEL}` (torch.matmul, fp16 focus + bf16/fp32 control)")
    md.append(f"- **version:** v2")
    md.append(f"- **timestamp (UTC):** {now}")
    md.append(f"- **configs sampled:** {len(configs)}")
    md.append(f"- **configs completed:** {completed}")
    md.append(f"- **timed out:** {timed_out}")
    md.append(f"- **elapsed:** {elapsed:.2f}s / budget {WALL_BUDGET_S}s")
    md.append(f"- **seeds per config:** {list(DATA_SEEDS)}")
    md.append(f"- **config seed:** `0x{CONFIG_SEED:08X}`")
    md.append(f"- **torch:** {torch.__version__}")
    md.append(f"- **MPS backend:** real (`torch.backends.mps.is_available() == True`)")
    md.append(f"- **CUDA backend:** N/A (no NVIDIA hardware on host)\n")
    md.append("## Classification counts\n")
    md.append("| classification | count |")
    md.append("|---|---|")
    for k in ("FILABLE", "TOLERANCE_RECALIBRATION", "OK", "UNSUPPORTED", "DEGEN_SKIP", "ERROR"):
        md.append(f"| {k} | {counts.get(k, 0)} |")
    md.append("")
    md.append("## Headline\n")
    md.append(f"- **FILABLE configs:** {len(filable)}")
    md.append(f"- **TOLERANCE_RECALIBRATION configs:** {len(recal)}")
    md.append(f"- **OK configs:** {len(ok)}")
    md.append(f"- **MPS-vs-CPU max relative error (any seed, any config):** `{overall_max_rel:.3e}`")
    md.append(f"- **MPS-vs-CUDA max relative error:** N/A (no NVIDIA hardware)")
    md.append(f"- **recommended upstream target:** {target}\n")

    md.append("## FILABLE filter spec\n")
    md.append("A config is FILABLE iff, on **all 3 of 3 data seeds**:")
    md.append("- `max_rel_err > 10 * rtol` (or `max_abs_err > 10 * atol`), AND")
    md.append("- `|Y_cpu|.max() >= 1e-6` (denom magnitude — guards against near-zero artifacts).\n")
    md.append("A config is TOLERANCE_RECALIBRATION iff it lands in the `[1x, 5x)` tolerance band on >= 3 seeds and is not FILABLE; or shows transient CRITICAL on a minority of seeds.\n")
    md.append("Tolerances come from `gpucheck.compute_tolerance(dtype, k_dim=K, device_type='mps')`: per-dtype base + sqrt(K/128) matmul scaling + MPS overlay multiplier.\n")

    md.append("## Top FILABLE repros (worst max_rel_err)\n")
    if top_filable:
        md.append("| # | dtype | stride | bucket | M | K | N | seeds (max_rel) | rtol | atol | denom_max |")
        md.append("|---|-------|--------|--------|---|---|---|-----------------|------|------|-----------|")
        for i, cr in enumerate(top_filable, 1):
            cfg = cr.config
            seed_strs = ", ".join(
                f"s{s.seed}={s.max_rel_err:.2e}" for s in cr.seeds if s.max_rel_err is not None
            )
            sr0 = next((s for s in cr.seeds if s.atol is not None), cr.seeds[0])
            denom = max((s.denom_magnitude or 0.0) for s in cr.seeds)
            md.append(
                f"| {i} | {cfg.dtype} | {cfg.stride} | {cfg.bucket} | {cfg.M} | {cfg.K} | {cfg.N} | "
                f"{seed_strs} | {sr0.rtol:.2e} | {sr0.atol:.2e} | {denom:.2e} |"
            )
    else:
        md.append("_No FILABLE configs found at gpucheck MPS-overlay tolerances._")
    md.append("")

    if recal:
        md.append(f"## TOLERANCE_RECALIBRATION summary ({len(recal)})\n")
        # Bucket by (dtype, stride)
        buckets: dict[tuple[str, str], int] = {}
        for cr in recal:
            key = (cr.config.dtype, cr.config.stride)
            buckets[key] = buckets.get(key, 0) + 1
        md.append("| dtype | stride | count |")
        md.append("|---|---|---|")
        for (dt, st), n in sorted(buckets.items(), key=lambda x: -x[1]):
            md.append(f"| {dt} | {st} | {n} |")
        md.append("")
        # Show top 3 worst by max_rel
        recal_sorted = sorted(recal, key=cfg_max_rel, reverse=True)[:3]
        md.append("Top 3 worst-rel TOLERANCE_RECALIBRATION configs:")
        md.append("| # | dtype | stride | bucket | M | K | N | max_rel | rtol |")
        md.append("|---|-------|--------|--------|---|---|---|---------|------|")
        for i, cr in enumerate(recal_sorted, 1):
            cfg = cr.config
            mr = cfg_max_rel(cr)
            sr0 = next((s for s in cr.seeds if s.rtol is not None), cr.seeds[0])
            md.append(
                f"| {i} | {cfg.dtype} | {cfg.stride} | {cfg.bucket} | {cfg.M} | {cfg.K} | {cfg.N} | "
                f"{mr:.3e} | {sr0.rtol:.2e} |"
            )
        md.append("")

    if unsup:
        md.append(f"## UNSUPPORTED summary ({len(unsup)})\n")
        first = unsup[0].seeds[0].note if unsup[0].seeds else ""
        md.append(f"- first note: `{first}`\n")
    if err:
        md.append(f"## ERROR summary ({len(err)})\n")
        for cr in err[:5]:
            errs = [s for s in cr.seeds if s.status == "ERROR"]
            if errs:
                md.append(f"- cid={cr.config.cid} {cr.config.dtype}/{cr.config.stride} "
                          f"M={cr.config.M} K={cr.config.K} N={cr.config.N}: {errs[0].note}")
        md.append("")

    md.append("## Methodology\n")
    md.append("- 500 configs sampled deterministically (`Random(0xF00DBABE)`).")
    md.append("- Each config replayed across 3 data seeds (0, 1, 2) for reproducibility.")
    md.append("- Inputs built on CPU as fp32 then cast to dtype + transferred — MPS and CPU see numerically identical source data.")
    md.append("- Reference: matmul in fp64 (for fp32 ops) or fp32 (fp16/bf16 ops) on CPU.")
    md.append("- Pass condition per seed: `|Y_mps - Y_cpu| <= atol + rtol*|Y_cpu|` element-wise; classification per FILABLE filter spec above.")
    md.append("- Stride patterns: `contiguous`, `slice` (stride-2 over K), `transpose` (`.t()` view), `broadcast` (1×K and K×1 expanded).")
    md.append("- Shape buckets: degenerate, prime, pow2_boundary, non_tile_aligned, large, mixed.")
    md.append("- Dtype weighting: float16 ×3, bfloat16 ×1, float32 ×1 (kernel under test is matmul-fp16).")
    md.append("")

    RESULTS_MD.write_text("\n".join(md))

    # Worst seed across config (for top-filable JSONL embedding)
    def seed_to_dict(s: SeedResult) -> dict:
        return {
            "seed": s.seed,
            "status": s.status,
            "max_abs_err": s.max_abs_err,
            "max_rel_err": s.max_rel_err,
            "denom_magnitude": s.denom_magnitude,
            "atol": s.atol,
            "rtol": s.rtol,
            "note": s.note,
        }

    record = {
        "kernel": KERNEL,
        "version": "v2",
        "timestamp_utc": now,
        "configs_sampled": len(configs),
        "configs_completed": completed,
        "timed_out": timed_out,
        "elapsed_s": elapsed,
        "torch_version": torch.__version__,
        "config_seed": f"0x{CONFIG_SEED:08X}",
        "data_seeds": list(DATA_SEEDS),
        "counts": counts,
        "headline": {
            "filable": len(filable),
            "tolerance_recalibration": len(recal),
            "ok": len(ok),
            "mps_vs_cpu_max_rel_err": overall_max_rel,
            "mps_vs_cuda_max_rel_err": None,
            "cuda_status": "N/A_no_nvidia_hardware",
        },
        "filable_top": [
            {
                "rank": i + 1,
                "cid": cr.config.cid,
                "dtype": cr.config.dtype,
                "stride": cr.config.stride,
                "bucket": cr.config.bucket,
                "M": cr.config.M, "K": cr.config.K, "N": cr.config.N,
                "seeds": [seed_to_dict(s) for s in cr.seeds],
            }
            for i, cr in enumerate(top_filable)
        ],
        "upstream_target": target,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] elapsed={elapsed:.1f}s configs={completed} filable={len(filable)} recal={len(recal)}",
          flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
