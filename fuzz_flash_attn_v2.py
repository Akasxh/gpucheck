"""Fuzz flash-attention v2 (torch SDPA / MPS) — v2 with FILABLE filter.

Runs 500 iterations of stride/contiguity + shape + dtype fuzzing on the MPS
SDPA path (which dispatches to flash-attn v2 on Apple Silicon for supported
configurations) and compares to a CPU fp32 reference.

Classification:
  FILABLE                : max_rel_err > 10 * rtol AND |ref|>=1e-6 AND reproducible across >=3 distinct data seeds
  TOLERANCE_RECALIBRATION: violates atol+rtol*|ref| envelope but fails FILABLE filter (small denom or non-reproducible)
  OK                     : passes envelope
"""
from __future__ import annotations

import json
import math
import platform
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = OUT_DIR / "RESULTS_flash-attn-v2.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

AGENT = "kernel-fuzzer-flash-attn-v2-v2"
KERNEL = "flash-attn-v2"
OP_PATH = "torch.nn.functional.scaled_dot_product_attention"
CONFIG_SEED = 0xF1A54_F2
TIME_BUDGET_SEC = 8 * 60 - 30
TARGET_ITERS = 500

# Reproducibility-confirmation: 3 distinct data seeds total (1 base + 2 retries).
N_REPRO_SEEDS = 3
RELERR_FILABLE_MULT = 10.0  # max_rel_err must exceed 10x rtol budget
DENOM_FLOOR = 1e-6           # ignore divergences where |ref| < this

SHAPE_CATEGORIES = ("degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large", "mixed")
DTYPES = ("float32", "float16", "bfloat16")
STRIDE_PATTERNS = ("contiguous", "transpose", "slice", "broadcast")


def sample_shape(rng: random.Random, cat: str) -> tuple[int, int, int, int]:
    if cat == "degenerate":
        return (rng.choice([1, 2]), rng.choice([1, 2]),
                rng.choice([1, 2, 3]), rng.choice([1, 2, 4, 8]))
    if cat == "prime":
        return (rng.choice([1, 2, 3]), rng.choice([1, 3, 5, 7]),
                rng.choice([7, 11, 13, 17, 19, 23, 29, 31, 37]),
                rng.choice([7, 13, 17, 23, 31]))
    if cat == "pow2_boundary":
        return (rng.choice([1, 2, 4]), rng.choice([2, 4, 8]),
                rng.choice([8, 16, 32, 64, 128]),
                rng.choice([16, 32, 64, 128]))
    if cat == "non_tile_aligned":
        return (rng.choice([1, 2, 3]), rng.choice([3, 5, 6]),
                rng.choice([15, 33, 65, 127]),
                rng.choice([24, 40, 48, 56, 72]))
    if cat == "large":
        return (rng.choice([1, 2]), rng.choice([4, 8]),
                rng.choice([256, 384, 512]),
                rng.choice([32, 64, 128]))
    if cat == "mixed":
        return (rng.choice([1, 2, 3]), rng.choice([1, 2, 4, 6]),
                rng.choice([5, 9, 17, 33, 65, 96, 129]),
                rng.choice([8, 16, 24, 32, 48, 64]))
    raise ValueError(cat)


def make_tensor(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: torch.device,
    pattern: str,
    rng_seed: int,
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(rng_seed)
    b, h, s, d = shape
    if pattern == "contiguous":
        x = torch.randn(shape, generator=g, dtype=torch.float32)
        return x.to(dtype=dtype, device=device).contiguous()
    if pattern == "transpose":
        base = torch.randn((b, h, d, s), generator=g, dtype=torch.float32)
        return base.to(dtype=dtype, device=device).transpose(-1, -2)
    if pattern == "slice":
        big = torch.randn((b, h, s * 2, d), generator=g, dtype=torch.float32)
        big = big.to(dtype=dtype, device=device)
        return big[:, :, ::2, :]
    if pattern == "broadcast":
        # stride-0 along the head dim — exercises broadcast read on a non-reduced axis.
        base = torch.randn((b, 1, s, d), generator=g, dtype=torch.float32)
        base = base.to(dtype=dtype, device=device).contiguous()
        return base.expand(b, h, s, d)
    raise ValueError(pattern)


def run_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


def measure_errors(
    a_mps: torch.Tensor, a_ref: torch.Tensor, atol: float, rtol: float,
) -> tuple[float, float, float, int, int]:
    """Return (max_rel_err_guarded, max_abs_err, denom_at_max_rel, n_violations, n_elements).

    max_rel_err_guarded: max |a-r|/|r| over elements where |r| > DENOM_FLOOR.
    denom_at_max_rel:    |r| at the element that produced the guarded max rel err.
    n_violations:        elements failing |a-r| > atol + rtol*|r|.
    """
    a = a_mps.detach().to(device="cpu", dtype=torch.float32)
    r = a_ref.detach().to(device="cpu", dtype=torch.float32)
    diff = (a - r).abs()
    n_el = int(r.numel())
    if n_el == 0:
        return 0.0, 0.0, 0.0, 0, 0
    abs_err = float(diff.max().item())
    bound = atol + rtol * r.abs()
    n_viol = int((diff > bound).sum().item())
    mask = r.abs() > DENOM_FLOOR
    if mask.any():
        rel_vals = diff[mask] / r.abs()[mask]
        idx = int(rel_vals.argmax().item())
        rel = float(rel_vals[idx].item())
        denom = float(r.abs()[mask].view(-1)[idx].item())
    else:
        rel, denom = 0.0, 0.0
    return rel, abs_err, denom, n_viol, n_el


@dataclass
class SeedResult:
    seed_q: int
    seed_k: int
    seed_v: int
    max_rel_err: float
    max_abs_err: float
    denom_at_rel: float
    n_violations: int
    n_elements: int
    failed_envelope: bool       # nviol > 0
    filable_hit: bool           # rel > 10*rtol AND denom >= 1e-6


@dataclass
class Config:
    cid: int
    shape: tuple[int, int, int, int]
    dtype: str
    stride: str
    bucket: str
    atol: float
    rtol: float
    seed_runs: list[SeedResult] = field(default_factory=list)
    status: str = "ok"          # filable | recalibration | ok | unsupported | error
    notes: str = ""


def run_one_seed(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    stride: str,
    seeds: tuple[int, int, int],
    atol: float,
    rtol: float,
    mps: torch.device,
    cpu: torch.device,
) -> tuple[SeedResult | None, str | None]:
    sq, sk, sv = seeds
    try:
        q_mps = make_tensor(shape, dtype, mps, stride, sq)
        k_mps = make_tensor(shape, dtype, mps, stride, sk)
        v_mps = make_tensor(shape, dtype, mps, stride, sv)
        q_cpu = make_tensor(shape, torch.float32, cpu, stride, sq)
        k_cpu = make_tensor(shape, torch.float32, cpu, stride, sk)
        v_cpu = make_tensor(shape, torch.float32, cpu, stride, sv)
    except Exception as e:  # noqa: BLE001
        return None, f"build:{type(e).__name__}:{str(e)[:160]}"

    try:
        out_mps = run_sdpa(q_mps, k_mps, v_mps)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as e:
        return None, f"mps_sdpa:{type(e).__name__}:{str(e)[:160]}"

    try:
        out_cpu = run_sdpa(q_cpu, k_cpu, v_cpu)
    except (RuntimeError, NotImplementedError, TypeError) as e:
        return None, f"cpu_sdpa:{type(e).__name__}:{str(e)[:160]}"

    rel, abs_e, denom, nviol, nel = measure_errors(out_mps, out_cpu, atol, rtol)
    failed = nviol > 0
    filable_hit = rel > RELERR_FILABLE_MULT * rtol and denom >= DENOM_FLOOR
    return SeedResult(sq, sk, sv, rel, abs_e, denom, nviol, nel, failed, filable_hit), None


def main() -> int:
    if not torch.backends.mps.is_available():
        record_skipped("torch.backends.mps.is_available()==False")
        return 0

    mps = torch.device("mps")
    cpu = torch.device("cpu")
    rng = random.Random(CONFIG_SEED)
    torch.manual_seed(CONFIG_SEED)

    configs: list[Config] = []
    unsupported_reasons: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    stride_counts: Counter[str] = Counter()

    start = time.perf_counter()
    attempted = 0
    completed = 0
    aborted = ""

    for i in range(TARGET_ITERS):
        if time.perf_counter() - start > TIME_BUDGET_SEC:
            aborted = "wall_budget"
            break
        attempted += 1
        cat = rng.choice(SHAPE_CATEGORIES)
        shape = sample_shape(rng, cat)
        dtype_name = rng.choice(DTYPES)
        stride = rng.choice(STRIDE_PATTERNS)
        bucket_counts[cat] += 1
        dtype_counts[dtype_name] += 1
        stride_counts[stride] += 1
        dtype = getattr(torch, dtype_name)
        head_dim = shape[-1]
        atol, rtol = compute_tolerance(dtype, k_dim=head_dim, device_type="mps")

        cfg = Config(cid=i, shape=shape, dtype=dtype_name, stride=stride,
                     bucket=cat, atol=atol, rtol=rtol)

        # Seed #1
        seeds_1 = (rng.randint(0, 2**31 - 1), rng.randint(0, 2**31 - 1), rng.randint(0, 2**31 - 1))
        sr1, err = run_one_seed(shape, dtype, stride, seeds_1, atol, rtol, mps, cpu)
        if sr1 is None:
            cfg.status = "unsupported"
            cfg.notes = err or ""
            unsupported_reasons[(err or "?").split(":", 1)[0]] += 1
            configs.append(cfg)
            continue
        cfg.seed_runs.append(sr1)
        completed += 1

        if sr1.filable_hit:
            # Run two more seeds for reproducibility
            for _ in range(N_REPRO_SEEDS - 1):
                seeds_n = (rng.randint(0, 2**31 - 1), rng.randint(0, 2**31 - 1), rng.randint(0, 2**31 - 1))
                srn, err = run_one_seed(shape, dtype, stride, seeds_n, atol, rtol, mps, cpu)
                if srn is None:
                    break
                cfg.seed_runs.append(srn)
            n_filable = sum(1 for s in cfg.seed_runs if s.filable_hit)
            if n_filable >= N_REPRO_SEEDS:
                cfg.status = "filable"
            elif any(s.failed_envelope for s in cfg.seed_runs):
                cfg.status = "recalibration"
            else:
                cfg.status = "ok"
        elif sr1.failed_envelope:
            cfg.status = "recalibration"
        else:
            cfg.status = "ok"

        configs.append(cfg)

    elapsed = time.perf_counter() - start
    write_outputs(configs, attempted, completed, elapsed, aborted,
                  unsupported_reasons, bucket_counts, dtype_counts, stride_counts)
    return 0


def record_skipped(reason: str) -> None:
    RESULTS_MD.write_text(
        f"# flash-attn-v2 fuzz — SKIPPED\n\n{reason}\n"
    )
    rec = {
        "agent": AGENT, "kernel": KERNEL, "status": "SKIPPED", "reason": reason,
        "torch_version": torch.__version__,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def write_outputs(
    configs: list[Config],
    attempted: int,
    completed: int,
    elapsed: float,
    aborted: str,
    unsupported_reasons: Counter[str],
    bucket_counts: Counter[str],
    dtype_counts: Counter[str],
    stride_counts: Counter[str],
) -> None:
    filable = [c for c in configs if c.status == "filable"]
    recal = [c for c in configs if c.status == "recalibration"]
    ok = [c for c in configs if c.status == "ok"]
    unsupported = [c for c in configs if c.status == "unsupported"]
    errors = [c for c in configs if c.status == "error"]

    def best_rel(c: Config) -> float:
        return max((s.max_rel_err for s in c.seed_runs), default=0.0)

    def best_abs(c: Config) -> float:
        return max((s.max_abs_err for s in c.seed_runs), default=0.0)

    overall_rel = max((best_rel(c) for c in configs), default=0.0)
    overall_abs = max((best_abs(c) for c in configs), default=0.0)

    by_dtype: dict[str, float] = {}
    for c in configs:
        if not c.seed_runs:
            continue
        by_dtype[c.dtype] = max(by_dtype.get(c.dtype, 0.0), best_rel(c))

    # Top 3 minimal repros: prefer FILABLE (smallest footprint), then largest-rel-err RECAL.
    def footprint(c: Config) -> int:
        b, h, s, d = c.shape
        return b * h * s * d
    if filable:
        top3_pool = sorted(filable, key=lambda c: (footprint(c), -best_rel(c)))[:3]
    else:
        top3_pool = sorted(recal, key=lambda c: -best_rel(c))[:3]

    top3 = [
        {
            "cid": c.cid,
            "shape": list(c.shape),
            "dtype": c.dtype,
            "stride": c.stride,
            "bucket": c.bucket,
            "atol": c.atol,
            "rtol": c.rtol,
            "status": c.status,
            "n_seeds_run": len(c.seed_runs),
            "n_filable_hits": sum(1 for s in c.seed_runs if s.filable_hit),
            "n_envelope_failures": sum(1 for s in c.seed_runs if s.failed_envelope),
            "max_rel_err": best_rel(c),
            "max_abs_err": best_abs(c),
            "max_denom_at_rel": max((s.denom_at_rel for s in c.seed_runs), default=0.0),
            "per_seed": [
                {
                    "seeds": [s.seed_q, s.seed_k, s.seed_v],
                    "max_rel_err": s.max_rel_err,
                    "max_abs_err": s.max_abs_err,
                    "denom_at_rel": s.denom_at_rel,
                    "n_violations": s.n_violations,
                    "n_elements": s.n_elements,
                    "filable_hit": s.filable_hit,
                }
                for s in c.seed_runs
            ],
        }
        for c in top3_pool
    ]

    target = "pytorch/pytorch (aten::scaled_dot_product_attention MPS)" if filable else "none"

    # ---------------- Markdown ----------------
    md = []
    md.append(f"# Fuzz Report — {KERNEL} (MPS SDPA)")
    md.append("")
    md.append(f"**Agent:** `{AGENT}`  ")
    md.append(f"**Op:** `{OP_PATH}`  ")
    md.append(f"**Reference:** CPU fp32 SDPA  ")
    md.append(f"**Date:** {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ")
    md.append(f"**Torch:** `{torch.__version__}` on `{platform.platform()}`  ")
    md.append(f"**MPS available:** `{torch.backends.mps.is_available()}`  ")
    md.append("")
    md.append("## Method")
    md.append("")
    md.append(
        "Stride/contiguity + shape + dtype fuzzing on MPS, comparing element-wise to a CPU fp32 reference. "
        "Each iteration samples a shape category, dtype, and stride pattern. Tolerances come from "
        "`gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=head_dim, device_type='mps')` — "
        "atol scales by `sqrt(head_dim/128)` (CUTLASS error model) with the MPS dtype overlay applied."
    )
    md.append("")
    md.append("**Classification:**")
    md.append("")
    md.append(f"- `FILABLE`              max_rel_err > {RELERR_FILABLE_MULT}× rtol AND |ref|≥{DENOM_FLOOR:g} AND reproducible across ≥{N_REPRO_SEEDS} distinct data seeds.")
    md.append("- `TOLERANCE_RECALIBRATION`  violates allclose envelope (`|a-b| > atol + rtol·|b|`) but fails the FILABLE filter (denom too small or non-reproducible).")
    md.append("- `OK`                   passes envelope.")
    md.append("")
    md.append("## Run summary")
    md.append("")
    md.append(f"- Iterations attempted: **{attempted}** / target {TARGET_ITERS}")
    md.append(f"- Iterations completed (one full seed pair): **{completed}**")
    md.append(f"- FILABLE: **{len(filable)}**  TOLERANCE_RECALIBRATION: **{len(recal)}**  OK: **{len(ok)}**  UNSUPPORTED: **{len(unsupported)}**  ERROR: **{len(errors)}**")
    md.append(f"- Wall time: **{elapsed:.2f}s** (budget {TIME_BUDGET_SEC}s){'  — ' + aborted if aborted else ''}")
    md.append(f"- Config seed: `{CONFIG_SEED:#x}`")
    md.append("")
    md.append("## MPS-vs-CPU error envelope")
    md.append("")
    md.append(f"- Overall **max relative error** (denom-guarded): `{overall_rel:.6g}`")
    md.append(f"- Overall **max absolute error**: `{overall_abs:.6g}`")
    md.append("")
    md.append("Per-dtype max rel err:")
    for dt, mx in sorted(by_dtype.items()):
        md.append(f"  - `{dt}` → `{mx:.4g}`")
    md.append("")
    md.append("Coverage breakdown:")
    md.append("")
    md.append(f"- shape buckets: {dict(bucket_counts)}")
    md.append(f"- dtypes: {dict(dtype_counts)}")
    md.append(f"- strides: {dict(stride_counts)}")
    md.append("")
    md.append("## FILABLE divergences")
    md.append("")
    if filable:
        md.append(f"Total: **{len(filable)}**")
        md.append("")
        for rank, c in enumerate(top3_pool[:3], 1):
            md.append(f"### #{rank}  shape=`{c.shape}` dtype=`{c.dtype}` stride=`{c.stride}` bucket=`{c.bucket}`")
            md.append(f"  - max-rel-err: `{best_rel(c):.6g}` (rtol budget `{c.rtol:.4g}`, threshold `{RELERR_FILABLE_MULT * c.rtol:.4g}`)")
            md.append(f"  - max-abs-err: `{best_abs(c):.6g}` (atol budget `{c.atol:.4g}`)")
            md.append(f"  - max denom at rel: `{max((s.denom_at_rel for s in c.seed_runs), default=0.0):.4g}`")
            md.append(f"  - reproducible: `{sum(1 for s in c.seed_runs if s.filable_hit)} / {len(c.seed_runs)}` seeds")
            md.append("")
    else:
        md.append("_None — no MPS-vs-CPU divergence met the 10×-rtol + |ref|≥1e-6 + 3-seed reproducibility bar._")
        md.append("")
    md.append("## TOLERANCE_RECALIBRATION (envelope busts that aren't filable)")
    md.append("")
    if recal:
        md.append(f"Total: **{len(recal)}** configs.")
        md.append("")
        # Group by (dtype, stride) to suggest xfail rules.
        rec_by_key: dict[tuple[str, str], int] = defaultdict(int)
        for c in recal:
            rec_by_key[(c.dtype, c.stride)] += 1
        md.append("By dtype × stride:")
        for (dt, st), n in sorted(rec_by_key.items(), key=lambda kv: -kv[1]):
            md.append(f"  - `{dt}` × `{st}` → {n}")
        md.append("")
    else:
        md.append("_None._")
        md.append("")
    md.append("## Unsupported / errored configurations")
    md.append("")
    if unsupported_reasons:
        for k, v in unsupported_reasons.most_common():
            md.append(f"- `{k}` × {v}")
        md.append("")
    else:
        md.append("_None._")
        md.append("")
    md.append("## Recommended upstream filing target")
    md.append("")
    md.append(f"`{target}`")
    if filable:
        md.append("")
        md.append(
            "Rationale: torch SDPA on MPS diverges from CPU fp32 reference beyond 10× rtol on a non-tiny "
            "denominator and the divergence reproduces across 3 distinct data seeds for a fixed "
            "(shape, dtype, stride) config. FlashAttention-v2 is integrated into PyTorch core; the MPS "
            "backend lives in `aten/src/ATen/native/mps/operations/Attention*`."
        )
    md.append("")
    md.append("---")
    md.append(f"*Generated by `{AGENT}`.*")
    RESULTS_MD.write_text("\n".join(md))

    # ---------------- JSONL record ----------------
    record = {
        "agent": AGENT,
        "kernel": KERNEL,
        "op_path": OP_PATH,
        "device_under_test": "mps",
        "reference": "cpu_fp32",
        "torch_version": torch.__version__,
        "host": f"{platform.system().lower()}/{platform.machine()} ({platform.platform()})",
        "config_seed": hex(CONFIG_SEED),
        "n_iterations_planned": TARGET_ITERS,
        "iters_attempted": attempted,
        "iters_completed": completed,
        "iters_unsupported": len(unsupported),
        "iters_errored": len(errors),
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal),
        "configs_ok": len(ok),
        "max_rel_err": overall_rel,
        "max_abs_err": overall_abs,
        "max_rel_err_by_dtype": by_dtype,
        "filable_filter": {
            "rel_err_multiplier_over_rtol": RELERR_FILABLE_MULT,
            "denom_floor": DENOM_FLOOR,
            "min_reproducing_seeds": N_REPRO_SEEDS,
        },
        "per_shape_bucket": dict(bucket_counts),
        "per_dtype": dict(dtype_counts),
        "per_stride": dict(stride_counts),
        "unsupported_reasons": dict(unsupported_reasons),
        "top_3_repros": top3,
        "filing_target": target,
        "elapsed_s": round(elapsed, 3),
        "wall_budget_s": TIME_BUDGET_SEC,
        "aborted": aborted,
        "results_md": str(RESULTS_MD),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        try:
            RESULTS_MD.write_text(
                f"# flash-attn-v2 fuzz — FATAL\n\n```\n{tb}\n```\n"
            )
        except Exception:
            pass
        try:
            with SWARM_JSONL.open("a") as f:
                f.write(json.dumps({
                    "agent": AGENT, "kernel": KERNEL, "status": "FATAL",
                    "traceback": tb[-1000:],
                }) + "\n")
        except Exception:
            pass
        print(tb, file=sys.stderr)
        sys.exit(1)
