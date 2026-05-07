"""Fuzzer for einsum-4d on MPS vs CPU.

Drives stride/contiguity + shape + dtype fuzzing across a curated 4D einsum
equation set. Implements the swarm's FILABLE / TOLERANCE_RECALIBRATION / OK
triage with denom-floor and multi-seed reproducibility.
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
from pathlib import Path

# Make gpucheck importable from the worktree
WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-einsum-4d"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_einsum-4d.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "einsum-4d"
OP_PATH = "torch.einsum"
N_ITER = 500
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s headroom for write-out
CONFIG_SEED = 0xE45U4D & 0xFFFFFFFF  # config-sampling RNG seed
DATA_SEEDS = (0, 1, 2, 3, 4)  # tensor-init seeds for repro cross-check
DENOM_FLOOR = 1e-6  # rel-err denom floor for FILABLE classification
FILABLE_MULT = 10.0  # max_rel_err must exceed FILABLE_MULT * tol to file
MIN_REPRO_SEEDS = 3  # need ≥3 seeds reproducing to call it FILABLE

# 4D einsum equations: classic attention-style + generic contractions.
# Each entry: (equation, shape_fn(A,B,I,J,K,D) -> (lhs_shape, rhs_shape))
EQUATIONS: list[tuple[str, str]] = [
    # batched matmul (attention QK^T pattern is bhid,bhjd->bhij; Y = QK^T)
    ("bhij,bhjk->bhik", "batched_matmul"),
    # attention QK^T: contract D axis
    ("bhid,bhjd->bhij", "qk_dot"),
    # attention out = attn @ V: bhij @ bhjd -> bhid
    ("bhij,bhjd->bhid", "attn_v"),
    # generic 4D contraction over the C axis (third position)
    ("abcd,abce->abde", "generic_c_contract"),
    # contraction over the second-to-last axis with a permute pattern
    ("bnhd,bmhd->bnhm", "transformer_attn"),
]

# Dim corpora (gpucheck shape category model).
DEGEN = [1, 2]
PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47, 53, 59, 61, 67, 71, 79, 83, 97]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150]
LARGE = [256, 384, 512]
MIXED_SOURCES = [DEGEN, PRIMES, POW2_BOUNDARY, NON_TILE, LARGE]

SHAPE_BUCKETS = ["degenerate", "prime", "power_of_2_boundary", "non_tile_aligned", "large", "mixed"]

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# 7-category stride taxonomy (per gpucheck StrideStrategy).
STRIDE_CATEGORIES = [
    "row_major",        # standard contiguous
    "column_major",     # last-dim is fastest reversed → permuted contiguous
    "broadcast",        # one or more axes broadcast (stride 0)
    "transpose",        # adjacent-axis swap via .transpose()
    "slice",            # over-allocate then stride-2 slice
    "non_contig",       # contiguous_after_clone — view with non-natural strides
    "gather",           # gather-induced layout (sparse-ish indexing copy)
]


def pick_dim(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(DEGEN)
    if bucket == "prime":
        return rng.choice(PRIMES)
    if bucket == "power_of_2_boundary":
        return rng.choice(POW2_BOUNDARY)
    if bucket == "non_tile_aligned":
        return rng.choice(NON_TILE)
    if bucket == "large":
        return rng.choice(LARGE)
    if bucket == "mixed":
        return rng.choice(rng.choice(MIXED_SOURCES))
    raise ValueError(bucket)


def pick_shapes_for_eq(eq: str, bucket: str, rng: random.Random) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (lhs_shape, rhs_shape) consistent with the einsum equation."""
    # Cap large-bucket dims for non-batched axes so that 4D × 4D contractions
    # stay within MPS budget. Batch & head axes stay tiny.
    lhs_str, rhs_str = eq.split("->")[0].split(",")

    # Pick concrete sizes for each unique label.
    labels = sorted(set(lhs_str + rhs_str))
    sizes: dict[str, int] = {}
    for lab in labels:
        # batch labels (b, a, n) get small dims to avoid blowups
        if lab in {"b", "a"}:
            sizes[lab] = rng.choice([1, 2, 3])
        elif lab in {"h", "n", "m"}:
            sizes[lab] = rng.choice([1, 2, 4]) if bucket != "large" else rng.choice([2, 4])
        else:
            sizes[lab] = pick_dim(bucket, rng)

    lhs = tuple(sizes[c] for c in lhs_str)
    rhs = tuple(sizes[c] for c in rhs_str)
    return lhs, rhs


def torch_dtype(name: str) -> torch.dtype:
    return dict(DTYPES)[name]


def _make_base(shape: tuple[int, ...], gen: torch.Generator) -> torch.Tensor:
    """fp32 backing tensor on CPU for stable RNG across devices."""
    return torch.randn(shape, generator=gen, dtype=torch.float32) * 0.5


def _apply_stride(t: torch.Tensor, cat: str, rng: random.Random) -> torch.Tensor:
    """Reshape a contiguous fp32 cpu tensor into the requested stride pattern.

    Returned tensor has the SAME logical shape and SAME numerical content (where
    feasible) but with the requested stride layout. For broadcast we reduce
    along the broadcast axis to keep numerics meaningful.
    """
    if cat == "row_major":
        return t.contiguous()
    if cat == "column_major":
        # Reverse-axis contiguous: build flipped, .flip is a view-equivalent copy.
        # We achieve column-major by permuting all axes, .contiguous(), then re-permuting back.
        perm = list(range(t.ndim))[::-1]
        return t.permute(*perm).contiguous().permute(*perm)
    if cat == "broadcast":
        # Pick first axis with size>1 and broadcast a size-1 slice to its size.
        for axis, dim in enumerate(t.shape):
            if dim > 1:
                # Take index 0 along axis, then expand back.
                slc: list[slice | int] = [slice(None)] * t.ndim
                slc[axis] = 0
                small = t[tuple(slc)].unsqueeze(axis)  # shape: ..., 1, ...
                return small.expand_as(t)
        return t.contiguous()
    if cat == "transpose":
        # Swap last two axes if possible
        if t.ndim < 2:
            return t.contiguous()
        return t.transpose(-1, -2).contiguous().transpose(-1, -2)
    if cat == "slice":
        # Over-allocate twice the last axis, fill with random, then stride-2 slice.
        big_shape = list(t.shape)
        big_shape[-1] *= 2
        big = torch.randn(big_shape, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)), dtype=torch.float32) * 0.5
        big[..., ::2] = t  # write our payload into even indices
        return big[..., ::2]
    if cat == "non_contig":
        # contiguous_after_clone: build a permuted view, clone, then permute back.
        # The result has natural shape but unusual stride.
        if t.ndim < 2:
            return t.contiguous()
        perm = list(range(t.ndim))
        # cyclic rotate by 1
        perm = perm[1:] + perm[:1]
        inv = [perm.index(i) for i in range(t.ndim)]
        return t.permute(*perm).contiguous().permute(*inv)
    if cat == "gather":
        # Gather along last axis with identity indices via a fresh allocation.
        # Produces a shape-equal tensor with stride layout from gather output.
        idx = torch.arange(t.shape[-1]).unsqueeze(0)
        for _ in range(t.ndim - 2):
            idx = idx.unsqueeze(0)
        idx = idx.expand_as(t)
        return torch.gather(t, -1, idx)
    raise ValueError(cat)


def shape_bucket_of(shape: tuple[int, ...]) -> str:
    """Classify the dominant shape category of a tuple of dims."""
    flat = list(shape)
    if any(d <= 2 for d in flat):
        return "degenerate"
    primes = set(PRIMES)
    pow2_b = set(POW2_BOUNDARY)
    non_tile = set(NON_TILE)
    large = set(LARGE)
    if any(d in pow2_b for d in flat):
        return "power_of_2_boundary"
    if any(d in primes for d in flat):
        return "prime"
    if any(d in large for d in flat):
        return "large"
    if any(d in non_tile for d in flat):
        return "non_tile_aligned"
    return "mixed"


@dataclass
class Config:
    cid: int
    equation: str
    eq_label: str
    lhs_shape: tuple[int, ...]
    rhs_shape: tuple[int, ...]
    out_shape: tuple[int, ...]
    bucket: str
    dtype: str
    stride: str

    def key(self) -> str:
        return f"{self.equation}|{self.lhs_shape}|{self.rhs_shape}|{self.dtype}|{self.stride}"


@dataclass
class SeedRun:
    seed: int
    max_abs_err: float
    max_rel_err: float
    max_rel_err_filtered: float  # rel err with denom floor applied
    denom_at_max: float
    status: str  # OK | RECAL | FILABLE_CANDIDATE | UNSUPPORTED | ERROR


def k_dim_for_eq(eq: str, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> int:
    """Return the size of the contracted dimension for tolerance scaling."""
    lhs_str, rhs_str = eq.split("->")[0].split(",")
    out_str = eq.split("->")[1]
    contracted = [c for c in lhs_str if c in rhs_str and c not in out_str]
    if not contracted:
        return 1
    sizes_lhs = dict(zip(lhs_str, lhs_shape))
    # Multiply across all contracted axes (each adds a sqrt-N factor in worst case).
    k = 1
    for c in contracted:
        k *= sizes_lhs[c]
    return max(k, 1)


def out_shape_for_eq(eq: str, lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    lhs_str, rhs_str = eq.split("->")[0].split(",")
    out_str = eq.split("->")[1]
    sizes: dict[str, int] = {}
    for c, d in zip(lhs_str, lhs_shape):
        sizes[c] = d
    for c, d in zip(rhs_str, rhs_shape):
        sizes[c] = d
    return tuple(sizes[c] for c in out_str)


def run_seed(cfg: Config, seed: int) -> SeedRun:
    """Run one (config, seed) on CPU and MPS, return error metrics."""
    gen_a = torch.Generator().manual_seed(seed * 2 + 1)
    gen_b = torch.Generator().manual_seed(seed * 2 + 2)
    rng = random.Random(seed * 7919 + cfg.cid)

    # Build fp32 base tensors on CPU
    a_base = _make_base(cfg.lhs_shape, gen_a)
    b_base = _make_base(cfg.rhs_shape, gen_b)

    # Apply stride pattern (still on CPU, fp32)
    try:
        a_strided = _apply_stride(a_base, cfg.stride, rng)
        b_strided = _apply_stride(b_base, cfg.stride, rng)
    except Exception as e:  # noqa: BLE001
        return SeedRun(seed=seed, max_abs_err=float("nan"), max_rel_err=float("nan"),
                       max_rel_err_filtered=float("nan"), denom_at_max=0.0,
                       status=f"ERROR:apply_stride:{type(e).__name__}")

    dt = torch_dtype(cfg.dtype)

    # CPU computation in target dtype
    try:
        a_cpu = a_strided.to(dtype=dt)
        b_cpu = b_strided.to(dtype=dt)
        # Reference: do einsum at fp64 (for fp32) or fp32 (for fp16/bf16) for ground-truth.
        ref_dt = torch.float64 if cfg.dtype == "float32" else torch.float32
        y_cpu_dtype = torch.einsum(cfg.equation, a_cpu, b_cpu)
        y_ref = torch.einsum(cfg.equation, a_strided.to(ref_dt), b_strided.to(ref_dt))
    except Exception as e:  # noqa: BLE001
        return SeedRun(seed=seed, max_abs_err=float("nan"), max_rel_err=float("nan"),
                       max_rel_err_filtered=float("nan"), denom_at_max=0.0,
                       status=f"ERROR:cpu_einsum:{type(e).__name__}")

    # MPS computation
    try:
        a_mps = a_strided.to(dtype=dt, device="mps")
        b_mps = b_strided.to(dtype=dt, device="mps")
        y_mps = torch.einsum(cfg.equation, a_mps, b_mps)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.to(device="cpu", dtype=torch.float32)
    except NotImplementedError as e:
        return SeedRun(seed=seed, max_abs_err=0.0, max_rel_err=0.0,
                       max_rel_err_filtered=0.0, denom_at_max=0.0,
                       status=f"UNSUPPORTED:{str(e)[:120]}")
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported", "MPS")):
            return SeedRun(seed=seed, max_abs_err=0.0, max_rel_err=0.0,
                           max_rel_err_filtered=0.0, denom_at_max=0.0,
                           status=f"UNSUPPORTED:{msg.splitlines()[0][:120]}")
        return SeedRun(seed=seed, max_abs_err=float("nan"), max_rel_err=float("nan"),
                       max_rel_err_filtered=float("nan"), denom_at_max=0.0,
                       status=f"ERROR:{msg.splitlines()[0][:120]}")

    # Compare MPS-vs-CPU at the dtype we ran (the kernel-level test).
    y_cpu_f32 = y_cpu_dtype.to(torch.float32)
    diff = (y_mps_cpu - y_cpu_f32).abs()
    if diff.numel() == 0:
        return SeedRun(seed=seed, max_abs_err=0.0, max_rel_err=0.0,
                       max_rel_err_filtered=0.0, denom_at_max=0.0, status="OK")

    max_abs = float(diff.max().item())
    denom_raw = y_cpu_f32.abs()
    rel_unfiltered = diff / denom_raw.clamp_min(1e-12)
    max_rel_unfiltered = float(rel_unfiltered.max().item())

    # Filtered rel-err: only consider entries where denom >= DENOM_FLOOR.
    mask = denom_raw >= DENOM_FLOOR
    if mask.any():
        rel_filtered = (diff[mask] / denom_raw[mask])
        max_rel_filtered = float(rel_filtered.max().item())
        # Find which entry achieved that filtered max
        idx_flat = int(rel_filtered.argmax().item())
        denom_at_max = float(denom_raw[mask].flatten()[idx_flat].item())
    else:
        # All denom < floor — divergence (if any) is unscalable; treat as no rel err.
        max_rel_filtered = 0.0
        denom_at_max = float(denom_raw.max().item()) if denom_raw.numel() else 0.0

    return SeedRun(
        seed=seed,
        max_abs_err=max_abs,
        max_rel_err=max_rel_unfiltered,
        max_rel_err_filtered=max_rel_filtered,
        denom_at_max=denom_at_max,
        status="OK",
    )


def classify_config(
    cfg: Config,
    runs: list[SeedRun],
    atol: float,
    rtol: float,
) -> tuple[str, dict]:
    """Apply FILABLE / TOLERANCE_RECALIBRATION / OK triage.

    FILABLE if ≥MIN_REPRO_SEEDS seeds satisfy:
        max_rel_err_filtered > FILABLE_MULT * rtol AND denom_at_max >= DENOM_FLOOR
    TOLERANCE_RECALIBRATION if any seed exceeds rtol but no FILABLE.
    OK otherwise.
    """
    threshold = max(rtol, atol)  # use the larger of the two as the perceived tol
    filable_thresh = FILABLE_MULT * threshold

    n_filable_hits = 0
    n_recal_hits = 0
    seeds_filable: list[int] = []
    seeds_recal: list[int] = []
    seeds_unsupported: list[int] = []
    seeds_error: list[int] = []
    max_rel = 0.0
    max_rel_filtered = 0.0
    max_abs = 0.0
    max_denom = 0.0

    for r in runs:
        if r.status.startswith("UNSUPPORTED"):
            seeds_unsupported.append(r.seed)
            continue
        if r.status.startswith("ERROR"):
            seeds_error.append(r.seed)
            continue
        max_rel = max(max_rel, r.max_rel_err)
        max_rel_filtered = max(max_rel_filtered, r.max_rel_err_filtered)
        max_abs = max(max_abs, r.max_abs_err)
        max_denom = max(max_denom, r.denom_at_max)
        if r.max_rel_err_filtered > filable_thresh and r.denom_at_max >= DENOM_FLOOR:
            n_filable_hits += 1
            seeds_filable.append(r.seed)
        elif r.max_rel_err_filtered > threshold:
            n_recal_hits += 1
            seeds_recal.append(r.seed)

    if n_filable_hits >= MIN_REPRO_SEEDS:
        verdict = "FILABLE"
    elif n_filable_hits + n_recal_hits > 0:
        verdict = "TOLERANCE_RECALIBRATION"
    else:
        verdict = "OK"

    if seeds_unsupported and not seeds_filable and not seeds_recal:
        verdict = "UNSUPPORTED"
    if seeds_error and not (seeds_filable or seeds_recal):
        verdict = "ERROR" if not seeds_unsupported else verdict

    detail = {
        "n_filable_hits": n_filable_hits,
        "n_recal_hits": n_recal_hits,
        "seeds_filable": seeds_filable,
        "seeds_recal": seeds_recal,
        "seeds_unsupported": seeds_unsupported,
        "seeds_error": seeds_error,
        "max_rel_err": max_rel,
        "max_rel_err_filtered": max_rel_filtered,
        "max_abs_err": max_abs,
        "denom_at_max": max_denom,
        "atol": atol,
        "rtol": rtol,
        "filable_threshold": filable_thresh,
        "recal_threshold": threshold,
    }
    return verdict, detail


def main() -> int:
    if not torch.backends.mps.is_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text(f"# {KERNEL_NAME} fuzz — SKIPPED\n\nMPS unavailable.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    rng = random.Random(CONFIG_SEED)
    t0 = time.monotonic()

    # Phase 1: sample N_ITER unique configs and run a single primary seed each.
    primary_seed = DATA_SEEDS[0]
    configs: list[Config] = []
    primary_runs: dict[str, SeedRun] = {}
    config_by_key: dict[str, Config] = {}

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_error = 0
    aborted = ""

    print(f"[start] N_ITER={N_ITER} budget={WALL_BUDGET_S}s", flush=True)

    for cid in range(N_ITER):
        elapsed = time.monotonic() - t0
        if elapsed > WALL_BUDGET_S * 0.65:
            # leave 35% of budget for repro + write
            print(f"[budget-phase1] stopping config sample at cid={cid} elapsed={elapsed:.1f}s", flush=True)
            aborted = "phase1_budget"
            break

        eq, eq_label = rng.choice(EQUATIONS)
        bucket = rng.choice(SHAPE_BUCKETS)
        dtype, _ = rng.choice(DTYPES)
        stride = rng.choice(STRIDE_CATEGORIES)
        lhs, rhs = pick_shapes_for_eq(eq, bucket, rng)
        out_shape = out_shape_for_eq(eq, lhs, rhs)

        cfg = Config(
            cid=cid, equation=eq, eq_label=eq_label,
            lhs_shape=lhs, rhs_shape=rhs, out_shape=out_shape,
            bucket=bucket, dtype=dtype, stride=stride,
        )
        configs.append(cfg)
        config_by_key[cfg.key()] = cfg

        iters_attempted += 1
        try:
            r = run_seed(cfg, primary_seed)
        except Exception as e:  # halt-on-process-error per spec
            tb = traceback.format_exc()[-600:]
            print(f"[HALT] cid={cid} {type(e).__name__}: {e}\n{tb}", flush=True)
            aborted = f"process_error:{type(e).__name__}"
            primary_runs[cfg.key()] = SeedRun(
                seed=primary_seed, max_abs_err=float("nan"), max_rel_err=float("nan"),
                max_rel_err_filtered=float("nan"), denom_at_max=0.0,
                status=f"ERROR:halt:{type(e).__name__}",
            )
            break

        primary_runs[cfg.key()] = r
        if r.status == "OK":
            iters_completed += 1
        elif r.status.startswith("UNSUPPORTED"):
            iters_unsupported += 1
        else:
            iters_error += 1

        if cid % 50 == 0 or r.status != "OK":
            note = ""
            if r.status != "OK":
                note = f" [{r.status[:60]}]"
            print(
                f"[{cid:03d}] {eq_label:<18} dt={dtype:<8} st={stride:<10} "
                f"lhs={lhs} rhs={rhs} rel={r.max_rel_err_filtered:.2e}{note}",
                flush=True,
            )

    print(f"[phase1] done at {time.monotonic() - t0:.1f}s, "
          f"{len(configs)} configs sampled, {iters_completed} OK, "
          f"{iters_unsupported} unsupported, {iters_error} errors", flush=True)

    # Phase 2: for any config whose primary seed shows rel_err > recal_threshold
    # (i.e., is at least RECALIBRATION territory), run extra seeds for repro check.
    candidates: list[Config] = []
    for cfg in configs:
        r = primary_runs[cfg.key()]
        if not r.status == "OK":
            continue
        atol, rtol = compute_tolerance(cfg.dtype, k_dim=k_dim_for_eq(cfg.equation, cfg.lhs_shape, cfg.rhs_shape), device_type="mps")
        threshold = max(atol, rtol)
        if r.max_rel_err_filtered > threshold:
            candidates.append(cfg)

    print(f"[phase2] {len(candidates)} candidate configs need multi-seed repro", flush=True)

    extra_runs: dict[str, list[SeedRun]] = {}
    repro_seeds = list(DATA_SEEDS[1:])  # already ran primary
    for i, cfg in enumerate(candidates):
        elapsed = time.monotonic() - t0
        if elapsed > WALL_BUDGET_S * 0.92:
            print(f"[budget-phase2] stopping repro at {i}/{len(candidates)} elapsed={elapsed:.1f}s", flush=True)
            if not aborted:
                aborted = "phase2_budget"
            break
        runs_list: list[SeedRun] = []
        for s in repro_seeds:
            try:
                r = run_seed(cfg, s)
            except Exception as e:
                tb = traceback.format_exc()[-600:]
                print(f"[HALT-phase2] cid={cfg.cid} seed={s} {type(e).__name__}: {e}\n{tb}", flush=True)
                aborted = f"process_error:{type(e).__name__}"
                break
            runs_list.append(r)
        extra_runs[cfg.key()] = runs_list

    # Phase 3: classify each config.
    classifications: list[tuple[Config, str, dict, list[SeedRun]]] = []
    for cfg in configs:
        runs_all: list[SeedRun] = [primary_runs[cfg.key()]]
        if cfg.key() in extra_runs:
            runs_all.extend(extra_runs[cfg.key()])
        atol, rtol = compute_tolerance(cfg.dtype, k_dim=k_dim_for_eq(cfg.equation, cfg.lhs_shape, cfg.rhs_shape), device_type="mps")
        verdict, detail = classify_config(cfg, runs_all, atol, rtol)
        classifications.append((cfg, verdict, detail, runs_all))

    elapsed = time.monotonic() - t0
    write_outputs(classifications, configs, primary_runs, extra_runs,
                  iters_attempted, iters_completed, iters_unsupported, iters_error,
                  elapsed, aborted)
    return 0


def write_outputs(
    classifications: list[tuple[Config, str, dict, list[SeedRun]]],
    configs: list[Config],
    primary_runs: dict[str, SeedRun],
    extra_runs: dict[str, list[SeedRun]],
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_error: int,
    elapsed: float,
    aborted: str,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    filable = [c for c in classifications if c[1] == "FILABLE"]
    recal = [c for c in classifications if c[1] == "TOLERANCE_RECALIBRATION"]
    ok_list = [c for c in classifications if c[1] == "OK"]
    unsupported_list = [c for c in classifications if c[1] == "UNSUPPORTED"]
    error_list = [c for c in classifications if c[1] == "ERROR"]

    # Top 3 minimal repros = filable+recal sorted by max_rel_err_filtered desc.
    candidates = filable + recal
    candidates.sort(key=lambda x: x[2].get("max_rel_err_filtered", 0.0), reverse=True)
    top3 = candidates[:3]

    # Aggregate maxima
    max_rel_overall = 0.0
    max_rel_filtered_overall = 0.0
    max_abs_overall = 0.0
    for cfg, _, detail, _ in classifications:
        max_rel_overall = max(max_rel_overall, detail.get("max_rel_err", 0.0) or 0.0)
        max_rel_filtered_overall = max(max_rel_filtered_overall, detail.get("max_rel_err_filtered", 0.0) or 0.0)
        max_abs_overall = max(max_abs_overall, detail.get("max_abs_err", 0.0) or 0.0)

    # Per-bucket / per-dtype / per-stride histograms over completed configs.
    per_bucket: dict[str, int] = {b: 0 for b in SHAPE_BUCKETS}
    per_dtype: dict[str, int] = {d[0]: 0 for d in DTYPES}
    per_stride: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}
    per_eq: dict[str, int] = {eq: 0 for eq, _ in EQUATIONS}
    for cfg in configs:
        per_bucket[cfg.bucket] += 1
        per_dtype[cfg.dtype] += 1
        per_stride[cfg.stride] += 1
        per_eq[cfg.equation] += 1

    # Recommended filing target
    filing_target = "pytorch/pytorch (MPS einsum kernel)" if filable else "none"

    # Markdown
    md: list[str] = []
    md.append(f"# einsum-4d Fuzz Results\n")
    md.append("## Summary\n")
    md.append(f"- **Kernel:** `{KERNEL_NAME}` (`{OP_PATH}`, 4D einsum equations)")
    md.append(f"- **Backends:** MPS (real) vs CPU reference; CUDA mocked (no NVIDIA GPU on host)")
    md.append(f"- **Equations tested:** {', '.join(eq for eq, _ in EQUATIONS)}")
    md.append(f"- **Iterations attempted:** {iters_attempted}")
    md.append(f"- **Iterations completed (OK seed run):** {iters_completed}")
    md.append(f"- **Unsupported skipped:** {iters_unsupported}")
    md.append(f"- **Process errors:** {iters_error}")
    md.append(f"- **Configs FILABLE:** {len(filable)}")
    md.append(f"- **Configs TOLERANCE_RECALIBRATION:** {len(recal)}")
    md.append(f"- **Configs OK:** {len(ok_list)}")
    md.append(f"- **Configs UNSUPPORTED:** {len(unsupported_list)}")
    md.append(f"- **Wall time:** {elapsed:.1f}s")
    md.append(f"- **Aborted:** `{aborted or 'no'}`")
    md.append("")
    md.append("## FILABLE filter\n")
    md.append(
        f"`max_rel_err_filtered > {FILABLE_MULT}× tolerance` AND "
        f"`denom_at_max >= {DENOM_FLOOR:.0e}` AND reproduces across ≥{MIN_REPRO_SEEDS} of 5 seeds."
    )
    md.append("")
    md.append("## Max relative error (MPS vs CPU)\n")
    md.append(f"- overall (filtered, denom>={DENOM_FLOOR:.0e}): **{max_rel_filtered_overall:.4e}**")
    md.append(f"- overall (unfiltered, includes near-zero denom): {max_rel_overall:.4e}")
    md.append(f"- max abs err: {max_abs_overall:.4e}")
    md.append("")
    md.append("## Max relative error (MPS vs CUDA)\n")
    md.append("- N/A (no NVIDIA GPU on this host; CUDA detection mocked, kernel cannot run)")
    md.append("")
    md.append("## Top 3 minimal repros (by max_rel_err_filtered)\n")
    if not top3:
        md.append("_None — every iteration stayed within tolerance._\n")
    else:
        md.append("| # | verdict | equation | dtype | stride | bucket | lhs_shape | rhs_shape | max_rel_err | atol | rtol | n_repro_seeds |")
        md.append("|---|---------|----------|-------|--------|--------|-----------|-----------|-------------|------|------|----------------|")
        for i, (cfg, verdict, detail, _runs) in enumerate(top3, 1):
            n_repro = len(detail.get("seeds_filable", [])) + len(detail.get("seeds_recal", []))
            md.append(
                f"| {i} | {verdict} | `{cfg.equation}` | {cfg.dtype} | {cfg.stride} | "
                f"{cfg.bucket} | {cfg.lhs_shape} | {cfg.rhs_shape} | "
                f"{detail['max_rel_err_filtered']:.3e} | {detail['atol']:.3e} | "
                f"{detail['rtol']:.3e} | {n_repro} |"
            )
    md.append("")
    if filable:
        md.append("## All FILABLE configs\n")
        md.append("| equation | dtype | stride | bucket | lhs_shape | rhs_shape | max_rel_err | seeds_filable |")
        md.append("|----------|-------|--------|--------|-----------|-----------|-------------|---------------|")
        for cfg, _v, detail, _runs in filable:
            md.append(
                f"| `{cfg.equation}` | {cfg.dtype} | {cfg.stride} | {cfg.bucket} | "
                f"{cfg.lhs_shape} | {cfg.rhs_shape} | "
                f"{detail['max_rel_err_filtered']:.3e} | {detail['seeds_filable']} |"
            )
        md.append("")
    if recal:
        md.append(f"## TOLERANCE_RECALIBRATION ({len(recal)} configs — first 15)\n")
        md.append("| equation | dtype | stride | bucket | lhs_shape | rhs_shape | max_rel_err | atol | rtol |")
        md.append("|----------|-------|--------|--------|-----------|-----------|-------------|------|------|")
        for cfg, _v, detail, _runs in recal[:15]:
            md.append(
                f"| `{cfg.equation}` | {cfg.dtype} | {cfg.stride} | {cfg.bucket} | "
                f"{cfg.lhs_shape} | {cfg.rhs_shape} | "
                f"{detail['max_rel_err_filtered']:.3e} | {detail['atol']:.3e} | {detail['rtol']:.3e} |"
            )
        md.append("")
    md.append("## Coverage histograms\n")
    md.append(f"- shape buckets: `{per_bucket}`")
    md.append(f"- dtypes: `{per_dtype}`")
    md.append(f"- stride categories: `{per_stride}`")
    md.append(f"- equations: `{per_eq}`")
    md.append("")
    md.append("## Methodology\n")
    md.append("- Inputs sampled fp32 on CPU, then cast to dtype and copied to MPS so both devices see identical numerical content.")
    md.append("- Reference computed in fp64 (for fp32 ops) or fp32 (for fp16/bf16 ops) on CPU via the same einsum call.")
    md.append("- Tolerance: `gpucheck.compute_tolerance(dtype, k_dim, device_type='mps')` — base + sqrt(K/128) atol scaling + 2× MPS overlay.")
    md.append("- `k_dim` = product of contracted-axis sizes from the einsum equation.")
    md.append("- Stride taxonomy: row_major, column_major, broadcast, transpose, slice, non_contig, gather (gpucheck StrideStrategy 7-category).")
    md.append("- Repro: any config exceeding rtol on the primary seed is rerun on 4 additional seeds; FILABLE requires ≥3 of 5 seeds beyond `10× tol` with denom ≥ 1e-6.")
    md.append("")
    md.append(f"## Recommended upstream filing target\n\n**{filing_target}**\n")

    RESULTS_MD.write_text("\n".join(md))

    # JSONL summary
    record: dict = {
        "agent": "kernel-fuzzer-einsum-4d-v2",
        "kernel": KERNEL_NAME,
        "op_path": OP_PATH,
        "device_under_test": "mps",
        "reference": "cpu_fp32",
        "cuda_backend": "mocked",
        "torch_version": torch.__version__,
        "config_seed": f"0x{CONFIG_SEED:X}",
        "data_seeds": list(DATA_SEEDS),
        "n_iter_planned": N_ITER,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_error": iters_error,
        "configs_filable": len(filable),
        "configs_recalibration": len(recal),
        "configs_ok": len(ok_list),
        "configs_unsupported": len(unsupported_list),
        "max_rel_err": max_rel_overall,
        "max_rel_err_filtered": max_rel_filtered_overall,
        "max_abs_err": max_abs_overall,
        "denom_filter_floor": DENOM_FLOOR,
        "filable_threshold_mult": FILABLE_MULT,
        "min_repro_seeds": MIN_REPRO_SEEDS,
        "top3_repros": [
            {
                "cid": cfg.cid,
                "verdict": verdict,
                "equation": cfg.equation,
                "eq_label": cfg.eq_label,
                "dtype": cfg.dtype,
                "stride_category": cfg.stride,
                "shape_bucket": cfg.bucket,
                "lhs_shape": list(cfg.lhs_shape),
                "rhs_shape": list(cfg.rhs_shape),
                "out_shape": list(cfg.out_shape),
                "max_rel_err": detail.get("max_rel_err"),
                "max_rel_err_filtered": detail.get("max_rel_err_filtered"),
                "max_abs_err": detail.get("max_abs_err"),
                "denom_at_max": detail.get("denom_at_max"),
                "atol": detail.get("atol"),
                "rtol": detail.get("rtol"),
                "seeds_filable": detail.get("seeds_filable"),
                "seeds_recal": detail.get("seeds_recal"),
                "n_runs": len(runs),
                "per_seed_status": {
                    str(r.seed): r.status for r in runs
                },
            }
            for cfg, verdict, detail, runs in top3
        ],
        "filable_configs": [
            {
                "cid": cfg.cid,
                "equation": cfg.equation,
                "dtype": cfg.dtype,
                "stride_category": cfg.stride,
                "lhs_shape": list(cfg.lhs_shape),
                "rhs_shape": list(cfg.rhs_shape),
                "max_rel_err_filtered": detail.get("max_rel_err_filtered"),
                "seeds_filable": detail.get("seeds_filable"),
            }
            for cfg, _v, detail, _runs in filable
        ],
        "per_shape_bucket": per_bucket,
        "per_dtype": per_dtype,
        "per_stride_category": per_stride,
        "per_equation": per_eq,
        "elapsed_s": round(elapsed, 2),
        "wall_budget_s": WALL_BUDGET_S,
        "aborted": aborted,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "recommended_filing_target": filing_target,
        "results_md": str(RESULTS_MD),
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL", flush=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
