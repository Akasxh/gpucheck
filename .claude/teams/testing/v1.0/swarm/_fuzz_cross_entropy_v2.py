"""V2 fuzz: cross_entropy on MPS vs CPU reference.

Spec (kernel-fuzzer-cross_entropy-v2):
- 500 configs sampled deterministically (config-RNG = 0xCE0FF517).
- Each config replayed across 5 data seeds (0,1,2,3,4); FILABLE bar = >=3 hits.
- Per-run divergence buckets:
    * abs > 10·atol AND rel > 10·rtol AND |y_ref|@argmax_rel >= 1e-6
                                                  -> FILABLE_HIT
    * abs > 10·atol only                          -> CRIT_ABS_ONLY
    * rel > 10·rtol only (denom-gated)            -> CRIT_REL_ONLY
    * abs in [atol, 5·atol)                       -> RECAL
    * rel in [rtol, 5·rtol) AND denom >= 1e-6     -> RECAL
    * else                                        -> OK
- A config is FILABLE iff FILABLE_HIT on >=3 of 5 seeds.
- A config is RECALIBRATION iff RECAL (or CRIT_*_ONLY counted as RECAL when
  >=3 seeds) on >=3 of 5 seeds and not FILABLE.
- 8-min wall budget; partial results on time-out.
- Halts hard on any unexpected harness exception (return code 2).
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

WORKTREE = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-cross_entropy")
sys.path.insert(0, str(WORKTREE / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_cross_entropy.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.cross_entropy"
N_CONFIGS = 500
DATA_SEEDS = (0, 1, 2, 3, 4)
WALL_BUDGET_S = 8 * 60 - 30  # leave 30s for write-out
CONFIG_SEED = 0xCE0FF517

# ---------------------------------------------------------------------------
# Sampling space
# ---------------------------------------------------------------------------
# (N, C) shapes — N is the batch axis, C is the class axis (reduction k_dim).
DEGENERATE: list[tuple[int, int]] = [(1, 2), (2, 2), (1, 4), (1, 8)]
PRIME: list[tuple[int, int]] = [
    (7, 13), (13, 31), (31, 127), (127, 13), (3, 257), (5, 127), (29, 31),
]
POW2_BOUNDARY: list[tuple[int, int]] = [
    (15, 16), (16, 17), (31, 32), (32, 33), (63, 64), (64, 65),
    (127, 128), (128, 129), (255, 256), (256, 257),
]
NON_TILE: list[tuple[int, int]] = [
    (33, 65), (65, 129), (129, 257), (130, 200), (45, 99), (54, 150),
]
LARGE: list[tuple[int, int]] = [
    (256, 1024), (512, 2048), (1024, 1000), (1024, 4096), (2048, 1024),
]
MIXED: list[tuple[int, int]] = [
    (127, 16), (1024, 3), (7, 128), (33, 128),
]

SHAPE_BUCKETS: dict[str, list[tuple[int, int]]] = {
    "degenerate": DEGENERATE,
    "prime": PRIME,
    "pow2_boundary": POW2_BOUNDARY,
    "non_tile_aligned": NON_TILE,
    "large": LARGE,
    "mixed": MIXED,
}
BUCKET_NAMES = list(SHAPE_BUCKETS.keys())

DTYPES_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
DTYPE_NAMES = list(DTYPES_BY_NAME.keys())

# Stride pattern names match gpucheck.fuzzing.strides.CATEGORIES semantically
# but we materialize them in-place because cross_entropy needs a 2-D logits
# tensor whose last dim is the class axis (and we want full control over what
# the MPS-side bytes look like).
STRIDE_CATEGORIES: tuple[str, ...] = (
    "contiguous",
    "transpose",        # build (C,N) contiguous, .t() -> (N,C) non-contig
    "slice_class",      # alloc (N, 2C), take [:, ::2]
    "slice_batch",      # alloc (2N, C), take [::2, :]
    "broadcast_class",  # (1, C) expanded to (N, C)
    "non_contig_perm",  # (N, 2, C) permute view -> (N, C) with stride (2C,1)
)

REDUCTIONS = ("none", "mean", "sum")


# ---------------------------------------------------------------------------
# Stride builders — operate on logits at (N, C); both backends see the same
# materialised bytes (we copy the value-laid-out tensor to MPS as a contiguous
# clone of the same logical layout, then the kernel runs on it).
# ---------------------------------------------------------------------------

def _make_logits_pair(
    N: int, C: int, dtype: torch.dtype, stride_cat: str, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a (logits_cpu, logits_mps) pair with the same logical (N,C) shape.

    The CPU tensor is the actual stride pattern (potentially non-contiguous);
    the MPS tensor is a contiguous clone of those exact values, so we test
    MPS's kernel given the same numerical inputs (without conflating MPS's
    stride handling at the *forward* op with the stride handling we're
    inducing pre-op).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)

    if stride_cat == "contiguous":
        x = torch.randn(N, C, generator=g, dtype=torch.float32).to(dtype)
    elif stride_cat == "transpose":
        x_t = torch.randn(C, N, generator=g, dtype=torch.float32).to(dtype)
        x = x_t.t()  # (N, C), non-contiguous
    elif stride_cat == "slice_class":
        full = torch.randn(N, 2 * C, generator=g, dtype=torch.float32).to(dtype)
        x = full[:, ::2]
    elif stride_cat == "slice_batch":
        full = torch.randn(2 * N, C, generator=g, dtype=torch.float32).to(dtype)
        x = full[::2, :]
    elif stride_cat == "broadcast_class":
        row = torch.randn(1, C, generator=g, dtype=torch.float32).to(dtype)
        x = row.expand(N, C)
    elif stride_cat == "non_contig_perm":
        full = torch.randn(N, 2, C, generator=g, dtype=torch.float32).to(dtype)
        x = full[:, 0, :]  # stride (2C, 1)
    else:
        raise ValueError(f"unknown stride category {stride_cat!r}")

    assert tuple(x.shape) == (N, C), (x.shape, (N, C))
    # MPS needs contiguous bytes; we want it to compute on identical values.
    x_mps = x.detach().contiguous().clone().to("mps")
    return x, x_mps


def _make_targets(
    N: int, C: int, seed: int, *, soft: bool, dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (target_cpu, target_mps).

    `soft=False` -> integer class indices in [0, C). PyTorch requires int64.
    `soft=True`  -> a per-row probability distribution (uses softmax of fresh
                    random logits) at the same dtype as the inputs.
    """
    g = torch.Generator(device="cpu").manual_seed(seed ^ 0xA5A5A5)
    if not soft:
        t = torch.randint(low=0, high=max(C, 1), size=(N,), generator=g)
        return t, t.detach().clone().to("mps")
    raw = torch.randn(N, C, generator=g, dtype=torch.float32)
    t = F.softmax(raw, dim=-1).to(dtype)
    return t, t.detach().contiguous().clone().to("mps")


# ---------------------------------------------------------------------------
# Error metric & classification
# ---------------------------------------------------------------------------

def _err_metrics(
    a_cpu: torch.Tensor, b_cpu: torch.Tensor,
) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err, |b|@argmax_rel)."""
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a - b).abs()
    abs_err = float(diff.max().item())
    denom = b.abs()
    safe = denom.clamp_min(1e-12)
    rel = diff / safe
    rel_flat = rel.reshape(-1)
    idx = int(rel_flat.argmax().item())
    rel_err = float(rel_flat[idx].item())
    denom_at_relmax = float(denom.reshape(-1)[idx].item())
    return abs_err, rel_err, denom_at_relmax


def _classify(
    abs_err: float, rel_err: float, denom_at_relmax: float,
    atol: float, rtol: float,
) -> str:
    rel_gate = denom_at_relmax >= 1e-6
    abs_critical = abs_err > 10.0 * atol
    rel_critical = (rel_err > 10.0 * rtol) and rel_gate
    if abs_critical and rel_critical:
        return "FILABLE_HIT"
    if abs_critical:
        return "CRIT_ABS_ONLY"
    if rel_critical:
        return "CRIT_REL_ONLY"
    abs_recal = atol <= abs_err < 5.0 * atol
    rel_recal = (rtol <= rel_err < 5.0 * rtol) and rel_gate
    if abs_recal or rel_recal:
        return "RECAL"
    return "OK"


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def _sample_configs(rng: random.Random, n: int) -> list[dict]:
    cfgs = []
    for cid in range(n):
        bucket = rng.choice(BUCKET_NAMES)
        N, C = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        reduction = rng.choice(REDUCTIONS)
        soft_target = rng.random() < 0.30  # 30% soft targets, 70% hard
        cfgs.append({
            "cid": cid,
            "N": N, "C": C,
            "shape_bucket": bucket,
            "dtype": dtype_name,
            "stride_category": stride_cat,
            "reduction": reduction,
            "soft_target": soft_target,
        })
    return cfgs


def _run_one(cfg: dict, data_seed: int) -> tuple[
    str, float, float, float, float, float, str | None,
]:
    """Returns (status, abs_err, rel_err, denom_at_relmax, atol, rtol, why)."""
    N, C = cfg["N"], cfg["C"]
    dtype_name = cfg["dtype"]
    dtype = DTYPES_BY_NAME[dtype_name]
    stride_cat = cfg["stride_category"]
    reduction = cfg["reduction"]
    soft = cfg["soft_target"]

    if N == 0 or C == 0:
        return "EMPTY", 0.0, 0.0, 0.0, 0.0, 0.0, None

    try:
        x_cpu, x_mps = _make_logits_pair(N, C, dtype, stride_cat, data_seed)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"build:{exc!s:.180}"

    try:
        t_cpu, t_mps = _make_targets(N, C, data_seed, soft=soft, dtype=dtype)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"target:{exc!s:.180}"

    try:
        y_cpu = F.cross_entropy(x_cpu, t_cpu, reduction=reduction)
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"cpu-op:{exc!s:.180}"

    try:
        y_mps = F.cross_entropy(x_mps, t_mps, reduction=reduction)
        torch.mps.synchronize()
        y_mps_cpu = y_mps.detach().to("cpu")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        return "UNSUPPORTED", 0.0, 0.0, 0.0, 0.0, 0.0, f"mps-op:{exc!s:.180}"

    abs_err, rel_err, denom_at_relmax = _err_metrics(y_mps_cpu, y_cpu)
    # k_dim = C because cross_entropy reduces along the class axis. MPS
    # overlay multiplier from gpucheck.assertions.tolerances applies.
    atol, rtol = compute_tolerance(dtype, k_dim=C, device_type="mps")
    status = _classify(abs_err, rel_err, denom_at_relmax, atol, rtol)
    return status, abs_err, rel_err, denom_at_relmax, atol, rtol, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_markdown(
    path: Path, s: dict, filable: list[dict], recal: list[dict],
) -> None:
    L: list[str] = []
    L.append(f"# {s['kernel']} - MPS fuzz v2 report")
    L.append("")
    L.append(f"- **kernel:** `{s['kernel']}`")
    L.append(f"- **agent:** `{s['agent']}`")
    L.append(f"- **device under test:** `{s['device_under_test']}` "
             f"(reference: `{s['reference']}`, CUDA backend: `{s['cuda_backend']}`)")
    L.append(f"- **iters_attempted:** {s['iters_attempted']}")
    L.append(f"- **iters_completed:** {s['iters_completed']}")
    L.append(f"- **iters_unsupported:** {s['iters_unsupported']}")
    L.append(f"- **iters_empty (numel==0):** {s['iters_empty']}")
    L.append(f"- **iters_skipped_dtype (bf16 unsupported):** {s['iters_skipped_dtype']}")
    L.append(f"- **bf16 supported on MPS for cross_entropy:** {s['bf16_supported']}")
    L.append(f"- **n_configs_planned:** {s['n_configs_planned']} "
             f"x seeds={s['data_seeds']}")
    L.append(f"- **n_configs_run (>=1 seed completed):** {s['n_configs_run']}")
    L.append(f"- **divergences_filable (FILABLE_HIT on >=3 seeds):** "
             f"{s['divergences_filable']}")
    L.append(f"- **divergences_recalibration (RECAL on >=3 seeds, not filable):** "
             f"{s['divergences_recalibration']}")
    L.append(f"- **max_abs_err:** {s['max_abs_err']:.3e}")
    L.append(f"- **max_rel_err:** {s['max_rel_err']:.3e}")
    L.append(f"- **elapsed:** {s['elapsed_s']} s "
             f"(budget {s['wall_budget_s']}s, timed_out={s['timed_out']})")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}")
    L.append(f"- **config_seed:** {s['config_seed']}, data_seeds: {s['data_seeds']}")
    L.append("")
    L.append("## Divergence rules (v2)")
    L.append("")
    L.append("Per-run status:")
    L.append("")
    L.append("- `FILABLE_HIT` - `abs_err > 10*atol` AND "
             "`rel_err > 10*rtol` AND `|y_ref|@argmax_rel >= 1e-6`.")
    L.append("- `CRIT_ABS_ONLY` - only the abs criterion fires.")
    L.append("- `CRIT_REL_ONLY` - only the denom-gated rel criterion fires.")
    L.append("- `RECAL` - `abs_err in [atol, 5*atol)` OR "
             "(`rel_err in [rtol, 5*rtol)` AND `|y_ref|@argmax_rel >= 1e-6`).")
    L.append("- `OK` - below 1x tolerance.")
    L.append("")
    L.append("Per-config verdict (across the 5 data seeds):")
    L.append("")
    L.append("- **FILABLE** - `FILABLE_HIT` on >= 3 of 5 seeds.")
    L.append("- **RECALIBRATION** - `RECAL` (or `CRIT_*_ONLY` counted into "
             "RECAL when each accumulates >= 3 seeds) on >= 3 of 5 seeds and "
             "not FILABLE. Single-criterion crit hits roll into RECAL because "
             "the spec's FILABLE bar requires both checks to fire.")
    L.append("")
    L.append("Tolerances from `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, k_dim=C, device_type='mps')` - includes the sqrt(C/128) "
             "reduction-axis scaling and the 2x MPS multiplier.")
    L.append("Convention matches `RESULTS_softmax.md` / `RESULTS_kl_div.md` / "
             "`RESULTS_gelu.md` (swarm-v2).")
    L.append("")
    L.append("## Top 3 repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No configs reached the FILABLE or RECALIBRATION buckets._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i} - {r['verdict']}")
            L.append("")
            L.append(f"- **shape:** `(N={r['N']}, C={r['C']})` "
                     f"(bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride_category:** `{r['stride_category']}`")
            L.append(f"- **reduction:** `{r['reduction']}`, "
                     f"**soft_target:** `{r['soft_target']}`")
            L.append(f"- **per-seed status:** `{r['per_seed_status']}`")
            L.append(
                f"- **filable_hit / crit_abs_only / crit_rel_only / recal "
                f"/ n_runs:** "
                f"{r['n_filable_hit_seeds']} / {r['n_crit_abs_only_seeds']} "
                f"/ {r['n_crit_rel_only_seeds']} / {r['n_recal_seeds']} "
                f"/ {r['n_runs']}"
            )
            L.append(f"- **max_abs_err:** {r['max_abs_err']:.3e} "
                     f"(atol={r['atol']:.2e}, 10x={10*r['atol']:.2e})")
            L.append(f"- **max_rel_err:** {r['max_rel_err']:.3e} "
                     f"(rtol={r['rtol']:.2e}, 10x={10*r['rtol']:.2e})")
            L.append(f"- **max_denom_magnitude (|y_ref|@argmax_rel):** "
                     f"{r['max_denom_magnitude']:.3e}")
            L.append("")

    if filable:
        L.append("## All FILABLE configs")
        L.append("")
        L.append("| cid | N | C | dtype | stride | red | soft | hits/n | max_abs | max_rel |")
        L.append("|-----|---|---|-------|--------|-----|------|--------|---------|---------|")
        for r in filable:
            c = r["cfg"]
            L.append(
                f"| {r['cid']} | {c['N']} | {c['C']} | `{c['dtype']}` | "
                f"`{c['stride_category']}` | `{c['reduction']}` | "
                f"{c['soft_target']} | {r['n_filable_hit']}/{r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        L.append("")

    if recal:
        L.append("## RECALIBRATION candidates (suggest xfail entries)")
        L.append("")
        L.append("| cid | N | C | dtype | stride | red | soft | recal/crit_abs/crit_rel of n | max_abs | max_rel |")
        L.append("|-----|---|---|-------|--------|-----|------|------------------------------|---------|---------|")
        for r in recal[:50]:
            c = r["cfg"]
            L.append(
                f"| {r['cid']} | {c['N']} | {c['C']} | `{c['dtype']}` | "
                f"`{c['stride_category']}` | `{c['reduction']}` | "
                f"{c['soft_target']} | "
                f"{r['n_recal']}/{r['n_crit_abs']}/{r['n_crit_rel']} of {r['n_runs']} "
                f"| {r['max_abs_err']:.3e} | {r['max_rel_err']:.3e} |"
            )
        if len(recal) > 50:
            L.append("")
            L.append(f"_... {len(recal) - 50} more recalibration entries omitted._")
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.cross_entropy` on CPU, comparison "
             "promoted to fp32.")
    L.append("- MPS execution synced via `torch.mps.synchronize()` before the "
             "device->host copy.")
    L.append("- Inputs built on CPU at fp32 then cast to native dtype; MPS sees "
             "a contiguous clone of the materialised CPU values, so MPS-vs-CPU "
             "diff isolates the cross_entropy kernel itself rather than stride "
             "lowering.")
    L.append("- bf16 probed once at startup; if unsupported, those configs are "
             "skipped (counted under iters_skipped_dtype).")
    L.append("- Targets: 70% hard (int64 class indices), 30% soft (per-row "
             "softmax distribution).")
    L.append("- Reductions sampled uniformly from {none, mean, sum}.")
    L.append("- Tolerance: `compute_tolerance(dtype, k_dim=C, device_type='mps')` "
             "where C is the class axis. Provides per-dtype base + sqrt(C/128) "
             "reduction-axis scaling + the 2x MPS overlay.")
    L.append("- Stride categories: " + ", ".join(STRIDE_CATEGORIES) + ".")
    L.append("- CUDA cross-device comparison is N/A (no NVIDIA GPU on host).")
    path.write_text("\n".join(L) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        rec = {
            "agent": "kernel-fuzzer-cross_entropy-v2",
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "top_3_repros": [],
            "torch_version": torch.__version__,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} v2 fuzz - SKIPPED\n\nMPS not available on this host.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return 0

    cfg_rng = random.Random(CONFIG_SEED)
    configs = _sample_configs(cfg_rng, N_CONFIGS)

    # Probe bf16 support once.
    bf16_supported = True
    try:
        _x = torch.randn(4, 4, dtype=torch.bfloat16, device="mps")
        _t = torch.zeros(4, dtype=torch.long, device="mps")
        _ = F.cross_entropy(_x, _t)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        bf16_supported = False
        print(f"[probe] bf16 unsupported on MPS for cross_entropy: {exc!s:.160}",
              file=sys.stderr)

    started = time.monotonic()
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_empty = 0
    iters_skipped_dtype = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0
    timed_out = False

    per_config: dict[int, dict] = {}

    # Config-major schedule so each config gets its full seed sweep before we
    # move to the next one — keeps reproducibility info intact under truncation.
    schedule = [(c, s) for c in configs for s in DATA_SEEDS]

    for c, s in schedule:
        if time.monotonic() - started > WALL_BUDGET_S:
            print(
                f"[budget] timeout at iter {iters_attempted} "
                f"(elapsed {time.monotonic()-started:.1f}s)",
                file=sys.stderr,
            )
            timed_out = True
            break

        if c["dtype"] == "bfloat16" and not bf16_supported:
            iters_skipped_dtype += 1
            iters_attempted += 1
            continue

        iters_attempted += 1
        try:
            status, abs_err, rel_err, denom_at_relmax, atol, rtol, why = _run_one(c, s)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001 - halt rule per spec
            print(f"[ERROR] cfg={c} seed={s}: {exc!r}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return 2

        if status == "UNSUPPORTED":
            iters_unsupported += 1
            continue
        if status == "EMPTY":
            iters_empty += 1
            iters_completed += 1
            continue

        iters_completed += 1
        max_abs_err_global = max(max_abs_err_global, abs_err)
        max_rel_err_global = max(max_rel_err_global, rel_err)

        slot = per_config.setdefault(c["cid"], {
            "cfg": c, "runs": {}, "atol": atol, "rtol": rtol,
        })
        slot["runs"][s] = {
            "status": status, "abs_err": abs_err, "rel_err": rel_err,
            "denom_at_relmax": denom_at_relmax, "seed": s,
        }
        if status == "FILABLE_HIT":
            print(
                f"[FILABLE_HIT] cid={c['cid']} seed={s} {c['shape_bucket']}/"
                f"{c['dtype']}/{c['stride_category']}/red={c['reduction']}/"
                f"soft={c['soft_target']} N={c['N']} C={c['C']} "
                f"abs={abs_err:.3e} rel={rel_err:.3e} denom@rel={denom_at_relmax:.3e}",
                file=sys.stderr,
            )

    elapsed = time.monotonic() - started

    # Aggregate per-config.
    filable: list[dict] = []
    recal: list[dict] = []
    for cid, slot in per_config.items():
        runs = slot["runs"]
        if not runs:
            continue
        n_runs = len(runs)
        n_filable_hit = sum(1 for r in runs.values() if r["status"] == "FILABLE_HIT")
        n_crit_abs = sum(1 for r in runs.values() if r["status"] == "CRIT_ABS_ONLY")
        n_crit_rel = sum(1 for r in runs.values() if r["status"] == "CRIT_REL_ONLY")
        n_recal = sum(1 for r in runs.values() if r["status"] == "RECAL")
        n_recal_eq = (
            n_recal
            + (n_crit_abs if n_crit_abs >= 3 else 0)
            + (n_crit_rel if n_crit_rel >= 3 else 0)
        )
        max_abs = max(r["abs_err"] for r in runs.values())
        max_rel = max(r["rel_err"] for r in runs.values())
        max_denom = max(r["denom_at_relmax"] for r in runs.values())
        record = {
            "cid": cid,
            "cfg": slot["cfg"],
            "atol": slot["atol"], "rtol": slot["rtol"],
            "n_runs": n_runs,
            "n_filable_hit": n_filable_hit,
            "n_crit_abs": n_crit_abs,
            "n_crit_rel": n_crit_rel,
            "n_recal": n_recal,
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "max_denom": max_denom,
            "per_seed": runs,
        }
        if n_filable_hit >= 3:
            filable.append(record)
        elif n_recal_eq >= 3:
            recal.append(record)

    filable.sort(
        key=lambda d: (-d["n_filable_hit"], -d["max_abs_err"], -d["max_rel_err"]),
    )
    recal.sort(
        key=lambda d: (
            -(d["n_recal"] + d["n_crit_abs"] + d["n_crit_rel"]),
            -d["max_abs_err"], -d["max_rel_err"],
        ),
    )
    top3_pool = filable + recal
    top3 = top3_pool[:3]

    summary = {
        "agent": "kernel-fuzzer-cross_entropy-v2",
        "kernel": KERNEL,
        "op_path": KERNEL,
        "device_under_test": "mps",
        "reference": "cpu_native_dtype_cast_to_fp32",
        "cuda_backend": "mocked",
        "torch_version": torch.__version__,
        "config_seed": f"0x{CONFIG_SEED:X}",
        "data_seeds": list(DATA_SEEDS),
        "n_configs_planned": N_CONFIGS,
        "n_configs_run": len(per_config),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_empty": iters_empty,
        "iters_skipped_dtype": iters_skipped_dtype,
        "bf16_supported": bf16_supported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recal),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": [
            {
                "cid": r["cid"],
                "N": r["cfg"]["N"], "C": r["cfg"]["C"],
                "dtype": r["cfg"]["dtype"],
                "stride_category": r["cfg"]["stride_category"],
                "shape_bucket": r["cfg"]["shape_bucket"],
                "reduction": r["cfg"]["reduction"],
                "soft_target": r["cfg"]["soft_target"],
                "n_filable_hit_seeds": r["n_filable_hit"],
                "n_crit_abs_only_seeds": r["n_crit_abs"],
                "n_crit_rel_only_seeds": r["n_crit_rel"],
                "n_recal_seeds": r["n_recal"],
                "n_runs": r["n_runs"],
                "max_abs_err": r["max_abs_err"],
                "max_rel_err": r["max_rel_err"],
                "max_denom_magnitude": r["max_denom"],
                "atol": r["atol"], "rtol": r["rtol"],
                "verdict": (
                    "FILABLE" if r["n_filable_hit"] >= 3 else "RECALIBRATION"
                ),
                "per_seed_status": {
                    s: rec["status"] for s, rec in r["per_seed"].items()
                },
            }
            for r in top3
        ],
        "elapsed_s": round(elapsed, 2),
        "timed_out": timed_out,
        "wall_budget_s": WALL_BUDGET_S,
        "host": "darwin/arm64 (Apple Silicon)",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_md": str(RESULTS_MD),
    }

    _write_markdown(RESULTS_MD, summary, filable, recal)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    print(
        f"[done] iters_completed={iters_completed} "
        f"filable={len(filable)} recal={len(recal)} "
        f"max_abs={max_abs_err_global:.3e} max_rel={max_rel_err_global:.3e} "
        f"elapsed={elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
