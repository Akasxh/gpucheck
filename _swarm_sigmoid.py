"""kernel-fuzzer-sigmoid (v2): MPS sigmoid divergence fuzzer."""
from __future__ import annotations

import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing import STRIDE_CATEGORIES, fuzz_shapes, fuzz_strides

KERNEL = "sigmoid"
SEEDS = [0, 1, 2, 3, 4]
DTYPES = [("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)]
TOTAL_ITERS = 1000
WALL_BUDGET_SEC = 12 * 60 - 75  # leave margin to write outputs

OUTPUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = OUTPUT_DIR / f"RESULTS_{KERNEL}.md"
SWARM_JSONL = OUTPUT_DIR / "swarm.jsonl"

start_time = time.time()


def time_left() -> float:
    return WALL_BUDGET_SEC - (time.time() - start_time)


def supported(dtype: torch.dtype) -> bool:
    try:
        x = torch.zeros((4,), dtype=dtype, device="mps")
        torch.sigmoid(x)
        return True
    except Exception:
        return False


def make_input(shape: tuple[int, ...], dtype: torch.dtype, category: str, seed: int):
    """Build a CPU tensor in the given stride category, then mirror on MPS."""
    corpus = fuzz_strides(shape, dtype, n=1, device="cpu", seed=seed, categories=(category,))
    if not corpus:
        return None
    cat, x_cpu = corpus[0]
    if x_cpu.numel() == 0:
        return None
    return cat, x_cpu


def classify(max_abs: float, max_rel: float, denom: float, atol: float, rtol: float) -> str:
    """OK | TOLERANCE_RECALIBRATION | FILABLE."""
    abs_thresh = atol
    rel_thresh = rtol
    abs_ratio = max_abs / abs_thresh if abs_thresh > 0 else 0.0
    rel_ratio = max_rel / rel_thresh if rel_thresh > 0 else 0.0

    rel_counts = denom >= 1e-6  # near-zero filter for relative-only divergence
    rel_filable = rel_ratio > 10.0 and rel_counts
    abs_filable = abs_ratio > 10.0
    if abs_filable or rel_filable:
        return "FILABLE"
    if abs_ratio > 1.0 or (rel_ratio > 1.0 and rel_counts):
        return "TOLERANCE_RECALIBRATION"
    return "OK"


def run() -> dict:
    mps_supported = {name: supported(dt) for name, dt in DTYPES}
    print(f"[swarm] MPS sigmoid support: {mps_supported}", flush=True)

    active_dtypes = [(n, d) for n, d in DTYPES if mps_supported[n]]
    unsupported = [n for n, _ in DTYPES if not mps_supported[n]]

    n_combos = len(SEEDS) * len(active_dtypes)
    iters_per_combo = max(1, TOTAL_ITERS // n_combos)

    iters_attempted = 0
    iters_completed = 0
    errors: list[dict] = []
    overall_max_abs = 0.0
    overall_max_rel_filtered = 0.0  # only count when denom >= 1e-6
    overall_max_rel_raw = 0.0

    # records keyed by signature for cross-seed reproducibility
    sig_seeds: dict[tuple, set[int]] = defaultdict(set)
    sig_records: dict[tuple, list[dict]] = defaultdict(list)
    sig_class: dict[tuple, str] = {}

    # Pre-generate shape pool per seed (ndim 2)
    shape_pools = {seed: fuzz_shapes(ndim=2, n=80, seed=seed) for seed in SEEDS}
    # Filter empty
    shape_pools = {s: [t for t in pool if all(d > 0 for d in t)] for s, pool in shape_pools.items()}

    cat_list = list(STRIDE_CATEGORIES)

    for seed in SEEDS:
        if time_left() <= 0:
            break
        torch.manual_seed(seed)
        for dt_idx, (dtype_name, dtype) in enumerate(active_dtypes):
            if time_left() <= 0:
                break
            atol, rtol = compute_tolerance(dtype, device_type="mps")
            shapes = shape_pools[seed]
            for i in range(iters_per_combo):
                if time_left() <= 0:
                    break
                iters_attempted += 1
                shape = shapes[i % len(shapes)]
                category = cat_list[(i + dt_idx) % len(cat_list)]
                sub_seed = seed * 100_003 + dt_idx * 1009 + i
                try:
                    built = make_input(shape, dtype, category, sub_seed)
                    if built is None:
                        continue
                    cat, x_cpu = built
                    x_mps = x_cpu.detach().to("mps")
                    out_cpu = torch.sigmoid(x_cpu)
                    out_mps = torch.sigmoid(x_mps).to("cpu")
                except Exception as e:
                    errors.append({
                        "seed": seed, "dtype": dtype_name, "shape": list(shape),
                        "category": category, "error": f"{type(e).__name__}: {e}",
                    })
                    continue

                # Compare in float32 to avoid second-round dtype quantization
                a = out_cpu.detach().float()
                b = out_mps.detach().float()
                diff = (a - b).abs()
                if diff.numel() == 0:
                    iters_completed += 1
                    continue
                max_abs = float(diff.max().item())
                denom = a.abs().clamp_min(1e-12)
                rel = diff / denom
                max_rel_raw = float(rel.max().item())
                # find denom at the location of the max rel
                idx = int(rel.argmax().item())
                denom_flat = denom.reshape(-1)
                denom_at_max_rel = float(denom_flat[idx].item())
                # rel filtered: only consider entries with denom >= 1e-6
                mask = denom >= 1e-6
                if mask.any():
                    rel_masked = torch.where(mask, rel, torch.zeros_like(rel))
                    max_rel_filtered = float(rel_masked.max().item())
                else:
                    max_rel_filtered = 0.0

                overall_max_abs = max(overall_max_abs, max_abs)
                overall_max_rel_filtered = max(overall_max_rel_filtered, max_rel_filtered)
                overall_max_rel_raw = max(overall_max_rel_raw, max_rel_raw)

                cls = classify(max_abs, max_rel_filtered, 1.0, atol, rtol)
                # ^ denom passed via the filtered metric; use 1.0 sentinel since denom check applied

                # signature for dedup across seeds: (dtype, category, shape, classification)
                sig = (dtype_name, cat, tuple(shape), cls)
                sig_seeds[sig].add(seed)
                sig_class[sig] = cls
                if len(sig_records[sig]) < 6:
                    sig_records[sig].append({
                        "seed": seed,
                        "dtype": dtype_name,
                        "shape": list(shape),
                        "category": cat,
                        "max_abs": max_abs,
                        "max_rel_filtered": max_rel_filtered,
                        "max_rel_raw": max_rel_raw,
                        "denom_at_max_rel_raw": denom_at_max_rel,
                        "atol": atol,
                        "rtol": rtol,
                        "abs_ratio": max_abs / atol if atol else 0.0,
                        "rel_ratio_filtered": max_rel_filtered / rtol if rtol else 0.0,
                    })
                iters_completed += 1

    # Filter: FILABLE requires reproducible across >=3 distinct seeds
    filable_sigs = [s for s, c in sig_class.items() if c == "FILABLE" and len(sig_seeds[s]) >= 3]
    recal_sigs = [s for s, c in sig_class.items() if c == "TOLERANCE_RECALIBRATION" and len(sig_seeds[s]) >= 3]

    # Top-3 worst by max_abs across FILABLE first, else recal, else any nonzero
    def worst_records(sigs: list[tuple]) -> list[dict]:
        rs: list[dict] = []
        for s in sigs:
            for r in sig_records[s]:
                rs.append(r)
        rs.sort(key=lambda r: (r["abs_ratio"], r["rel_ratio_filtered"]), reverse=True)
        return rs[:3]

    top3 = worst_records(filable_sigs) or worst_records(recal_sigs) or sorted(
        (r for rs in sig_records.values() for r in rs),
        key=lambda r: (r["abs_ratio"], r["rel_ratio_filtered"]),
        reverse=True,
    )[:3]

    summary = {
        "kernel": KERNEL,
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "divergences_filable": len(filable_sigs),
        "divergences_recalibration": len(recal_sigs),
        "max_abs_err": overall_max_abs,
        "max_rel_err": overall_max_rel_filtered,
        "max_rel_err_unfiltered": overall_max_rel_raw,
        "unsupported_dtypes": unsupported,
        "errors": len(errors),
        "first_errors": errors[:5],
        "top_3_repros": top3,
        "filable_signatures": [
            {"dtype": s[0], "category": s[1], "shape": list(s[2]), "seeds": sorted(sig_seeds[s])}
            for s in filable_sigs
        ],
        "recalibration_signatures": [
            {"dtype": s[0], "category": s[1], "shape": list(s[2]), "seeds": sorted(sig_seeds[s])}
            for s in recal_sigs[:25]
        ],
        "elapsed_sec": round(time.time() - start_time, 2),
        "wall_budget_sec": WALL_BUDGET_SEC,
        "torch_version": torch.__version__,
    }
    return summary


def render_md(s: dict) -> str:
    lines = [
        f"# Swarm fuzz results — {KERNEL}",
        "",
        f"- kernel: `{s['kernel']}`",
        f"- iters_attempted: {s['iters_attempted']}",
        f"- iters_completed: {s['iters_completed']}",
        f"- divergences_filable: {s['divergences_filable']}",
        f"- divergences_recalibration: {s['divergences_recalibration']}",
        f"- max_abs_err: {s['max_abs_err']:.6e}",
        f"- max_rel_err (denom>=1e-6): {s['max_rel_err']:.6e}",
        f"- max_rel_err (raw): {s['max_rel_err_unfiltered']:.6e}",
        f"- unsupported_dtypes: {s['unsupported_dtypes'] or 'none'}",
        f"- errors: {s['errors']}",
        f"- elapsed_sec: {s['elapsed_sec']}",
        f"- torch: {s['torch_version']}",
        "",
        "## Top 3 repros",
        "",
    ]
    if not s["top_3_repros"]:
        lines.append("_None — all comparisons were within tolerance._")
    for i, r in enumerate(s["top_3_repros"], 1):
        lines.append(f"### #{i}")
        lines.append("```json")
        lines.append(json.dumps(r, indent=2))
        lines.append("```")
        lines.append("")

    if s["filable_signatures"]:
        lines.append("## FILABLE signatures (>=3 seeds)")
        lines.append("")
        for sig in s["filable_signatures"]:
            lines.append(f"- dtype={sig['dtype']} category={sig['category']} shape={sig['shape']} seeds={sig['seeds']}")
        lines.append("")
    if s["recalibration_signatures"]:
        lines.append("## TOLERANCE_RECALIBRATION signatures (>=3 seeds, first 25)")
        lines.append("")
        for sig in s["recalibration_signatures"]:
            lines.append(f"- dtype={sig['dtype']} category={sig['category']} shape={sig['shape']} seeds={sig['seeds']}")
        lines.append("")
    if s["first_errors"]:
        lines.append("## First errors (process-level)")
        lines.append("```json")
        lines.append(json.dumps(s["first_errors"], indent=2))
        lines.append("```")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    try:
        summary = run()
    except Exception as e:
        traceback.print_exc()
        summary = {
            "kernel": KERNEL,
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "max_rel_err_unfiltered": 0.0,
            "unsupported_dtypes": [],
            "errors": 1,
            "first_errors": [{"fatal": f"{type(e).__name__}: {e}"}],
            "top_3_repros": [],
            "filable_signatures": [],
            "recalibration_signatures": [],
            "elapsed_sec": round(time.time() - start_time, 2),
            "wall_budget_sec": WALL_BUDGET_SEC,
            "torch_version": torch.__version__,
        }
    RESULTS_MD.write_text(render_md(summary))
    with SWARM_JSONL.open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    print(f"[swarm] wrote {RESULTS_MD}")
    print(f"[swarm] appended jsonl line to {SWARM_JSONL}")
    print(f"[swarm] iters_attempted={summary['iters_attempted']} iters_completed={summary['iters_completed']} "
          f"filable={summary['divergences_filable']} recal={summary['divergences_recalibration']}")
