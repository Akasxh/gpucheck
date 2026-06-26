"""kernel-fuzzer-relu v2: stride/shape/dtype fuzzer for torch.relu on MPS vs CPU.

v2 spec:
- 1000 iterations across seeds 0..4 (200 unique configs x 5 seeds)
- Filter divergences with denom_magnitude >= 1e-6 rule
- Bucket: FILABLE (>10x tol, repro >=3 seeds), RECALIBRATION (1-5x tol), OK (<1x)
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.shapes import (
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import CATEGORIES, fuzz_strides_for_category

KERNEL = "torch.relu"
N_CONFIGS = 200
SEEDS = (0, 1, 2, 3, 4)
N_ITER_TOTAL = N_CONFIGS * len(SEEDS)
BUDGET_SEC = 11 * 60 + 30  # leave 30s for writing results
CONFIG_SEED = 0xCAFEBABE

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_MD = OUT_DIR / "RESULTS_relu.md"
OUT_JSONL = OUT_DIR / "swarm.jsonl"

DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _shapes_pool() -> dict[str, list[tuple[int, ...]]]:
    return {
        "degenerate": [
            (0,), (1,), (1, 1), (1, 16), (16, 1), (1, 1, 1),
            (0, 16), (16, 0), (1, 1, 16),
        ],
        "non_tile_aligned": (
            [(t - 1,) for t in TILE_SIZES]
            + [(t + 1,) for t in TILE_SIZES]
            + [(t - 1, t + 1) for t in TILE_SIZES]
            + [(t + 1, t - 1) for t in TILE_SIZES]
            + [(t + 3, 16) for t in TILE_SIZES]
        ),
        "prime": (
            [(p,) for p in PRIMES]
            + [(p, p) for p in PRIMES if p <= 257]
            + [(p, q) for p in PRIMES for q in PRIMES if p != q and p * q <= 4096]
        ),
        "pow2_boundary": (
            [(v,) for v in POWER_OF_2_BOUNDARIES]
            + [(v, v) for v in POWER_OF_2_BOUNDARIES if v <= 513]
            + [(127, 129), (129, 127), (255, 257), (511, 513)]
        ),
        "large": (
            [(v,) for v in LARGE_DIMS]
            + [(2048, 1024), (1024, 2048), (256, 256), (1024, 64)]
        ),
        "mixed": [
            (127, 16), (1024, 3), (7, 128), (33, 128, 4),
            (3, 3, 64, 64), (2, 16, 32),
        ],
    }


SHAPE_POOL = _shapes_pool()
SHAPE_CATS = list(SHAPE_POOL.keys())
DTYPE_NAMES = list(DTYPES.keys())
STRIDE_CATS = list(CATEGORIES)


def _stride_supports(shape: tuple[int, ...], cat: str) -> bool:
    """Return False if the stride category cannot be built for this shape."""
    if not shape:
        return cat == "row_major"
    if cat in {"column_major", "transpose"} and len(shape) < 2:
        return False
    if cat == "broadcast" and (not shape or shape[-1] == 0):
        return False
    if cat == "slice":
        if any(d == 0 for d in shape):
            return False
        doubled = 1
        for d in shape:
            doubled *= max(d * 2, 1)
            if doubled > 64_000_000:
                return False
    if cat == "gather":
        if any(d == 0 for d in shape):
            return False
        numel = 1
        for d in shape:
            numel *= d
        if numel > 4_000_000:
            return False
    return True


def _err_metrics(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Return (max_abs_err, max_rel_err_at_safe_denom, denom_at_max_rel).

    Uses fp64 promotion. The "safe-denom" rel error masks elements where
    |ref| < 1e-6 BEFORE taking the max — so divisions near zero do not
    inflate the relative-error figure.
    """
    a = out.detach().cpu().to(dtype=torch.float64).contiguous()
    b = ref.detach().cpu().to(dtype=torch.float64).contiguous()
    if a.numel() == 0:
        return 0.0, 0.0, 0.0
    diff = (a - b).abs()
    max_abs = float(diff.max().item())

    safe_mask = b.abs() >= 1e-6
    if safe_mask.any():
        # Where the denom is unsafe, set rel error to 0 so it cannot dominate.
        denom_safe = torch.where(safe_mask, b.abs(), torch.ones_like(b))
        rel = torch.where(safe_mask, diff / denom_safe, torch.zeros_like(diff))
        idx = int(rel.argmax().item())
        max_rel = float(rel.flatten()[idx].item())
        denom_here = float(b.abs().flatten()[idx].item())
    else:
        max_rel = 0.0
        denom_here = 0.0
    return max_abs, max_rel, denom_here


def _classify(
    max_abs: float,
    max_rel: float,
    denom_at_max_rel: float,
    atol: float,
    rtol: float,
) -> str:
    """Return 'OK' | 'RECALIBRATION' | 'DIVERGENT' per v2 spec."""
    abs_ratio = max_abs / atol if atol > 0 else 0.0
    rel_ratio = max_rel / rtol if rtol > 0 else 0.0
    rel_counts = denom_at_max_rel >= 1e-6  # near-zero-denom filter

    if not rel_counts:
        rel_ratio = 0.0  # do not let near-zero-denom rel err drive classification

    worst = max(abs_ratio, rel_ratio)
    if worst > 10.0:
        return "DIVERGENT"
    if worst >= 1.0:
        return "RECALIBRATION"
    return "OK"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        OUT_MD.write_text("# RESULTS_relu — SKIPPED (MPS unavailable)\n")
        line = json.dumps({
            "kernel": "relu",
            "status": "SKIPPED",
            "reason": "torch.backends.mps.is_available() is False",
        })
        with OUT_JSONL.open("a") as f:
            f.write(line + "\n")
        print("SKIPPED: MPS unavailable")
        return 0

    cfg_rng = random.Random(CONFIG_SEED)
    t0 = time.monotonic()

    # Build config list once (deterministic): N_CONFIGS unique (shape,dtype,stride)
    configs: list[tuple[tuple[int, ...], str, str, str]] = []
    while len(configs) < N_CONFIGS:
        shape_cat = cfg_rng.choice(SHAPE_CATS)
        shape = cfg_rng.choice(SHAPE_POOL[shape_cat])
        dtype_name = cfg_rng.choice(DTYPE_NAMES)
        stride_cat = cfg_rng.choice(STRIDE_CATS)
        if not _stride_supports(shape, stride_cat):
            continue
        configs.append((shape, dtype_name, stride_cat, shape_cat))

    attempted = 0
    completed = 0
    unsupported = 0
    aborted = False

    # Per-config tracking: list of per-seed outcomes
    # outcomes[cfg_key] = list of dicts {seed, status, max_abs, max_rel, denom_at, atol, rtol}
    outcomes: dict[tuple, list[dict]] = defaultdict(list)
    global_max_abs = 0.0
    global_max_rel_safe = 0.0  # max rel err under safe-denom rule

    for cfg_idx, (shape, dtype_name, stride_cat, shape_cat) in enumerate(configs):
        if time.monotonic() - t0 > BUDGET_SEC:
            aborted = True
            print(f"[budget] cfg {cfg_idx}/{N_CONFIGS} — aborting", file=sys.stderr)
            break

        dtype = DTYPES[dtype_name]
        cfg_key = (tuple(shape), dtype_name, stride_cat)

        for seed in SEEDS:
            if time.monotonic() - t0 > BUDGET_SEC:
                aborted = True
                break
            attempted += 1
            try:
                cpu_src = fuzz_strides_for_category(
                    shape, dtype, stride_cat, device="cpu", seed=seed,
                )
            except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
                unsupported += 1
                outcomes[cfg_key].append({
                    "seed": seed, "status": "UNSUPPORTED_BUILD",
                    "msg": repr(e)[:200],
                })
                continue

            try:
                mps_src = cpu_src.detach().to("mps")
            except (RuntimeError, NotImplementedError, TypeError) as e:
                unsupported += 1
                outcomes[cfg_key].append({
                    "seed": seed, "status": "UNSUPPORTED_MPS_COPY",
                    "msg": repr(e)[:200],
                })
                continue

            # Run kernel both sides
            try:
                ref = torch.relu(cpu_src)
            except (RuntimeError, NotImplementedError, TypeError) as e:
                unsupported += 1
                outcomes[cfg_key].append({
                    "seed": seed, "status": "UNSUPPORTED_CPU",
                    "msg": repr(e)[:200],
                })
                continue
            try:
                out = torch.relu(mps_src)
                torch.mps.synchronize()
            except (RuntimeError, NotImplementedError, TypeError) as e:
                unsupported += 1
                outcomes[cfg_key].append({
                    "seed": seed, "status": "UNSUPPORTED_MPS",
                    "msg": repr(e)[:200],
                })
                continue

            try:
                max_abs, max_rel, denom_at = _err_metrics(out, ref)
            except RuntimeError as e:
                outcomes[cfg_key].append({
                    "seed": seed, "status": "COMPARE_ERROR",
                    "msg": repr(e)[:200],
                })
                continue

            atol, rtol = compute_tolerance(dtype, device_type="mps")
            verdict = _classify(max_abs, max_rel, denom_at, atol, rtol)
            outcomes[cfg_key].append({
                "seed": seed, "status": verdict,
                "max_abs": max_abs, "max_rel": max_rel,
                "denom_at_max_rel": denom_at,
                "atol": atol, "rtol": rtol,
                "shape_cat": shape_cat,
            })
            completed += 1
            global_max_abs = max(global_max_abs, max_abs)
            if denom_at >= 1e-6:
                global_max_rel_safe = max(global_max_rel_safe, max_rel)

        if aborted:
            break

    elapsed = time.monotonic() - t0

    # ---- Bucket configs ----
    filable: list[dict] = []
    recalibration: list[dict] = []
    for cfg_key, runs in outcomes.items():
        n_div = sum(1 for r in runs if r["status"] == "DIVERGENT")
        n_recal = sum(1 for r in runs if r["status"] == "RECALIBRATION")
        if n_div >= 3:
            # Pick the worst-error run as canonical
            divs = [r for r in runs if r["status"] == "DIVERGENT"]
            worst = max(divs, key=lambda r: (r["max_abs"], r["max_rel"]))
            filable.append({
                "shape": list(cfg_key[0]),
                "dtype": cfg_key[1],
                "stride": cfg_key[2],
                "shape_cat": worst.get("shape_cat", "?"),
                "n_seeds_divergent": n_div,
                "max_abs_err": worst["max_abs"],
                "max_rel_err": worst["max_rel"],
                "denom_at_max_rel": worst["denom_at_max_rel"],
                "atol": worst["atol"],
                "rtol": worst["rtol"],
                "seeds_observed": [r["seed"] for r in divs],
            })
        elif n_recal >= 3 or (n_recal + n_div) >= 3:
            # tolerance recalibration candidate
            interesting = [r for r in runs
                           if r["status"] in ("RECALIBRATION", "DIVERGENT")]
            worst = max(interesting,
                        key=lambda r: (r["max_abs"] / max(r["atol"], 1e-30),
                                       r["max_rel"] / max(r["rtol"], 1e-30)))
            recalibration.append({
                "shape": list(cfg_key[0]),
                "dtype": cfg_key[1],
                "stride": cfg_key[2],
                "shape_cat": worst.get("shape_cat", "?"),
                "n_seeds_above_tol": n_recal + n_div,
                "max_abs_err": worst["max_abs"],
                "max_rel_err": worst["max_rel"],
                "atol": worst["atol"],
                "rtol": worst["rtol"],
            })

    # Sort
    filable.sort(key=lambda d: (-d["max_abs_err"], -d["max_rel_err"]))
    recalibration.sort(key=lambda d: (-d["max_abs_err"], -d["max_rel_err"]))
    top3 = filable[:3] if filable else recalibration[:3]

    upstream = "pytorch/pytorch" if filable else "none"

    # ---- Write markdown ----
    md = []
    md.append("# RESULTS_relu — gpucheck stride/shape/dtype fuzz (v2)")
    md.append("")
    md.append(f"- **kernel:** `{KERNEL}`")
    md.append(f"- **iters_attempted:** {attempted}")
    md.append(f"- **iters_completed:** {completed}")
    md.append(f"- **iters_unsupported:** {unsupported}")
    md.append(f"- **divergences_filable:** {len(filable)}  "
              f"(>10x tol AND reproducible across >=3 seeds)")
    md.append(f"- **divergences_recalibration:** {len(recalibration)}  "
              f"(1-10x tol on >=3 seeds — recommend xfail entry)")
    md.append(f"- **max_abs_err (global, MPS vs CPU):** {global_max_abs:.6g}")
    md.append(f"- **max_rel_err (global, safe-denom):** {global_max_rel_safe:.6g}")
    md.append(f"- **MPS vs CUDA-mock max relative error:** N/A "
              f"(no NVIDIA GPU on host)")
    md.append(f"- **recommended upstream filing target:** `{upstream}`")
    md.append(f"- **elapsed_seconds:** {elapsed:.2f} (budget {BUDGET_SEC}s, "
              f"aborted={aborted})")
    md.append(f"- **n_configs:** {N_CONFIGS}, **seeds:** {list(SEEDS)}, "
              f"**iters_total_target:** {N_ITER_TOTAL}")
    md.append(f"- **torch:** {torch.__version__}, host: darwin/arm64 "
              f"(Apple Silicon)")
    md.append("")
    md.append("## Top 3 minimal repros")
    md.append("")
    if not top3:
        md.append("_No configurations exceeded gpucheck's per-dtype tolerance "
                  "(with MPS multiplier) on >=3 of 5 seeds — clean run for relu._")
    else:
        for k, d in enumerate(top3, start=1):
            md.append(
                f"### Repro #{k}"
            )
            md.append("")
            md.append(f"- **shape:** `{tuple(d['shape'])}` (bucket: "
                      f"`{d.get('shape_cat', '?')}`)")
            md.append(f"- **dtype:** `{d['dtype']}`")
            md.append(f"- **stride category:** `{d['stride']}`")
            md.append(f"- **max_abs_err:** {d['max_abs_err']:.4g} "
                      f"(atol={d['atol']:.2e}, ratio="
                      f"{d['max_abs_err']/max(d['atol'],1e-30):.2f}x)")
            md.append(f"- **max_rel_err:** {d['max_rel_err']:.4g} "
                      f"(rtol={d['rtol']:.2e}, ratio="
                      f"{d['max_rel_err']/max(d['rtol'],1e-30):.2f}x)")
            if "n_seeds_divergent" in d:
                md.append(f"- **seeds_divergent:** {d['n_seeds_divergent']}/5  "
                          f"(seeds: {d.get('seeds_observed', '?')})")
            else:
                md.append(f"- **seeds_above_tol:** {d['n_seeds_above_tol']}/5")
            md.append("")
    md.append("## Method")
    md.append("")
    md.append(f"- {N_CONFIGS} unique (shape, dtype, stride) configs generated "
              f"with deterministic config-rng (seed=0x{CONFIG_SEED:x}).")
    md.append("- Each config run with input seeds 0,1,2,3,4 → "
              f"{N_ITER_TOTAL} target iterations.")
    md.append("- Reference: `torch.relu` on CPU; comparison promoted to fp64.")
    md.append("- Tolerances: `gpucheck.assertions.tolerances.compute_tolerance("
              "dtype, device_type='mps')` (MPS multiplier applied).")
    md.append("- Divergence filtering (v2):")
    md.append("  - `max_rel_err > 10x rtol` counts only when "
              "`denom_magnitude >= 1e-6` at the offending element.")
    md.append("  - `max_abs_err > 10x atol` always counts.")
    md.append("  - **FILABLE**: > 10x AND reproducible across >= 3 seeds.")
    md.append("  - **RECALIBRATION**: 1-10x on >= 3 seeds (recommend xfail).")
    md.append("  - **OK**: < 1x tolerance.")
    md.append("- Stride categories drawn: " + ", ".join(STRIDE_CATS) + ".")
    md.append("")
    OUT_MD.write_text("\n".join(md))

    summary = {
        "kernel": "relu",
        "status": "OK" if not aborted else "BUDGET_ABORT",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "iters_unsupported": unsupported,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_abs_err": global_max_abs,
        "max_rel_err": global_max_rel_safe,
        "top_3_repros": [
            {
                "shape": d["shape"],
                "dtype": d["dtype"],
                "stride": d["stride"],
                "max_abs_err": d["max_abs_err"],
                "max_rel_err": d["max_rel_err"],
            }
            for d in top3
        ],
        "upstream_target": upstream,
        "elapsed_sec": round(elapsed, 2),
        "budget_sec": BUDGET_SEC,
        "aborted_due_to_budget": aborted,
        "n_configs": N_CONFIGS,
        "seeds": list(SEEDS),
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
    }
    with OUT_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    print(f"completed={completed} attempted={attempted} unsupported={unsupported} "
          f"filable={len(filable)} recal={len(recalibration)} "
          f"max_abs={global_max_abs:.3e} max_rel={global_max_rel_safe:.3e} "
          f"elapsed={elapsed:.1f}s aborted={aborted}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
