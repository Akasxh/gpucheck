"""kernel-fuzzer-layernorm v2: stride/contig + shape + dtype fuzz of layer_norm vs CPU.

Spec: 1000 iters over seeds {0,1,2,3,4}, dtypes {fp32, fp16, bf16}.
Divergence filter:
  - max_abs_err > 10x tol  -> always counts
  - max_rel_err > 10x tol  -> counts ONLY if denom_magnitude >= 1e-6
  - reproducible across >=3 distinct seeds  -> FILABLE
  - 1x..5x tolerance       -> TOLERANCE_RECALIBRATION (recommend xfail)
  - <1x                    -> OK
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Inject worktree src so gpucheck imports resolve to the merged release/v1.0.
WT = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-layernorm")
sys.path.insert(0, str(WT / "src"))

import torch
import torch.nn.functional as F

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.backends.mps import MPSBackend
from gpucheck.fuzzing.shapes import (
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import (
    CATEGORIES as STRIDE_CATEGORIES,
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_layernorm.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.layer_norm"
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
ITERS_PER_SEED = 200            # 5 * 200 = 1000
WALL_BUDGET_S = 12 * 60 - 45    # leave 45 s for write + summarise

# layer_norm needs the last dim >= 2 (var of 1 elt -> NaN). Buckets enforce it.
DEGENERATE: list[tuple[int, ...]] = [
    (2,), (1, 2), (1, 1, 2),
    (16, 2), (3, 5),
    (1, 16, 2),
]
NON_TILE: list[tuple[int, ...]] = [
    (TILE_SIZES[0] - 1, TILE_SIZES[0] + 1),     # 31 x 33
    (TILE_SIZES[1] - 1, TILE_SIZES[1] + 3),     # 63 x 67
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),     # 127 x 129
    (TILE_SIZES[2] + 1, TILE_SIZES[1] - 1),     # 129 x 63
    (16, TILE_SIZES[2] - 1),                    # 16 x 127
    (4, 16, TILE_SIZES[1] + 3),                 # 4 x 16 x 67
]
PRIME_S: list[tuple[int, ...]] = [
    (PRIMES[0], PRIMES[1]),
    (PRIMES[1], PRIMES[2]),
    (PRIMES[2], PRIMES[3]),
    (16, PRIMES[1]),
    (4, 8, PRIMES[2]),
]
POW2_BOUNDARY: list[tuple[int, ...]] = [
    (16, 64), (32, 128), (64, 256),
    (16, 65), (32, 127), (64, 257),
    (4, 16, 128),
]
LARGE: list[tuple[int, ...]] = [
    (256, 512),
    (128, 1024),
    (LARGE_DIMS[0], 64),                        # tall thin
    (32, 32, 256),                              # 3D
]
MIXED: list[tuple[int, ...]] = [
    (33, 16, 32),
    (3, 7, 64),
    (1, 1, 1, 64),
    (2, 3, 4, 5),
    (8, PRIMES[2], 4),
]

SHAPE_BUCKETS: dict[str, list[tuple[int, ...]]] = {
    "degenerate": DEGENERATE,
    "non_tile_aligned": NON_TILE,
    "prime": PRIME_S,
    "power_of_2_boundary": POW2_BOUNDARY,
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


def _max_abs_err(a_cpu: torch.Tensor, b_cpu: torch.Tensor) -> float:
    a = a_cpu.to(torch.float32)
    b = b_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0
    return float((a - b).abs().max().item())


def _max_rel_err_with_denom(
    mps_cpu: torch.Tensor, ref_cpu: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_argmax_rel)."""
    a = mps_cpu.to(torch.float32)
    b = ref_cpu.to(torch.float32)
    if a.numel() == 0:
        return 0.0, 0.0
    diff = (a - b).abs()
    eps = 1e-12
    denom = b.abs().clamp_min(eps)
    rel = diff / denom
    flat_rel = rel.flatten()
    flat_denom = b.abs().flatten()
    idx = int(torch.argmax(flat_rel).item())
    return float(flat_rel[idx].item()), float(flat_denom[idx].item())


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _config_key(shape: tuple[int, ...], dtype_name: str, stride_cat: str) -> str:
    return f"{tuple(shape)!r}|{dtype_name}|{stride_cat}"


def _classify(
    abs_err: float, rel_err: float, atol: float, rtol: float,
    denom_at_rel: float,
) -> str:
    """One-of: ok, recalibration, divergence_abs, divergence_rel."""
    abs_div = abs_err > 10.0 * atol
    rel_div = (rel_err > 10.0 * rtol) and (denom_at_rel >= 1e-6)
    if abs_div:
        return "divergence_abs"
    if rel_div:
        return "divergence_rel"
    abs_recal = atol < abs_err <= 5.0 * atol
    rel_recal = rtol < rel_err <= 5.0 * rtol
    if abs_recal or rel_recal:
        return "recalibration"
    return "ok"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = MPSBackend()

    if not backend.is_available():
        skipped = {
            "kernel": KERNEL,
            "status": "SKIPPED",
            "reason": "MPSBackend.is_available() is False",
            "iters_attempted": 0,
            "iters_completed": 0,
            "divergences_filable": 0,
            "divergences_recalibration": 0,
            "max_abs_err": 0.0,
            "max_rel_err": 0.0,
            "top_3_repros": [],
        }
        RESULTS_MD.write_text(
            f"# {KERNEL} fuzz v2 - SKIPPED\n\nMPSBackend reports unavailable.\n"
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    started = time.monotonic()
    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    iters_hard_error = 0

    max_abs_err_global = 0.0
    max_rel_err_global = 0.0          # filtered (denom >= 1e-6)
    max_rel_err_unfiltered = 0.0      # diagnostic

    config_div_seeds: dict[str, set[int]] = defaultdict(set)
    config_repros: dict[str, dict] = {}

    recalibration_records: list[dict] = []
    divergence_records: list[dict] = []

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}
    abort_reason = ""

    for seed in SEEDS:
        sub_rng = random.Random(seed)
        for i in range(ITERS_PER_SEED):
            elapsed = time.monotonic() - started
            if elapsed > WALL_BUDGET_S:
                abort_reason = (
                    f"wall budget {WALL_BUDGET_S}s exceeded at seed={seed} "
                    f"iter={i} (elapsed {elapsed:.1f}s)"
                )
                print(f"[budget] {abort_reason}", file=sys.stderr)
                break

            iters_attempted += 1

            bucket = sub_rng.choice(BUCKET_NAMES)
            shape = sub_rng.choice(SHAPE_BUCKETS[bucket])
            dtype_name = sub_rng.choice(DTYPE_NAMES)
            dtype = DTYPES_BY_NAME[dtype_name]
            stride_cat = sub_rng.choice(STRIDE_CATEGORIES)

            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            tensor_seed = sub_rng.randrange(2**31 - 1)

            normalized_shape = (shape[-1],)
            if shape[-1] < 2:
                iters_completed += 1
                continue

            try:
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=tensor_seed,
                    )
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=tensor_seed,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build] seed={seed} iter={i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                use_affine = sub_rng.random() < 0.5
                if use_affine:
                    w_cpu = torch.randn(
                        normalized_shape, dtype=torch.float32,
                    ).to(dtype=dtype)
                    b_cpu = torch.randn(
                        normalized_shape, dtype=torch.float32,
                    ).to(dtype=dtype)
                    w_mps = w_cpu.to(device="mps")
                    b_mps = b_cpu.to(device="mps")
                else:
                    w_cpu = b_cpu = w_mps = b_mps = None

                try:
                    y_mps = F.layer_norm(
                        x_mps, normalized_shape, weight=w_mps, bias=b_mps,
                    )
                    backend.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] seed={seed} iter={i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.layer_norm(
                        x_cpu, normalized_shape, weight=w_cpu, bias=b_cpu,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] seed={seed} iter={i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                if (not torch.isfinite(y_cpu).all()
                        or not torch.isfinite(y_mps_cpu).all()):
                    rec = {
                        "kind": "non_finite",
                        "iter": i,
                        "seed": seed,
                        "shape": list(shape),
                        "dtype": dtype_name,
                        "stride_category": stride_cat,
                        "shape_bucket": bucket,
                        "max_abs_err": float("inf"),
                        "max_rel_err": float("inf"),
                        "denom_at_rel": float("inf"),
                        "atol": float("nan"),
                        "rtol": float("nan"),
                        "x_layout_cpu": _stride_tag(x_cpu),
                        "x_layout_mps": _stride_tag(x_mps),
                        "tensor_seed": tensor_seed,
                        "use_affine": use_affine,
                    }
                    divergence_records.append(rec)
                    ck = _config_key(shape, dtype_name, stride_cat)
                    config_div_seeds[ck].add(seed)
                    config_repros.setdefault(ck, rec)
                    continue

                abs_err = _max_abs_err(y_mps_cpu, y_cpu)
                rel_err, denom_at_rel = _max_rel_err_with_denom(y_mps_cpu, y_cpu)

                max_abs_err_global = max(max_abs_err_global, abs_err)
                max_rel_err_unfiltered = max(max_rel_err_unfiltered, rel_err)
                if denom_at_rel >= 1e-6:
                    max_rel_err_global = max(max_rel_err_global, rel_err)

                k_dim = 1
                for d in normalized_shape:
                    k_dim *= int(d)
                atol, rtol = compute_tolerance(
                    dtype, k_dim=k_dim, device_type="mps",
                )

                cls = _classify(abs_err, rel_err, atol, rtol, denom_at_rel)

                base_rec = {
                    "kind": cls,
                    "iter": i,
                    "seed": seed,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "shape_bucket": bucket,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "denom_at_rel": denom_at_rel,
                    "atol": atol,
                    "rtol": rtol,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                    "tensor_seed": tensor_seed,
                    "use_affine": use_affine,
                }

                if cls in ("divergence_abs", "divergence_rel"):
                    divergence_records.append(base_rec)
                    ck = _config_key(shape, dtype_name, stride_cat)
                    config_div_seeds[ck].add(seed)
                    config_repros.setdefault(ck, base_rec)
                    print(
                        f"[DIVERGE/{cls}] seed={seed} iter={i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                        f"abs={abs_err:.3e} rel={rel_err:.3e} "
                        f"denom={denom_at_rel:.3e} "
                        f"(atol={atol:.2e} rtol={rtol:.2e})",
                        file=sys.stderr,
                    )
                elif cls == "recalibration":
                    recalibration_records.append(base_rec)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                iters_hard_error += 1
                print(
                    f"[HARD-ERROR] seed={seed} iter={i} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                summary = _build_summary(
                    started=started,
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported=iters_unsupported,
                    iters_hard_error=iters_hard_error,
                    max_abs_err=max_abs_err_global,
                    max_rel_err=max_rel_err_global,
                    max_rel_err_unfiltered=max_rel_err_unfiltered,
                    config_div_seeds=config_div_seeds,
                    config_repros=config_repros,
                    divergence_records=divergence_records,
                    recalibration_records=recalibration_records,
                    per_bucket_counts=per_bucket_counts,
                    per_dtype_counts=per_dtype_counts,
                    per_stride_counts=per_stride_counts,
                    aborted="hard_error",
                    abort_detail=repr(exc),
                )
                _emit(summary)
                return 2

        if abort_reason:
            break

    summary = _build_summary(
        started=started,
        iters_attempted=iters_attempted,
        iters_completed=iters_completed,
        iters_unsupported=iters_unsupported,
        iters_hard_error=iters_hard_error,
        max_abs_err=max_abs_err_global,
        max_rel_err=max_rel_err_global,
        max_rel_err_unfiltered=max_rel_err_unfiltered,
        config_div_seeds=config_div_seeds,
        config_repros=config_repros,
        divergence_records=divergence_records,
        recalibration_records=recalibration_records,
        per_bucket_counts=per_bucket_counts,
        per_dtype_counts=per_dtype_counts,
        per_stride_counts=per_stride_counts,
        aborted=("budget" if abort_reason else ""),
        abort_detail=abort_reason,
    )
    _emit(summary)
    return 0


def _build_summary(
    *,
    started: float,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    iters_hard_error: int,
    max_abs_err: float,
    max_rel_err: float,
    max_rel_err_unfiltered: float,
    config_div_seeds: dict[str, set[int]],
    config_repros: dict[str, dict],
    divergence_records: list[dict],
    recalibration_records: list[dict],
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
    aborted: str,
    abort_detail: str,
) -> dict:
    elapsed = time.monotonic() - started

    filable_configs = {
        ck: sorted(seeds) for ck, seeds in config_div_seeds.items() if len(seeds) >= 3
    }
    unconfirmed_configs = {
        ck: sorted(seeds) for ck, seeds in config_div_seeds.items() if len(seeds) < 3
    }

    recal_by_config: dict[str, dict] = {}
    for r in recalibration_records:
        ck = _config_key(tuple(r["shape"]), r["dtype"], r["stride_category"])
        recal_by_config.setdefault(ck, r)

    def _repro_block(ck: str, seeds: list[int]) -> dict:
        rep = config_repros[ck]
        return {
            "config_key": ck,
            "shape": rep["shape"],
            "dtype": rep["dtype"],
            "stride_category": rep["stride_category"],
            "shape_bucket": rep["shape_bucket"],
            "use_affine": rep["use_affine"],
            "max_abs_err": rep["max_abs_err"],
            "max_rel_err": rep["max_rel_err"],
            "denom_at_rel": rep["denom_at_rel"],
            "atol": rep["atol"],
            "rtol": rep["rtol"],
            "x_layout_cpu": rep["x_layout_cpu"],
            "x_layout_mps": rep["x_layout_mps"],
            "tensor_seed": rep["tensor_seed"],
            "seeds_with_divergence": seeds,
        }

    filable_sorted = sorted(
        filable_configs.items(),
        key=lambda kv: (-config_repros[kv[0]]["max_abs_err"], kv[0]),
    )
    unconfirmed_sorted = sorted(
        unconfirmed_configs.items(),
        key=lambda kv: (-config_repros[kv[0]]["max_abs_err"], kv[0]),
    )

    top_repros: list[dict] = []
    for ck, seeds in filable_sorted:
        if len(top_repros) >= 3:
            break
        top_repros.append(_repro_block(ck, seeds))
    for ck, seeds in unconfirmed_sorted:
        if len(top_repros) >= 3:
            break
        top_repros.append(_repro_block(ck, seeds))

    return {
        "kernel": KERNEL,
        "status": (
            "OK" if iters_hard_error == 0 and not aborted
            else ("ABORTED_BUDGET" if aborted == "budget" else "ABORTED_ERROR")
        ),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "iters_hard_error": iters_hard_error,
        "divergences_filable": len(filable_configs),
        "divergences_unconfirmed_unique_configs": len(unconfirmed_configs),
        "divergences_total_records": len(divergence_records),
        "divergences_recalibration": len(recal_by_config),
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "max_rel_err_unfiltered": max_rel_err_unfiltered,
        "elapsed_seconds": round(elapsed, 2),
        "top_3_repros": top_repros,
        "filable_configs": [
            {"config_key": ck, "seeds_with_divergence": seeds}
            for ck, seeds in filable_sorted
        ],
        "recalibration_recommendations": [
            {
                "config_key": ck,
                "shape": r["shape"],
                "dtype": r["dtype"],
                "stride_category": r["stride_category"],
                "max_abs_err": r["max_abs_err"],
                "max_rel_err": r["max_rel_err"],
                "atol": r["atol"],
                "rtol": r["rtol"],
                "suggested_xfail_id": (
                    f"layer_norm.{r['dtype']}.{r['stride_category']}"
                ),
            }
            for ck, r in sorted(recal_by_config.items())
        ],
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "iters_per_seed": ITERS_PER_SEED,
        "wall_budget_s": WALL_BUDGET_S,
        "aborted": aborted,
        "abort_detail": abort_detail,
        "recommended_filing_target": (
            "pytorch/pytorch" if filable_configs else "none"
        ),
    }


def _emit(summary: dict) -> None:
    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(path: Path, s: dict) -> None:
    L: list[str] = []
    L.append(f"# {s['kernel']} - MPS fuzz v2 report")
    L.append("")
    L.append("- **Kernel:** `{}`".format(s["kernel"]))
    L.append("- **Status:** {}".format(s["status"]))
    L.append("- **iters_attempted:** {}".format(s["iters_attempted"]))
    L.append("- **iters_completed:** {}".format(s["iters_completed"]))
    L.append("- **iters_unsupported (MPS rejected):** {}".format(
        s["iters_unsupported"]))
    L.append("- **iters_hard_error:** {}".format(s["iters_hard_error"]))
    L.append("- **divergences_filable (>=3 seeds, >10x tol):** {}".format(
        s["divergences_filable"]))
    L.append("- **divergences_recalibration (1x..5x tol, unique configs):** {}".format(
        s["divergences_recalibration"]))
    L.append("- **divergences_unconfirmed (>10x tol, <3 seeds, unique configs):** "
             "{}".format(s["divergences_unconfirmed_unique_configs"]))
    L.append("- **max_abs_err:** {:.3e}".format(s["max_abs_err"]))
    L.append("- **max_rel_err (denom>=1e-6):** {:.3e}".format(s["max_rel_err"]))
    L.append("- **max_rel_err (unfiltered, near-zero artifacts shown):** "
             "{:.3e}".format(s["max_rel_err_unfiltered"]))
    L.append("- **elapsed:** {} s (budget {} s)".format(
        s["elapsed_seconds"], s["wall_budget_s"]))
    L.append("- **torch:** {} on {}".format(s["torch_version"], s["host"]))
    L.append("- **seeds:** {} x {} iters = {} attempted".format(
        s["seeds"], s["iters_per_seed"], len(s["seeds"]) * s["iters_per_seed"]))
    if s.get("aborted"):
        L.append("- **aborted:** {} ({})".format(s["aborted"], s["abort_detail"]))
    L.append("")

    L.append("## Sampling distribution")
    L.append("")
    L.append("| dimension | counts |")
    L.append("|---|---|")
    L.append("| shape bucket | {} |".format(s["per_shape_bucket"]))
    L.append("| dtype | {} |".format(s["per_dtype"]))
    L.append("| stride category | {} |".format(s["per_stride_category"]))
    L.append("")

    L.append("## Top 3 repros (FILABLE first; then top unconfirmed)")
    L.append("")
    if not s["top_3_repros"]:
        L.append(
            "_No (>10x tol) divergences with non-trivial denominator passed the "
            "v2 filter. layer_norm output is well-behaved on MPS for this corpus._"
        )
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append("### Repro #{}".format(i))
            L.append("")
            L.append("- **shape:** `{}`  (bucket: `{}`)".format(
                tuple(r["shape"]), r["shape_bucket"]))
            L.append("- **dtype:** `{}`".format(r["dtype"]))
            L.append("- **stride category:** `{}`".format(r["stride_category"]))
            L.append("- **affine (weight+bias) used:** {}".format(r["use_affine"]))
            ratio_a = (r["max_abs_err"] / r["atol"]) if r["atol"] else float("inf")
            ratio_r = (r["max_rel_err"] / r["rtol"]) if r["rtol"] else float("inf")
            L.append(
                "- **max_abs_err:** {:.3e}  (atol={:.2e}, ratio={:.2f}x)".format(
                    r["max_abs_err"], r["atol"], ratio_a,
                )
            )
            L.append(
                "- **max_rel_err:** {:.3e}  (rtol={:.2e}, ratio={:.2f}x, "
                "denom={:.3e})".format(
                    r["max_rel_err"], r["rtol"], ratio_r, r["denom_at_rel"],
                )
            )
            L.append("- **CPU layout:** `{}`".format(r["x_layout_cpu"]))
            L.append("- **MPS layout:** `{}`".format(r["x_layout_mps"]))
            L.append("- **tensor seed:** {}".format(r["tensor_seed"]))
            L.append("- **seeds reproducing divergence:** {} "
                     "(>=3 == FILABLE)".format(r["seeds_with_divergence"]))
            L.append("")

    if s["recalibration_recommendations"]:
        L.append("## TOLERANCE_RECALIBRATION recommendations (xfail / atol bumps)")
        L.append("")
        L.append("Configs whose error landed in (1x .. 5x] tolerance on at least "
                 "one seed. Recommend an entry in `[tool.gpucheck.mps.xfail]` or "
                 "a per-dtype atol bump.")
        L.append("")
        L.append("| suggested_xfail_id | shape | dtype | stride | abs/atol | "
                 "rel/rtol |")
        L.append("|---|---|---|---|---|---|")
        for r in s["recalibration_recommendations"][:25]:
            ratio_a = (r["max_abs_err"] / r["atol"]) if r["atol"] else float("inf")
            ratio_r = (r["max_rel_err"] / r["rtol"]) if r["rtol"] else float("inf")
            L.append(
                "| `{xid}` | `{shape}` | `{dtype}` | `{strd}` "
                "| {ra:.2f}x | {rr:.2f}x |".format(
                    xid=r["suggested_xfail_id"],
                    shape=tuple(r["shape"]),
                    dtype=r["dtype"],
                    strd=r["stride_category"],
                    ra=ratio_a, rr=ratio_r,
                )
            )
        L.append("")

    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.layer_norm` on CPU, output cast "
             "to float32 for error metrics.")
    L.append("- Backend: `gpucheck.backends.mps.MPSBackend`; "
             "`backend.synchronize()` (device-level sync, deadlock-safe per "
             "pytorch#162872) called before pulling MPS output back to CPU.")
    L.append("- Tolerance: `compute_tolerance(dtype, k_dim=last_dim, "
             "device_type='mps')`; k_dim scales atol per gpucheck's "
             "CUTLASS-style sqrt(k/128) error model.")
    L.append("- Divergence filter (v2 spec):")
    L.append("  - `max_abs_err > 10x atol` -> always counts.")
    L.append("  - `max_rel_err > 10x rtol` -> counts only when "
             "`|y_cpu_at_argmax_rel| >= 1e-6`; otherwise dropped as a "
             "near-zero-denom artifact.")
    L.append("  - **FILABLE** = same `(shape, dtype, stride)` config diverged "
             "on >=3 of the 5 seeds (0..4).")
    L.append("- Recalibration bucket: error in (1x .. 5x] tol on at least one "
             "seed -> recommend xfail or atol bump rather than a bug filing.")
    L.append("- Stride categories from `gpucheck.fuzzing.strides`: " +
             ", ".join(STRIDE_CATEGORIES) + ".")
    L.append("- CUDA backend mocked on this host (no NVIDIA GPU); MPS-vs-CUDA "
             "comparison N/A by spec.")
    L.append("")
    L.append("**Recommended filing target:** `{}`".format(
        s["recommended_filing_target"]))

    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
