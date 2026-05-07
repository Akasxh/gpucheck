"""Fuzz the ELU kernel: MPS vs CPU reference.

Spec (kernel-fuzzer-elu v2):
- 1000 iterations across seeds {0, 1, 2, 3, 4}, dtypes {fp32, fp16, bf16}.
- We sample 200 unique (shape, dtype, stride_category) configs, then replay each
  with all 5 seeds = 1000 total iterations. Replay-with-different-seeds is what
  lets us decide reproducibility.
- Divergence filtering:
    * max_abs_err > 10x tolerance         -> divergence (always counts)
    * max_rel_err > 10x tolerance         -> divergence ONLY if denom >= 1e-6
                                              (else: near-zero artifact, ignore)
    * 1x .. 10x tolerance                 -> TOLERANCE_RECALIBRATION
    * < 1x tolerance                       -> OK
    * Reproducible across >=3 distinct seeds -> FILABLE
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Worktree gpucheck source
SRC = Path("/Users/cero/Code/gpucheck-worktrees/fuzz-elu/src")
sys.path.insert(0, str(SRC))

import random  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402
from gpucheck.fuzzing.shapes import (  # noqa: E402
    LARGE_DIMS,
    POWER_OF_2_BOUNDARIES,
    PRIMES,
    TILE_SIZES,
)
from gpucheck.fuzzing.strides import (  # noqa: E402
    CATEGORIES as STRIDE_CATEGORIES,
)
from gpucheck.fuzzing.strides import (
    fuzz_strides_for_category,
)

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_elu.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.nn.functional.elu"
N_CONFIGS = 200
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
TOTAL_ITERS = N_CONFIGS * len(SEEDS)  # 1000
TIME_BUDGET_S = 11 * 60 + 0  # 11 minutes; caller has 12 min wall budget

META_SEED = 0xE10  # for sampling configs deterministically

# Divergence-filter thresholds (multiples of (atol, rtol))
TOL_RECAL_LO = 1.0
TOL_DIVERGENCE = 10.0
DENOM_EPS = 1e-6  # near-zero rel-err artifact threshold

DEGENERATE = [(1,), (1, 1), (16, 1), (1, 16), (1, 1, 1)]
NON_TILE = [
    (TILE_SIZES[0] - 1,),  # 31
    (TILE_SIZES[0] + 1,),  # 33
    (TILE_SIZES[1] - 1, 16),
    (TILE_SIZES[1] + 3, 16),
    (TILE_SIZES[2] - 1, TILE_SIZES[2] + 1),
    (TILE_SIZES[2] + 1, 16),
]
PRIME_S = [(p,) for p in PRIMES] + [(PRIMES[0], PRIMES[1]), (PRIMES[2], 16)]
POW2_BOUNDARY = [(v,) for v in POWER_OF_2_BOUNDARIES] + [(128, 129), (256, 255)]
LARGE = [(LARGE_DIMS[0],), (256, 256), (1024, 64)]
MIXED = [(127, 16), (1024, 3), (7, 128), (33, 128, 4)]

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


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return 0.0
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    return float((af - bf).abs().max().item())


def _max_rel_err_with_denom(
    a: torch.Tensor, b: torch.Tensor,
) -> tuple[float, float]:
    """Return (max_rel_err, denom_magnitude_at_max) using the reference ``b``.

    denom_magnitude is |b| at the element that achieved the max relative error.
    Callers can decide whether the rel-err is a near-zero artifact.
    """
    if a.numel() == 0:
        return 0.0, 0.0
    af = a.to(torch.float32)
    bf = b.to(torch.float32)
    diff = (af - bf).abs()
    denom = bf.abs().clamp_min(1e-12)
    rel = diff / denom
    flat_idx = int(rel.argmax().item())
    rel_max = float(rel.flatten()[flat_idx].item())
    denom_at_max = float(bf.abs().flatten()[flat_idx].item())
    return rel_max, denom_at_max


def _stride_tag(t: torch.Tensor) -> str:
    return f"shape={tuple(t.shape)}, strides={tuple(t.stride())}, contig={t.is_contiguous()}"


def _classify(
    abs_err: float, rel_err: float, denom_at_max: float,
    atol: float, rtol: float,
) -> str:
    """Bucket a single iteration's error into OK / RECAL / DIVERGE."""
    abs_ratio = abs_err / atol if atol > 0 else 0.0
    # Near-zero rel-err artifact: kill the rel signal, not the abs.
    rel_artifact = denom_at_max < DENOM_EPS
    rel_ratio = (rel_err / rtol) if (rtol > 0 and not rel_artifact) else 0.0

    if abs_ratio > TOL_DIVERGENCE or rel_ratio > TOL_DIVERGENCE:
        return "DIVERGE"
    if abs_ratio > TOL_RECAL_LO or rel_ratio > TOL_RECAL_LO:
        return "RECAL"
    return "OK"


def _build_configs(rng: random.Random) -> list[tuple[str, tuple[int, ...], str, str]]:
    """Generate N_CONFIGS unique-ish (bucket, shape, dtype, stride_cat) configs.

    'Unique-ish' = sampled with replacement; duplicates are fine because we
    replay each across 5 seeds anyway. Deterministic given META_SEED.
    """
    configs = []
    for _ in range(N_CONFIGS):
        bucket = rng.choice(BUCKET_NAMES)
        shape = rng.choice(SHAPE_BUCKETS[bucket])
        dtype_name = rng.choice(DTYPE_NAMES)
        stride_cat = rng.choice(STRIDE_CATEGORIES)
        configs.append((bucket, shape, dtype_name, stride_cat))
    return configs


def _config_key(c: tuple[str, tuple[int, ...], str, str]) -> str:
    return f"{c[0]}|{tuple(c[1])}|{c[2]}|{c[3]}"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        skipped = {
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
        }
        RESULTS_MD.write_text(f"# {KERNEL} fuzz - SKIPPED\n\nMPS not available.\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps(skipped) + "\n")
        return 0

    cfg_rng = random.Random(META_SEED)
    configs = _build_configs(cfg_rng)

    started = time.monotonic()

    iters_attempted = 0
    iters_completed = 0
    iters_unsupported = 0
    max_abs_err_global = 0.0
    max_rel_err_global = 0.0  # only counted when not a near-zero artifact

    # bucket -> list of records (one per seed-instance that crossed a threshold)
    diverge_records: list[dict] = []   # > 10x tol
    recal_records: list[dict] = []     # 1x..10x tol

    # Per-config seed-coverage for FILABLE reproducibility check.
    # cfg_key -> set of seeds where this config was DIVERGE.
    diverge_seeds_per_cfg: dict[str, set[int]] = defaultdict(set)
    # Cache one representative diverge record per cfg_key for output.
    diverge_repr_per_cfg: dict[str, dict] = {}
    recal_seeds_per_cfg: dict[str, set[int]] = defaultdict(set)

    per_bucket_counts: dict[str, int] = {b: 0 for b in BUCKET_NAMES}
    per_dtype_counts: dict[str, int] = {d: 0 for d in DTYPE_NAMES}
    per_stride_counts: dict[str, int] = {s: 0 for s in STRIDE_CATEGORIES}

    aborted = False

    for cfg_idx, cfg in enumerate(configs):
        bucket, shape, dtype_name, stride_cat = cfg
        dtype = DTYPES_BY_NAME[dtype_name]
        cfg_key = _config_key(cfg)

        for seed_i in SEEDS:
            if time.monotonic() - started > TIME_BUDGET_S:
                aborted = True
                print(
                    f"[budget] aborting at cfg {cfg_idx}/{N_CONFIGS} "
                    f"seed={seed_i} (elapsed {time.monotonic() - started:.1f}s)",
                    file=sys.stderr,
                )
                break

            iters_attempted += 1
            per_bucket_counts[bucket] += 1
            per_dtype_counts[dtype_name] += 1
            per_stride_counts[stride_cat] += 1

            try:
                # Build CPU + MPS tensors with the same seed and stride layout.
                try:
                    x_cpu = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="cpu", seed=seed_i,
                    )
                    x_mps = fuzz_strides_for_category(
                        shape, dtype, stride_cat, device="mps", seed=seed_i,
                    )
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-build] cfg={cfg_idx} seed={seed_i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                # Skip empty tensors (no signal).
                if x_cpu.numel() == 0:
                    iters_completed += 1
                    continue

                try:
                    y_mps = F.elu(x_mps)
                    torch.mps.synchronize()
                    y_mps_cpu = y_mps.detach().to("cpu")
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-mps] cfg={cfg_idx} seed={seed_i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                try:
                    y_cpu = F.elu(x_cpu)
                except (RuntimeError, NotImplementedError, TypeError) as exc:
                    msg = str(exc).splitlines()[0][:200]
                    iters_unsupported += 1
                    print(
                        f"[unsupported-cpu] cfg={cfg_idx} seed={seed_i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {msg}",
                        file=sys.stderr,
                    )
                    continue

                iters_completed += 1

                abs_err = _max_abs_err(y_mps_cpu, y_cpu)
                rel_err, denom_at_max = _max_rel_err_with_denom(y_mps_cpu, y_cpu)
                # Only update global rel max if it's not a near-zero artifact.
                if denom_at_max >= DENOM_EPS:
                    max_rel_err_global = max(max_rel_err_global, rel_err)
                max_abs_err_global = max(max_abs_err_global, abs_err)

                atol, rtol = compute_tolerance(dtype, device_type="mps")
                verdict = _classify(abs_err, rel_err, denom_at_max, atol, rtol)

                if verdict == "OK":
                    continue

                rec = {
                    "cfg_idx": cfg_idx,
                    "seed": seed_i,
                    "shape": list(shape),
                    "dtype": dtype_name,
                    "stride_category": stride_cat,
                    "shape_bucket": bucket,
                    "max_abs_err": abs_err,
                    "max_rel_err": rel_err,
                    "denom_at_max_rel": denom_at_max,
                    "atol": atol,
                    "rtol": rtol,
                    "abs_ratio": abs_err / atol if atol > 0 else 0.0,
                    "rel_ratio": (rel_err / rtol) if (
                        rtol > 0 and denom_at_max >= DENOM_EPS
                    ) else 0.0,
                    "x_layout_cpu": _stride_tag(x_cpu),
                    "x_layout_mps": _stride_tag(x_mps),
                    "verdict": verdict,
                }
                if verdict == "DIVERGE":
                    diverge_records.append(rec)
                    diverge_seeds_per_cfg[cfg_key].add(seed_i)
                    diverge_repr_per_cfg.setdefault(cfg_key, rec)
                    print(
                        f"[DIVERGE] cfg={cfg_idx} seed={seed_i} "
                        f"{bucket}/{dtype_name}/{stride_cat} shape={shape} "
                        f"abs={abs_err:.3e}({rec['abs_ratio']:.1f}x) "
                        f"rel={rel_err:.3e}({rec['rel_ratio']:.1f}x) "
                        f"denom={denom_at_max:.2e}",
                        file=sys.stderr,
                    )
                else:  # RECAL
                    recal_records.append(rec)
                    recal_seeds_per_cfg[cfg_key].add(seed_i)

            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                # Halt and report any process error immediately.
                print(
                    f"[ERROR] cfg={cfg_idx} seed={seed_i} "
                    f"{bucket}/{dtype_name}/{stride_cat} shape={shape}: {exc!r}",
                    file=sys.stderr,
                )
                traceback.print_exc(file=sys.stderr)
                # Still write whatever we have before exiting.
                _emit_results(
                    started=started,
                    aborted=True,
                    error=repr(exc),
                    iters_attempted=iters_attempted,
                    iters_completed=iters_completed,
                    iters_unsupported=iters_unsupported,
                    max_abs_err_global=max_abs_err_global,
                    max_rel_err_global=max_rel_err_global,
                    diverge_records=diverge_records,
                    recal_records=recal_records,
                    diverge_seeds_per_cfg=diverge_seeds_per_cfg,
                    diverge_repr_per_cfg=diverge_repr_per_cfg,
                    recal_seeds_per_cfg=recal_seeds_per_cfg,
                    per_bucket_counts=per_bucket_counts,
                    per_dtype_counts=per_dtype_counts,
                    per_stride_counts=per_stride_counts,
                )
                return 2

        if aborted:
            break

    _emit_results(
        started=started,
        aborted=aborted,
        error=None,
        iters_attempted=iters_attempted,
        iters_completed=iters_completed,
        iters_unsupported=iters_unsupported,
        max_abs_err_global=max_abs_err_global,
        max_rel_err_global=max_rel_err_global,
        diverge_records=diverge_records,
        recal_records=recal_records,
        diverge_seeds_per_cfg=diverge_seeds_per_cfg,
        diverge_repr_per_cfg=diverge_repr_per_cfg,
        recal_seeds_per_cfg=recal_seeds_per_cfg,
        per_bucket_counts=per_bucket_counts,
        per_dtype_counts=per_dtype_counts,
        per_stride_counts=per_stride_counts,
    )
    return 0


def _numel(shape: list[int] | tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= max(d, 1)
    return n


def _emit_results(
    *,
    started: float,
    aborted: bool,
    error: str | None,
    iters_attempted: int,
    iters_completed: int,
    iters_unsupported: int,
    max_abs_err_global: float,
    max_rel_err_global: float,
    diverge_records: list[dict],
    recal_records: list[dict],
    diverge_seeds_per_cfg: dict[str, set[int]],
    diverge_repr_per_cfg: dict[str, dict],
    recal_seeds_per_cfg: dict[str, set[int]],
    per_bucket_counts: dict[str, int],
    per_dtype_counts: dict[str, int],
    per_stride_counts: dict[str, int],
) -> None:
    elapsed = time.monotonic() - started

    # FILABLE = DIVERGE config reproducible across >=3 distinct seeds.
    filable_cfg_keys = {
        k for k, seeds in diverge_seeds_per_cfg.items() if len(seeds) >= 3
    }
    # Recalibration count: configs where any seed showed RECAL OR a DIVERGE
    # config that did NOT meet the >=3 seed reproducibility bar (treat as
    # tolerance noise, recommend xfail/recalibration).
    recal_cfg_keys = set(recal_seeds_per_cfg.keys())
    not_filable_diverge_cfgs = {
        k for k in diverge_seeds_per_cfg if k not in filable_cfg_keys
    }
    recal_or_noise_cfg_keys = recal_cfg_keys | not_filable_diverge_cfgs

    # Top-3 repros, prioritized: filable first, then by seed count, then by
    # rel-err-ratio, then smaller numel (more minimal). For ties, abs-ratio.
    def repro_sort_key(cfg_key: str) -> tuple:
        rec = diverge_repr_per_cfg[cfg_key]
        seeds_n = len(diverge_seeds_per_cfg[cfg_key])
        is_filable = cfg_key in filable_cfg_keys
        return (
            0 if is_filable else 1,
            -seeds_n,
            -max(rec["abs_ratio"], rec["rel_ratio"]),
            _numel(rec["shape"]),
        )

    diverge_cfg_keys_sorted = sorted(diverge_repr_per_cfg.keys(), key=repro_sort_key)
    top3_keys = diverge_cfg_keys_sorted[:3]
    top3 = []
    for k in top3_keys:
        rec = dict(diverge_repr_per_cfg[k])
        rec["seeds_reproducing"] = sorted(diverge_seeds_per_cfg[k])
        rec["filable"] = k in filable_cfg_keys
        top3.append(rec)

    summary = {
        "kernel": KERNEL,
        "status": "ABORTED_BUDGET" if aborted and not error else (
            "ERROR" if error else "OK"
        ),
        "iters_attempted": iters_attempted,
        "iters_completed": iters_completed,
        "iters_unsupported": iters_unsupported,
        "divergences_filable": len(filable_cfg_keys),
        "divergences_recalibration": len(recal_or_noise_cfg_keys),
        "max_abs_err": max_abs_err_global,
        "max_rel_err": max_rel_err_global,
        "top_3_repros": top3,
        "elapsed_seconds": round(elapsed, 2),
        "per_shape_bucket": per_bucket_counts,
        "per_dtype": per_dtype_counts,
        "per_stride_category": per_stride_counts,
        "torch_version": torch.__version__,
        "mps_available": True,
        "host": "darwin/arm64 (Apple Silicon)",
        "seeds": list(SEEDS),
        "n_configs": N_CONFIGS,
        "total_iters_planned": TOTAL_ITERS,
        "meta_seed_hex": f"0x{META_SEED:x}",
        "filing_target_recommended": (
            "pytorch/pytorch" if filable_cfg_keys else "none"
        ),
        "error": error,
    }

    _write_markdown(RESULTS_MD, summary)
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")


def _write_markdown(path: Path, s: dict) -> None:
    L = []
    L.append(f"# {s['kernel']} - MPS fuzz report")
    L.append("")
    L.append(f"- **Kernel:** `{s['kernel']}`")
    L.append(f"- **Status:** {s['status']}")
    if s.get("error"):
        L.append(f"- **Error:** `{s['error']}`")
    L.append(f"- **Iterations attempted:** {s['iters_attempted']}")
    L.append(f"- **Iterations completed:** {s['iters_completed']}")
    L.append(f"- **Iterations unsupported (skipped):** {s['iters_unsupported']}")
    L.append(f"- **Iterations planned:** {s['total_iters_planned']} "
             f"({s['n_configs']} configs x {len(s['seeds'])} seeds)")
    L.append(f"- **Divergences (FILABLE, repro >=3 seeds):** {s['divergences_filable']}")
    L.append(f"- **Divergences (TOLERANCE_RECALIBRATION):** {s['divergences_recalibration']}")
    L.append(f"- **MPS vs CPU max absolute error:** {s['max_abs_err']:.3e}")
    L.append(f"- **MPS vs CPU max relative error (denom >= 1e-6):** "
             f"{s['max_rel_err']:.3e}")
    L.append(f"- **Recommended filing target:** `{s['filing_target_recommended']}`")
    L.append(f"- **Elapsed:** {s['elapsed_seconds']} s")
    L.append(f"- **torch:** {s['torch_version']}, host: {s['host']}, "
             f"seeds: {s['seeds']}, meta-seed: {s['meta_seed_hex']}")
    L.append("")
    L.append("## Sampling distribution")
    L.append("")
    L.append("| dimension | counts |")
    L.append("|---|---|")
    L.append(f"| shape bucket | {s['per_shape_bucket']} |")
    L.append(f"| dtype | {s['per_dtype']} |")
    L.append(f"| stride category | {s['per_stride_category']} |")
    L.append("")
    L.append("## Top 3 minimal repros")
    L.append("")
    if not s["top_3_repros"]:
        L.append("_No DIVERGE-class records (>10x tolerance with non-near-zero denom)._")
    else:
        for i, r in enumerate(s["top_3_repros"], 1):
            L.append(f"### Repro #{i}  -- `{'FILABLE' if r['filable'] else 'NOT_FILABLE'}`")
            L.append("")
            L.append(f"- **shape:** `{tuple(r['shape'])}`  (bucket: `{r['shape_bucket']}`)")
            L.append(f"- **dtype:** `{r['dtype']}`")
            L.append(f"- **stride category:** `{r['stride_category']}`")
            L.append(f"- **seeds reproducing DIVERGE:** "
                     f"`{r['seeds_reproducing']}`  "
                     f"(>=3 -> filable)")
            L.append(f"- **max abs err:** {r['max_abs_err']:.3e}  "
                     f"(atol={r['atol']:.2e}, ratio={r['abs_ratio']:.2f}x)")
            L.append(f"- **max rel err:** {r['max_rel_err']:.3e}  "
                     f"(rtol={r['rtol']:.2e}, ratio={r['rel_ratio']:.2f}x)")
            L.append(f"- **denom magnitude at max-rel:** {r['denom_at_max_rel']:.3e}  "
                     f"(>=1e-6 ? {r['denom_at_max_rel'] >= 1e-6})")
            L.append(f"- **CPU layout:** `{r['x_layout_cpu']}`")
            L.append(f"- **MPS layout:** `{r['x_layout_mps']}`")
            L.append(f"- **example seed:** {r['seed']}")
            L.append("")
    L.append("## Method notes")
    L.append("")
    L.append("- Reference: `torch.nn.functional.elu` on CPU, FP32-promoted for the "
             "error metric.")
    L.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance("
             "dtype, device_type='mps')` (MPS 2x overlay applied).")
    L.append("- Divergence classification per spec:")
    L.append("    * **OK:** abs_ratio < 1.0 and rel_ratio < 1.0")
    L.append("    * **TOLERANCE_RECALIBRATION:** 1.0 <= ratio <= 10.0 (either)")
    L.append("    * **DIVERGE:** ratio > 10.0 (either); rel-err filtered when "
             "denom < 1e-6 (near-zero artifact)")
    L.append("    * **FILABLE:** DIVERGE config reproduced under >=3 distinct seeds "
             "from {0,1,2,3,4}")
    L.append("- Stride categories: row_major, column_major, broadcast, transpose, "
             "slice, non_contig, gather (`gpucheck.fuzzing.strides`).")
    L.append("- Sampling: 200 (bucket, shape, dtype, stride) configs sampled with "
             "meta-seed; each replayed under all 5 seeds = 1000 iterations.")
    L.append("- ELU is elementwise; no k_dim scaling applied.")
    L.append("- CUDA backend is mocked (no NVIDIA GPU on host); cross-device "
             "MPS-vs-CUDA comparison N/A.")
    L.append("")
    path.write_text("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
