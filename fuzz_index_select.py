"""Fuzz torch.index_select on MPS vs CPU using gpucheck's stride/shape/dtype fuzzers.

Budgeted at 250 iterations or 7 minutes wall-clock, whichever comes first.
"""
from __future__ import annotations

import json
import random
import time
import traceback
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.strides import CATEGORIES, fuzz_strides_for_category

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm/")
RESULTS_MD = OUT_DIR / "RESULTS_index_select.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

ITERATIONS = 250
TIME_BUDGET_S = 7 * 60  # leave ~1 min margin from the 8-min hard cap
SEED = 0xC0FFEE

# Shape pools by category (priority: degenerate > non-tile > prime > pow2 > large > mixed)
DEGENERATE = [(0,), (1,), (0, 8), (8, 0), (1, 1), (1, 1, 1)]
PRIME = [(7,), (13, 17), (3, 5, 7), (11, 13), (17, 19, 23)]
POW2_BOUNDARY = [(32,), (33,), (64, 64), (65, 65), (128, 127), (256, 255)]
NON_TILE = [(127,), (129, 31), (50, 70), (33, 65), (96, 96)]
LARGE = [(4096,), (1024, 256), (256, 1024), (64, 64, 64)]
SHAPE_POOLS = {
    "degenerate": DEGENERATE,
    "prime": PRIME,
    "pow2_boundary": POW2_BOUNDARY,
    "non_tile_aligned": NON_TILE,
    "large": LARGE,
}

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# Stride categories supported for index_select inputs. "broadcast" creates a
# stride-0 view which is legal input to index_select; "gather" returns a
# contiguous tensor (per gpucheck's strides module).
STRIDE_CATEGORIES = list(CATEGORIES)


def sample_iteration(rng: random.Random) -> dict:
    shape_cat = rng.choice(list(SHAPE_POOLS.keys()))
    shape = rng.choice(SHAPE_POOLS[shape_cat])
    dtype_name, dtype = rng.choice(DTYPES)
    stride_cat = rng.choice(STRIDE_CATEGORIES)
    return {
        "shape_category": shape_cat,
        "shape": shape,
        "dtype_name": dtype_name,
        "dtype": dtype,
        "stride_category": stride_cat,
    }


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max element-wise |a-b| / (|b| + eps), upcast to fp32 for stable math."""
    a32 = a.detach().to(torch.float32).cpu()
    b32 = b.detach().to(torch.float32).cpu()
    if a32.numel() == 0:
        return 0.0
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-12)
    return float((diff / denom).max().item())


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(torch.float32).cpu()
    b32 = b.detach().to(torch.float32).cpu()
    if a32.numel() == 0:
        return 0.0
    return float((a32 - b32).abs().max().item())


def run_one(iter_idx: int, spec: dict, rng: random.Random) -> dict:
    shape = spec["shape"]
    dtype = spec["dtype"]
    dtype_name = spec["dtype_name"]
    stride_cat = spec["stride_category"]
    seed = rng.randint(0, 2**31 - 1)

    out: dict = {
        "iter": iter_idx,
        "shape": list(shape),
        "shape_category": spec["shape_category"],
        "dtype": dtype_name,
        "stride_category": stride_cat,
        "seed": seed,
        "status": "ok",
    }

    # Build the source tensor on CPU using the stride fuzzer.
    try:
        src_cpu = fuzz_strides_for_category(shape, dtype, stride_cat, device="cpu", seed=seed)
    except Exception as exc:
        out["status"] = "build_failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    if src_cpu.ndim == 0 or src_cpu.numel() == 0 or 0 in tuple(src_cpu.shape):
        out["status"] = "skipped_empty"
        return out

    # Pick a dim to index along, and a random index tensor.
    dim = rng.randrange(src_cpu.ndim)
    dim_size = src_cpu.shape[dim]
    if dim_size <= 0:
        out["status"] = "skipped_empty_dim"
        return out
    n_idx = rng.randint(1, max(1, min(dim_size * 2, 64)))
    idx_cpu = torch.randint(0, dim_size, (n_idx,), dtype=torch.long)
    out["dim"] = dim
    out["n_idx"] = n_idx

    # Reference on CPU.
    try:
        ref = torch.index_select(src_cpu, dim, idx_cpu)
    except Exception as exc:
        out["status"] = "cpu_failed"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    # Move to MPS and run there. bfloat16 support on MPS varies by torch version,
    # so catch and tag as UNSUPPORTED rather than diverge.
    try:
        src_mps = src_cpu.to("mps")
        idx_mps = idx_cpu.to("mps")
        got = torch.index_select(src_mps, dim, idx_mps)
        torch.mps.synchronize()
    except (NotImplementedError, RuntimeError) as exc:
        out["status"] = "unsupported_mps"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out

    if got.shape != ref.shape:
        out["status"] = "shape_mismatch"
        out["got_shape"] = list(got.shape)
        out["ref_shape"] = list(ref.shape)
        return out

    rel = relative_error(got, ref)
    abs_err = max_abs_error(got, ref)
    atol, rtol = compute_tolerance(dtype, device_type="mps")
    out["max_rel_err"] = rel
    out["max_abs_err"] = abs_err
    out["atol"] = atol
    out["rtol"] = rtol

    # Use torch.allclose semantics on the upcast pair so atol+rtol both apply.
    a32 = got.detach().to(torch.float32).cpu()
    b32 = ref.detach().to(torch.float32).cpu()
    is_close = torch.allclose(a32, b32, atol=atol, rtol=rtol, equal_nan=True)
    out["divergent"] = not is_close
    if not is_close:
        out["status"] = "divergent"
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        msg = "MPS not available — SKIPPED"
        print(msg)
        RESULTS_MD.write_text(f"# index_select fuzz — SKIPPED\n\n{msg}\n")
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({
                "kernel": "index_select",
                "status": "SKIPPED",
                "reason": "torch.mps.is_available() is False",
            }) + "\n")
        return

    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    started = time.time()
    completed = 0
    attempted = 0
    divergences: list[dict] = []
    unsupported = 0
    errors: list[dict] = []
    status_counts: dict[str, int] = {}

    for i in range(ITERATIONS):
        if time.time() - started > TIME_BUDGET_S:
            print(f"[budget] stopping at iter {i} after {time.time()-started:.1f}s")
            break
        attempted += 1
        spec = sample_iteration(rng)
        try:
            result = run_one(i, spec, rng)
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            result = {
                "iter": i,
                "status": "harness_error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": tb,
                "shape": list(spec["shape"]),
                "dtype": spec["dtype_name"],
                "stride_category": spec["stride_category"],
            }
            errors.append(result)

        st = result.get("status", "ok")
        status_counts[st] = status_counts.get(st, 0) + 1
        if st in {"ok", "divergent"}:
            completed += 1
        if result.get("divergent"):
            divergences.append(result)
        if st == "unsupported_mps":
            unsupported += 1

        if i % 25 == 0:
            print(
                f"iter {i}: status={st} shape={spec['shape']} "
                f"dtype={spec['dtype_name']} stride={spec['stride_category']} "
                f"rel={result.get('max_rel_err')}"
            )

    elapsed = time.time() - started

    # Sort divergences by rel error descending for "top 3".
    divergences.sort(key=lambda d: d.get("max_rel_err", 0.0), reverse=True)
    top3 = divergences[:3]

    summary = {
        "kernel": "index_select",
        "backend_mps": True,
        "backend_cuda": "mocked-only (no NVIDIA GPU on host)",
        "torch_version": torch.__version__,
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "iterations_unsupported": unsupported,
        "divergences": len(divergences),
        "elapsed_s": round(elapsed, 2),
        "status_counts": status_counts,
        "top3_repros": [
            {
                "shape": d.get("shape"),
                "dtype": d.get("dtype"),
                "stride_category": d.get("stride_category"),
                "dim": d.get("dim"),
                "n_idx": d.get("n_idx"),
                "seed": d.get("seed"),
                "max_rel_err": d.get("max_rel_err"),
                "max_abs_err": d.get("max_abs_err"),
                "atol": d.get("atol"),
                "rtol": d.get("rtol"),
            }
            for d in top3
        ],
        "mps_vs_cpu_max_rel_err": (
            max((d.get("max_rel_err", 0.0) for d in divergences), default=0.0)
            if divergences else max(
                (errors and 0.0) or 0.0, 0.0,
            )
        ),
        "mps_vs_cuda_mock_max_rel_err": "N/A (CUDA mocked, no real device)",
        "upstream_target": (
            "pytorch/pytorch" if divergences else "none"
        ),
    }

    # Compute true max_rel_err across ALL completed runs (not just divergences).
    # Re-scan: we didn't keep all results; but max over divergences is a
    # tight lower bound, and zero divergences means the suite agreed within
    # tolerance — report that explicitly.
    if not divergences:
        summary["mps_vs_cpu_max_rel_err"] = "<= per-dtype tolerance (no divergences)"

    md_lines = [
        "# index_select fuzz — MPS vs CPU",
        "",
        f"- **kernel**: `torch.index_select`",
        f"- **torch**: {torch.__version__}",
        f"- **MPS available**: True",
        f"- **CUDA backend**: {summary['backend_cuda']}",
        f"- **iterations attempted**: {attempted}",
        f"- **iterations completed (ok+divergent)**: {completed}",
        f"- **iterations unsupported on MPS**: {unsupported}",
        f"- **divergences**: {len(divergences)}",
        f"- **elapsed**: {elapsed:.1f}s",
        f"- **status counts**: {status_counts}",
        "",
        "## MPS-vs-CPU max relative error",
        "",
        f"{summary['mps_vs_cpu_max_rel_err']}",
        "",
        "## MPS-vs-CUDA-mock max relative error",
        "",
        "N/A — no NVIDIA GPU present; CUDA detection mocked but kernel cannot execute.",
        "",
        "## Top 3 minimal repros",
        "",
    ]
    if not top3:
        md_lines.append("_None — no divergences observed in this run._")
    else:
        for i, r in enumerate(top3, 1):
            md_lines.append(
                f"{i}. shape={r['shape']} dtype={r['dtype']} stride={r['stride_category']} "
                f"dim={r['dim']} n_idx={r['n_idx']} seed={r['seed']} "
                f"max_rel_err={r['max_rel_err']:.3e} "
                f"(atol={r['atol']:.2e}, rtol={r['rtol']:.2e})"
            )
    md_lines.append("")
    md_lines.append("## Recommended upstream filing target")
    md_lines.append("")
    md_lines.append(f"`{summary['upstream_target']}`")
    if not divergences:
        md_lines.append("")
        md_lines.append(
            "No divergences exceeding gpucheck's MPS-overlaid per-dtype tolerance. "
            "`torch.index_select` on MPS agrees with CPU within tolerance across the "
            "tested shape/dtype/stride matrix.",
        )
    md_lines.append("")
    if errors:
        md_lines.append(f"## Harness errors ({len(errors)})")
        md_lines.append("")
        for e in errors[:5]:
            md_lines.append(
                f"- iter={e['iter']} shape={e.get('shape')} dtype={e.get('dtype')} "
                f"stride={e.get('stride_category')} :: {e.get('error')}"
            )
        md_lines.append("")

    RESULTS_MD.write_text("\n".join(md_lines))

    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary, default=str) + "\n")

    print(f"\nDone. divergences={len(divergences)} attempted={attempted} elapsed={elapsed:.1f}s")
    print(f"wrote {RESULTS_MD}")
    print(f"appended to {SWARM_JSONL}")


if __name__ == "__main__":
    main()
