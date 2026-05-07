"""Stride/shape/dtype fuzzer for torch SDPA *causal* attention on MPS vs CPU.

500 iterations. FILABLE filter:
  max_rel_err > 10 * threshold
  AND denom_magnitude (max |ref|) >= 1e-6
  AND reproducible across >= 3 distinct seeds (same shape/dtype/stride).

Below FILABLE: TOLERANCE_RECALIBRATION (rel_err > threshold but not filable) or OK.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

# Make gpucheck importable from this worktree
sys.path.insert(0, "/Users/cero/Code/gpucheck-worktrees/fuzz-attention-causal/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_attention-causal.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "scaled_dot_product_attention[is_causal=True]"
N_ITER = 500
BUDGET_S = 8 * 60 - 45  # leave 45s headroom for triage + writes
SEED = 0xCA51

DEGEN_DIMS = [1]
PRIME_DIMS = [7, 13, 31]
POW2_DIMS = [127, 128, 129, 255, 256]
NON_TILE_DIMS = [31, 33, 63, 65, 129, 131]
LARGE_DIMS = [512, 1024]

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]
STRIDE_CATS = ["contiguous", "slice", "transpose", "broadcast"]


def make_shape(rng: random.Random, category: str) -> tuple[int, int, int, int]:
    if category == "degenerate":
        s = rng.choice([0, 1])
        return (1, 1, s, 8)
    if category == "prime":
        s = rng.choice(PRIME_DIMS)
        d = rng.choice([8, 16, 32, 64])
        return (1, 1, s, d)
    if category == "pow2":
        s = rng.choice(POW2_DIMS)
        d = rng.choice([16, 32, 64, 128])
        return (1, 2, s, d)
    if category == "non_tile":
        s = rng.choice(NON_TILE_DIMS)
        d = rng.choice([16, 32, 64])
        return (1, 1, s, d)
    if category == "large":
        s = rng.choice(LARGE_DIMS)
        d = rng.choice([32, 64, 128])
        return (1, 2, s, d)
    raise ValueError(category)


def make_qkv(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    device: str,
    stride_cat: str,
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, S, D = shape

    if stride_cat == "contiguous":
        def base() -> torch.Tensor:
            t = torch.randn((B, H, S, D), generator=gen, dtype=torch.float32) * 0.5
            return t.to(dtype=dtype, device=device).contiguous()
        return base(), base(), base()

    if stride_cat == "slice":
        big = (B, H, S * 2, D)
        def bigt() -> torch.Tensor:
            t = torch.randn(big, generator=gen, dtype=torch.float32).mul_(0.5)
            return t.to(dtype=dtype, device=device).contiguous()
        return bigt()[:, :, ::2, :], bigt()[:, :, ::2, :], bigt()[:, :, ::2, :]

    if stride_cat == "transpose":
        tshape = (B, H, D, S)
        def bigt() -> torch.Tensor:
            t = torch.randn(tshape, generator=gen, dtype=torch.float32).mul_(0.5)
            return t.to(dtype=dtype, device=device).contiguous().transpose(-1, -2)
        return bigt(), bigt(), bigt()

    if stride_cat == "broadcast":
        small = (B, 1, S, D)
        def bigt() -> torch.Tensor:
            t = torch.randn(small, generator=gen, dtype=torch.float32).mul_(0.5)
            return t.to(dtype=dtype, device=device).contiguous().expand(B, H, S, D)
        return bigt(), bigt(), bigt()

    raise ValueError(stride_cat)


def _max_rel_err_and_denom(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Return (max_rel_err, max_abs_ref). NaN-safe: NaN mismatch -> +inf."""
    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    if torch.isnan(a32).any() != torch.isnan(b32).any() or torch.isinf(a32).any() != torch.isinf(b32).any():
        return float("inf"), float(b32.abs().nan_to_num().max().item())
    mask = torch.isfinite(a32) & torch.isfinite(b32)
    if not mask.any():
        return 0.0, 0.0
    diff = (a32[mask] - b32[mask]).abs()
    denom = b32[mask].abs().clamp_min(1e-7)
    return float((diff / denom).max().item()), float(b32[mask].abs().max().item())


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    mask = torch.isfinite(a32) & torch.isfinite(b32)
    if not mask.any():
        return 0.0
    return float((a32[mask] - b32[mask]).abs().max().item())


def run_one(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    stride_cat: str,
    seed: int,
) -> dict:
    """Run a single config at a given seed. Returns the result record."""
    B, H, S, D = shape
    out: dict = {
        "shape": list(shape), "dtype": str(dtype).split(".")[-1],
        "stride": stride_cat, "seed": seed,
        "status": "OK", "max_rel_err": None, "max_abs_err": None,
        "denom_mag": None, "tol_atol": None, "tol_rtol": None,
        "threshold": None, "diverged": False, "note": "",
    }

    if S == 0:
        out["status"] = "SKIP_EMPTY"
        return out

    gen = torch.Generator()
    gen.manual_seed(seed)
    try:
        q_cpu, k_cpu, v_cpu = make_qkv(shape, dtype, "cpu", stride_cat, gen)
    except Exception as e:
        out["status"] = "BUILD_ERR_CPU"
        out["note"] = f"{type(e).__name__}: {e}"[:240]
        return out

    try:
        q_m = q_cpu.detach().to(device="mps")
        k_m = k_cpu.detach().to(device="mps")
        v_m = v_cpu.detach().to(device="mps")
    except Exception as e:
        out["status"] = "MPS_COPY_ERR"
        out["note"] = f"{type(e).__name__}: {e}"[:240]
        return out

    try:
        ref = F.scaled_dot_product_attention(
            q_cpu.to(torch.float32),
            k_cpu.to(torch.float32),
            v_cpu.to(torch.float32),
            is_causal=True,
        )
    except Exception as e:
        out["status"] = "CPU_REF_ERR"
        out["note"] = f"{type(e).__name__}: {e}"[:240]
        return out

    try:
        out_mps = F.scaled_dot_product_attention(q_m, k_m, v_m, is_causal=True)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError) as e:
        out["status"] = "UNSUPPORTED_MPS"
        out["note"] = f"{type(e).__name__}: {e}"[:240]
        return out

    rel, denom_mag = _max_rel_err_and_denom(out_mps, ref)
    abs_err = _max_abs_err(out_mps, ref)
    out["max_rel_err"] = rel
    out["max_abs_err"] = abs_err
    out["denom_mag"] = denom_mag

    k_dim = max(S, D)
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
    out["tol_atol"] = atol
    out["tol_rtol"] = rtol
    out["threshold"] = atol
    out["diverged"] = rel > atol
    return out


def run_iter(idx: int, rng: random.Random) -> dict:
    cat = rng.choice(["degenerate", "prime", "pow2", "non_tile", "large"])
    dtype_name, dtype = rng.choice(DTYPES)
    stride_cat = rng.choice(STRIDE_CATS)
    shape = make_shape(rng, cat)
    seed = rng.randint(0, 2**31 - 1)
    rec = run_one(tuple(shape), dtype, stride_cat, seed)
    rec.update({"iter": idx, "shape_cat": cat})
    return rec


def classify(rec: dict, repro_seeds: int) -> str:
    """Classify a record:

    - FILABLE     : rel > 10*threshold AND denom >= 1e-6 AND repro_seeds >= 3
    - TOL_RECAL   : rel > threshold but not filable (and finite-numerics)
    - OK          : rel <= threshold (or skipped/unsupported/error class)
    """
    status = rec.get("status")
    if status != "OK":
        return status  # SKIP_EMPTY / UNSUPPORTED_MPS / *_ERR
    rel = rec.get("max_rel_err")
    thr = rec.get("threshold")
    denom = rec.get("denom_mag") or 0.0
    if rel is None or thr is None:
        return "OK"
    if rel > 10 * thr and denom >= 1e-6 and repro_seeds >= 3:
        return "FILABLE"
    if rel > thr:
        return "TOLERANCE_RECALIBRATION"
    return "OK"


def reproduce(rec: dict, n_extra_seeds: int = 4) -> int:
    """Return the count of distinct seeds where the same config exceeds 10x threshold,
    including the original. Cap at n_extra_seeds+1.
    """
    if rec.get("status") != "OK" or rec.get("max_rel_err") is None:
        return 0
    thr = rec.get("threshold") or 0.0
    target = 10.0 * thr
    shape = tuple(rec["shape"])
    dt_name = rec["dtype"]
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dt_name]
    stride_cat = rec["stride"]

    count = 1 if (rec["max_rel_err"] > target and (rec.get("denom_mag") or 0.0) >= 1e-6) else 0
    base_seed = int(rec.get("seed", 0))
    extra_seeds = [(base_seed * 2654435761 + 17 * (i + 1)) & 0x7FFFFFFF for i in range(n_extra_seeds)]
    for s in extra_seeds:
        try:
            r = run_one(shape, dtype, stride_cat, s)
        except Exception:
            continue
        if r.get("status") == "OK" and r.get("max_rel_err") is not None:
            if r["max_rel_err"] > target and (r.get("denom_mag") or 0.0) >= 1e-6:
                count += 1
    return count


def main() -> int:
    if not torch.backends.mps.is_available():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_MD.write_text(
            f"# {KERNEL_NAME} fuzz results\n\nSTATUS: SKIPPED — torch.mps unavailable.\n",
        )
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    rng = random.Random(SEED)
    t0 = time.time()
    records: list[dict] = []
    completed = 0

    for i in range(N_ITER):
        if time.time() - t0 > BUDGET_S:
            print(f"[budget] stopping fuzz loop at iter={i}", flush=True)
            break
        try:
            rec = run_iter(i, rng)
        except Exception as e:
            rec = {
                "iter": i, "status": "EXC", "err": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-600:],
            }
            records.append(rec)
            print(f"[halt-report] iter={i} {rec['err']}", flush=True)
            continue
        records.append(rec)
        if rec["status"] == "OK":
            completed += 1
        if i % 50 == 0:
            print(
                f"[{i:3d}] cat={rec.get('shape_cat'):<10} "
                f"dt={rec.get('dtype'):<9} st={rec.get('stride'):<11} "
                f"shape={rec.get('shape')} status={rec['status']} "
                f"rel={rec.get('max_rel_err')}",
                flush=True,
            )

    fuzz_elapsed = time.time() - t0
    print(f"[fuzz] done in {fuzz_elapsed:.1f}s; running triage…", flush=True)

    # Triage divergent records: reproduce-across-seeds for the candidates
    # whose rel_err > threshold AND denom >= 1e-6.
    candidates = [
        r for r in records
        if r.get("status") == "OK"
        and r.get("max_rel_err") is not None
        and r.get("threshold") is not None
        and r["max_rel_err"] > r["threshold"]
        and (r.get("denom_mag") or 0.0) >= 1e-6
    ]
    print(f"[triage] {len(candidates)} candidates above threshold w/ denom>=1e-6", flush=True)

    for r in candidates:
        if time.time() - t0 > (BUDGET_S + 30):
            r["repro_seeds"] = 1  # untested under budget
            r["classification"] = "TOLERANCE_RECALIBRATION"
            continue
        n = reproduce(r, n_extra_seeds=4)
        r["repro_seeds"] = n
        r["classification"] = classify(r, n)

    # For non-candidates, classify directly.
    for r in records:
        if "classification" in r:
            continue
        r["classification"] = classify(r, repro_seeds=1)

    write_outputs(records, completed, time.time() - t0)
    return 0


def write_outputs(records: list[dict], completed: int, elapsed: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    filable = [r for r in records if r.get("classification") == "FILABLE"]
    tol_recal = [r for r in records if r.get("classification") == "TOLERANCE_RECALIBRATION"]
    unsupported = [r for r in records if r.get("status") == "UNSUPPORTED_MPS"]
    other_err = [
        r for r in records
        if r.get("status") not in {"OK", "UNSUPPORTED_MPS", "SKIP_EMPTY"}
    ]

    def numel(r: dict) -> int:
        return int(math.prod(r.get("shape") or [1]))

    top3 = sorted(filable, key=numel)[:3]

    rels = [r["max_rel_err"] for r in records if r.get("max_rel_err") is not None]
    max_rel_overall = max(rels) if rels else 0.0

    target = "pytorch/pytorch" if filable else "none"

    md: list[str] = []
    md.append(f"# Fuzz results — {KERNEL_NAME}\n")
    md.append(f"- kernel: `{KERNEL_NAME}`")
    md.append(f"- iterations attempted: {len(records)}")
    md.append(f"- iterations completed (status=OK): {completed}")
    md.append(f"- FILABLE (rel>10x thr ∧ denom≥1e-6 ∧ ≥3 seeds): {len(filable)}")
    md.append(f"- TOLERANCE_RECALIBRATION (rel>thr but not filable): {len(tol_recal)}")
    md.append(f"- unsupported-on-MPS: {len(unsupported)}")
    md.append(f"- other errors: {len(other_err)}")
    md.append(f"- elapsed: {elapsed:.1f}s")
    md.append(f"- MPS-vs-CPU max relative error: {max_rel_overall:.4g}")
    md.append(
        "- MPS-vs-CUDA-mock max relative error: N/A "
        "(no NVIDIA hardware present)",
    )
    md.append(f"- recommended upstream filing target: **{target}**")
    md.append("")
    md.append("## Tolerance model")
    md.append(
        "- gpucheck per-dtype defaults (fp32=1e-4, fp16=1e-2, bf16=5e-2), "
        "MPS multiplier ×2, atol scaled by sqrt(k/128) where k=max(S,D). "
        "Threshold = scaled atol. FILABLE requires `rel > 10*threshold`.",
    )
    md.append("")
    md.append("## Top 3 minimal FILABLE repros")
    if not top3:
        md.append("_None — no filable divergences found within budget._")
    else:
        for i, r in enumerate(top3, 1):
            md.append(
                f"{i}. shape={tuple(r['shape'])} dtype={r['dtype']} "
                f"stride={r['stride']} cat={r['shape_cat']} "
                f"max_rel_err={r['max_rel_err']:.4g} "
                f"threshold={r['threshold']:.4g} "
                f"denom_mag={r['denom_mag']:.4g} "
                f"repro_seeds={r.get('repro_seeds', 1)}/5",
            )
    md.append("")

    if tol_recal:
        md.append("## Tolerance-recalibration sample (rel>thr, not filable)")
        seen: set[str] = set()
        for r in sorted(tol_recal, key=lambda x: -(x["max_rel_err"] or 0))[:5]:
            key = f"{r['dtype']}-{r['stride']}-{r['shape_cat']}"
            if key in seen:
                continue
            seen.add(key)
            md.append(
                f"- {key} shape={tuple(r['shape'])} "
                f"rel={r['max_rel_err']:.4g} thr={r['threshold']:.4g} "
                f"ratio={r['max_rel_err']/max(r['threshold'],1e-30):.2f}x "
                f"denom={r['denom_mag']:.3g}",
            )
        md.append("")

    md.append("## Status / classification breakdown")
    status_ct: dict[str, int] = {}
    cls_ct: dict[str, int] = {}
    for r in records:
        s = r.get("status", "?")
        status_ct[s] = status_ct.get(s, 0) + 1
        c = r.get("classification", "?")
        cls_ct[c] = cls_ct.get(c, 0) + 1
    md.append("### status:")
    for s, c in sorted(status_ct.items(), key=lambda kv: -kv[1]):
        md.append(f"- {s}: {c}")
    md.append("### classification:")
    for c, n in sorted(cls_ct.items(), key=lambda kv: -kv[1]):
        md.append(f"- {c}: {n}")
    md.append("")

    if unsupported:
        md.append("## Sample UNSUPPORTED_MPS notes")
        seen2: set[str] = set()
        for r in unsupported:
            note = r.get("note", "")[:160]
            if note in seen2:
                continue
            seen2.add(note)
            md.append(
                f"- shape={tuple(r['shape'])} dtype={r['dtype']} "
                f"stride={r['stride']}: {note}",
            )
            if len(seen2) >= 5:
                break
        md.append("")

    RESULTS_MD.write_text("\n".join(md) + "\n")

    summary = {
        "kernel": KERNEL_NAME,
        "iterations_attempted": len(records),
        "iterations_completed": completed,
        "filable": len(filable),
        "tolerance_recalibration": len(tol_recal),
        "unsupported": len(unsupported),
        "errors": len(other_err),
        "max_rel_err_mps_vs_cpu": max_rel_overall,
        "max_rel_err_mps_vs_cuda_mock": None,
        "filing_target": target,
        "top3_repros": [
            {
                "shape": list(r["shape"]),
                "dtype": r["dtype"],
                "stride": r["stride"],
                "shape_cat": r["shape_cat"],
                "max_rel_err": r["max_rel_err"],
                "threshold": r.get("threshold"),
                "denom_mag": r.get("denom_mag"),
                "repro_seeds": r.get("repro_seeds"),
                "tol_atol": r["tol_atol"],
                "tol_rtol": r["tol_rtol"],
            }
            for r in top3
        ],
        "elapsed_s": elapsed,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    sys.exit(main())
