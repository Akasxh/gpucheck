"""Stride/shape/dtype fuzzer for padded SDPA on MPS vs CPU.

Variant of `_attention_fuzz.py` that exercises padded attention — i.e. SDPA
with an additive key-padding (and sometimes query-padding) mask, where some
sequence positions are -inf-masked. The kernel under test is the *masked*
path of `torch.nn.functional.scaled_dot_product_attention`.

Filable filter (strict):
  - max_rel_err > 10 × k-scaled tolerance
  - denom_magnitude (|ref|) at the worst-error position >= 1e-6
  - reproducible across >= 3 seeds (re-runs with seeds 1..3 must all clear
    the 10x bar with denom>=1e-6).
Below 10× tol but above 1× tol -> TOLERANCE_RECALIBRATION.
Below 1× tol -> OK.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

# gpucheck importable from this worktree
sys.path.insert(0, "/Users/cero/Code/gpucheck-worktrees/fuzz-attention-padded/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_attention-padded.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "scaled_dot_product_attention[padded]"
N_ITER = 500
BUDGET_S = 8 * 60 - 45  # leave 45s headroom for finalisation
SEED = 0xA77E

DTYPES = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]
STRIDE_CATS = ["contiguous", "slice", "transpose", "broadcast"]
PAD_MODES = ["key", "query", "both", "key_bool"]
SHAPE_CATS = ["degenerate", "prime", "pow2", "non_tile", "large"]


def make_shape(rng: random.Random, category: str) -> tuple[int, int, int, int]:
    """Return (B, H, S, D)."""
    if category == "degenerate":
        s = rng.choice([1, 2])
        return (rng.choice([1, 2]), 1, s, 8)
    if category == "prime":
        s = rng.choice([7, 13, 31])
        return (rng.choice([1, 2]), rng.choice([1, 2]), s, rng.choice([8, 16, 32, 64]))
    if category == "pow2":
        s = rng.choice([128, 256])
        return (rng.choice([1, 2]), rng.choice([1, 2, 4]), s, rng.choice([16, 32, 64, 128]))
    if category == "non_tile":
        s = rng.choice([31, 33, 63, 65, 129, 131])
        return (rng.choice([1, 2]), rng.choice([1, 2]), s, rng.choice([16, 32, 64]))
    if category == "large":
        s = rng.choice([512, 1024])
        return (1, rng.choice([1, 2]), s, rng.choice([32, 64, 128]))
    raise ValueError(category)


def _strided(t_cont: torch.Tensor, stride_cat: str) -> torch.Tensor:
    """Re-shape a contiguous (B,H,S,D) tensor to the requested stride pattern."""
    B, H, S, D = t_cont.shape
    if stride_cat == "contiguous":
        return t_cont.contiguous()
    if stride_cat == "slice":
        big = torch.empty((B, H, S * 2, D), dtype=t_cont.dtype, device=t_cont.device)
        big[:, :, ::2, :] = t_cont
        return big[:, :, ::2, :]
    if stride_cat == "transpose":
        # build (B, H, D, S) and transpose last two axes
        flipped = torch.empty((B, H, D, S), dtype=t_cont.dtype, device=t_cont.device)
        flipped.copy_(t_cont.transpose(-1, -2).contiguous())
        return flipped.transpose(-1, -2)
    if stride_cat == "broadcast":
        # collapse heads to 1 then expand
        small = t_cont[:, :1, :, :].contiguous()
        return small.expand(B, H, S, D)
    raise ValueError(stride_cat)


def make_qkv_pair(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    stride_cat: str,
    gen_cpu: torch.Generator,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Build matched (CPU, MPS) Q/K/V tensors with the requested stride pattern."""
    B, H, S, D = shape
    q_c = torch.randn((B, H, S, D), generator=gen_cpu, dtype=torch.float32) * 0.5
    k_c = torch.randn((B, H, S, D), generator=gen_cpu, dtype=torch.float32) * 0.5
    v_c = torch.randn((B, H, S, D), generator=gen_cpu, dtype=torch.float32) * 0.5

    q_cpu = _strided(q_c.to(dtype=dtype), stride_cat)
    k_cpu = _strided(k_c.to(dtype=dtype), stride_cat)
    v_cpu = _strided(v_c.to(dtype=dtype), stride_cat)

    # MPS: copy contiguous content over then re-apply same stride pattern.
    q_mps = _strided(q_c.to(dtype=dtype, device="mps"), stride_cat)
    k_mps = _strided(k_c.to(dtype=dtype, device="mps"), stride_cat)
    v_mps = _strided(v_c.to(dtype=dtype, device="mps"), stride_cat)

    return (q_cpu, k_cpu, v_cpu), (q_mps, k_mps, v_mps)


def make_pad_mask(
    shape: tuple[int, int, int, int],
    pad_mode: str,
    rng: random.Random,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build (attn_mask, q_valid_mask).

    attn_mask is broadcast-compatible with (B, H, S_q, S_kv).
    q_valid_mask is shape (B, S_q) bool — True = valid query position.
    """
    B, H, S, D = shape

    # Pick per-batch valid lengths in [1, S]
    def lens() -> torch.Tensor:
        return torch.tensor(
            [rng.randint(1, max(1, S)) for _ in range(B)], dtype=torch.long
        )

    q_valid = torch.ones((B, S), dtype=torch.bool)

    if pad_mode == "key":
        kl = lens()
        # additive mask: 0 where keep, -inf where pad. Shape (B, 1, 1, S).
        ar = torch.arange(S).unsqueeze(0)  # (1, S)
        keep_kv = ar < kl.unsqueeze(1)  # (B, S)
        attn_mask = torch.zeros((B, 1, 1, S), dtype=torch.float32)
        attn_mask.masked_fill_(~keep_kv.view(B, 1, 1, S), float("-inf"))
        return attn_mask.to(device=device, dtype=dtype), q_valid

    if pad_mode == "key_bool":
        kl = lens()
        ar = torch.arange(S).unsqueeze(0)
        keep_kv = ar < kl.unsqueeze(1)
        # Boolean mask: True = participate.
        attn_mask = keep_kv.view(B, 1, 1, S).to(device=device)
        return attn_mask, q_valid

    if pad_mode == "query":
        ql = lens()
        ar = torch.arange(S).unsqueeze(0)
        keep_q = ar < ql.unsqueeze(1)  # (B, S)
        # Build (B, 1, S_q, 1) so it broadcasts; pad rows -> -inf.
        attn_mask = torch.zeros((B, 1, S, 1), dtype=torch.float32)
        attn_mask.masked_fill_(~keep_q.view(B, 1, S, 1), float("-inf"))
        # NB: a row that is fully -inf gives nan after softmax — which is
        # exactly what we want to compare. Track the q_valid for downstream.
        return attn_mask.to(device=device, dtype=dtype), keep_q

    if pad_mode == "both":
        ql = lens()
        kl = lens()
        ar = torch.arange(S).unsqueeze(0)
        keep_q = ar < ql.unsqueeze(1)
        keep_kv = ar < kl.unsqueeze(1)
        # (B, 1, S_q, S_kv): -inf where row is invalid OR col is invalid
        keep_full = keep_q.unsqueeze(2) & keep_kv.unsqueeze(1)  # (B, S_q, S_kv)
        attn_mask = torch.zeros((B, 1, S, S), dtype=torch.float32)
        attn_mask.masked_fill_(~keep_full.unsqueeze(1), float("-inf"))
        return attn_mask.to(device=device, dtype=dtype), keep_q

    raise ValueError(pad_mode)


def _err_over_valid(
    out_mps: torch.Tensor,
    out_cpu_ref: torch.Tensor,
    q_valid: torch.Tensor,
) -> tuple[float, float, float]:
    """Compute (max_abs_err, max_rel_err, denom_at_worst) restricted to valid q rows.

    out_*: (B, H, S, D); q_valid: (B, S) bool.
    Returns (0, 0, 0) when no valid rows exist.
    """
    a = out_mps.detach().to(device="cpu", dtype=torch.float32)
    b = out_cpu_ref.detach().to(device="cpu", dtype=torch.float32)
    B, H, S, D = a.shape

    valid = q_valid.to(dtype=torch.bool)  # (B, S)
    if not valid.any():
        return 0.0, 0.0, 0.0

    # Broadcast valid mask to (B,H,S,D)
    mask = valid.view(B, 1, S, 1).expand(B, H, S, D)
    diff = (a - b).abs()
    diff = diff.masked_fill(~mask, 0.0)
    abs_err = float(diff.max().item())

    denom = b.abs().clamp_min(1e-12)
    rel = diff / denom
    rel = rel.masked_fill(~mask, 0.0)
    # Find the index of max rel and read the denom there.
    flat_idx = int(rel.flatten().argmax().item())
    rel_max = float(rel.flatten()[flat_idx].item())
    denom_at = float(b.abs().flatten()[flat_idx].item())
    return abs_err, rel_max, denom_at


def run_once(
    cat: str,
    dtype_name: str,
    dtype: torch.dtype,
    stride_cat: str,
    pad_mode: str,
    shape: tuple[int, int, int, int],
    seed: int,
) -> dict | None:
    """Run a single MPS-vs-CPU comparison. Returns record (or None on hard skip)."""
    B, H, S, D = shape
    if S == 0:
        return {"status": "SKIP_EMPTY"}

    gen = torch.Generator()
    gen.manual_seed(seed)

    try:
        (q_c, k_c, v_c), (q_m, k_m, v_m) = make_qkv_pair(shape, dtype, stride_cat, gen)
    except Exception as e:
        return {"status": "BUILD_ERR", "note": f"{type(e).__name__}: {e}"[:200]}

    # Build masks deterministically off seed.
    rng = random.Random(seed ^ 0xBEEF)
    try:
        attn_mask_cpu, q_valid = make_pad_mask(shape, pad_mode, rng, "cpu", dtype)
        # Re-seed an identical RNG for MPS so the mask matches bit-for-bit.
        rng2 = random.Random(seed ^ 0xBEEF)
        attn_mask_mps, _ = make_pad_mask(shape, pad_mode, rng2, "mps", dtype)
    except Exception as e:
        return {"status": "MASK_ERR", "note": f"{type(e).__name__}: {e}"[:200]}

    # CPU reference in fp32 (mask cast to fp32).
    try:
        am_ref = attn_mask_cpu.to(torch.float32) if attn_mask_cpu.dtype != torch.bool else attn_mask_cpu
        ref = F.scaled_dot_product_attention(
            q_c.to(torch.float32),
            k_c.to(torch.float32),
            v_c.to(torch.float32),
            attn_mask=am_ref,
        )
    except Exception as e:
        return {"status": "CPU_REF_ERR", "note": f"{type(e).__name__}: {e}"[:200]}

    try:
        out_mps = F.scaled_dot_product_attention(q_m, k_m, v_m, attn_mask=attn_mask_mps)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError) as e:
        return {"status": "UNSUPPORTED_MPS", "note": f"{type(e).__name__}: {e}"[:200]}

    # Sanitize NaNs to 0 inside masked-out query rows (which are nan-by-spec).
    # We compute errors only over q_valid positions; outside positions are masked.
    abs_err, rel_err, denom_at = _err_over_valid(out_mps, ref, q_valid)
    return {
        "status": "OK",
        "max_abs_err": abs_err,
        "max_rel_err": rel_err,
        "denom_at_worst": denom_at,
    }


def classify(rec: dict, threshold: float) -> str:
    """Bucket a record vs threshold."""
    if rec.get("status") != "OK":
        return rec.get("status", "?")
    rel = rec.get("max_rel_err") or 0.0
    if rel > 10.0 * threshold:
        return "DIVERGE_10X"
    if rel > 1.0 * threshold:
        return "RECALIBRATE"
    return "OK"


def reproduce(
    cat: str,
    dtype_name: str,
    dtype: torch.dtype,
    stride_cat: str,
    pad_mode: str,
    shape: tuple[int, int, int, int],
    threshold: float,
) -> tuple[bool, list[float], list[float]]:
    """Re-run with seeds 1..3 to test reproducibility of a 10x divergence."""
    rels: list[float] = []
    denoms: list[float] = []
    for s in (1, 2, 3):
        r = run_once(cat, dtype_name, dtype, stride_cat, pad_mode, shape, seed=s)
        if not r or r.get("status") != "OK":
            return False, rels, denoms
        rels.append(r["max_rel_err"])
        denoms.append(r["denom_at_worst"])
    # all three must clear 10x bar AND have denom >= 1e-6 at the worst element.
    ok = all(rel > 10.0 * threshold and den >= 1e-6 for rel, den in zip(rels, denoms))
    return ok, rels, denoms


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

    filable: list[dict] = []
    recalibration: list[dict] = []

    for i in range(N_ITER):
        if time.time() - t0 > BUDGET_S:
            print(f"[budget] stopping at iter={i}", flush=True)
            break

        cat = rng.choice(SHAPE_CATS)
        dtype_name, dtype = rng.choice(DTYPES)
        stride_cat = rng.choice(STRIDE_CATS)
        pad_mode = rng.choice(PAD_MODES)
        shape = make_shape(rng, cat)
        seed = rng.randint(0, 2**31 - 1)

        rec: dict = {
            "iter": i,
            "shape_cat": cat,
            "shape": list(shape),
            "dtype": dtype_name,
            "stride": stride_cat,
            "pad_mode": pad_mode,
            "seed": seed,
            "status": "OK",
            "max_abs_err": None,
            "max_rel_err": None,
            "denom_at_worst": None,
            "threshold": None,
            "bucket": "OK",
            "reproducible_3seed": None,
            "note": "",
        }

        try:
            r = run_once(cat, dtype_name, dtype, stride_cat, pad_mode, shape, seed)
        except Exception as e:
            rec["status"] = "EXC"
            rec["note"] = f"{type(e).__name__}: {e}"[:200]
            rec["trace"] = traceback.format_exc()[-400:]
            records.append(rec)
            print(f"[halt-report] iter={i} {rec['note']}", flush=True)
            continue

        if r is None:
            rec["status"] = "SKIP_EMPTY"
            records.append(rec)
            continue

        rec.update({k: r.get(k) for k in (
            "status", "max_abs_err", "max_rel_err", "denom_at_worst", "note",
        ) if k in r})

        if rec["status"] == "OK":
            B, H, S, D = shape
            atol, rtol = compute_tolerance(dtype, k_dim=max(S, D), device_type="mps")
            rec["threshold"] = atol
            rec["bucket"] = classify(rec, atol)
            completed += 1

            if rec["bucket"] == "DIVERGE_10X":
                # Strict denom guard before 3-seed reverification.
                if (rec.get("denom_at_worst") or 0.0) < 1e-6:
                    rec["bucket"] = "OK_NEAR_ZERO_DENOM"
                else:
                    ok, rels, denoms = reproduce(
                        cat, dtype_name, dtype, stride_cat, pad_mode, shape, atol,
                    )
                    rec["reproducible_3seed"] = ok
                    rec["repro_rels"] = rels
                    rec["repro_denoms"] = denoms
                    if ok:
                        filable.append(rec)
                    else:
                        # Strong on seed=seed, weak on seeds 1..3 -> recalibration.
                        rec["bucket"] = "RECALIBRATE"
                        recalibration.append(rec)
            elif rec["bucket"] == "RECALIBRATE":
                recalibration.append(rec)

        records.append(rec)

        if i % 50 == 0:
            print(
                f"[{i:3d}] cat={cat:<10} dt={dtype_name:<9} st={stride_cat:<11} "
                f"pad={pad_mode:<9} shape={shape} bucket={rec['bucket']} "
                f"rel={rec.get('max_rel_err')}",
                flush=True,
            )

    elapsed = time.time() - t0
    write_outputs(records, completed, elapsed, filable, recalibration)
    return 0


def write_outputs(
    records: list[dict],
    completed: int,
    elapsed: float,
    filable: list[dict],
    recalibration: list[dict],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    by_bucket: dict[str, int] = {}
    for r in records:
        if r.get("status") == "OK":
            by_bucket[r["bucket"]] = by_bucket.get(r["bucket"], 0) + 1

    def numel(r: dict) -> int:
        return int(math.prod(r.get("shape") or [1]))

    top3 = sorted(filable, key=numel)[:3]

    max_rel_overall = max(
        (r.get("max_rel_err") or 0.0) for r in records if r.get("max_rel_err") is not None
    ) if any(r.get("max_rel_err") is not None for r in records) else 0.0
    max_abs_overall = max(
        (r.get("max_abs_err") or 0.0) for r in records if r.get("max_abs_err") is not None
    ) if any(r.get("max_abs_err") is not None for r in records) else 0.0

    target = "pytorch/pytorch" if filable else "none"

    md: list[str] = []
    md.append(f"# Fuzz results — {KERNEL_NAME}\n")
    md.append(f"- kernel: `{KERNEL_NAME}`")
    md.append(f"- iterations attempted: {len(records)}")
    md.append(f"- iterations completed (status=OK): {completed}")
    md.append(f"- divergences FILABLE (>10× tol, denom>=1e-6, reproducible /3 seeds): {len(filable)}")
    md.append(f"- TOLERANCE_RECALIBRATION (1-10× tol, or 10× but unstable across seeds): {len(recalibration)}")
    md.append(f"- elapsed: {elapsed:.1f}s")
    md.append(f"- MPS-vs-CPU max relative error: {max_rel_overall:.4g}")
    md.append(f"- MPS-vs-CPU max absolute error: {max_abs_overall:.4g}")
    md.append("- MPS-vs-CUDA-mock max relative error: N/A (no NVIDIA hardware present)")
    md.append(f"- recommended upstream filing target: **{target}**")
    md.append("")
    md.append("## Tolerance model")
    md.append(
        "- gpucheck per-dtype defaults (fp32=1e-4, fp16=1e-2, bf16=5e-2), MPS multiplier 2x, "
        "atol scaled by sqrt(k/128) where k = max(S, D). Single threshold = scaled atol."
    )
    md.append("- FILABLE: max_rel_err > 10× threshold AND denom_magnitude >= 1e-6 AND reproducible across seeds {1,2,3}.")
    md.append("- RECALIBRATE: above 1× threshold but failed at least one of the FILABLE conditions.")
    md.append("- OK: max_rel_err <= 1× threshold (or worst-error denom < 1e-6).")
    md.append("")
    md.append("## Top 3 minimal filable repros")
    if not top3:
        md.append("_None — no FILABLE divergences within 8-minute budget._")
    else:
        for i, r in enumerate(top3, 1):
            md.append(
                f"{i}. shape={tuple(r['shape'])}  dtype={r['dtype']}  "
                f"stride={r['stride']}  pad={r['pad_mode']}  shape_cat={r['shape_cat']}  "
                f"seed={r['seed']}  max_rel_err={r['max_rel_err']:.4g}  "
                f"threshold={r['threshold']:.4g}  "
                f"denom@worst={r['denom_at_worst']:.4g}  "
                f"repro_rels={['{:.3g}'.format(x) for x in r.get('repro_rels', [])]}",
            )
    md.append("")
    md.append("## Status breakdown")
    for s, c in sorted(by_status.items(), key=lambda kv: -kv[1]):
        md.append(f"- {s}: {c}")
    md.append("")
    md.append("## Bucket breakdown (OK iters only)")
    for b, c in sorted(by_bucket.items(), key=lambda kv: -kv[1]):
        md.append(f"- {b}: {c}")
    md.append("")
    md.append("## Recalibration sample (top 5 by max_rel_err)")
    rc_sorted = sorted(
        (r for r in recalibration if r.get("max_rel_err") is not None),
        key=lambda r: -r["max_rel_err"],
    )[:5]
    if not rc_sorted:
        md.append("_None._")
    else:
        for r in rc_sorted:
            md.append(
                f"- shape={tuple(r['shape'])} dtype={r['dtype']} stride={r['stride']} "
                f"pad={r['pad_mode']} rel={r['max_rel_err']:.4g} thr={r['threshold']:.4g} "
                f"denom@worst={r.get('denom_at_worst', 0):.3g}"
            )
    md.append("")

    RESULTS_MD.write_text("\n".join(md) + "\n")

    summary = {
        "kernel": KERNEL_NAME,
        "iterations_attempted": len(records),
        "iterations_completed": completed,
        "divergences_filable": len(filable),
        "divergences_recalibration": len(recalibration),
        "max_rel_err_mps_vs_cpu": max_rel_overall,
        "max_abs_err_mps_vs_cpu": max_abs_overall,
        "max_rel_err_mps_vs_cuda_mock": None,
        "filing_target": target,
        "filter": "max_rel_err>10x tol AND denom>=1e-6 AND repro across seeds 1..3",
        "top3_repros": [
            {
                "shape": list(r["shape"]),
                "dtype": r["dtype"],
                "stride": r["stride"],
                "pad_mode": r["pad_mode"],
                "shape_cat": r["shape_cat"],
                "seed": r["seed"],
                "max_rel_err": r["max_rel_err"],
                "max_abs_err": r["max_abs_err"],
                "denom_at_worst": r["denom_at_worst"],
                "threshold": r["threshold"],
                "repro_rels": r.get("repro_rels"),
                "repro_denoms": r.get("repro_denoms"),
            }
            for r in top3
        ],
        "elapsed_s": elapsed,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    sys.exit(main())
