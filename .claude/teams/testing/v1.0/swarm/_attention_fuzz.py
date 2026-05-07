"""Stride/shape/dtype fuzzer for torch SDPA attention on MPS vs CPU.

Driven by gpucheck's tolerance + stride-category model.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

# Make gpucheck importable
sys.path.insert(0, "/Users/cero/Code/gpucheck-worktrees/fuzz-attention/src")

import torch
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_attention.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "scaled_dot_product_attention"
N_ITER = 250
BUDGET_S = 8 * 60 - 30  # leave 30s headroom for writing
SEED = 0x5A11

# Shape categories — attention shape is (B, H, S, D). We sample axis sizes
# from gpucheck's category corpus and assemble a 4D shape.
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
    """Return (B, H, S, D) for attention.

    D (head_dim) is constrained to 8/16/32/64/128 only when not "degenerate"
    so SDPA actually has something to do — but we still let the chosen
    category drive S (sequence length) which is the bug-rich axis.
    """
    if category == "degenerate":
        # Use a 0 or 1 in seq, plus minimal batch/heads.
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
    """Build Q, K, V tensors with the requested stride pattern."""
    B, H, S, D = shape

    def base() -> torch.Tensor:
        # Generate in fp32 on CPU then move (MPS generators do not support seed
        # from CPU generator directly; this is fine for fuzzing).
        t = torch.randn((B, H, S, D), generator=gen, dtype=torch.float32) * 0.5
        return t.to(dtype=dtype, device=device).contiguous()

    if stride_cat == "contiguous":
        return base(), base(), base()

    if stride_cat == "slice":
        # Build doubled-S then [:, :, ::2, :] -> stride-2 along seq dim.
        big = (B, H, S * 2, D)
        bigt = lambda: torch.randn(big, generator=gen, dtype=torch.float32).mul_(0.5).to(
            dtype=dtype, device=device,
        ).contiguous()
        q = bigt()[:, :, ::2, :]
        k = bigt()[:, :, ::2, :]
        v = bigt()[:, :, ::2, :]
        return q, k, v

    if stride_cat == "transpose":
        # Build (B, H, D, S) contiguous, transpose last two -> (B, H, S, D).
        tshape = (B, H, D, S)
        bigt = lambda: torch.randn(tshape, generator=gen, dtype=torch.float32).mul_(0.5).to(
            dtype=dtype, device=device,
        ).contiguous().transpose(-1, -2)
        return bigt(), bigt(), bigt()

    if stride_cat == "broadcast":
        # Broadcast along H: build (B, 1, S, D) then expand to (B, H, S, D).
        small = (B, 1, S, D)
        bigt = lambda: torch.randn(small, generator=gen, dtype=torch.float32).mul_(0.5).to(
            dtype=dtype, device=device,
        ).contiguous().expand(B, H, S, D)
        return bigt(), bigt(), bigt()

    raise ValueError(stride_cat)


def _max_rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max |a-b| / (|b| + eps) computed in fp32 on CPU."""
    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    diff = (a32 - b32).abs()
    denom = b32.abs().clamp_min(1e-7)
    return float((diff / denom).max().item())


def _max_abs_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    return float((a32 - b32).abs().max().item())


def run_iter(idx: int, rng: random.Random) -> dict:
    cat = rng.choice(["degenerate", "prime", "pow2", "non_tile", "large"])
    dtype_name, dtype = rng.choice(DTYPES)
    stride_cat = rng.choice(STRIDE_CATS)
    shape = make_shape(rng, cat)
    B, H, S, D = shape

    rec: dict = {
        "iter": idx,
        "shape_cat": cat,
        "shape": list(shape),
        "dtype": dtype_name,
        "stride": stride_cat,
        "status": "OK",
        "max_rel_err": None,
        "max_abs_err": None,
        "tol_atol": None,
        "tol_rtol": None,
        "diverged": False,
        "note": "",
    }

    # Degenerate shapes with S=0 are degenerate-by-definition; SDPA on empty
    # is well-defined as empty out. Skip with note.
    if S == 0:
        rec["status"] = "SKIP_EMPTY"
        return rec

    gen = torch.Generator()
    gen.manual_seed(rng.randint(0, 2**31 - 1))

    try:
        q_cpu, k_cpu, v_cpu = make_qkv(shape, dtype, "cpu", stride_cat, gen)
    except Exception as e:  # pragma: no cover
        rec["status"] = "BUILD_ERR_CPU"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec

    # Re-seed so MPS gets the same numerical content.
    gen2 = torch.Generator()
    gen2.manual_seed(rng.randint(0, 2**31 - 1))
    # We want Q/K/V to be IDENTICAL on CPU and MPS up to dtype rounding.
    # Easiest: build on CPU then move to MPS.
    try:
        q_mps = q_cpu.detach().to(device="mps").contiguous() if stride_cat == "contiguous" else None
    except Exception:
        q_mps = None

    # Build MPS tensors by re-applying the same stride pattern to a fresh
    # buffer — but the simplest deterministic mirror is: build the *contig*
    # data on CPU, then construct MPS tensors that match the strided layout.
    # For that, we just .to(device="mps") the strided CPU tensor; it copies.
    try:
        q_m = q_cpu.detach().to(device="mps")
        k_m = k_cpu.detach().to(device="mps")
        v_m = v_cpu.detach().to(device="mps")
    except Exception as e:
        rec["status"] = "MPS_COPY_ERR"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec

    # Run reference (CPU, in fp32 for numeric stability), and MPS (in dtype).
    # SDPA on CPU bf16/fp16 is supported in 2.x; we still cast to fp32 as the
    # reference to remove CPU dtype-quantization noise.
    try:
        ref = F.scaled_dot_product_attention(
            q_cpu.to(torch.float32),
            k_cpu.to(torch.float32),
            v_cpu.to(torch.float32),
        )
    except Exception as e:
        rec["status"] = "CPU_REF_ERR"
        rec["note"] = f"{type(e).__name__}: {e}"
        return rec

    try:
        out_mps = F.scaled_dot_product_attention(q_m, k_m, v_m)
        # Sync MPS work
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError) as e:
        rec["status"] = "UNSUPPORTED_MPS"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    rel = _max_rel_err(out_mps, ref)
    abs_err = _max_abs_err(out_mps, ref)
    rec["max_rel_err"] = rel
    rec["max_abs_err"] = abs_err

    # Tolerance: gpucheck per-dtype tol scaled by sqrt(k/128) for matmul-class.
    # For SDPA, the inner reduction is over S (softmax over keys) AND D (Q·K^T).
    # We scale by max(S, D) to cover both contractions conservatively.
    k_dim = max(S, D)
    atol, rtol = compute_tolerance(dtype, k_dim=k_dim, device_type="mps")
    rec["tol_atol"] = atol
    rec["tol_rtol"] = rtol

    # Divergence rule: max-rel-err > k-scaled tolerance for the dtype.
    # gpucheck's compute_tolerance applies the sqrt(k/128) factor to atol only,
    # so atol is the one carrying the spec's k-scaling. We compare against atol
    # (since per-dtype default has atol == rtol, this is the scaled per-dtype tol).
    rec["threshold"] = atol
    rec["diverged"] = rel > atol

    return rec


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
            print(f"[budget] stopping at iter={i}", flush=True)
            break
        try:
            rec = run_iter(i, rng)
        except Exception as e:  # halt-and-report (but keep going for the run)
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
        if i % 25 == 0:
            print(
                f"[{i:3d}] cat={rec.get('shape_cat'):<10} "
                f"dt={rec.get('dtype'):<9} st={rec.get('stride'):<11} "
                f"shape={rec.get('shape')} st={rec['status']} "
                f"rel={rec.get('max_rel_err')}",
                flush=True,
            )

    elapsed = time.time() - t0
    write_outputs(records, completed, elapsed)
    return 0


def write_outputs(records: list[dict], completed: int, elapsed: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    divergences = [r for r in records if r.get("diverged")]
    unsupported = [r for r in records if r.get("status") == "UNSUPPORTED_MPS"]
    other_err = [
        r for r in records
        if r.get("status") not in {"OK", "UNSUPPORTED_MPS", "SKIP_EMPTY"}
    ]

    # Top 3 minimal repros: smallest total numel among divergences.
    def numel(r: dict) -> int:
        return int(math.prod(r.get("shape") or [1]))

    top3 = sorted(divergences, key=numel)[:3]

    max_rel_overall = max(
        (r.get("max_rel_err") or 0.0) for r in records if r.get("max_rel_err") is not None
    ) if any(r.get("max_rel_err") is not None for r in records) else 0.0

    # Recommend filing target.
    if divergences:
        # SDPA on MPS — file against pytorch/pytorch (per gpucheck SYNTHESIS).
        target = "pytorch/pytorch"
    else:
        target = "none"

    md_lines: list[str] = []
    md_lines.append(f"# Fuzz results — {KERNEL_NAME}\n")
    md_lines.append(f"- kernel: `{KERNEL_NAME}`")
    md_lines.append(f"- iterations attempted: {len(records)}")
    md_lines.append(f"- iterations completed (status=OK): {completed}")
    md_lines.append(f"- divergences found: {len(divergences)}")
    md_lines.append(f"- unsupported-on-MPS: {len(unsupported)}")
    md_lines.append(f"- other errors: {len(other_err)}")
    md_lines.append(f"- elapsed: {elapsed:.1f}s")
    md_lines.append(f"- MPS-vs-CPU max relative error: {max_rel_overall:.4g}")
    md_lines.append(
        "- MPS-vs-CUDA-mock max relative error: N/A "
        "(no NVIDIA hardware present; mocked detection cannot produce real numerics)",
    )
    md_lines.append(f"- recommended upstream filing target: **{target}**")
    md_lines.append("")
    md_lines.append("## Tolerance model")
    md_lines.append(
        "- gpucheck per-dtype defaults (fp32=1e-4, fp16=1e-2, bf16=5e-2), "
        "MPS multiplier 2x, atol scaled by sqrt(k/128) where k = max(S, D). "
        "(rtol is not k-scaled by gpucheck, so we use the scaled atol as the "
        "single per-dtype threshold.)",
    )
    md_lines.append("- divergence rule: `max_rel_err > k_scaled_threshold`.")
    md_lines.append("")
    md_lines.append("## Top 3 minimal repros")
    if not top3:
        md_lines.append("_None — no divergences found within budget._")
    else:
        for i, r in enumerate(top3, 1):
            md_lines.append(
                f"{i}. shape={tuple(r['shape'])}  dtype={r['dtype']}  "
                f"stride={r['stride']}  shape_cat={r['shape_cat']}  "
                f"max_rel_err={r['max_rel_err']:.4g}  "
                f"threshold={r.get('threshold', r['tol_rtol']):.4g}",
            )
    md_lines.append("")
    md_lines.append("## Status breakdown")
    by_status: dict[str, int] = {}
    for r in records:
        s = r.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1
    for s, c in sorted(by_status.items(), key=lambda kv: -kv[1]):
        md_lines.append(f"- {s}: {c}")
    md_lines.append("")
    if unsupported:
        md_lines.append("## Sample UNSUPPORTED_MPS notes")
        seen: set[str] = set()
        for r in unsupported:
            note = r.get("note", "")[:160]
            if note in seen:
                continue
            seen.add(note)
            md_lines.append(
                f"- shape={tuple(r['shape'])} dtype={r['dtype']} stride={r['stride']}: {note}",
            )
            if len(seen) >= 5:
                break
        md_lines.append("")

    RESULTS_MD.write_text("\n".join(md_lines) + "\n")

    summary = {
        "kernel": KERNEL_NAME,
        "iterations_attempted": len(records),
        "iterations_completed": completed,
        "divergences": len(divergences),
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
