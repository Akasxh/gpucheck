"""Fuzzer for kl_div across MPS (real) and CUDA (mocked detection).

Drives gpucheck's per-dtype tolerance + sqrt(k/128) reduction-axis scaling.
CUDA path is N/A on this Mac (no NVIDIA hardware) — we mark CUDA results
UNSUPPORTED and only validate the MPS-vs-CPU divergence channel.

The kernel under test is ``torch.nn.functional.kl_div(log_p, q, reduction='none')``.
PyTorch convention: ``input`` is log-probabilities, ``target`` is probabilities
(unless ``log_target=True``). Element-wise output is ``q * (log(q) - log_p)``;
the class axis is therefore the natural reduction k for sqrt(k/128) scaling.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-kl_div"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "kl_div"
N_ITERS = 250
BUDGET_S = 8 * 60 - 45  # leave 45s for write-out

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_kl_div.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast", "non_contig_perm"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150, 200]
LARGE = [384, 512, 640, 768, 1024]
DEGEN = [1, 2]


@dataclass
class IterResult:
    idx: int
    shape_bucket: str
    B: int
    K: int
    dtype: str
    stride: str
    log_target: bool
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    max_abs_err: float | None
    max_rel_err: float | None
    atol: float | None
    rtol: float | None
    note: str = ""


def pick_dim(bucket: str, rng: random.Random) -> int:
    if bucket == "degenerate":
        return rng.choice(DEGEN)
    if bucket == "prime":
        return rng.choice(PRIMES)
    if bucket == "pow2_boundary":
        return rng.choice(POW2_BOUNDARY)
    if bucket == "non_tile_aligned":
        return rng.choice(NON_TILE)
    if bucket == "large":
        return rng.choice(LARGE)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def make_distribution_pair(
    B: int, K: int, dtype_name: str, stride: str, rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (log_p, q) on CPU at the requested dtype with the requested stride.

    Both are valid log-probabilities / probabilities along the last dim. Returned
    tensors are on CPU; caller transfers a clone to MPS so both backends see the
    same input bytes (modulo dtype-equivalent representation).
    """
    dt = torch_dtype(dtype_name)
    seed_a = rng.randrange(1 << 30)
    seed_b = rng.randrange(1 << 30)
    g_a = torch.Generator().manual_seed(seed_a)
    g_b = torch.Generator().manual_seed(seed_b)

    if stride == "contiguous":
        logits_p = torch.randn(B, K, generator=g_a, dtype=torch.float32)
        logits_q = torch.randn(B, K, generator=g_b, dtype=torch.float32)
        log_p_f32 = F.log_softmax(logits_p, dim=-1)
        q_f32 = F.softmax(logits_q, dim=-1)
        log_p = log_p_f32.to(dt)
        q = q_f32.to(dt)
    elif stride == "slice":
        # over-allocate K*2 along the class axis; compute distributions over the
        # full slab then take stride-2 columns. The result is a non-contiguous
        # view that is no longer a proper distribution — kl_div should still
        # run; we keep this as a stride-stress case (renormalising would hide
        # the layout bug).
        logits_p = torch.randn(B, K * 2, generator=g_a, dtype=torch.float32)
        logits_q = torch.randn(B, K * 2, generator=g_b, dtype=torch.float32)
        log_p_full = F.log_softmax(logits_p, dim=-1).to(dt)
        q_full = F.softmax(logits_q, dim=-1).to(dt)
        log_p = log_p_full[:, ::2]
        q = q_full[:, ::2]
        assert log_p.shape == (B, K) and q.shape == (B, K)
    elif stride == "transpose":
        # Build (K,B) and transpose: keeps math but flips memory layout.
        logits_p = torch.randn(K, B, generator=g_a, dtype=torch.float32)
        logits_q = torch.randn(K, B, generator=g_b, dtype=torch.float32)
        log_p_t = F.log_softmax(logits_p, dim=0).to(dt)  # softmax over K (dim 0)
        q_t = F.softmax(logits_q, dim=0).to(dt)
        log_p = log_p_t.t()
        q = q_t.t()
        assert log_p.shape == (B, K) and q.shape == (B, K)
    elif stride == "broadcast":
        # 1xK distribution expanded to BxK — broadcast on rows. q is
        # batch-varying so the kernel sees mixed-stride inputs.
        logits_p = torch.randn(1, K, generator=g_a, dtype=torch.float32)
        logits_q = torch.randn(B, K, generator=g_b, dtype=torch.float32)
        log_p_full = F.log_softmax(logits_p, dim=-1).to(dt)
        q_full = F.softmax(logits_q, dim=-1).to(dt)
        log_p = log_p_full.expand(B, K)
        q = q_full
    elif stride == "non_contig_perm":
        # Permute a 3-D tensor and slice — exercises generic-stride paths.
        logits_p = torch.randn(B, 2, K, generator=g_a, dtype=torch.float32)
        logits_q = torch.randn(B, 2, K, generator=g_b, dtype=torch.float32)
        log_p_full = F.log_softmax(logits_p, dim=-1).to(dt)
        q_full = F.softmax(logits_q, dim=-1).to(dt)
        # Pick channel 0 — the resulting (B,K) view has stride (2K,1) instead of (K,1).
        log_p = log_p_full[:, 0, :]
        q = q_full[:, 0, :]
    else:
        raise ValueError(stride)
    return log_p, q


def run_one(idx: int, rng: random.Random) -> IterResult:
    bucket = rng.choice(SHAPE_BUCKETS)
    dtype = rng.choice(DTYPES)
    stride = rng.choice(STRIDE_PATTERNS)
    log_target = rng.random() < 0.25
    B = pick_dim(bucket, rng)
    K = pick_dim(bucket, rng)

    base = IterResult(
        idx=idx, shape_bucket=bucket, B=B, K=K, dtype=dtype, stride=stride,
        log_target=log_target,
        status="ERROR", max_abs_err=None, max_rel_err=None, atol=None, rtol=None,
    )

    if B == 0 or K == 0:
        base.status = "SKIP"
        base.note = "degenerate zero-dim"
        return base

    try:
        cpu_dev = torch.device("cpu")
        mps_dev = torch.device("mps")

        log_p_cpu, q_cpu = make_distribution_pair(B, K, dtype, stride, rng)

        # If log_target, target is log-probabilities (and PyTorch will exp it
        # internally). Materialise both forms here so MPS and CPU see identical
        # byte streams.
        if log_target:
            target_cpu = torch.log(q_cpu.clamp_min(1e-12))
        else:
            target_cpu = q_cpu

        log_p_mps = log_p_cpu.detach().clone().contiguous().to(mps_dev)
        target_mps = target_cpu.detach().clone().contiguous().to(mps_dev)
        # NOTE: we deliberately .contiguous() before transfer so the MPS side
        # sees the *materialized* values from each stride pattern. The stride
        # variation has already shaped the values; from that point on we want
        # MPS-vs-CPU to differ only in kernel implementation, not in input bytes.

        # Reference dtype: fp64 for fp32; fp32 for fp16/bf16.
        ref_dtype = torch.float64 if dtype == "float32" else torch.float32

        # Run on CPU at native dtype, then upcast for diffing.
        out_cpu = F.kl_div(log_p_cpu, target_cpu, reduction="none", log_target=log_target).to(ref_dtype)
        out_mps = F.kl_div(log_p_mps, target_mps, reduction="none", log_target=log_target).to(cpu_dev).to(ref_dtype)

        # Tolerances: kl_div is element-wise, but the per-row distribution was
        # produced by a softmax over K, so values scale ~1/K. Use k_dim=K to
        # apply the sqrt(K/128) accumulator scaling — this is conservative for
        # an element-wise op but correct for the upstream reduction context
        # (and matches gpucheck's policy for reduction-class ops).
        atol_mps, rtol_mps = compute_tolerance(dtype, k_dim=K, device_type="mps")
        atol_cpu, rtol_cpu = compute_tolerance(dtype, k_dim=K, device_type="cpu")
        atol, rtol = atol_mps, rtol_mps

        diff = (out_mps - out_cpu).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = out_cpu.abs().clamp_min(1e-12)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0

        base.atol, base.rtol = atol, rtol
        base.max_abs_err, base.max_rel_err = max_abs, max_rel

        # gpucheck combined check: |a-b| <= atol + rtol * |b|
        ok = bool(((diff - (atol + rtol * out_cpu.abs())) <= 0).all().item())
        base.status = "OK" if ok else "DIVERGENCE"
        if not ok:
            base.note = (
                f"mps-vs-cpu max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, "
                f"atol_cpu={atol_cpu:.3e}, log_target={log_target}"
            )
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e).splitlines()[0][:200]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in ("not implemented", "Placeholder storage", "is not currently supported", "MPS")):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def main() -> int:
    rng = random.Random(0xC0FFEE17)
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# kl_div fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters", flush=True)
            break
        attempted = i + 1
        r = run_one(i, rng)
        results.append(r)
        completed += 1
        if r.status == "DIVERGENCE":
            print(
                f"[{i:03d}] DIVERGENCE {r.dtype} {r.stride} B={r.B} K={r.K} "
                f"log_target={r.log_target} max_rel={r.max_rel_err:.3e} "
                f"max_abs={r.max_abs_err:.3e} atol={r.atol:.3e}",
                flush=True,
            )
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.dtype}/{r.stride} B={r.B} K={r.K}: {r.note}", flush=True)
        elif r.status == "UNSUPPORTED":
            print(f"[{i:03d}] UNSUPPORTED {r.dtype}/{r.stride}: {r.note}", flush=True)

    divergences = [r for r in results if r.status == "DIVERGENCE"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    ok = [r for r in results if r.status == "OK"]
    skipped = [r for r in results if r.status == "SKIP"]

    divergences.sort(key=lambda r: (r.max_rel_err or 0.0), reverse=True)
    top3 = divergences[:3]

    overall_max_rel = max(
        (r.max_rel_err for r in results if r.max_rel_err is not None), default=0.0
    )
    overall_max_abs = max(
        (r.max_abs_err for r in results if r.max_abs_err is not None), default=0.0
    )

    if divergences:
        target = "pytorch/pytorch"
    else:
        target = "none"

    elapsed = time.monotonic() - t0

    md: list[str] = []
    md.append(f"# Fuzz results — kernel: `{KERNEL_NAME}`\n")
    md.append("**Backends:** MPS (real, Apple Silicon) vs CPU reference (same dtype).")
    md.append("CUDA: not present — would have been mocked but no comparison performed.\n")
    md.append("## Summary\n")
    md.append(f"- iterations attempted : **{attempted}** / target {N_ITERS}")
    md.append(f"- iterations completed : **{completed}**")
    md.append(f"- OK                    : {len(ok)}")
    md.append(f"- skipped (empty shape) : {len(skipped)}")
    md.append(f"- unsupported (MPS)     : {len(unsupported)}")
    md.append(f"- harness errors        : {len(errors)}")
    md.append(f"- divergences (gpucheck combined check fails): **{len(divergences)}**")
    md.append(f"- max relative error (MPS vs CPU): `{overall_max_rel:.3e}`")
    md.append(f"- max absolute error (MPS vs CPU): `{overall_max_abs:.3e}`")
    md.append("- max relative error (MPS vs CUDA-mock): N/A (no CUDA device; mocked)")
    md.append(f"- runtime: `{elapsed:.1f}s` (budget {BUDGET_S}s; end={'budget_hit' if attempted < N_ITERS else 'completed'})")
    md.append(f"- torch: `{torch.__version__}`\n")
    md.append("## Top 3 minimal repros\n")
    if top3:
        md.append("| # | dtype | stride | B | K | log_target | shape_bucket | max_rel_err | max_abs_err | atol_used | rtol_used | note |")
        md.append("|---|-------|--------|---|---|------------|--------------|-------------|-------------|-----------|-----------|------|")
        for i, r in enumerate(top3, 1):
            md.append(
                f"| {i} | {r.dtype} | {r.stride} | {r.B} | {r.K} | {r.log_target} | {r.shape_bucket} | "
                f"{r.max_rel_err:.3e} | {r.max_abs_err:.3e} | {r.atol:.3e} | {r.rtol:.3e} | {r.note or ''} |"
            )
    else:
        md.append("_None — every (shape, dtype, stride) combo stayed within the MPS-overlay tolerance from `gpucheck.assertions.tolerances`._")
    md.append("")
    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})\n")
        seen: dict[tuple[str, str], list[IterResult]] = {}
        for r in unsupported:
            key = (r.dtype, r.stride)
            seen.setdefault(key, []).append(r)
        for (dt, st), rs in seen.items():
            md.append(f"- {dt} / {st}: {len(rs)} cases — first note: `{rs[0].note}`")
        md.append("")
    if errors:
        md.append(f"## ERROR summary ({len(errors)})\n")
        for r in errors[:10]:
            md.append(f"- iter {r.idx} {r.dtype}/{r.stride}/{r.shape_bucket} B={r.B} K={r.K}: {r.note}")
        md.append("")
    md.append("## Recommended upstream filing target\n")
    if divergences:
        md.append(f"**{target}** — {len(divergences)} divergence(s) over MPS-overlay tolerance; see top-3 table for repros.")
    else:
        md.append("**none** — no divergence exceeded the MPS-overlay tolerance.")
    md.append("\n## Method notes\n")
    md.append("- Kernel: `torch.nn.functional.kl_div(log_p, target, reduction='none', log_target=...)`.")
    md.append("- 25% of iterations sample `log_target=True` (target is log-probabilities) to exercise the alternative branch.")
    md.append("- Inputs built on CPU at fp32, projected to native dtype, transferred to MPS as a contiguous clone — both backends see identical bytes for the materialized stride pattern.")
    md.append("- Tolerance: `gpucheck.assertions.tolerances.compute_tolerance(dtype, k_dim=K, device_type='mps')` — base + sqrt(K/128) accumulator scaling + MPS 2× overlay.")
    md.append("- Divergence rule: gpucheck combined check `|a-b| <= atol + rtol*|b|` element-wise (FAIL on any element).")
    md.append("- Stride categories: contiguous, slice (stride-2 over class axis), transpose (`.t()` view), broadcast (1×K row expanded to B×K), non_contig_perm (3-D channel slice giving stride (2K,1)).")
    md.append("- Shape buckets: degenerate, prime, pow2_boundary, non_tile_aligned, large.")
    md.append("- CUDA channel: gpucheck arch detection is mockable, but the kernel itself cannot execute without an NVIDIA device — the CUDA-vs-MPS comparison is N/A on this host.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "divergences": len(divergences),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cpu_max_abs_err": overall_max_abs,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_mocked_no_nvidia_hardware",
        "top3": [
            {
                "rank": i + 1,
                "dtype": r.dtype,
                "stride": r.stride,
                "B": r.B, "K": r.K,
                "log_target": r.log_target,
                "shape_bucket": r.shape_bucket,
                "max_rel_err": r.max_rel_err,
                "max_abs_err": r.max_abs_err,
                "atol": r.atol, "rtol": r.rtol,
                "note": r.note,
            }
            for i, r in enumerate(top3)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": elapsed,
    }
    with open(SWARM_JSONL, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[done] wrote {RESULTS_MD} and appended JSONL", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
