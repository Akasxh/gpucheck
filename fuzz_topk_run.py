"""Fuzz torch.topk on MPS vs CPU. Stride/contiguity + shape + dtype sweep."""
from __future__ import annotations

import json
import random
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_MD = OUT_DIR / "RESULTS_topk.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL = "torch.topk"
ITERS_TARGET = 250
BUDGET_S = 7 * 60  # leave 1m headroom for writing outputs

SHAPE_CATEGORIES = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large"]
STRIDE_CATEGORIES = ["contiguous", "slice", "transpose", "broadcast"]
DTYPES = [("float32", torch.float32), ("float16", torch.float16), ("bfloat16", torch.bfloat16)]


def sample_shape(cat: str, rng: random.Random) -> tuple[int, ...]:
    if cat == "degenerate":
        # 0-elements or 1-elements along dims, but topk needs k>=1 along dim
        choices = [(1, 1), (1, 2), (2, 1), (1, 8), (8, 1), (1, 1, 4)]
        return rng.choice(choices)
    if cat == "prime":
        primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        return (rng.choice(primes), rng.choice(primes))
    if cat == "pow2_boundary":
        n = rng.choice([15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257])
        m = rng.choice([16, 32, 64, 128])
        return (m, n)
    if cat == "non_tile_aligned":
        # Tile sizes typically 16/32/128 — pick non-multiples
        return (rng.choice([13, 17, 33, 65, 129]), rng.choice([13, 17, 33, 65, 129]))
    if cat == "large":
        return (rng.choice([512, 1024, 2048]), rng.choice([512, 1024]))
    raise ValueError(cat)


def make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    stride_cat: str,
    device: torch.device,
    rng: random.Random,
) -> tuple[torch.Tensor, str]:
    """Build a tensor in target stride pattern. Returns (tensor, stride_repr)."""
    seed = rng.randint(0, 2**31 - 1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    if stride_cat == "contiguous":
        t = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        return t.contiguous(), f"contiguous{tuple(t.stride())}"

    if stride_cat == "slice":
        # Build 2x along last dim, then slice every other element
        s2 = list(shape)
        s2[-1] = shape[-1] * 2 + 1
        big = torch.randn(s2, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        t = big[..., 1::2][..., : shape[-1]]
        assert t.shape == shape
        return t, f"slice{tuple(t.stride())}"

    if stride_cat == "transpose":
        if len(shape) < 2:
            t = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
            return t, f"contiguous{tuple(t.stride())}"
        # Make a transposed view (non-contiguous)
        rev = tuple(reversed(shape))
        big = torch.randn(rev, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        t = big.transpose(-1, -2) if len(shape) == 2 else big.transpose(0, -1)
        # Match exact shape
        if t.shape != shape:
            t = torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
            return t, f"contiguous{tuple(t.stride())}"
        return t, f"transpose{tuple(t.stride())}"

    if stride_cat == "broadcast":
        # Broadcast along last dim — stride 0 → repeated values, then break ties
        # by adding a tiny noise tensor to keep topk meaningful but still stride 0
        # in the base. We'll keep it pure broadcast (stride 0) for the fuzz signal.
        small = torch.randn((shape[0], 1) if len(shape) == 2 else (1,) * (len(shape) - 1) + (1,),
                            generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        t = small.expand(shape)
        return t, f"broadcast{tuple(t.stride())}"

    raise ValueError(stride_cat)


@dataclass
class Repro:
    shape: list[int]
    dtype: str
    stride_cat: str
    stride_repr: str
    k: int
    dim: int
    largest: bool
    sorted_: bool
    max_abs_err: float
    max_rel_err: float
    atol: float
    rtol: float
    note: str = ""


def run_one(rng: random.Random) -> tuple[str, Repro | None, str]:
    """Returns (status, repro_if_div, message). status in {OK, DIV, UNSUPPORTED, ERROR, SKIP}."""
    shape_cat = rng.choice(SHAPE_CATEGORIES)
    stride_cat = rng.choice(STRIDE_CATEGORIES)
    dtype_name, dtype = rng.choice(DTYPES)
    shape = sample_shape(shape_cat, rng)

    # bf16 on MPS: torch>=2.6 mostly supports it, but some ops fall back
    # — we still try and mark UNSUPPORTED if it fails.

    dim = rng.choice([-1, 0]) if len(shape) >= 2 else -1
    n_along = shape[dim]
    if n_along < 1:
        return "SKIP", None, "empty dim"
    k = rng.randint(1, max(1, min(n_along, 8)))
    largest = rng.choice([True, False])
    sorted_ = rng.choice([True, False])

    try:
        t_cpu, stride_repr = make_tensor(shape, dtype, stride_cat, torch.device("cpu"), rng)
    except Exception as e:
        return "ERROR", None, f"build_cpu: {e}"

    # Build MPS tensor mirroring the same contents+stride pattern by copying CPU
    try:
        if stride_cat == "contiguous":
            t_mps = t_cpu.detach().clone().to("mps")
        else:
            # Replicate the construction on MPS independently (same RNG path
            # is non-trivial across devices; instead, copy values from CPU into
            # an MPS tensor with the same stride layout where possible).
            t_mps = t_cpu.detach().clone().to("mps")
            # For stride patterns we want the MPS-side op to see the *non-contig*
            # layout too. Recreate the layout on MPS:
            if stride_cat == "slice":
                s2 = list(shape)
                s2[-1] = shape[-1] * 2 + 1
                big_cpu = torch.empty(s2, dtype=dtype)
                big_cpu[..., 1::2][..., : shape[-1]] = t_cpu
                t_mps = big_cpu.to("mps")[..., 1::2][..., : shape[-1]]
            elif stride_cat == "transpose":
                # t_cpu is a transposed view; replicate the layout on MPS
                # by transposing a contiguous copy of the underlying data.
                contig_cpu = t_cpu.contiguous()
                if len(shape) == 2:
                    t_mps = contig_cpu.t().contiguous().to("mps").t()
                else:
                    t_mps = contig_cpu.to("mps").transpose(0, -1)
                    if t_mps.shape != shape:
                        t_mps = t_cpu.to("mps")
            elif stride_cat == "broadcast":
                # Build a broadcast view on MPS with stride 0.
                base_shape = tuple(1 if i == len(shape) - 1 or (len(shape) == 2 and i == 1) else shape[i]
                                   for i in range(len(shape)))
                # Simpler: take first slice along dim -1, expand
                sl = [slice(None)] * len(shape)
                sl[-1] = slice(0, 1)
                t_mps = t_cpu[tuple(sl)].to("mps").expand(shape)
                # Recreate t_cpu broadcast view as well so they match values
                t_cpu = t_cpu[tuple(sl)].expand(shape)
    except Exception as e:
        return "ERROR", None, f"build_mps: {e}"

    # Run topk on both
    try:
        v_cpu, i_cpu = torch.topk(t_cpu, k=k, dim=dim, largest=largest, sorted=sorted_)
    except Exception as e:
        return "ERROR", None, f"cpu_topk: {e}"

    try:
        v_mps, i_mps = torch.topk(t_mps, k=k, dim=dim, largest=largest, sorted=sorted_)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError, TypeError) as e:
        msg = str(e)
        if "not implement" in msg.lower() or "not support" in msg.lower() or "MPS" in msg:
            return "UNSUPPORTED", None, f"mps_topk: {msg[:160]}"
        return "ERROR", None, f"mps_topk: {msg[:160]}"

    # Compare values (cast both to float32 for fair comparison).
    # When sorted=False, topk returns values in unspecified order, so we
    # canonicalize by sorting along `dim` before comparing — the *set* of
    # top-k values should match across backends.
    v_cpu_f = v_cpu.detach().to(torch.float32)
    v_mps_f = v_mps.detach().to("cpu").to(torch.float32)
    if not sorted_:
        v_cpu_f, _ = torch.sort(v_cpu_f, dim=dim)
        v_mps_f, _ = torch.sort(v_mps_f, dim=dim)

    if v_cpu_f.shape != v_mps_f.shape:
        repro = Repro(list(shape), dtype_name, stride_cat, stride_repr, k, dim,
                      largest, sorted_, float("nan"), float("nan"), 0.0, 0.0,
                      note=f"shape mismatch cpu={tuple(v_cpu_f.shape)} mps={tuple(v_mps_f.shape)}")
        return "DIV", repro, "shape mismatch"

    diff = (v_cpu_f - v_mps_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    denom = v_cpu_f.abs().clamp_min(1e-12)
    rel = diff / denom
    max_rel = float(rel.max().item()) if rel.numel() else 0.0

    atol, rtol = compute_tolerance(dtype, device_type="mps")
    # topk is selection (no accumulation), so no sqrt(k/128) matmul scaling.
    # Use max(atol, rtol*|x|) tolerance: pass if max_abs <= atol + rtol*|max|.
    cpu_max = float(v_cpu_f.abs().max().item()) if v_cpu_f.numel() else 0.0
    threshold = atol + rtol * cpu_max
    diverged = max_abs > threshold

    if diverged:
        repro = Repro(list(shape), dtype_name, stride_cat, stride_repr, k, int(dim),
                      largest, sorted_, max_abs, max_rel, atol, rtol)
        return "DIV", repro, ""
    return "OK", None, ""


def main() -> int:
    rng = random.Random(0xC0FFEE)
    t0 = time.time()
    completed = 0
    attempted = 0
    ok = 0
    skipped = 0
    unsupported = 0
    errors = 0
    divergences: list[Repro] = []
    error_msgs: list[str] = []
    unsupported_msgs: list[str] = []

    max_rel_seen = 0.0

    for i in range(ITERS_TARGET):
        if time.time() - t0 > BUDGET_S:
            break
        attempted += 1
        try:
            status, repro, msg = run_one(rng)
        except Exception as e:
            errors += 1
            error_msgs.append(f"[iter {i}] {e}")
            traceback.print_exc()
            continue

        if status == "OK":
            ok += 1
            completed += 1
        elif status == "DIV":
            divergences.append(repro)
            completed += 1
            if repro and repro.max_rel_err == repro.max_rel_err:  # not nan
                max_rel_seen = max(max_rel_seen, repro.max_rel_err)
        elif status == "UNSUPPORTED":
            unsupported += 1
            unsupported_msgs.append(msg)
        elif status == "ERROR":
            errors += 1
            error_msgs.append(f"[iter {i}] {msg}")
        elif status == "SKIP":
            skipped += 1

    elapsed = time.time() - t0

    # Pick top 3 divergences by max_rel_err (NaN treated as max)
    def sort_key(r: Repro) -> float:
        v = r.max_rel_err
        return float("inf") if v != v else v  # NaN→inf so shape-mismatches sort to top

    top3 = sorted(divergences, key=sort_key, reverse=True)[:3]

    # Recommend filing target
    if divergences:
        target = "pytorch/pytorch"
    else:
        target = "none"

    # Write markdown
    md_lines = [
        f"# Fuzz results — {KERNEL}",
        "",
        f"- **Kernel:** `{KERNEL}`",
        f"- **Backends compared:** MPS (real, Apple Silicon) vs CPU (reference)",
        f"- **CUDA backend:** mocked / unavailable on this host (no NVIDIA GPU)",
        f"- **Iterations attempted:** {attempted}",
        f"- **Iterations completed (OK+DIV):** {completed}",
        f"  - OK: {ok}",
        f"  - Divergences: {len(divergences)}",
        f"  - Unsupported (op not on MPS): {unsupported}",
        f"  - Errors: {errors}",
        f"  - Skipped (empty dim): {skipped}",
        f"- **Wall time:** {elapsed:.1f}s (budget {BUDGET_S}s)",
        f"- **Max relative error MPS-vs-CPU:** {max_rel_seen:.6g}",
        f"- **Max relative error MPS-vs-CUDA-mock:** N/A (no CUDA device on host; detection mocked, kernel not executed)",
        f"- **Recommended upstream filing target:** `{target}`",
        "",
        "## Top divergences",
    ]
    if not top3:
        md_lines.append("_None._")
    else:
        for idx, r in enumerate(top3, 1):
            md_lines.append(
                f"{idx}. shape={r.shape} dtype={r.dtype} stride={r.stride_cat} "
                f"k={r.k} dim={r.dim} largest={r.largest} sorted={r.sorted_} "
                f"max_abs_err={r.max_abs_err:.6g} max_rel_err={r.max_rel_err:.6g} "
                f"atol={r.atol:.3g} rtol={r.rtol:.3g} stride_repr={r.stride_repr}"
                + (f" note={r.note}" if r.note else "")
            )

    if unsupported_msgs:
        md_lines += ["", "## UNSUPPORTED samples", ""]
        for m in unsupported_msgs[:5]:
            md_lines.append(f"- {m}")
    if error_msgs:
        md_lines += ["", "## Errors (first 5)", ""]
        for m in error_msgs[:5]:
            md_lines.append(f"- {m}")

    RESULTS_MD.write_text("\n".join(md_lines) + "\n")

    # JSONL line
    record = {
        "kernel": KERNEL,
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "ok": ok,
        "divergences": len(divergences),
        "unsupported": unsupported,
        "errors": errors,
        "skipped": skipped,
        "elapsed_s": round(elapsed, 2),
        "max_rel_err_mps_vs_cpu": max_rel_seen,
        "max_rel_err_mps_vs_cuda_mock": None,
        "top_repros": [asdict(r) for r in top3],
        "recommended_target": target,
        "torch_version": torch.__version__,
        "mps_available": True,
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"DONE iters={attempted} ok={ok} div={len(divergences)} "
          f"unsup={unsupported} err={errors} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
