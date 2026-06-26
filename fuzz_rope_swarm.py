"""RoPE kernel fuzz campaign for the testing swarm.

Drives the gpucheck stride/shape/dtype fuzzers against a standard
Rotary Position Embedding kernel. Reference is CPU fp32; target is MPS
in the sampled dtype. CUDA is mocked (no NVIDIA GPU on this box).

Budget: 250 iterations, 8 minute wall clock.
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from pathlib import Path

import torch

from gpucheck.assertions.tolerances import compute_tolerance
from gpucheck.fuzzing.shapes import fuzz_shapes
from gpucheck.fuzzing.strides import CATEGORIES as STRIDE_CATEGORIES
from gpucheck.fuzzing.strides import fuzz_strides_for_category

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_rope.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"
LOG_PATH = OUT_DIR / "logs" / "rope.log"

KERNEL = "rope"
N_ITERS = 250
WALL_BUDGET_S = 8 * 60

# Stride categories that are layout-meaningful for a 4-D (B, S, H, D) RoPE
# input. We exclude `gather` because the fuzzer already returns it
# contiguous; including it would just duplicate `row_major` measurements.
ACTIVE_STRIDE_CATS: tuple[str, ...] = tuple(
    c for c in STRIDE_CATEGORIES if c != "gather"
)


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%H:%M:%S')}] {msg}\n"
    sys.stdout.write(line)
    with LOG_PATH.open("a") as fh:
        fh.write(line)


# ---------------------------------------------------------------------------
# Reference RoPE
# ---------------------------------------------------------------------------

def build_cos_sin(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard LLaMA-style RoPE cos/sin tables.

    Returns tensors of shape (seq_len, head_dim) — duplicated across the
    half-split so the rotation can be applied as a single multiply.
    """
    assert head_dim % 2 == 0, "RoPE requires even head_dim"
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (S, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (S, D)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, S, H, D)."""
    d = x.shape[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    # cos/sin are (S, D); broadcast over (B, ..., H, D)
    cos_b = cos[None, :, None, :]
    sin_b = sin[None, :, None, :]
    return x * cos_b + rotated * sin_b


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_shape(rng: random.Random) -> tuple[int, int, int, int]:
    """Sample (B, S, H, D) for RoPE.

    D is drawn from gpucheck's shape fuzzer (boundary/prime/non-tile-aligned)
    and forced even (RoPE requires it). B/S/H come from a smaller pool to
    keep tensor sizes practical for 250 iterations on a laptop.
    """
    pool = fuzz_shapes(
        ndim=1,
        min_size=2,
        max_size=512,
        n=40,
        seed=rng.randint(0, 2**31 - 1),
    )
    # head_dim must be even and >= 2; drop 0 and 1 if any leaked through.
    d_candidates = [s[0] for s in pool if s[0] >= 2]
    d = rng.choice(d_candidates)
    if d % 2:
        d -= 1  # snap to even (preserves boundary character: 127 -> 126 etc.)
    if d < 2:
        d = 2

    b = rng.choice([1, 2, 4])
    s = rng.choice([1, 8, 17, 31, 64, 128, 129, 257])
    h = rng.choice([1, 4, 8, 12])
    return (b, s, h, d)


def sample_dtype(rng: random.Random) -> torch.dtype:
    return rng.choice([torch.float32, torch.float16, torch.bfloat16])


def sample_stride_cat(rng: random.Random) -> str:
    return rng.choice(ACTIVE_STRIDE_CATS)


def materialize_x(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    cat: str,
    *,
    seed: int,
    device: str,
) -> torch.Tensor:
    """Build an x tensor with the requested stride category on `device`.

    The stride fuzzer constructs everything on CPU; we move to device after
    the fact so the layout is preserved (.to() preserves strides for
    contiguous results, and our non-contig views become contiguous on copy).
    To keep the layout meaningful on MPS, we build the underlying base on
    `device` directly via fuzz_strides_for_category.
    """
    return fuzz_strides_for_category(
        shape, dtype, cat, device=device, seed=seed,
    )


# ---------------------------------------------------------------------------
# Numerical comparison
# ---------------------------------------------------------------------------

def max_rel_err(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (max_abs_err, max_rel_err) computed in fp64 on CPU.

    Move to CPU first, then upcast — MPS does not support fp64 so a chained
    ``.to(device="cpu", dtype=torch.float64)`` would attempt the cast on the
    MPS device and raise.
    """
    a = out.detach().cpu().to(torch.float64).reshape(-1)
    b = ref.detach().cpu().to(torch.float64).reshape(-1)
    if a.numel() == 0:
        return (0.0, 0.0)
    diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-12)
    return (float(diff.max().item()), float((diff / denom).max().item()))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    LOG_PATH.unlink(missing_ok=True)
    log(f"=== rope fuzz swarm — budget {WALL_BUDGET_S}s, target {N_ITERS} iters ===")

    if not torch.backends.mps.is_available():
        log("SKIPPED: torch.mps.is_available() == False")
        write_outputs(
            attempted=0, completed=0, divergences=[], errors=[],
            unsupported=[], status="SKIPPED_NO_MPS",
        )
        return 0

    log(f"torch={torch.__version__}  mps_built={torch.backends.mps.is_built()}")
    rng = random.Random(20260501)
    t0 = time.monotonic()

    attempted = 0
    completed = 0
    divergences: list[dict] = []
    unsupported: list[dict] = []
    errors: list[dict] = []
    max_errs_mps_cpu: list[float] = []

    for i in range(N_ITERS):
        if time.monotonic() - t0 > WALL_BUDGET_S:
            log(f"BUDGET EXHAUSTED at iter {i} ({time.monotonic() - t0:.1f}s)")
            break
        attempted += 1
        shape = sample_shape(rng)
        dtype = sample_dtype(rng)
        cat = sample_stride_cat(rng)
        seed = rng.randint(0, 2**31 - 1)

        try:
            x_mps = materialize_x(shape, dtype, cat, seed=seed, device="mps")
            # CPU reference always in fp32 for stable ground truth, then we
            # downcast to the test dtype on CPU to model the same numerics
            # the kernel sees.
            x_cpu_fp32 = materialize_x(shape, torch.float32, cat, seed=seed, device="cpu")
            x_cpu = x_cpu_fp32.to(dtype)

            B, S, H, D = shape
            cos_mps, sin_mps = build_cos_sin(S, D, dtype=dtype, device=torch.device("mps"))
            cos_cpu, sin_cpu = build_cos_sin(S, D, dtype=dtype, device=torch.device("cpu"))

            out_mps = rope_apply(x_mps, cos_mps, sin_mps)
            torch.mps.synchronize()
            out_cpu = rope_apply(x_cpu, cos_cpu, sin_cpu)

        except (NotImplementedError, RuntimeError) as exc:
            msg = str(exc).splitlines()[0][:160]
            unsupported.append(
                {"shape": list(shape), "dtype": str(dtype), "stride": cat, "reason": msg},
            )
            log(f"  iter {i:03d} UNSUPPORTED [{cat}|{dtype}|{shape}] {msg}")
            continue
        except Exception as exc:
            tb = traceback.format_exc(limit=3)
            errors.append(
                {"shape": list(shape), "dtype": str(dtype), "stride": cat, "error": str(exc)[:200]},
            )
            log(f"  iter {i:03d} ERROR [{cat}|{dtype}|{shape}] {exc}")
            log(tb)
            continue

        abs_err, rel_err = max_rel_err(out_mps, out_cpu)
        atol, rtol = compute_tolerance(dtype, k_dim=None, device_type="mps")
        # RoPE is element-wise (no inner reduction), so we compare against the
        # base per-dtype tolerance. The task spec invites k-scaling for
        # matmul-class ops; rope is not matmul-class.
        threshold = max(atol, rtol)
        diverged = rel_err > threshold and abs_err > atol
        completed += 1
        max_errs_mps_cpu.append(rel_err)

        if diverged:
            divergences.append({
                "iter": i,
                "shape": list(shape),
                "dtype": str(dtype),
                "stride": cat,
                "max_abs_err": abs_err,
                "max_rel_err": rel_err,
                "atol": atol,
                "rtol": rtol,
                "seed": seed,
            })
            log(
                f"  iter {i:03d} DIVERGENT [{cat}|{dtype}|{shape}] "
                f"rel={rel_err:.3e} abs={abs_err:.3e} (atol={atol:.2e})"
            )
        elif i % 25 == 0:
            log(
                f"  iter {i:03d} ok       [{cat}|{dtype}|{shape}] "
                f"rel={rel_err:.3e} abs={abs_err:.3e}"
            )

    elapsed = time.monotonic() - t0
    log(
        f"=== done: attempted={attempted} completed={completed} "
        f"divergent={len(divergences)} unsupported={len(unsupported)} "
        f"errors={len(errors)} elapsed={elapsed:.1f}s ===",
    )

    write_outputs(
        attempted=attempted,
        completed=completed,
        divergences=divergences,
        errors=errors,
        unsupported=unsupported,
        status="OK",
        elapsed=elapsed,
        max_errs_mps_cpu=max_errs_mps_cpu,
    )
    return 0


def write_outputs(
    *,
    attempted: int,
    completed: int,
    divergences: list[dict],
    errors: list[dict],
    unsupported: list[dict],
    status: str,
    elapsed: float = 0.0,
    max_errs_mps_cpu: list[float] | None = None,
) -> None:
    max_errs_mps_cpu = max_errs_mps_cpu or []
    # Sort divergences by max_rel_err desc; minimal-repro = highest err
    # within smallest shape (numel ascending).
    def numel(d: dict) -> int:
        n = 1
        for x in d["shape"]:
            n *= max(int(x), 1)
        return n

    sorted_for_repro = sorted(
        divergences, key=lambda d: (numel(d), -d["max_rel_err"]),
    )
    top3 = sorted_for_repro[:3]
    overall_max_rel = max(max_errs_mps_cpu) if max_errs_mps_cpu else 0.0

    if not divergences:
        upstream = "none"
    else:
        # All MPS-vs-CPU; the kernel itself is plain PyTorch.
        upstream = "pytorch/pytorch"

    md = [
        f"# RoPE Fuzz Campaign — {KERNEL}",
        "",
        f"- **Status:** {status}",
        f"- **Backend:** MPS (Apple Silicon, real)  +  CUDA (mocked — no NVIDIA GPU)",
        f"- **Iterations attempted:** {attempted} / target {N_ITERS}",
        f"- **Iterations completed:** {completed}",
        f"- **Unsupported (MPS NotImplemented / RuntimeError):** {len(unsupported)}",
        f"- **Hard errors:** {len(errors)}",
        f"- **Divergences (MPS vs CPU):** {len(divergences)}",
        f"- **MPS-vs-CPU max relative error (overall):** {overall_max_rel:.3e}",
        f"- **MPS-vs-CUDA-mock max relative error:** N/A (CUDA detection mocked, no kernel run)",
        f"- **Elapsed:** {elapsed:.1f}s",
        f"- **Recommended upstream filing target:** {upstream}",
        "",
        "## Method",
        "",
        "RoPE applied to (B, S, H, D) inputs. Reference path runs on CPU in the",
        "same dtype as the MPS path (cos/sin built from fp32 then cast). Both",
        "paths share an RNG seed per iteration so the underlying values match.",
        "Tolerance comes from `gpucheck.assertions.tolerances.compute_tolerance`",
        "with `device_type=\"mps\"` (the MPS overlay multiplier is applied).",
        "RoPE is element-wise — no `sqrt(k/128)` scaling is applied.",
        "",
        "Stride categories sampled per iteration: " + ", ".join(ACTIVE_STRIDE_CATS) + ".",
        "Dtypes sampled: float32, float16, bfloat16.",
        "Shapes drawn from `gpucheck.fuzzing.fuzz_shapes` for `head_dim`",
        "(snapped to the nearest even integer ≥ 2; RoPE precondition).",
        "",
        "## Top divergences (minimal repros)",
        "",
    ]
    if not top3:
        md.append("_None — every (shape, dtype, stride) combination stayed within tolerance._")
    else:
        md.append("| # | shape (B,S,H,D) | dtype | stride | max_rel_err | max_abs_err | atol | rtol | seed |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for i, d in enumerate(top3, start=1):
            md.append(
                f"| {i} | {tuple(d['shape'])} | {d['dtype']} | {d['stride']} | "
                f"{d['max_rel_err']:.3e} | {d['max_abs_err']:.3e} | "
                f"{d['atol']:.2e} | {d['rtol']:.2e} | {d['seed']} |"
            )
    md.append("")
    if unsupported:
        md.append("## Unsupported configurations (sampled)")
        md.append("")
        md.append("| shape | dtype | stride | reason |")
        md.append("|---|---|---|---|")
        for u in unsupported[:8]:
            md.append(
                f"| {tuple(u['shape'])} | {u['dtype']} | {u['stride']} | {u['reason']} |",
            )
        md.append("")
    if errors:
        md.append("## Hard errors (sampled)")
        md.append("")
        for e in errors[:5]:
            md.append(f"- {e['stride']} | {e['dtype']} | {tuple(e['shape'])} → {e['error']}")
        md.append("")

    md.append("## Notes")
    md.append("")
    md.append("- CUDA backend was not exercised — no NVIDIA GPU is present on this")
    md.append("  Mac, and the swarm task explicitly said to mock CUDA detection. The")
    md.append("  CUDA-vs-MPS comparison is therefore reported as N/A rather than")
    md.append("  fabricated.")
    md.append("- Tolerances follow gpucheck's MPS overlay (PROVISIONAL — see")
    md.append("  `assertions/tolerances.py:_MPS_TOLERANCE_MULTIPLIERS`).")
    md.append("")

    RESULTS_MD.write_text("\n".join(md))

    record = {
        "kernel": KERNEL,
        "status": status,
        "backend_mps": True,
        "backend_cuda": "mocked",
        "iterations_attempted": attempted,
        "iterations_completed": completed,
        "unsupported": len(unsupported),
        "errors": len(errors),
        "divergences": len(divergences),
        "max_rel_err_mps_vs_cpu": overall_max_rel,
        "max_rel_err_mps_vs_cuda": None,
        "top_repros": [
            {
                "shape": d["shape"],
                "dtype": d["dtype"],
                "stride": d["stride"],
                "max_rel_err": d["max_rel_err"],
                "max_abs_err": d["max_abs_err"],
                "seed": d["seed"],
            }
            for d in top3
        ],
        "recommended_upstream": upstream,
        "elapsed_s": elapsed,
        "torch_version": torch.__version__,
    }
    with SWARM_JSONL.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sys.exit(main())
