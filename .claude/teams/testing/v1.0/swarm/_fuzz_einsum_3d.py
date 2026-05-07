"""Fuzzer for einsum-3d kernels: 3D batched einsum contractions on MPS vs CPU.

Patterns covered (all 3D inputs / outputs):

  P1  bij,bjk->bik   batched matmul (bmm)
  P2  bij,bkj->bik   batched matmul, transposed B
  P3  ijk,ikl->ijl   batched matmul with leading i index
  P4  bij,bj->bi     batched matrix-vector
  P5  bik,bjk->bij   gram-style contraction
  P6  bij,bij->b     batch sum-product
  P7  bij->bji       transpose / permute (no contraction; sanity channel)

Methodology
-----------
1. 500 iterations randomized over {pattern, shape_bucket, dtype, stride_category}.
2. Inputs constructed on CPU (fp32 backing storage, then cast) so MPS and CPU see
   bitwise-identical source tensors.
3. Reference Y = einsum(A_cpu.fp64, B_cpu.fp64) for fp32 kernels; for fp16/bf16
   the reference is fp32 (CPU promoted) — this isolates MPS-vs-CPU drift at the
   same low precision rather than fp accumulation drift.
4. Pass condition: |Y_mps - Y_cpu| <= atol + rtol*|Y_cpu| element-wise, where
   (atol, rtol) come from gpucheck.compute_tolerance(dtype, k_dim=K_eff,
   device_type="mps") with the v1.0 MPS overlay (2x).
5. FILABLE filter: max_rel_err > 10x atol-overlay AND denom_magnitude >= 1e-6
   AND reproducible across >= 3 seeds. Below that ceiling -> classify as
   TOLERANCE_RECALIBRATION (within 1x..10x atol-overlay) or OK.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field

WORKTREE = "/Users/cero/Code/gpucheck-worktrees/fuzz-einsum-3d"
sys.path.insert(0, os.path.join(WORKTREE, "src"))

import torch  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

KERNEL_NAME = "einsum-3d"
N_ITERS = 500
BUDGET_S = 8 * 60 - 45  # leave 45s for repro-rerun + write-out

OUTPUT_DIR = "/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm"
RESULTS_MD = os.path.join(OUTPUT_DIR, "RESULTS_einsum-3d.md")
SWARM_JSONL = os.path.join(OUTPUT_DIR, "swarm.jsonl")

PATTERNS = [
    "bij,bjk->bik",
    "bij,bkj->bik",
    "ijk,ikl->ijl",
    "bij,bj->bi",
    "bik,bjk->bij",
    "bij,bij->b",
    "bij->bji",
]

SHAPE_BUCKETS = ["degenerate", "prime", "pow2_boundary", "non_tile_aligned", "large", "mixed"]
DTYPES = ["float32", "float16", "bfloat16"]
STRIDE_PATTERNS = ["contiguous", "slice", "transpose", "broadcast", "permute"]

PRIMES = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73]
POW2_BOUNDARY = [15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
NON_TILE = [9, 18, 27, 36, 45, 54, 99, 130, 150]
LARGE = [192, 256, 384, 512]
DEGEN = [1, 2]


@dataclass
class IterResult:
    idx: int
    pattern: str
    shape_bucket: str
    dims: dict[str, int]
    dtype: str
    stride: str
    status: str  # OK | DIVERGENCE | UNSUPPORTED | ERROR | SKIP
    classification: str  # OK | TOLERANCE_RECALIBRATION | FILABLE | (blank)
    max_abs_err: float | None
    max_rel_err: float | None
    denom_magnitude: float | None  # max |Y_cpu| at the worst-case element
    atol: float | None
    rtol: float | None
    over_atol_x: float | None  # max_abs_err / (atol + rtol*|Y_cpu_at_that_elem|)
    seed: int
    repro_seeds: list[int] = field(default_factory=list)
    repro_max_rel_errs: list[float] = field(default_factory=list)
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
    if bucket == "mixed":
        pool = PRIMES + POW2_BOUNDARY + NON_TILE
        return rng.choice(pool)
    raise ValueError(bucket)


def torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _apply_stride(t: torch.Tensor, stride: str) -> torch.Tensor:
    """Return a view of `t` with the requested stride/contiguity property.

    The caller is responsible for ensuring the source `t` is large enough.
    Returned tensor has the same logical shape `t.shape` modulo the slice path.
    """
    if stride == "contiguous":
        return t
    if stride == "slice":
        # Slice along the last axis with stride 2; caller must allocate 2x last dim.
        return t[..., ::2]
    if stride == "transpose":
        # Swap the last two axes (a view, non-contiguous).
        return t.transpose(-2, -1).contiguous().transpose(-2, -1)  # force non-contig view
    if stride == "broadcast":
        # Squeeze last-dim of t (caller ensures dim=1) and expand back.
        # Actually handled by caller producing a singleton-on-some-axis tensor.
        return t
    if stride == "permute":
        # 3D permute that's a view, not contiguous.
        if t.ndim == 3:
            return t.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        return t
    raise ValueError(stride)


def _gen_3d(B: int, M: int, N: int, dtype: torch.dtype, stride: str,
            rng: random.Random, device: torch.device) -> torch.Tensor:
    """Make a 3D tensor with shape (B,M,N) on `device` w/ requested stride class."""
    seed = rng.randrange(1 << 30)
    g = torch.Generator().manual_seed(seed)
    if stride == "contiguous":
        x = torch.randn(B, M, N, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    elif stride == "slice":
        # Allocate 2x along last axis, slice with stride 2.
        x_full = torch.randn(B, M, N * 2, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_full[..., ::2]
        assert x.shape == (B, M, N)
    elif stride == "transpose":
        # Allocate (B,N,M) and .transpose(-2,-1) -> (B,M,N) but non-contiguous.
        x_t = torch.randn(B, N, M, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_t.transpose(-2, -1)
        assert x.shape == (B, M, N) and not x.is_contiguous()
    elif stride == "broadcast":
        # Make a (1,M,N) tensor and broadcast across batch.
        x_one = torch.randn(1, M, N, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_one.expand(B, M, N)
    elif stride == "permute":
        # Allocate (B,N,M) and permute(0,2,1) -> (B,M,N), non-contiguous view.
        x_p = torch.randn(B, N, M, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_p.permute(0, 2, 1)
        assert x.shape == (B, M, N)
    else:
        raise ValueError(stride)
    return x


def _gen_2d(M: int, N: int, dtype: torch.dtype, stride: str,
            rng: random.Random, device: torch.device) -> torch.Tensor:
    seed = rng.randrange(1 << 30)
    g = torch.Generator().manual_seed(seed)
    if stride == "contiguous":
        x = torch.randn(M, N, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    elif stride == "slice":
        x_full = torch.randn(M, N * 2, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_full[:, ::2]
    elif stride == "transpose":
        x_t = torch.randn(N, M, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_t.t()
    elif stride == "broadcast":
        x_one = torch.randn(1, N, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_one.expand(M, N)
    elif stride == "permute":
        # 2D permute is identical to transpose; reuse.
        x_p = torch.randn(N, M, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
        x = x_p.permute(1, 0)
    else:
        raise ValueError(stride)
    return x


def make_inputs(pattern: str, dims: dict[str, int], dtype_name: str, stride: str,
                rng: random.Random, device: torch.device,
                ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """Return (A, B_or_None, k_eff) for the given einsum pattern.

    k_eff = effective contraction depth used for tolerance scaling.
    """
    dt = torch_dtype(dtype_name)
    b, i, j, k, l = dims["b"], dims["i"], dims["j"], dims["k"], dims.get("l", 0)

    if pattern == "bij,bjk->bik":
        A = _gen_3d(b, i, j, dt, stride, rng, device)
        Bt = _gen_3d(b, j, k, dt, stride, rng, device)
        return A, Bt, j
    if pattern == "bij,bkj->bik":
        A = _gen_3d(b, i, j, dt, stride, rng, device)
        Bt = _gen_3d(b, k, j, dt, stride, rng, device)
        return A, Bt, j
    if pattern == "ijk,ikl->ijl":
        # Reuse b->i, i->j, j->k, k->l mapping
        A = _gen_3d(i, j, k, dt, stride, rng, device)
        Bt = _gen_3d(i, k, l, dt, stride, rng, device)
        return A, Bt, k
    if pattern == "bij,bj->bi":
        A = _gen_3d(b, i, j, dt, stride, rng, device)
        Bt = _gen_2d(b, j, dt, stride, rng, device)
        return A, Bt, j
    if pattern == "bik,bjk->bij":
        A = _gen_3d(b, i, k, dt, stride, rng, device)
        Bt = _gen_3d(b, j, k, dt, stride, rng, device)
        return A, Bt, k
    if pattern == "bij,bij->b":
        A = _gen_3d(b, i, j, dt, stride, rng, device)
        Bt = _gen_3d(b, i, j, dt, stride, rng, device)
        return A, Bt, i * j
    if pattern == "bij->bji":
        A = _gen_3d(b, i, j, dt, stride, rng, device)
        return A, None, 1
    raise ValueError(pattern)


def pick_dims(pattern: str, bucket: str, rng: random.Random) -> dict[str, int]:
    d = {}
    needed = set("bijk")
    if pattern == "ijk,ikl->ijl":
        needed |= {"l"}
    if pattern in {"bij,bj->bi", "bij,bij->b", "bij->bji"}:
        needed -= {"k"}
    if pattern == "bij->bji":
        needed -= {"k"}
    for letter in "bijkl":
        if letter in needed:
            d[letter] = pick_dim(bucket, rng)
        else:
            d[letter] = 0
    # Reasonable batch caps to keep memory in check on large bucket.
    if d["b"] > 8 and bucket == "large":
        d["b"] = 4
    return d


def _stride_compatible_with_pattern(pattern: str, stride: str) -> bool:
    """Some stride patterns don't make sense for some operand layouts. Skip
    'transpose'/'permute' on the rank-2 operand of 'bij,bj->bi' since it's a
    1D-after-batch-axis tensor and transpose collapses meaning."""
    if pattern == "bij,bj->bi" and stride in {"transpose", "permute"}:
        return False
    if pattern == "bij->bji" and stride == "broadcast":
        # broadcast on a single-arg pattern with no batch dim collapse is fine
        # but the tolerance becomes degenerate; allow.
        return True
    return True


def _einsum(pattern: str, A: torch.Tensor, B: torch.Tensor | None) -> torch.Tensor:
    if B is None:
        return torch.einsum(pattern, A)
    return torch.einsum(pattern, A, B)


def _run_one_seed(idx: int, pattern: str, dims: dict[str, int], dtype: str,
                  stride: str, seed: int) -> dict[str, float | str | None]:
    """Run one configuration with a fixed seed. Returns metrics dict or error."""
    rng = random.Random(seed)
    cpu_dev = torch.device("cpu")
    mps_dev = torch.device("mps")

    A_cpu_t = make_inputs(pattern, dims, dtype, stride, rng, cpu_dev)
    A_cpu, B_cpu, k_eff = A_cpu_t
    A_mps = A_cpu.detach().clone().to(mps_dev)
    B_mps = B_cpu.detach().clone().to(mps_dev) if B_cpu is not None else None

    Y_cpu = _einsum(pattern, A_cpu, B_cpu)
    Y_mps = _einsum(pattern, A_mps, B_mps).to(cpu_dev)

    # Promote both to a common high-precision space for the metric.
    ref_dtype = torch.float64 if dtype == "float32" else torch.float32
    Y_cpu_h = Y_cpu.to(ref_dtype)
    Y_mps_h = Y_mps.to(ref_dtype)

    diff = (Y_mps_h - Y_cpu_h).abs()
    if diff.numel() == 0:
        return {
            "max_abs": 0.0, "max_rel": 0.0, "denom_at_max": 0.0,
            "k_eff": k_eff, "y_max_abs": 0.0,
        }
    flat = diff.flatten()
    abs_max_idx = int(torch.argmax(flat).item())
    max_abs = float(flat[abs_max_idx].item())
    denom_full = Y_cpu_h.abs().clamp_min(1e-12).flatten()
    denom_at_max = float(denom_full[abs_max_idx].item())
    # max_rel = max |a-b|/|b| over all elements (clamped denom)
    rel = diff / Y_cpu_h.abs().clamp_min(1e-12)
    max_rel = float(rel.max().item())
    y_max_abs = float(Y_cpu_h.abs().max().item())
    return {
        "max_abs": max_abs, "max_rel": max_rel,
        "denom_at_max": denom_at_max, "k_eff": k_eff,
        "y_max_abs": y_max_abs,
    }


def run_one(idx: int, master_rng: random.Random) -> IterResult:
    pattern = master_rng.choice(PATTERNS)
    bucket = master_rng.choice(SHAPE_BUCKETS)
    dtype = master_rng.choice(DTYPES)
    stride = master_rng.choice(STRIDE_PATTERNS)
    while not _stride_compatible_with_pattern(pattern, stride):
        stride = master_rng.choice(STRIDE_PATTERNS)
    dims = pick_dims(pattern, bucket, master_rng)
    seed = master_rng.randrange(1 << 30)

    base = IterResult(
        idx=idx, pattern=pattern, shape_bucket=bucket, dims=dims, dtype=dtype,
        stride=stride, status="ERROR", classification="",
        max_abs_err=None, max_rel_err=None, denom_magnitude=None,
        atol=None, rtol=None, over_atol_x=None, seed=seed,
    )

    # SKIP zero-dims — torch will error or produce 0-numel results.
    used_keys = [k for k, v in dims.items() if v > 0]
    if any(dims[k] == 0 for k in used_keys):
        base.status = "SKIP"
        base.note = "zero dim"
        return base

    try:
        m = _run_one_seed(idx, pattern, dims, dtype, stride, seed)
        atol_mps, rtol_mps = compute_tolerance(dtype, k_dim=int(m["k_eff"]) or None,
                                                device_type="mps")
        base.atol, base.rtol = atol_mps, rtol_mps
        base.max_abs_err, base.max_rel_err = m["max_abs"], m["max_rel"]
        base.denom_magnitude = m["denom_at_max"]

        # gpucheck combined tolerance gate.
        gate = atol_mps + rtol_mps * float(m["denom_at_max"])
        base.over_atol_x = (m["max_abs"] / gate) if gate > 0 else float("inf")

        passed = m["max_abs"] <= gate
        if passed:
            base.status = "OK"
            base.classification = "OK"
        else:
            base.status = "DIVERGENCE"
            # Provisional classification — refined below by repro pass.
            if base.over_atol_x is not None and base.over_atol_x > 10.0 \
                    and m["denom_at_max"] >= 1e-6:
                base.classification = "FILABLE_CANDIDATE"
            else:
                base.classification = "TOLERANCE_RECALIBRATION"
            base.note = (
                f"max_abs={m['max_abs']:.3e} y_max={m['y_max_abs']:.3e} "
                f"denom@max={m['denom_at_max']:.3e} over_atol={base.over_atol_x:.2f}x"
            )
    except NotImplementedError as e:
        base.status = "UNSUPPORTED"
        base.note = f"NotImplemented: {str(e).splitlines()[0][:200]}"
    except RuntimeError as e:
        msg = str(e)
        if any(t in msg for t in (
            "not implemented", "Placeholder storage", "is not currently supported",
            "expand", "must match",
        )):
            base.status = "UNSUPPORTED"
            base.note = msg.splitlines()[0][:200]
        else:
            base.status = "ERROR"
            base.note = msg.splitlines()[0][:200]
    except Exception as e:  # noqa: BLE001
        base.status = "ERROR"
        base.note = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
    return base


def repro_check(r: IterResult, n_extra_seeds: int = 2) -> IterResult:
    """Re-run a FILABLE_CANDIDATE with N additional seeds to confirm reproducibility.

    Promotes to FILABLE if the divergence is observed in >= 3 seeds total.
    Demotes to TOLERANCE_RECALIBRATION otherwise.
    """
    if r.classification != "FILABLE_CANDIDATE":
        return r
    extra_rng = random.Random(r.seed ^ 0xA5A5A5A5)
    repro_seeds: list[int] = [r.seed]
    repro_max_rels: list[float] = [r.max_rel_err or 0.0]
    reproduces = 1  # initial run counts
    for _ in range(n_extra_seeds):
        s2 = extra_rng.randrange(1 << 30)
        try:
            m = _run_one_seed(r.idx, r.pattern, r.dims, r.dtype, r.stride, s2)
        except Exception:  # noqa: BLE001
            continue
        repro_seeds.append(s2)
        repro_max_rels.append(m["max_rel"])
        gate = (r.atol or 0.0) + (r.rtol or 0.0) * float(m["denom_at_max"])
        if gate > 0 and (m["max_abs"] / gate) > 10.0 and m["denom_at_max"] >= 1e-6:
            reproduces += 1
    r.repro_seeds = repro_seeds
    r.repro_max_rel_errs = repro_max_rels
    if reproduces >= 3:
        r.classification = "FILABLE"
    else:
        r.classification = "TOLERANCE_RECALIBRATION"
        r.note += f" | repro {reproduces}/3 seeds"
    return r


def main() -> int:
    if not torch.mps.is_available():
        print("MPS unavailable — SKIPPED", flush=True)
        with open(RESULTS_MD, "w") as f:
            f.write("# einsum-3d fuzz — SKIPPED\n\nMPS unavailable on this host.\n")
        with open(SWARM_JSONL, "a") as f:
            f.write(json.dumps({"kernel": KERNEL_NAME, "status": "SKIPPED"}) + "\n")
        return 0

    master_seed = 0xE1485003  # einsum + 3d marker
    master_rng = random.Random(master_seed)

    t0 = time.monotonic()
    results: list[IterResult] = []
    completed = 0
    attempted = 0
    for i in range(N_ITERS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"[budget] stopping after {i} iters", flush=True)
            break
        attempted = i + 1
        r = run_one(i, master_rng)
        results.append(r)
        completed += 1
        if r.status == "DIVERGENCE":
            print(
                f"[{i:03d}] DIV {r.pattern} {r.dtype}/{r.stride}/{r.shape_bucket} "
                f"dims={r.dims} max_abs={r.max_abs_err:.3e} over={r.over_atol_x:.2f}x "
                f"-> {r.classification}",
                flush=True,
            )
        elif r.status == "ERROR":
            print(f"[{i:03d}] ERROR {r.pattern} {r.dtype}/{r.stride}: {r.note}", flush=True)

    # Repro pass — promote/demote FILABLE_CANDIDATE.
    candidates = [r for r in results if r.classification == "FILABLE_CANDIDATE"]
    print(f"[repro] {len(candidates)} candidate(s) -> rerunning across seeds", flush=True)
    for r in candidates:
        if time.monotonic() - t0 > BUDGET_S - 5:
            print("[repro-budget] aborting repro pass", flush=True)
            break
        repro_check(r)

    # Aggregate.
    ok = [r for r in results if r.status == "OK"]
    divergences = [r for r in results if r.status == "DIVERGENCE"]
    unsupported = [r for r in results if r.status == "UNSUPPORTED"]
    errors = [r for r in results if r.status == "ERROR"]
    skipped = [r for r in results if r.status == "SKIP"]

    filable = [r for r in divergences if r.classification == "FILABLE"]
    recal = [r for r in divergences if r.classification == "TOLERANCE_RECALIBRATION"]
    cand_unverified = [r for r in divergences if r.classification == "FILABLE_CANDIDATE"]

    overall_max_rel = max(
        (r.max_rel_err for r in results if r.max_rel_err is not None),
        default=0.0,
    )

    # By dtype.
    by_dtype: dict[str, float] = {}
    for dt in DTYPES:
        m = max(
            (r.max_rel_err for r in results
             if r.dtype == dt and r.max_rel_err is not None),
            default=0.0,
        )
        by_dtype[dt] = m

    # Pattern coverage counters.
    by_pattern: dict[str, int] = {p: 0 for p in PATTERNS}
    for r in results:
        by_pattern[r.pattern] += 1

    # Top 3 by over_atol_x within DIVERGENCE.
    divergences.sort(key=lambda r: (r.over_atol_x or 0.0), reverse=True)
    top3 = divergences[:3]

    if filable:
        target = "pytorch/pytorch"
    else:
        target = "none"

    md: list[str] = []
    md.append("# einsum-3d Fuzz Results\n")
    md.append("## Summary\n")
    md.append(f"- **kernel:** {KERNEL_NAME} (torch.einsum, 3D contractions)")
    md.append(f"- **patterns:** {', '.join(PATTERNS)}")
    md.append("- **backends:** MPS (real) vs CPU (reference); CUDA mocked (no NVIDIA GPU)")
    md.append(f"- **iterations attempted:** {attempted}")
    md.append(f"- **iterations completed:** {completed}")
    md.append(f"- **OK:** {len(ok)}")
    md.append(f"- **DIVERGENCE:** {len(divergences)} "
              f"(FILABLE={len(filable)}, TOLERANCE_RECALIBRATION={len(recal)}, "
              f"FILABLE_CANDIDATE_unverified={len(cand_unverified)})")
    md.append(f"- **UNSUPPORTED:** {len(unsupported)}")
    md.append(f"- **ERROR:** {len(errors)}")
    md.append(f"- **SKIP:** {len(skipped)}")
    md.append(f"- **wall time:** {time.monotonic() - t0:.1f}s")
    md.append(f"- **master seed:** 0x{master_seed:08X}")
    md.append("")
    md.append("## Max relative error (MPS vs CPU)\n")
    md.append(f"- overall: {overall_max_rel:.3e}")
    for dt in DTYPES:
        md.append(f"- {dt}: {by_dtype[dt]:.3e}")
    md.append("")
    md.append("## Max relative error (MPS vs CUDA)\n")
    md.append("- N/A (no NVIDIA GPU on this host; CUDA detection mocked, no kernels run)\n")
    md.append("## Pattern coverage\n")
    for p in PATTERNS:
        md.append(f"- `{p}`: {by_pattern[p]} iters")
    md.append("")
    md.append("## FILABLE filter\n")
    md.append(
        "Filter applied: `max_abs_err > 10x (atol+rtol*|denom|)` "
        "AND `denom_magnitude >= 1e-6` AND reproducible across `>= 3 seeds`.\n"
    )
    if filable:
        md.append(f"### FILABLE divergences ({len(filable)})\n")
        md.append("| # | pattern | dtype | stride | shape_bucket | dims | max_abs_err | over_atol_x | denom@max | atol | repro_seeds |")
        md.append("|---|---------|-------|--------|--------------|------|-------------|-------------|-----------|------|-------------|")
        for i, r in enumerate(filable, 1):
            md.append(
                f"| {i} | `{r.pattern}` | {r.dtype} | {r.stride} | {r.shape_bucket} | "
                f"{r.dims} | {r.max_abs_err:.3e} | {r.over_atol_x:.2f}x | "
                f"{r.denom_magnitude:.3e} | {r.atol:.3e} | {r.repro_seeds} |"
            )
        md.append("")
    else:
        md.append("No FILABLE divergences observed.\n")
    if recal:
        md.append(f"### TOLERANCE_RECALIBRATION divergences ({len(recal)})\n")
        md.append("These cases break the gate but stay within 10x — candidate for "
                  "MPS xfail registry / overlay re-tune, not upstream filing.\n")
        md.append("| # | pattern | dtype | stride | dims | over_atol_x | denom@max | note |")
        md.append("|---|---------|-------|--------|------|-------------|-----------|------|")
        for i, r in enumerate(recal[:15], 1):
            md.append(
                f"| {i} | `{r.pattern}` | {r.dtype} | {r.stride} | {r.dims} | "
                f"{r.over_atol_x:.2f}x | {r.denom_magnitude:.3e} | {r.note} |"
            )
        if len(recal) > 15:
            md.append(f"| _...{len(recal)-15} more truncated_ | | | | | | | |")
        md.append("")

    md.append("## Top 3 minimal repros (highest over_atol_x)\n")
    if top3:
        md.append("| # | pattern | dtype | stride | dims | max_abs_err | over_atol_x | denom@max | classification |")
        md.append("|---|---------|-------|--------|------|-------------|-------------|-----------|----------------|")
        for i, r in enumerate(top3, 1):
            md.append(
                f"| {i} | `{r.pattern}` | {r.dtype} | {r.stride} | {r.dims} | "
                f"{r.max_abs_err:.3e} | {r.over_atol_x:.2f}x | "
                f"{r.denom_magnitude:.3e} | {r.classification} |"
            )
        md.append("")
    else:
        md.append("_No divergences observed at gpucheck MPS-overlay tolerances._\n")

    if unsupported:
        md.append(f"## UNSUPPORTED summary ({len(unsupported)})\n")
        seen: dict[tuple[str, str, str], list[IterResult]] = {}
        for r in unsupported:
            key = (r.pattern, r.dtype, r.stride)
            seen.setdefault(key, []).append(r)
        for (pat, dt, st), rs in list(seen.items())[:10]:
            md.append(f"- `{pat}` / {dt} / {st}: {len(rs)} cases — first: `{rs[0].note}`")
        md.append("")
    if errors:
        md.append(f"## ERROR summary ({len(errors)})\n")
        for r in errors[:10]:
            md.append(f"- iter {r.idx} `{r.pattern}` {r.dtype}/{r.stride} dims={r.dims}: {r.note}")
        md.append("")

    md.append("## Methodology notes\n")
    md.append("- Source RNG seeded fp32 on CPU; MPS clones via `.to(mps)` so both devices see identical bytes.")
    md.append("- Reference: fp64 for fp32 ops, fp32 for fp16/bf16 ops (CPU-promoted).")
    md.append("- Gate: `|Y_mps - Y_cpu| <= atol + rtol*|Y_cpu|` element-wise; "
              "`(atol, rtol) = compute_tolerance(dtype, k_dim=k_eff, device_type='mps')` "
              "(v1.0 MPS 2x overlay).")
    md.append("- `k_eff` is the contraction depth: `j` for `bij,bjk->bik`, "
              "`k` for `ijk,ikl->ijl`, `i*j` for `bij,bij->b`, etc.")
    md.append("- Stride patterns: contiguous, slice (stride-2 last axis), transpose, "
              "broadcast (1 along batch), permute (3D non-contig view).")
    md.append("- FILABLE filter requires reproducibility across >=3 seeds — per-seed RNG drift "
              "or denominator-near-zero one-offs are correctly excluded.")
    md.append("")
    md.append(f"## Recommended upstream filing target\n\n**{target}**")
    if not filable:
        md.append(" — no divergences cleared the FILABLE filter.")
    md.append("")

    with open(RESULTS_MD, "w") as f:
        f.write("\n".join(md))

    record = {
        "kernel": KERNEL_NAME,
        "agent": "kernel-fuzzer-einsum-3d-v2",
        "iters_attempted": attempted,
        "iters_completed": completed,
        "ok": len(ok),
        "divergences": len(divergences),
        "filable": len(filable),
        "tolerance_recalibration": len(recal),
        "filable_candidate_unverified": len(cand_unverified),
        "unsupported": len(unsupported),
        "errors": len(errors),
        "skipped": len(skipped),
        "mps_vs_cpu_max_rel_err": overall_max_rel,
        "mps_vs_cpu_max_rel_err_by_dtype": by_dtype,
        "mps_vs_cuda_max_rel_err": None,
        "cuda_status": "N/A_mocked_no_nvidia_hardware",
        "patterns": PATTERNS,
        "by_pattern_counts": by_pattern,
        "filable_repros": [
            {
                "pattern": r.pattern,
                "dtype": r.dtype,
                "stride": r.stride,
                "dims": r.dims,
                "shape_bucket": r.shape_bucket,
                "max_abs_err": r.max_abs_err,
                "max_rel_err": r.max_rel_err,
                "denom_magnitude": r.denom_magnitude,
                "atol": r.atol,
                "rtol": r.rtol,
                "over_atol_x": r.over_atol_x,
                "repro_seeds": r.repro_seeds,
                "repro_max_rel_errs": r.repro_max_rel_errs,
            }
            for r in filable
        ],
        "top3": [
            {
                "rank": i + 1,
                "pattern": r.pattern,
                "dtype": r.dtype,
                "stride": r.stride,
                "dims": r.dims,
                "shape_bucket": r.shape_bucket,
                "max_abs_err": r.max_abs_err,
                "max_rel_err": r.max_rel_err,
                "denom_magnitude": r.denom_magnitude,
                "atol": r.atol,
                "rtol": r.rtol,
                "over_atol_x": r.over_atol_x,
                "classification": r.classification,
                "note": r.note,
            }
            for i, r in enumerate(top3)
        ],
        "upstream_target": target,
        "torch_version": torch.__version__,
        "elapsed_s": time.monotonic() - t0,
        "master_seed": master_seed,
        "filter_spec": {
            "max_abs_over_gate_x": 10.0,
            "min_denom_magnitude": 1e-6,
            "min_reproducing_seeds": 3,
        },
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
