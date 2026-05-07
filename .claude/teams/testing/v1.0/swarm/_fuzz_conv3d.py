"""Stride/contiguity + shape + dtype fuzzer for ``torch.nn.functional.conv3d``
on MPS, compared against a CPU-fp32 reference.

FILABLE filter (per prompt):
  max_rel_err > 10x tolerance  AND  denom_magnitude >= 1e-6
  AND reproducible across >= 3 seeds.

Below that:
  - TOLERANCE_RECALIBRATION when max_rel_err > tolerance (i.e. divergence)
    but the FILABLE bar is not met.
  - OK when within tolerance.

Driven by gpucheck's tolerance + stride-category model (matches the conv2d
fuzz harness and the rest of the swarm).
"""
from __future__ import annotations

import json
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Make gpucheck importable (worktree path).
sys.path.insert(0, "/Users/cero/Code/gpucheck-worktrees/fuzz-conv3d/src")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from gpucheck.assertions.tolerances import compute_tolerance  # noqa: E402

OUT_DIR = Path("/Users/cero/Code/gpucheck/.claude/teams/testing/v1.0/swarm")
RESULTS_MD = OUT_DIR / "RESULTS_conv3d.md"
SWARM_JSONL = OUT_DIR / "swarm.jsonl"

KERNEL_NAME = "conv3d"
OP_PATH = "torch.nn.functional.conv3d"
N_ITER = 500
BUDGET_S = 8 * 60 - 45  # leave 45s headroom for re-runs + writing
MASTER_SEED = 0xC03D_F022  # "conv3d fuzz" mnemonic
FILABLE_TOL_MULT = 10.0
DENOM_MIN = 1e-6
REPRO_SEEDS_REQUIRED = 3

DTYPES: list[tuple[str, torch.dtype]] = [
    ("float32", torch.float32),
    ("float16", torch.float16),
    ("bfloat16", torch.bfloat16),
]

# 7 categories per gpucheck StrideStrategy.
STRIDE_CATS = [
    "row_major",          # plain contiguous
    "channels_last_3d",   # NDHWC-equivalent layout (column-major-ish for 5D)
    "broadcast",          # broadcast along an axis (expand)
    "transpose",          # transpose two spatial axes
    "slice",              # ::2 stride along D
    "non_contig",         # contiguous-after-clone of a non-contig view
    "gather",             # gather-induced non-contig (index_select on D)
]

# Shape buckets (N, C_in, D, H, W, C_out, kD, kH, kW). Conv3d is memory-heavy;
# stay modest so 500 iters fit in 8 min on Apple Silicon.
SHAPE_CATS = ["degenerate", "non_tile_aligned", "prime", "power_of_2_boundary", "large", "mixed"]


def _make_shape(rng: random.Random, cat: str) -> dict[str, int]:
    """Return conv3d shape dict — input (N,C_in,D,H,W) + (C_out,kD,kH,kW)."""
    if cat == "degenerate":
        # D or H or W = 1, smallest channels.
        d = rng.choice([1, 2])
        h = rng.choice([1, 4])
        w = rng.choice([1, 4])
        return {"N": 1, "C_in": rng.choice([1, 2]), "D": d, "H": h, "W": w,
                "C_out": rng.choice([1, 4]), "kD": 1, "kH": 1, "kW": 1}
    if cat == "prime":
        return {"N": 1, "C_in": rng.choice([3, 5, 7]), "D": rng.choice([7, 11, 13]),
                "H": rng.choice([13, 17]), "W": rng.choice([13, 17]),
                "C_out": rng.choice([7, 11]), "kD": 3, "kH": 3, "kW": 3}
    if cat == "power_of_2_boundary":
        return {"N": 1, "C_in": rng.choice([8, 16]), "D": rng.choice([8, 16, 17]),
                "H": rng.choice([16, 32, 33]), "W": rng.choice([16, 32]),
                "C_out": rng.choice([8, 16]), "kD": rng.choice([1, 3]),
                "kH": 3, "kW": 3}
    if cat == "non_tile_aligned":
        # Choose dims that don't cleanly tile to 32/64.
        return {"N": 1, "C_in": rng.choice([3, 5, 9]), "D": rng.choice([9, 11, 15]),
                "H": rng.choice([15, 17, 31]), "W": rng.choice([15, 17, 31]),
                "C_out": rng.choice([5, 9]), "kD": 3, "kH": 3, "kW": 3}
    if cat == "large":
        # Largest we tolerate — keep memory reasonable.
        return {"N": 1, "C_in": rng.choice([8, 16]), "D": rng.choice([16, 24]),
                "H": rng.choice([32, 48]), "W": rng.choice([32, 48]),
                "C_out": rng.choice([16, 32]), "kD": 3, "kH": 3, "kW": 3}
    if cat == "mixed":
        return {"N": 1, "C_in": rng.choice([4, 6, 8]), "D": rng.choice([5, 8, 12]),
                "H": rng.choice([16, 24, 31]), "W": rng.choice([16, 24, 31]),
                "C_out": rng.choice([8, 12]), "kD": rng.choice([1, 3]),
                "kH": 3, "kW": 3}
    raise ValueError(cat)


def _build_inputs(
    s: dict[str, int],
    dtype: torch.dtype,
    stride_cat: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (input, weight) on CPU with the requested stride layout."""
    g = torch.Generator()
    g.manual_seed(seed)

    in_shape = (s["N"], s["C_in"], s["D"], s["H"], s["W"])
    w_shape = (s["C_out"], s["C_in"], s["kD"], s["kH"], s["kW"])

    base_in = torch.randn(in_shape, generator=g, dtype=torch.float32) * 0.5
    base_w = torch.randn(w_shape, generator=g, dtype=torch.float32) * 0.5

    if stride_cat == "row_major":
        x = base_in.to(dtype=dtype).contiguous()
    elif stride_cat == "channels_last_3d":
        # Permute to NDHWC then back — gives non-default but valid 5D strides.
        # Use torch.channels_last_3d when available.
        try:
            x = base_in.to(dtype=dtype).contiguous(memory_format=torch.channels_last_3d)
        except RuntimeError:
            x = base_in.to(dtype=dtype).contiguous()
    elif stride_cat == "broadcast":
        # Build (N, C_in, 1, H, W) and expand along D — non-contiguous view.
        small = torch.randn((s["N"], s["C_in"], 1, s["H"], s["W"]),
                            generator=g, dtype=torch.float32).mul_(0.5)
        x = small.to(dtype=dtype).contiguous().expand(in_shape)
    elif stride_cat == "transpose":
        # Build (N, C_in, D, W, H) contiguous and transpose last two -> (N, C_in, D, H, W).
        t = torch.randn((s["N"], s["C_in"], s["D"], s["W"], s["H"]),
                        generator=g, dtype=torch.float32).mul_(0.5)
        x = t.to(dtype=dtype).contiguous().transpose(-1, -2)
    elif stride_cat == "slice":
        # Build (N, C_in, 2D, H, W) and slice ::2 along D.
        big = torch.randn((s["N"], s["C_in"], s["D"] * 2, s["H"], s["W"]),
                          generator=g, dtype=torch.float32).mul_(0.5)
        x = big.to(dtype=dtype).contiguous()[:, :, ::2, :, :]
    elif stride_cat == "non_contig":
        # Take a non-contig view (transpose H/W), .clone() — yields contiguous-after-clone.
        big = torch.randn((s["N"], s["C_in"], s["D"], s["W"], s["H"]),
                          generator=g, dtype=torch.float32).mul_(0.5)
        x = big.to(dtype=dtype).contiguous().transpose(-1, -2).clone()
    elif stride_cat == "gather":
        # index_select along D with arbitrary order — non-contig storage.
        big = torch.randn((s["N"], s["C_in"], s["D"], s["H"], s["W"]),
                          generator=g, dtype=torch.float32).mul_(0.5)
        idx = torch.arange(s["D"] - 1, -1, -1)  # reverse order
        x = big.to(dtype=dtype).contiguous().index_select(2, idx)
    else:
        raise ValueError(stride_cat)

    w = base_w.to(dtype=dtype).contiguous()
    return x, w


def _max_rel_err_and_denom(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Returns (max_rel_err, max_abs_err, denom_at_max_rel)."""
    a32 = a.detach().to(device="cpu", dtype=torch.float32)
    b32 = b.detach().to(device="cpu", dtype=torch.float32)
    diff = (a32 - b32).abs()
    if diff.numel() == 0:
        return 0.0, 0.0, 0.0
    denom = b32.abs().clamp_min(1e-12)
    rel = diff / denom
    max_idx = int(rel.flatten().argmax().item())
    return (
        float(rel.flatten()[max_idx].item()),
        float(diff.max().item()),
        float(b32.abs().flatten()[max_idx].item()),
    )


def _run_one(
    s: dict[str, int],
    dtype: torch.dtype,
    stride_cat: str,
    seed: int,
) -> dict:
    """Run one conv3d comparison. Returns a record dict."""
    rec = {
        "seed": seed,
        "shape": s,
        "dtype": str(dtype).removeprefix("torch."),
        "stride": stride_cat,
        "status": "OK",
        "max_rel_err": None,
        "max_abs_err": None,
        "denom_at_max_rel": None,
        "atol": None,
        "rtol": None,
        "k_dim": s["C_in"] * s["kD"] * s["kH"] * s["kW"],
        "diverged": False,
        "filable": False,
        "note": "",
    }

    try:
        x_cpu, w_cpu = _build_inputs(s, dtype, stride_cat, seed)
    except (RuntimeError, ValueError) as e:
        rec["status"] = "BUILD_ERR"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    # CPU fp32 reference.
    try:
        ref = F.conv3d(x_cpu.to(torch.float32), w_cpu.to(torch.float32))
    except (RuntimeError, NotImplementedError) as e:
        rec["status"] = "CPU_REF_ERR"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    # MPS run in test dtype.
    try:
        x_m = x_cpu.detach().to(device="mps")
        w_m = w_cpu.detach().to(device="mps")
        out_m = F.conv3d(x_m, w_m)
        torch.mps.synchronize()
    except (RuntimeError, NotImplementedError) as e:
        rec["status"] = "UNSUPPORTED_MPS"
        rec["note"] = f"{type(e).__name__}: {e}"[:240]
        return rec

    rel, abserr, denom = _max_rel_err_and_denom(out_m, ref)
    atol, rtol = compute_tolerance(dtype, k_dim=rec["k_dim"], device_type="mps")
    rec["max_rel_err"] = rel
    rec["max_abs_err"] = abserr
    rec["denom_at_max_rel"] = denom
    rec["atol"] = atol
    rec["rtol"] = rtol
    # Divergence: max-abs > k-scaled atol.
    rec["diverged"] = abserr > atol
    # FILABLE candidate (single-seed level): rel_err > 10x rtol AND denom>=1e-6.
    # The 3-seed reproducibility check is applied in the post-pass.
    rec["filable_candidate"] = (rel > FILABLE_TOL_MULT * rtol) and (denom >= DENOM_MIN)

    return rec


def _signature(rec: dict) -> tuple:
    """Reproducibility key — what we hold constant across seeds."""
    s = rec["shape"]
    return (
        rec["dtype"],
        rec["stride"],
        s["N"], s["C_in"], s["D"], s["H"], s["W"],
        s["C_out"], s["kD"], s["kH"], s["kW"],
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available():
        msg = f"# {KERNEL_NAME} fuzz results\n\nSTATUS: SKIPPED — torch.mps unavailable.\n"
        RESULTS_MD.write_text(msg)
        with SWARM_JSONL.open("a") as f:
            f.write(json.dumps({
                "kernel": KERNEL_NAME, "status": "SKIPPED",
                "reason": "MPS unavailable",
            }) + "\n")
        return 0

    rng = random.Random(MASTER_SEED)
    t0 = time.time()
    records: list[dict] = []
    completed = 0
    unsupported = 0
    errored = 0
    skipped = 0

    per_shape: dict[str, int] = defaultdict(int)
    per_dtype: dict[str, int] = defaultdict(int)
    per_stride: dict[str, int] = defaultdict(int)

    for i in range(N_ITER):
        if time.time() - t0 > BUDGET_S:
            print(f"[budget] stopping at iter={i}", flush=True)
            break
        cat = rng.choice(SHAPE_CATS)
        dtype_name, dtype = rng.choice(DTYPES)
        stride_cat = rng.choice(STRIDE_CATS)
        s = _make_shape(rng, cat)
        seed = rng.randint(0, 2**31 - 1)
        try:
            rec = _run_one(s, dtype, stride_cat, seed)
        except Exception as e:  # halt-and-report on unexpected process error
            print(f"[FATAL] iter={i} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            return 2
        rec["iter"] = i
        rec["shape_cat"] = cat
        records.append(rec)
        per_shape[cat] += 1
        per_dtype[dtype_name] += 1
        per_stride[stride_cat] += 1
        if rec["status"] == "OK":
            completed += 1
        elif rec["status"] == "UNSUPPORTED_MPS":
            unsupported += 1
        elif rec["status"].endswith("ERR"):
            errored += 1
        else:
            skipped += 1

    elapsed_phase1 = time.time() - t0

    # ---- 3-seed reproducibility for FILABLE candidates ----
    # Group filable_candidate records by signature; for each unique signature,
    # gather distinct seeds where it triggered. If <3, top-up with explicit
    # extra seeds until we either confirm REPRO_SEEDS_REQUIRED or run 3 extra
    # checks and fail to reproduce.
    by_sig: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("filable_candidate"):
            by_sig[_signature(r)].append(r)

    repro_results: list[dict] = []
    for sig, hits in by_sig.items():
        if time.time() - t0 > BUDGET_S + 30:
            break
        seeds_seen = {h["seed"] for h in hits}
        # Already reproduced naturally?
        confirmed_seeds = list(seeds_seen)
        # Top up with explicit fresh seeds derived from MASTER_SEED.
        topup_rng = random.Random(MASTER_SEED ^ hash(sig))
        attempts = 0
        max_topups = max(0, REPRO_SEEDS_REQUIRED - len(confirmed_seeds)) + 2
        # Pull a representative shape/dtype/stride from the first hit.
        rep = hits[0]
        s = rep["shape"]
        dtype = {"float32": torch.float32, "float16": torch.float16,
                 "bfloat16": torch.bfloat16}[rep["dtype"]]
        stride_cat = rep["stride"]
        rels = [h["max_rel_err"] for h in hits]
        denoms = [h["denom_at_max_rel"] for h in hits]
        while len(confirmed_seeds) < REPRO_SEEDS_REQUIRED and attempts < max_topups:
            extra_seed = topup_rng.randint(0, 2**31 - 1)
            attempts += 1
            if extra_seed in seeds_seen:
                continue
            seeds_seen.add(extra_seed)
            try:
                r2 = _run_one(s, dtype, stride_cat, extra_seed)
            except Exception as e:
                print(f"[repro-fatal] {type(e).__name__}: {e}", flush=True)
                continue
            if r2.get("filable_candidate"):
                confirmed_seeds.append(extra_seed)
                rels.append(r2["max_rel_err"])
                denoms.append(r2["denom_at_max_rel"])
        is_filable = len(confirmed_seeds) >= REPRO_SEEDS_REQUIRED
        repro_results.append({
            "signature": {
                "dtype": rep["dtype"], "stride": rep["stride"],
                "shape": rep["shape"], "k_dim": rep["k_dim"],
            },
            "atol": rep["atol"], "rtol": rep["rtol"],
            "seeds_confirmed": confirmed_seeds[:REPRO_SEEDS_REQUIRED],
            "n_seeds_tried": len(seeds_seen),
            "rels_observed": rels,
            "denoms_observed": denoms,
            "filable": is_filable,
        })

    # Mark the records.
    filable_sigs = {tuple(sorted(r["signature"]["shape"].items()) +
                          [r["signature"]["dtype"], r["signature"]["stride"]])
                    for r in repro_results if r["filable"]}
    for r in records:
        sig_key = tuple(sorted(r["shape"].items()) + [r["dtype"], r["stride"]])
        r["filable"] = sig_key in filable_sigs

    elapsed = time.time() - t0

    # ---- Aggregate ----
    # Compute global maxes from completed records only.
    completed_records = [r for r in records if r["status"] == "OK"]
    if completed_records:
        max_rel = max(r["max_rel_err"] for r in completed_records)
        max_abs = max(r["max_abs_err"] for r in completed_records)
        # Denominator-aware max rel: only consider records with denom >= 1e-6
        meaningful = [r for r in completed_records if (r["denom_at_max_rel"] or 0.0) >= DENOM_MIN]
        max_rel_meaningful = (
            max(r["max_rel_err"] for r in meaningful) if meaningful else 0.0
        )
    else:
        max_rel = max_abs = max_rel_meaningful = 0.0

    diverged = [r for r in completed_records if r["diverged"]]
    filable_records = [r for r in completed_records if r["filable"]]
    recalibration_records = [
        r for r in diverged
        if not r["filable"] and (r["denom_at_max_rel"] or 0.0) >= DENOM_MIN
    ]

    # Top-3 FILABLE repros by (over_rtol_factor) — most cite-worthy first.
    def _over_rtol(r: dict) -> float:
        rt = r.get("rtol") or 1e-12
        return (r.get("max_rel_err") or 0.0) / rt

    filable_records.sort(key=_over_rtol, reverse=True)
    top_filable = []
    seen_sigs: set[tuple] = set()
    for r in filable_records:
        sig_key = tuple(sorted(r["shape"].items()) + [r["dtype"], r["stride"]])
        if sig_key in seen_sigs:
            continue
        seen_sigs.add(sig_key)
        top_filable.append({
            "shape": r["shape"], "dtype": r["dtype"], "stride": r["stride"],
            "shape_cat": r["shape_cat"], "k_dim": r["k_dim"],
            "max_rel_err": r["max_rel_err"], "max_abs_err": r["max_abs_err"],
            "denom_at_max_rel": r["denom_at_max_rel"],
            "atol": r["atol"], "rtol": r["rtol"],
            "over_rtol_x": _over_rtol(r),
        })
        if len(top_filable) >= 3:
            break

    if not top_filable:
        # When nothing is FILABLE, surface the top recalibration candidates instead.
        recalibration_records.sort(key=_over_rtol, reverse=True)
        top_recalibration = []
        seen_sigs2: set[tuple] = set()
        for r in recalibration_records:
            sig_key = tuple(sorted(r["shape"].items()) + [r["dtype"], r["stride"]])
            if sig_key in seen_sigs2:
                continue
            seen_sigs2.add(sig_key)
            top_recalibration.append({
                "shape": r["shape"], "dtype": r["dtype"], "stride": r["stride"],
                "shape_cat": r["shape_cat"], "k_dim": r["k_dim"],
                "max_abs_err": r["max_abs_err"], "atol": r["atol"], "rtol": r["rtol"],
                "max_rel_err": r["max_rel_err"], "denom_at_max_rel": r["denom_at_max_rel"],
                "over_atol_x": (r["max_abs_err"] or 0.0) / (r["atol"] or 1e-12),
            })
            if len(top_recalibration) >= 3:
                break
    else:
        top_recalibration = []

    # ---- Write RESULTS_conv3d.md ----
    md_lines: list[str] = []
    md_lines.append(f"# Conv3d fuzz results — {KERNEL_NAME}\n")
    md_lines.append(f"- Kernel: `{OP_PATH}`")
    md_lines.append(f"- Iterations attempted: {len(records)}")
    md_lines.append(f"- Iterations completed (compared): {completed}")
    md_lines.append(f"- Iterations UNSUPPORTED on MPS: {unsupported}")
    md_lines.append(f"- Iterations errored: {errored}")
    md_lines.append(f"- Iterations skipped (other): {skipped}")
    md_lines.append(f"- Divergences (max_abs > MPS-overlay atol): {len(diverged)}")
    md_lines.append(f"- FILABLE (>10x rtol, denom>=1e-6, repro across >=3 seeds): {len(filable_records)}")
    md_lines.append(f"- TOLERANCE_RECALIBRATION (diverged but not FILABLE): {len(recalibration_records)}")
    md_lines.append(f"- Elapsed: {elapsed:.1f}s (phase 1 fuzz: {elapsed_phase1:.1f}s)")
    md_lines.append(
        f"- Backend: MPS real (torch {torch.__version__}, "
        f"`torch.backends.mps.is_available()={torch.backends.mps.is_available()}`)"
    )
    md_lines.append("- CUDA: not exercised — no NVIDIA GPU on host (mocked detection only)")
    md_lines.append("")
    md_lines.append("## Error magnitudes (MPS vs CPU-fp32 reference)")
    md_lines.append("")
    md_lines.append(f"- max abs error: **{max_abs:.3e}**")
    md_lines.append(f"- max rel error (raw, includes near-zero-denom artifacts): **{max_rel:.3e}**")
    md_lines.append(
        f"- max rel error (denom>={DENOM_MIN:.0e} only — 'meaningful'): "
        f"**{max_rel_meaningful:.3e}**"
    )
    md_lines.append("")
    md_lines.append("## Classification (per gpucheck FILABLE filter)")
    md_lines.append("")
    md_lines.append(
        "FILABLE = max_rel_err > 10× rtol AND denom_magnitude >= 1e-6 "
        "AND reproducible across >= 3 distinct seeds."
    )
    md_lines.append("")
    if top_filable:
        md_lines.append("### Top FILABLE repros (by rel_err / rtol multiplier)")
        md_lines.append("")
        for i, r in enumerate(top_filable, 1):
            s = r["shape"]
            md_lines.append(
                f"{i}. shape=(N={s['N']}, C_in={s['C_in']}, D={s['D']}, H={s['H']}, "
                f"W={s['W']}, C_out={s['C_out']}, k={s['kD']}x{s['kH']}x{s['kW']}) "
                f"dtype={r['dtype']} stride={r['stride']} (cat={r['shape_cat']})"
            )
            md_lines.append(
                f"   rel_err={r['max_rel_err']:.3e}, denom={r['denom_at_max_rel']:.3e}, "
                f"abs_err={r['max_abs_err']:.3e}, atol={r['atol']:.3e}, rtol={r['rtol']:.3e}, "
                f"k_dim={r['k_dim']} → {r['over_rtol_x']:.1f}× rtol"
            )
        md_lines.append("")
        md_lines.append("## Recommended upstream filing target: `pytorch/pytorch`")
        md_lines.append("")
        md_lines.append(
            "These divergences exceed the gpucheck MPS-overlay rtol by >10× with a "
            "non-trivial reference magnitude AND reproduce across multiple seeds. "
            "File against `pytorch/pytorch` MPS backend; include a fixed-seed minimal "
            "repro and `torch --version` (this run: " + torch.__version__ + ")."
        )
    else:
        md_lines.append("### No FILABLE divergences observed")
        md_lines.append("")
        md_lines.append("All divergences (if any) are either within 10× tolerance, hit a "
                        "near-zero denominator (artifact), or fail to reproduce across "
                        f">= {REPRO_SEEDS_REQUIRED} seeds.")
        if top_recalibration:
            md_lines.append("")
            md_lines.append("### Top TOLERANCE_RECALIBRATION candidates (by abs/atol multiplier)")
            md_lines.append("")
            for i, r in enumerate(top_recalibration, 1):
                s = r["shape"]
                md_lines.append(
                    f"{i}. shape=(N={s['N']}, C_in={s['C_in']}, D={s['D']}, H={s['H']}, "
                    f"W={s['W']}, C_out={s['C_out']}, k={s['kD']}x{s['kH']}x{s['kW']}) "
                    f"dtype={r['dtype']} stride={r['stride']} (cat={r['shape_cat']})"
                )
                md_lines.append(
                    f"   abs_err={r['max_abs_err']:.3e}, atol={r['atol']:.3e}, "
                    f"rel_err={r['max_rel_err']:.3e}, denom={r['denom_at_max_rel']:.3e}, "
                    f"k_dim={r['k_dim']} → {r['over_atol_x']:.2f}× atol"
                )
        md_lines.append("")
        md_lines.append("## Recommended upstream filing target: `none`")
        md_lines.append("")
        md_lines.append("Action: revisit the gpucheck MPS tolerance overlay and/or the "
                        "MPS xfail registry rather than filing upstream.")

    md_lines.append("")
    md_lines.append("## Coverage breakdown")
    md_lines.append("")
    md_lines.append(f"- per shape bucket: {dict(per_shape)}")
    md_lines.append(f"- per dtype: {dict(per_dtype)}")
    md_lines.append(f"- per stride category: {dict(per_stride)}")
    md_lines.append("")
    md_lines.append(f"Master seed: `{MASTER_SEED:#x}` (deterministic — same seed reproduces same iter sequence).")
    md_lines.append("")

    RESULTS_MD.write_text("\n".join(md_lines))

    # ---- Append JSONL row ----
    summary = {
        "kernel": KERNEL_NAME,
        "agent": "kernel-fuzzer-conv3d-v2",
        "op_path": OP_PATH,
        "status": "OK" if completed > 0 else "EMPTY",
        "device_under_test": "mps",
        "reference": "cpu_fp32",
        "cuda_status": "N/A_mocked_no_nvidia_hardware",
        "torch_version": torch.__version__,
        "master_seed": MASTER_SEED,
        "iters_attempted": len(records),
        "iters_completed": completed,
        "iters_unsupported": unsupported,
        "iters_errored": errored,
        "iters_skipped": skipped,
        "divergences": len(diverged),
        "filable": len(filable_records),
        "tolerance_recalibration": len(recalibration_records),
        "max_rel_err_mps_vs_cpu_raw": max_rel,
        "max_abs_err_mps_vs_cpu": max_abs,
        "max_rel_err_mps_vs_cpu_meaningful": max_rel_meaningful,
        "denom_min_threshold": DENOM_MIN,
        "filable_tol_multiplier": FILABLE_TOL_MULT,
        "repro_seeds_required": REPRO_SEEDS_REQUIRED,
        "top_filable": top_filable,
        "top_recalibration": top_recalibration,
        "per_shape_bucket": dict(per_shape),
        "per_dtype": dict(per_dtype),
        "per_stride_category": dict(per_stride),
        "recommended_filing_target": "pytorch/pytorch" if top_filable else "none",
        "elapsed_s": round(elapsed, 2),
        "results_md": str(RESULTS_MD),
    }
    with SWARM_JSONL.open("a") as f:
        f.write(json.dumps(summary) + "\n")

    print(f"[done] iters={completed}/{len(records)} divergences={len(diverged)} "
          f"filable={len(filable_records)} elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
