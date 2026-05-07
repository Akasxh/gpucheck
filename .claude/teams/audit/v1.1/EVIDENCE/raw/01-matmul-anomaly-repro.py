"""Reproduce the matmul fp32 1024^3 MPS anomaly with shape neighbors and warm/cold paths.

Charter:
- Different seeds (was 0..4 — try 7,11,42,1337,2026)
- Different dtype priors (warm up with fp16 first vs cold)
- Different shape neighbors (1023^3, 1025^3, 1024x1023x1024)
- Compare to which dispatch path is taken
"""
from __future__ import annotations

import os
import sys
import time
import json
import platform

import torch

# Optional: enable PYTORCH_MPS_PREFER_METAL via env to test the metal-shader
# fallback path. Default off here (we want the default path as users see).
PREFER_METAL = os.environ.get("PYTORCH_MPS_PREFER_METAL", "0") == "1"

print(f"torch={torch.__version__}")
print(f"mps_avail={torch.backends.mps.is_available()}")
print(f"platform={platform.platform()}")
print(f"PYTORCH_MPS_PREFER_METAL={PREFER_METAL}")
assert torch.backends.mps.is_available()


def _bench(M: int, K: int, N: int, dtype: torch.dtype, n_iters: int = 50,
           warmup: int = 10, seed: int = 0,
           dtype_warmup: torch.dtype | None = None) -> dict:
    """Bench a single (M,K,N,dtype) cell, return median+iqr ms and basic stats."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a32 = torch.randn(M, K, dtype=torch.float32, generator=gen)
    b32 = torch.randn(K, N, dtype=torch.float32, generator=gen)
    a = a32.to(dtype).to("mps")
    b = b32.to(dtype).to("mps")

    # Optional warmup with a different dtype to test caching/dispatch interaction
    if dtype_warmup is not None and dtype_warmup is not dtype:
        wa = a32.to(dtype_warmup).to("mps")
        wb = b32.to(dtype_warmup).to("mps")
        for _ in range(5):
            (wa @ wb).contiguous()
        torch.mps.synchronize()
        del wa, wb

    # Real warmup in target dtype
    for _ in range(warmup):
        c = a @ b
    torch.mps.synchronize()

    times: list[float] = []
    for _ in range(n_iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    median = times[n_iters // 2]
    p25 = times[n_iters // 4]
    p75 = times[3 * n_iters // 4]
    p_min = times[0]
    p_max = times[-1]

    flops = 2.0 * M * K * N
    gflops = flops / (median / 1000.0) / 1e9

    return {
        "shape": (M, K, N),
        "dtype": str(dtype).split(".")[-1],
        "median_ms": round(median, 4),
        "p25_ms": round(p25, 4),
        "p75_ms": round(p75, 4),
        "min_ms": round(p_min, 4),
        "max_ms": round(p_max, 4),
        "gflops": round(gflops, 1),
        "n": n_iters,
        "seed": seed,
        "dtype_warmup": str(dtype_warmup).split(".")[-1] if dtype_warmup else None,
    }


def main() -> int:
    out: list[dict] = []

    print("\n=== A. Confirmation with new seed (7) at 1024^3 baseline ===")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        r = _bench(1024, 1024, 1024, dt, n_iters=50, warmup=10, seed=7)
        out.append({"experiment": "A_baseline_seed7", **r})
        print(f"  {r['dtype']:8s}: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== B. Multi-seed sweep at 1024^3 fp32 (5 seeds) ===")
    for seed in (7, 11, 42, 1337, 2026):
        r = _bench(1024, 1024, 1024, torch.float32, n_iters=50, warmup=10, seed=seed)
        out.append({"experiment": "B_fp32_multiseed", **r})
        print(f"  seed={seed:5d}: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== C. Shape neighbors at fp32 (sensitivity to exact shape) ===")
    shapes = [
        (1023, 1023, 1023),
        (1024, 1024, 1024),
        (1025, 1025, 1025),
        (1024, 1023, 1024),
        (1023, 1024, 1023),
        (1024, 1024, 1023),
        (512, 512, 512),
        (2048, 2048, 2048),
        (768, 768, 768),
        (1000, 1000, 1000),
    ]
    for s in shapes:
        r = _bench(*s, dtype=torch.float32, n_iters=30, warmup=10, seed=7)
        out.append({"experiment": "C_shape_neighbors_fp32", **r})
        print(f"  {s[0]:>4}x{s[1]:>4}x{s[2]:>4}: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== D. Same neighbors at fp16 (cross-check fp16 also slow at 1024?) ===")
    for s in shapes:
        r = _bench(*s, dtype=torch.float16, n_iters=30, warmup=10, seed=7)
        out.append({"experiment": "D_shape_neighbors_fp16", **r})
        print(f"  {s[0]:>4}x{s[1]:>4}x{s[2]:>4}: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== E. Warmup-dtype interaction at 1024^3 fp32 ===")
    # Cold (fresh process is best, but here we just hit fp32 with no prior warmup)
    r = _bench(1024, 1024, 1024, torch.float32, n_iters=30, warmup=10,
               seed=7, dtype_warmup=None)
    out.append({"experiment": "E_warmup_cold", **r})
    print(f"  no_warmup_dtype: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    r = _bench(1024, 1024, 1024, torch.float32, n_iters=30, warmup=10,
               seed=7, dtype_warmup=torch.float16)
    out.append({"experiment": "E_warmup_fp16_first", **r})
    print(f"  fp16_warmup_then_fp32: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    r = _bench(1024, 1024, 1024, torch.float32, n_iters=30, warmup=10,
               seed=7, dtype_warmup=torch.bfloat16)
    out.append({"experiment": "E_warmup_bf16_first", **r})
    print(f"  bf16_warmup_then_fp32: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== F. Power-of-2 sweep fp32 (does 1024 stand out?) ===")
    for n in (256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 2048):
        r = _bench(n, n, n, torch.float32, n_iters=30, warmup=10, seed=7)
        out.append({"experiment": "F_p2_sweep_fp32", **r})
        print(f"  {n:>4}^3 fp32: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    print("\n=== G. Same sweep fp16 ===")
    for n in (256, 384, 512, 640, 768, 896, 1024, 1152, 1280, 1408, 1536, 2048):
        r = _bench(n, n, n, torch.float16, n_iters=30, warmup=10, seed=7)
        out.append({"experiment": "G_p2_sweep_fp16", **r})
        print(f"  {n:>4}^3 fp16: median={r['median_ms']:.3f} ms ({r['gflops']:.0f} GFLOPs)")

    out_path = "/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/raw/repro-results.json"
    payload = {
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "PYTORCH_MPS_PREFER_METAL": PREFER_METAL,
        "results": out,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
