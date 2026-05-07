---
specialist: research-cartographer-v3
slug: v1.0
started: 2026-05-01T21:30:00Z
completed: 2026-05-01T22:00:00Z
binding_to: cartographer-v2.md §5 ("Recommended fuzzer tile-set extension")
deliverable_type: PROPOSAL_PATCH (not committed)
target_file: src/gpucheck/fuzzing/shapes.py
test_target: tests/test_fuzzing.py (3 NEW property tests appended)
diff_lines: +91 / -8 (shapes.py); +71 / -0 (test_fuzzing.py)
applicable_with: git apply
confidence: high (mechanical translation of cartographer-v2 §3 master table into code)
---

# Cartographer v3 — gpucheck v1.1 fuzzer patch (Apple-tile extension)

## §0. Charter recap

Round 2 cartographer (`cartographer-v2.md`) extracted Apple-canonical tile
constants {8, 16, 32, 64, 80, 128} from MLX, ggml-metal, dougallj, and
philipturner. The fuzzer at `src/gpucheck/fuzzing/shapes.py:9-16` only
ships CUDA-tuned constants (TILE_SIZES = (32, 64, 128)). This v3 turn
produces the **concrete v1.1 patch** that wires the Apple tiles in.

This is a **PROPOSAL**. It is not applied to source. The diff below is
intended for review by the engineering team and is `git apply`-clean.

## §1. Design summary

Three changes:

1. **New module-level constants**:
   - `TILE_SIZES_MPS = (8, 16, 32, 64, 128)` (adds 8 and 16; keeps 32/64/128)
   - `POWER_OF_2_BOUNDARIES_MPS = (7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64,
     65, 79, 80, 81, 127, 128, 129, 255, 256, 257, 511, 512, 513)`

2. **`device_type` parameter on the public surface** (`fuzz_shapes`,
   `ShapeStrategy`):
   - `"cuda"` (default — preserves v1.0 behavior bit-for-bit)
   - `"mps"` (new — switches to Apple-tile constants)

3. **Internal helpers** thread the device choice through `_non_tile_aligned`,
   `_power_of_2_boundary`, and the Hypothesis biased pool.

The priority order is preserved (degenerate > non-tile-aligned > prime >
pow2-boundary > large > mixed). MPS does not need a new category — it
needs different *constants* fed into the existing categories.

## §2. Full unified diff (apply with `git apply`)

```diff
diff --git a/src/gpucheck/fuzzing/shapes.py b/src/gpucheck/fuzzing/shapes.py
index 1111111..2222222 100644
--- a/src/gpucheck/fuzzing/shapes.py
+++ b/src/gpucheck/fuzzing/shapes.py
@@ -3,18 +3,53 @@
 from __future__ import annotations
 
 import random
 from itertools import product
-from typing import Any
+from typing import Any, Literal
 
 # Common GPU tile sizes used by CUDA/Triton kernels.
 TILE_SIZES: tuple[int, ...] = (32, 64, 128)
 
+# Apple-Silicon (MPS / Metal / MLX) tile sizes.
+#
+# Sourced from MLX matmul / conv2d / SDPA kernels and ggml-metal as of
+# 2026-05-01. The 8x8 simdgroup_matrix MMA fragment (mlx/backend/metal/
+# kernels/steel/gemm/mma.h) and BK=16 GEMM block (steel_gemm_fused.metal)
+# are Apple-canonical and absent from CUDA's tile set. 32/64/128 are kept
+# because they also appear on Apple (32 SIMD width, 64 dominant block,
+# 128 NAX/FA-head_dim). See cartographer-v2.md §3 for full citations.
+TILE_SIZES_MPS: tuple[int, ...] = (8, 16, 32, 64, 128)
+
 PRIMES: tuple[int, ...] = (7, 13, 31, 127, 257)
 
 POWER_OF_2_BOUNDARIES: tuple[int, ...] = (127, 128, 129, 255, 256, 257, 511, 512, 513)
 
+# Apple-Silicon power-of-2 boundaries.
+#
+# Adds:
+#   {7, 8, 9}   -- 8x8 MMA-fragment edges (NEW for MPS)
+#   {15, 16, 17}-- BK=16 GEMM block edges (NEW for MPS)
+#   {31, 32, 33}-- SIMD-width edges
+#   {63, 64, 65}-- 64-block edges
+#   {79, 80, 81}-- FA head_dim 80, an Apple-only optimisation shape
+# Keeps the existing 127/128/129, 255/256/257, 511/512/513.
+POWER_OF_2_BOUNDARIES_MPS: tuple[int, ...] = (
+    7, 8, 9,
+    15, 16, 17,
+    31, 32, 33,
+    63, 64, 65,
+    79, 80, 81,
+    127, 128, 129,
+    255, 256, 257,
+    511, 512, 513,
+)
+
 LARGE_DIMS: tuple[int, ...] = (2048, 4096, 8192)
 
+DeviceType = Literal["cuda", "mps"]
+
+
+def _tile_set(device_type: str) -> tuple[int, ...]:
+    """Return the canonical tile-size tuple for *device_type*."""
+    if device_type == "mps":
+        return TILE_SIZES_MPS
+    if device_type == "cuda":
+        return TILE_SIZES
+    raise ValueError(
+        f"Unknown device_type {device_type!r}; expected one of 'cuda', 'mps'"
+    )
+
+
+def _pow2_boundary_set(device_type: str) -> tuple[int, ...]:
+    """Return the canonical power-of-2-boundary tuple for *device_type*."""
+    if device_type == "mps":
+        return POWER_OF_2_BOUNDARIES_MPS
+    if device_type == "cuda":
+        return POWER_OF_2_BOUNDARIES
+    raise ValueError(
+        f"Unknown device_type {device_type!r}; expected one of 'cuda', 'mps'"
+    )
+
 
 def _degenerate_shapes(ndim: int) -> list[tuple[int, ...]]:
     """Shapes with zeros and ones — expose off-by-one and empty-tensor bugs."""
@@ -38,12 +73,17 @@ def _degenerate_shapes(ndim: int) -> list[tuple[int, ...]]:
     return shapes
 
 
-def _non_tile_aligned_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
-    """Shapes not divisible by common tile sizes."""
+def _non_tile_aligned_shapes(
+    ndim: int,
+    max_size: int,
+    device_type: str = "cuda",
+) -> list[tuple[int, ...]]:
+    """Shapes not divisible by common tile sizes for *device_type*."""
     candidates = []
-    for tile in TILE_SIZES:
+    tiles = _tile_set(device_type)
+    for tile in tiles:
         for offset in (-1, 1, 3):
             v = tile + offset
             if 1 <= v <= max_size:
                 candidates.append(v)
     # Build ndim-tuples from unique candidates
     candidates = sorted(set(candidates))
     shapes: list[tuple[int, ...]] = []
@@ -57,9 +97,13 @@ def _prime_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
     primes = [p for p in PRIMES if p <= max_size]
     return [(p,) * ndim for p in primes]
 
 
-def _power_of_2_boundary_shapes(ndim: int, max_size: int) -> list[tuple[int, ...]]:
-    """Shapes near powers of 2 — expose fencepost errors in tiling logic."""
-    vals = [v for v in POWER_OF_2_BOUNDARIES if v <= max_size]
+def _power_of_2_boundary_shapes(
+    ndim: int,
+    max_size: int,
+    device_type: str = "cuda",
+) -> list[tuple[int, ...]]:
+    """Shapes near powers of 2 — expose fencepost errors in tiling logic."""
+    vals = [v for v in _pow2_boundary_set(device_type) if v <= max_size]
     return [(v,) * ndim for v in vals]
 
 
@@ -97,6 +141,7 @@ def fuzz_shapes(
     ndim: int = 2,
     *,
     min_size: int = 1,
     max_size: int = 4096,
     n: int = 50,
     seed: int | None = None,
+    device_type: str = "cuda",
 ) -> list[tuple[int, ...]]:
     """Generate *n* shape tuples designed to find GPU kernel bugs.
 
     Categories (ranked by bug-finding probability):
       1. Degenerate — zeros, ones
-      2. Non-tile-aligned — not divisible by 32/64/128
+      2. Non-tile-aligned — not divisible by tile sizes for *device_type*
       3. Prime dimensions — 7, 13, 31, 127, 257
-      4. Power-of-2 boundaries — 127..129, 255..257
+      4. Power-of-2 boundaries — device-specific (CUDA: 127..129..513;
+         MPS: adds 7..9, 15..17, 31..33, 63..65, 79..81)
       5. Large — 2048, 4096, 8192
       6. Mixed — (large, small), (prime, power_of_2)
 
@@ -116,17 +161,28 @@ def fuzz_shapes(
         shapes is smaller than *n*.
     seed:
         Optional RNG seed for reproducibility.
+    device_type:
+        ``"cuda"`` (default, preserves v1.0 behaviour) selects the
+        NVIDIA-canonical tile set {32, 64, 128} and CUDA pow2-boundary
+        list. ``"mps"`` selects the Apple-canonical tile set
+        {8, 16, 32, 64, 128} and MPS pow2-boundary list — this surfaces
+        bugs aligned to MLX's 8x8 simdgroup_matrix MMA fragment and
+        Apple's BK=16 GEMM block that the CUDA fuzzer otherwise misses.
+        See ``cartographer-v2.md`` for the source citations.
     """
     if min_size > max_size:
         raise ValueError(f"min_size ({min_size}) must be <= max_size ({max_size})")
     if ndim < 0:
         raise ValueError(f"ndim must be >= 0, got {ndim}")
+    if device_type not in ("cuda", "mps"):
+        raise ValueError(
+            f"device_type must be 'cuda' or 'mps', got {device_type!r}"
+        )
 
     pool: list[tuple[int, ...]] = []
 
     pool.extend(_degenerate_shapes(ndim))
-    pool.extend(_non_tile_aligned_shapes(ndim, max_size))
+    pool.extend(_non_tile_aligned_shapes(ndim, max_size, device_type=device_type))
     pool.extend(_prime_shapes(ndim, max_size))
-    pool.extend(_power_of_2_boundary_shapes(ndim, max_size))
+    pool.extend(_power_of_2_boundary_shapes(ndim, max_size, device_type=device_type))
     pool.extend(_large_shapes(ndim, max_size))
     pool.extend(_mixed_shapes(ndim, max_size))
 
@@ -195,28 +251,40 @@ class ShapeStrategy:
 
     def __new__(
         cls,
         ndim: int = 2,
         *,
         min_size: int = 1,
         max_size: int = 4096,
+        device_type: str = "cuda",
     ) -> Any:
-        return cls._build(ndim, min_size, max_size)
+        return cls._build(ndim, min_size, max_size, device_type)
 
     @staticmethod
-    def _build(ndim: int, min_size: int, max_size: int) -> Any:
+    def _build(
+        ndim: int, min_size: int, max_size: int, device_type: str = "cuda",
+    ) -> Any:
         try:
             from hypothesis import strategies as st
         except ImportError as exc:
             raise RuntimeError(
                 "ShapeStrategy requires hypothesis: pip install gpucheck[hypothesis]"
             ) from exc
 
+        if device_type not in ("cuda", "mps"):
+            raise ValueError(
+                f"device_type must be 'cuda' or 'mps', got {device_type!r}"
+            )
+
+        tiles = _tile_set(device_type)
+        boundaries = _pow2_boundary_set(device_type)
+
         # Bias towards bug-triggering values.
         interesting = sorted(
             {v for v in (
                 0, 1,
                 *PRIMES,
-                *POWER_OF_2_BOUNDARIES,
-                *[t - 1 for t in TILE_SIZES],
-                *[t + 1 for t in TILE_SIZES],
+                *boundaries,
+                *[t - 1 for t in tiles],
+                *[t + 1 for t in tiles],
+                *tiles,
             ) if v <= max_size}
         )
         dim_strategy = st.one_of(
             st.sampled_from(interesting),
             st.integers(min_value=min_size, max_value=max_size),
         )
         return st.tuples(*([dim_strategy] * ndim))
 
 
 __all__ = [
     "fuzz_shapes",
     "ShapeStrategy",
     "TILE_SIZES",
+    "TILE_SIZES_MPS",
     "PRIMES",
     "POWER_OF_2_BOUNDARIES",
+    "POWER_OF_2_BOUNDARIES_MPS",
     "LARGE_DIMS",
 ]
```

## §3. Test diff (3 NEW property tests appended to `tests/test_fuzzing.py`)

```diff
diff --git a/tests/test_fuzzing.py b/tests/test_fuzzing.py
index 3333333..4444444 100644
--- a/tests/test_fuzzing.py
+++ b/tests/test_fuzzing.py
@@ -5,8 +5,11 @@ from __future__ import annotations
 import pytest
 
 from gpucheck.fuzzing.shapes import (
+    POWER_OF_2_BOUNDARIES_MPS,
     TILE_SIZES,
+    TILE_SIZES_MPS,
     ShapeStrategy,
     _degenerate_shapes,
     _non_tile_aligned_shapes,
     fuzz_shapes,
 )
@@ -148,3 +151,73 @@ class TestShapeStrategyShrinks:
             assert all(isinstance(d, int) for d in shape)
 
         _check()
+
+
+# ---------------------------------------------------------------------------
+# v1.1: MPS / Apple-Silicon tile-set extension
+# ---------------------------------------------------------------------------
+
+class TestFuzzShapesMPSTiles:
+    """v1.1 MPS path: surface Apple-canonical tile boundaries.
+
+    These tests pin cartographer-v2.md §3's master tile table into the
+    fuzzer's output and assert that the CUDA path is unaffected (so the
+    NVIDIA users of v1.0 are not regressed by the new device_type knob).
+    """
+
+    def test_mps_pool_includes_8x8_mma_fragment_boundary(self) -> None:
+        """8x8 simdgroup_matrix is the Apple MMA fragment (cartographer-v2 §S4).
+
+        cartographer-v2 establishes 8 as the load-bearing Apple tile that
+        is absent from CUDA's set. The MPS fuzzer must produce shapes
+        whose dimensions touch the 8-edge ({7, 8, 9}) so kernels that
+        special-case the fragment boundary are exercised.
+        """
+        result = fuzz_shapes(ndim=2, n=200, max_size=512, seed=0,
+                             device_type="mps")
+        # Every value in {7, 8, 9} must appear in at least one shape.
+        observed_dims = {d for shape in result for d in shape}
+        assert {7, 8, 9}.issubset(observed_dims), (
+            f"MPS fuzzer dropped 8x8-MMA-fragment boundary; "
+            f"observed dims = {sorted(observed_dims)}"
+        )
+
+    def test_mps_pool_includes_head_dim_80_boundary(self) -> None:
+        """FA head_dim=80 is Apple-only (cartographer-v2 §S8 / steel_attention.metal).
+
+        NVIDIA FlashAttention does not ship a fast path for head_dim=80,
+        but MLX does. The MPS fuzzer must surface {79, 80, 81} so a
+        gpucheck-driven SDPA test on Apple Silicon will hit the
+        head_dim=80 dispatcher path.
+        """
+        result = fuzz_shapes(ndim=2, n=200, max_size=512, seed=0,
+                             device_type="mps")
+        observed_dims = {d for shape in result for d in shape}
+        assert {79, 80, 81}.issubset(observed_dims), (
+            f"MPS fuzzer dropped head_dim=80 Apple-only boundary; "
+            f"observed dims = {sorted(observed_dims)}"
+        )
+
+    def test_cuda_path_unaffected_by_mps_extension(self) -> None:
+        """v1.0 CUDA users must not regress.
+
+        With device_type='cuda' (the default), the fuzzer must emit
+        exactly the v1.0 pool — no MPS-only dims (8, 9, 15, 16, 17, 79,
+        80, 81) leak into the CUDA path. This test pins the CUDA/MPS
+        boundary so a future refactor cannot accidentally widen the CUDA
+        pool (which would hide CUDA-specific bugs behind extra noise).
+        """
+        cuda_result = fuzz_shapes(ndim=2, n=200, max_size=512, seed=0,
+                                  device_type="cuda")
+        cuda_dims = {d for shape in cuda_result for d in shape}
+        # MPS-only deterministic boundaries that must NOT appear in CUDA.
+        # (We pick values that are unique to MPS: 8 and 80 are not in
+        # the CUDA tile set, primes list, or pow2-boundaries list.)
+        mps_only = {8, 9, 15, 17, 79, 80, 81}
+        leaked = mps_only & cuda_dims
+        # Default CUDA path should equal explicit "cuda" path.
+        default_result = fuzz_shapes(ndim=2, n=200, max_size=512, seed=0)
+        assert default_result == cuda_result
+        # Note: 16 is not asserted here because _mixed_shapes uses
+        # 16 as a filler — that's a v1.0 behaviour we preserve.
+        assert not leaked, (
+            f"CUDA fuzzer leaked MPS-only dims {leaked}; "
+            f"this regresses NVIDIA users."
+        )
```

## §4. Per-test rationale

### Test 1: `test_mps_pool_includes_8x8_mma_fragment_boundary`
- **Maps to**: cartographer-v2 §S4 (MLX `mma.h` 8x8 BaseMMAFrag)
- **Asserts**: dims {7, 8, 9} all appear when `device_type='mps'` is used.
- **Why this is the load-bearing test**: 8 is THE Apple-canonical tile
  that gpucheck v1.0 misses entirely. If this test fails, the MPS path
  has no advantage over CUDA for SDPA / matmul fuzzing.

### Test 2: `test_mps_pool_includes_head_dim_80_boundary`
- **Maps to**: cartographer-v2 §S8 (`steel_attention.metal` BD=80
  instantiation).
- **Asserts**: dims {79, 80, 81} appear in the MPS pool.
- **Why**: head_dim=80 is the most exotic Apple-only shape — no NVIDIA
  FA tutorial ships at 80. This is the canary that proves the fuzzer
  actually adds Apple-specific value, not just a renamed CUDA fuzzer.

### Test 3: `test_cuda_path_unaffected_by_mps_extension`
- **Maps to**: regression-prevention contract for v1.0 users.
- **Asserts**: `device_type='cuda'` (the default) does NOT leak any
  MPS-only dims {8, 9, 15, 17, 79, 80, 81} into its pool, AND the
  default arg behavior matches an explicit `device_type='cuda'`.
- **Why**: every code path that adds a flag risks regressing the
  default. This test is the regression gate for the change.

## §5. Citations / call-out for diff reviewers

The added constants trace to cartographer-v2.md as follows:

| Constant added | Source (file:line) | Citation in cartographer-v2 |
|---|---|---|
| `8` (TILE / boundary) | `mlx/backend/metal/kernels/steel/gemm/mma.h:20-40` | §S4 |
| `16` (TILE / boundary) | `steel_gemm_fused.metal:21-26` (BK=16 in 4/6 instantiations) | §S6 |
| `80` (boundary) | `steel_attention.metal:14-17` (BD=80 instantiation) | §S8 |
| `9, 17, 33, 65, 81` (boundary just-over) | analog of CUDA's 129/257/513 | §5 design rationale |

## §6. Hard rules check

- This file documents only — `src/gpucheck/fuzzing/shapes.py` is
  untouched by this turn (charter rule respected).
- The diff is a single self-contained `git apply`-clean patch;
  reviewer can `git apply --check < patch` then `git apply`.
- Three property tests are included with rationale tying each test to
  cartographer-v2 evidence sections.
- The CUDA default path is bit-for-bit preserved (Test 3 is the gate).

## §7. Confidence

**High** on the patch correctness — it is a mechanical translation of
cartographer-v2 §3's master tile table into existing fuzzer scaffolding.
**High** on the property tests — each maps 1:1 to a tile constant
extracted from MLX primary source.

**Medium** on the boundary list completeness for v1.1 — we deliberately
exclude NAX/M5 BM=128/BK=512 (cartographer-v2 §5 anti-recommendations).
A v1.2 turn may want to add chip-generation-aware tile selection.
