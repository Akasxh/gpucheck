# tracer-cpp-cache-source summary

## Cache-key fields (LinearAlgebra.mm:1024 + OperationUtils.mm:277-303)

`"mm_out_mps_impl"` + per-input `{dtype, logical sizes, conj-flag}`.

No strides, no contiguity, no device id, no transpose (transpose only included for bmm; alpha/beta only for addmm/baddbmm).

**Our case: cache key is literally `mm_out_mps_impl:f32[1024,1024]:f32[1024,1024]`.**

## Kernel-pick verdict

**PyTorch wrapper does NOT pick the kernel.** It only chooses MPSGraph-path vs naive-Metal-shader via `use_metal_mm` (`LinearAlgebra.mm:1015`), which fires only for dims > 2¹⁵.

For 1024×1024 fp32 the wrapper unconditionally calls `[graph matrixMultiplicationWithPrimaryTensor:secondaryTensor:]` (`LinearAlgebra.mm:614`) and **Apple's closed-source MPSGraph framework** picks the kernel.

Whatever kernel MPSGraph compiles at first call is **baked into the cached `MPSCachedGraph*`** and reused verbatim forever after.

## Does `torch.mps.empty_cache()` invalidate?

**NO.**
- Chain: `torch/mps/__init__.py:86-90` → `torch/csrc/mps/Module.cpp:113-117` → `aten/src/ATen/mps/MPSAllocator.mm:551-553` (`release_cached_buffers()`).
- Only memory pools. **The `MPSGraphCache` class (`OperationUtils.h:338-421`) has NO `clear`/`erase`/`evict`/`invalidate`/`reset`/`drop` method.**
- No Python binding for graph-cache eviction exists.

## Debug knobs

Existing MPS env vars:
- `PYTORCH_DEBUG_MPS_ALLOCATOR` (`MPSAllocator.mm:18`)
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (`MPSAllocator.mm:21`)
- `PYTORCH_MPS_LOW_WATERMARK_RATIO` (`MPSAllocator.mm:26`)
- `PYTORCH_MPS_FAST_MATH` (`OperationUtils.mm:453`)

**NO `PYTORCH_MPS_DISABLE_GRAPH_CACHE` or equivalent exists.**

## v2.10 → v2.11 delta

`mm_out_mps_impl` body byte-identical. `MPSGraphCache` mechanism unchanged. No commits in the path touched cache key/lookup.

**Behavior diff (if any) must come from MPSGraph itself or macOS shader cache** — neither readable from PyTorch source.

## What we can tell the maintainer

The bug surface is entirely inside Apple's closed-source MPSGraph. PyTorch's user-visible workaround surface is empty. To give users a workaround, PyTorch needs to:

1. Add a `torch.mps.invalidate_graph_cache()` Python API that walks `MPSGraphCache` and `erase`s entries (would require adding `erase`/`clear` to the class first).
2. OR add `PYTORCH_MPS_DISABLE_GRAPH_CACHE=1` env var that bypasses cache lookup (would re-pay compile cost every call but might pick a different kernel).
3. OR escalate to Apple to fix MPSGraph's kernel-selection heuristic for the (1024, 1024, fp32) cache key.

For users TODAY, the only workaround is "touch a different shape" — which is what the issue should document.
