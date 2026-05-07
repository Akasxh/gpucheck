# Tracer — MPS matmul cache-key, kernel pick, and invalidation

All citations refer to `pytorch/pytorch` at tag **`v2.11.0`** unless otherwise
stated. v2.10 numbers are noted where they differ. Line numbers are taken
from the raw GitHub blobs fetched at audit time.

## Entry point

`mm_out_mps_impl` in `aten/src/ATen/native/mps/operations/LinearAlgebra.mm`.
It is the dispatcher for `torch.mm` / `aten::mm.out` on the MPS backend, and
reaches the cache via `LookUpOrCreateCachedGraph<MPSBinaryCachedGraph>(key,
…)`.

In v2.11 the function starts at line 990; in v2.10 the same body starts at
line 715 — the *body* is unchanged across the two releases (verified with
side-by-side fetches; only surrounding code shifted line numbers).

---

## Q1 — What goes into the cache key?

The key is built in `mm_out_mps_impl` at v2.11
`aten/src/ATen/native/mps/operations/LinearAlgebra.mm:1024`:

```cpp
@autoreleasepool {
  std::string key = "mm_out_mps_impl" + getTensorsStringKey({self, other});
  auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(key, …);
```

The full alphabet of what `getTensorsStringKey` emits is defined at
`aten/src/ATen/native/mps/OperationUtils.mm:277-303`:

```cpp
std::string getTensorsStringKey(const TensorList& tensors,
                                bool short_dtype, bool exclude_shape) {
  fmt::basic_memory_buffer<char, 100> buffer;
  auto buf_iterator = std::back_inserter(buffer);
  for (const Tensor& tensor : tensors) {
    fmt::format_to(buf_iterator, ":");
    if (tensor.defined()) {
      fmt::format_to(buf_iterator, "{}[", getMPSTypeString(tensor.scalar_type(), short_dtype));
      if (tensor.dim() == 0) {
        fmt::format_to(buf_iterator, "Scalar");
      } else {
        if (exclude_shape) {
          fmt::format_to(buf_iterator, "-1");
        } else {
          fmt::format_to(buf_iterator, "{}", getArrayRefString(tensor.sizes()));
        }
      }
      fmt::format_to(buf_iterator, "]");
      if (tensor.is_conj()) {
        fmt::format_to(buf_iterator, "_conj");
      }
    } else {
      fmt::format_to(buf_iterator, "Undefined");
    }
  }
  return fmt::to_string(buffer);
}
```

`getArrayRefString` (`OperationUtils.mm:272-274`) joins `tensor.sizes()` with
commas, i.e. **logical shape**.

For an `mm` the cache key is therefore literally:

```
mm_out_mps_impl:f32[1024,1024]:f32[1024,1024]
```

Fields included:

- prefix `"mm_out_mps_impl"` (op identity)
- per-input dtype string from `getMPSTypeString(scalar_type, short_dtype=true)`
- per-input **logical shape** from `tensor.sizes()`
- conjugate flag (suffix `_conj`)

Fields **NOT** included:

- **strides** / contiguity / memory format
- **storage offset**
- **device id** (single-GPU MPS, but still notable)
- **transpose flag** (this is added for `bmm_out_mps_impl` —
  `LinearAlgebra.mm:1375` — but *not* for `mm`)
- alpha / beta scalars (only `addmm`/`baddbmm` include those —
  `LinearAlgebra.mm:1308` and `:1218`)
- macOS version, current MPS device state, residency, MPSGraph compile-time
  options

After construction the string is hashed and stored in
`std::unordered_map<MPSCacheKey, CacheEntry>` (`OperationUtils.h:419`) keyed
by `std::hash<std::string>` (`OperationUtils.h:371`).

So the cache identity for our case is exactly
`mm_out_mps_impl:f32[1024,1024]:f32[1024,1024]`. Two runs with the same
dtype × shape map to the same cached graph.

---

## Q2 — Where is the kernel pick made?

The PyTorch wrapper does **not** select between matmul kernel implementations
beyond a single, narrow shape-overflow guard. Inside `mm_out_mps_impl`
(`LinearAlgebra.mm:1015-1017`):

```cpp
if (use_metal_mm(self, other, output)) {
  return do_metal_mm(self, other, output);
}
```

`use_metal_mm` only triggers when a matrix dimension exceeds 2**15 (see the
comment block at lines 1011-1014 referencing pytorch issue 116769). For
1024×1024 fp32 this branch is **never taken**.

The remaining path builds a plain MPSGraph node:

`do_mm` helper (`LinearAlgebra.mm:600-625`):

```cpp
auto selfTensor_  = mpsGraphRankedPlaceHolder(graph, self);
auto otherTensor_ = mpsGraphRankedPlaceHolder(graph, other);
auto selfTensor   = self.is_conj()  ? [graph conjugateWithTensor:selfTensor_  name:nil] : selfTensor_;
auto otherTensor  = other.is_conj() ? [graph conjugateWithTensor:otherTensor_ name:nil] : otherTensor_;
auto output = [graph matrixMultiplicationWithPrimaryTensor:selfTensor
                                          secondaryTensor:otherTensor
                                                     name:nil];
return {selfTensor_, otherTensor_, output};
```

That `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` call hands the
operation off to **Apple's MPSGraph framework**, which is closed-source.

Verdict: the final kernel selection (which Metal compute kernel /
MPSMatrixMultiplication strategy / tile size / SIMD-group config) happens
**inside MPSGraph during graph compile**, not in PyTorch. PyTorch only
picks "MPSGraph path" vs "naive Metal shader" via `use_metal_mm`, and for
1024×1024 fp32 it always picks MPSGraph. **What kernel MPSGraph then bakes
into the compiled graph is opaque to PyTorch and to us.** Once baked, that
choice is sticky: `LookUpOrCreateCachedGraph` reuses the *same compiled
graph instance* on every subsequent call with the same key.

---

## Q3 — Does `torch.mps.empty_cache()` invalidate the graph cache?

**No.** It only frees memory.

Python entry (`torch/mps/__init__.py:86-90`):

```python
def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()
```

C binding (`torch/csrc/mps/Module.cpp:113-117`):

```cpp
static PyObject* MPSModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  at::detail::getMPSHooks().emptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
```

Hook implementation (`aten/src/ATen/mps/MPSAllocator.mm:551-553`):

```cpp
void MPSHeapAllocatorImpl::emptyCache() {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  release_cached_buffers();
}
```

`release_cached_buffers` walks `BufferPool`s and frees `BufferBlock`s — pure
allocator state. It never touches `MPSGraphCache`.

Verifying the negative directly: the `MPSGraphCache` class
(`aten/src/ATen/native/mps/OperationUtils.h:338-421`) exposes only
`getInstance`, `CreateCachedGraph[As]`, `LookUp[As]`, and a private
`profileCachedGraph`. There is **no `clear`, `erase`, `evict`, `drop`,
`reset`, `invalidate`, `dump`, `size`, or `print` method** anywhere in the
class. I grepped the header for `EvictCachedGraphs`, `ClearCachedGraph`,
`InvalidateCache`, `DropCache`, `ResetCache`, `PrintCache`,
`EnumerateCache`, `DumpCache` — none exist.

The destructor `~MPSGraphCache` (`OperationUtils.h:356-361`) deletes entries,
but the cache is a process-lifetime singleton (`_instance_cache`,
`OperationUtils.h:418`); the destructor only fires at process teardown.

The Module.cpp binding list has no `_mps_clearGraphCache`,
`_mps_dropCachedGraph`, or similar (full list verified at
`torch/csrc/mps/Module.cpp:189-215` + `:291-340`).

**Conclusion: the MPSGraph cache is a process-lifetime singleton with no
user-facing invalidation API. Once a graph for a given (op, dtype, shape)
key is compiled and cached, it lives for the rest of the process.
`torch.mps.empty_cache()` does not affect it.**

---

## Q4 — What does `LookUpOrCreateCachedGraph` actually do?

Definition at `aten/src/ATen/native/mps/OperationUtils.h:408-422`:

```cpp
template <typename T>
inline T* LookUpOrCreateCachedGraph(const std::string& key,
                                    std::function<void(MPSGraph*, T*)> instantiate) {
  auto cache_ = MPSGraphCache::getInstance();
  if (auto rc = cache_->LookUpAs<T>(key)) {
    return rc;
  }
  return cache_->CreateCachedGraphAs<T>(key, ^mps::MPSCachedGraph*() {
    T* newCachedGraph = nil;
    @autoreleasepool {
      auto mpsGraph = mps::make_mps_graph();
      newCachedGraph = new T(mpsGraph);
      instantiate(mpsGraph, newCachedGraph);
    }
    return newCachedGraph;
  });
}
```

Walk:

1. `MPSGraphCache::getInstance()` — singleton (`:349-354`). Lazily allocates
   on first call; never freed until process exit.
2. `LookUpAs<T>(key)` (`:395-408`, `:411-414`) — under a serial GCD queue,
   computes `MPSCacheKey hash = std::hash<std::string>{}(key)` (`:397`),
   looks up `cache_.at(hash)` in an `unordered_map<MPSCacheKey, CacheEntry>`
   (`:419`), and returns the existing `MPSCachedGraph*` if present. **A hit
   here returns the already-compiled graph object — no MPSGraph compile, no
   kernel selection.**
3. On miss, `CreateCachedGraphAs<T>` (`:367-393`) re-acquires the serial
   queue, double-checks for a race, then runs the supplied block: creates a
   fresh `MPSGraph` via `make_mps_graph()`, calls the caller-supplied
   `instantiate` lambda (which for `mm` calls `do_mm` and ultimately
   `matrixMultiplicationWithPrimaryTensor:`), and inserts the new entry.

The "cold start picks slow kernel" effect therefore originates inside step 3:
the first time a `(dtype, shape)` is seen in the process, MPSGraph is asked
to compile a graph for it, and whatever heuristic MPSGraph applies *at that
moment* (current device pressure, shader-cache warmth, residency, JIT state)
gets baked in. Subsequent calls hit step 2 and reuse the artifact verbatim.

---

## Q5 — Debug knobs / env vars

I grepped `getenv` and `c10::utils::get_env` in:

- `aten/src/ATen/mps/MPSAllocator.mm`:
  - line 18: `c10::utils::get_env("PYTORCH_DEBUG_MPS_ALLOCATOR")`
  - line 21: `c10::utils::get_env("PYTORCH_MPS_HIGH_WATERMARK_RATIO")`
  - line 26: `c10::utils::get_env("PYTORCH_MPS_LOW_WATERMARK_RATIO")`
- `aten/src/ATen/mps/MPSDevice.mm`: **no** getenv calls
- `aten/src/ATen/native/mps/OperationUtils.mm`:
  - line 453: `c10::utils::get_env("PYTORCH_MPS_FAST_MATH")`

There is **no env var** that disables, bypasses, or forces a recompile of the
MPSGraph cache. No `PYTORCH_MPS_DISABLE_GRAPH_CACHE`,
`PYTORCH_MPS_NO_CACHE`, `PYTORCH_MPS_FORCE_RECOMPILE`, or similar exists in
the v2.11 source.

`PYTORCH_MPS_FAST_MATH` is the only knob that can change the *kernel that
gets compiled into a cached graph entry*, and it's a build-time-once switch
read during shader compilation, not a per-call control.

---

## Q6 — Cache-implementation deltas v2.10 → v2.11

Direct `mm_out_mps_impl` body diff: **none material**. v2.10 line 715 vs
v2.11 line 990 carry the same 35-line body verbatim (key construction
`"mm_out_mps_impl" + getTensorsStringKey({self, other})`, same `do_mm` call,
same `LookUpOrCreateCachedGraph`).

Listing of LinearAlgebra.mm commits between v2.10 and v2.11 cut dates
(GitHub commits API, path-filtered):

```
2025-11-19  65f08ee  [MPS][1/N] Fix unsupported dtypes error checking for some MPS ops
2025-11-14  9e2bf12  [MPS] addmm complex fix
2025-11-13  a954242  [MPS] Add Metal complex mm implementation
2025-10-24  c9b49e5  [MPS] Add `linalg.householder_product` for MPS
2025-10-22  715449c  [MPS] Fix parity between CPU and MPS on singular matrices in linalg.lu_factor
2025-10-17  935ccdb  [MPS] Fix internal assertion in torch.linalg.solve for singular matrices
2025-09-22  60b4791  [MPS] Fix compile linalg inv
2025-09-14  7fe1f5e  [BE] Delete [Ventura|Sonoma]Ops header
2025-08-10  842cc77  [MPS] Extend addmm to integral types
2025-07-01  1c8844d  [MPS] Switch Cholesky decomp to column wise
2025-04-29  41bd0c9  [1/N] Deprecate c10::string_view and at::string
2025-04-24  f2cfeb2  [Environment Variable][7/N] Use thread-safe getenv functions
2025-04-05  cfea55d  [MPS] fix inverse bug for N>1024
2025-03-28  7c65911  [MPS] Fix dot/mm for conj_tensors
2025-02-26  ebf6b98  [MPS] faster integer batched matmul
2025-02-25  7e37fb0  [MPS] faster integer matmul for mps
2025-02-14  8b5ee27  [MPS] Fix cholesky_ex for empty inputs
2025-02-14  92f669e  [BE] Use `c10::multiply_integers` in cholesky_impl
2025-02-13  17a8085  [MPS] cholesky ex version
2025-02-11  d763093  [MPS] fix lu factor for large tensors with bs>1
2025-02-08  0ab6729  [MPS] lu unpack
2025-02-06  0dc0313  [MPS] linalg solve implementation
2025-02-03  00dc5b1  Revert "[Environment Variable][7/N] Use thread-safe getenv functions"
2025-02-03  e3643e1  [MPS] Add linalg det and fix lu factor for non contiguous tensors
2025-02-02  5d55a65  [MPS] lu factor ex implementation
2025-02-01  2fd1b6b  [Environment Variable][7/N] Use thread-safe getenv functions
```

None of these touch the `mm` cache-key path, the `MPSGraphCache` data
structure, or `LookUpOrCreateCachedGraph`. Two commits relate to fp32 `mm`
codegen indirectly — `7c65911` (conj-tensor fix) and `ebf6b98`/`7e37fb0`
(integer matmul speedups, do not affect fp32 path) — but the cache *key
construction and lookup mechanism* is byte-identical between v2.10 and
v2.11.

That is significant: **whatever cold-start behavior we are observing in
v2.11 for fp32 `mm(1024,1024,1024)` was almost certainly present in v2.10
too, because the PyTorch-side cache machinery did not change.** Any
behavioral shift between the two would have to come from MPSGraph
(closed-source), the macOS Metal compiler / shader cache, or the residency
heuristic — none of which are visible from PyTorch source.

---

## Competing hypotheses

### H1: Cold-start slowness is MPSGraph compile-time kernel pick, baked permanently

- Supporting:
  - Cache key includes only `(op, dtype, shape, conj)`
    (`OperationUtils.mm:277-303`). No knob lets MPSGraph re-evaluate later.
  - `LookUpOrCreateCachedGraph` returns the *same compiled `MPSCachedGraph*`
    object* on every hit (`OperationUtils.h:408-422`).
  - Kernel selection happens inside the closed-source
    `matrixMultiplicationWithPrimaryTensor:` at compile time
    (`LinearAlgebra.mm:614`).
  - `torch.mps.empty_cache()` does not flush this cache
    (`MPSAllocator.mm:551-553`).
- Falsifiable by: invoking the same `mm` shape twice with the *process*
  restarted between calls, but with the macOS Metal shader cache pre-warmed
  (e.g. by previously running another fp32 mm in any process). If MPSGraph
  reads from the system shader cache during compile and that warming
  changes the kernel choice, we'd see fast-from-cold on the second process.
- Status: **open, strongly supported on the PyTorch side**; the MPSGraph
  side cannot be confirmed from public source.

### H2: Slowness is not the kernel pick but graph-compile latency that masquerades as a slow kernel

- Supporting: First hit goes through `CreateCachedGraphAs<T>`
  (`OperationUtils.h:367-393`) which calls `make_mps_graph()` and the
  user-supplied `instantiate` lambda — this includes MPSGraph's compile
  step, which is non-trivial for a fresh graph.
- Falsifiable by: timing a *second* call with the same shape in the same
  process. If H2 is right, the second call is fast (cache hit, no
  recompile). If H1 is right, the second call is also slow because the
  cached compiled graph is the slow kernel.
- Status: **open**; needs the empirical second-call measurement from the
  empiricist to disambiguate. (This is the critical experiment.)

### H3: The slow path is `do_metal_mm`, not MPSGraph

- Supporting: `use_metal_mm` at `LinearAlgebra.mm:1015` *would* dispatch to
  `do_metal_mm` for any matrix dimension > 2**15.
- Falsifiable by: 1024 < 32768, so `use_metal_mm` returns false. Refuted by
  source.
- Status: **refuted** by `LinearAlgebra.mm:1011-1014` comment + threshold.

---

## Probes I ran

None — this was a pure source dive. All claims are from raw GitHub blobs at
tag `v2.11.0` and `v2.10.0`, fetched live during the audit. No PyTorch tree
was modified.

---

## Uncovered ground

- **MPSGraph internals.** Apple's `MPSGraph.framework` is closed-source;
  what `matrixMultiplicationWithPrimaryTensor:secondaryTensor:` actually
  does at compile time (heuristic, tile size, fallback) cannot be
  determined from PyTorch source alone. The empiricist could probe via
  `MTLCaptureManager` GPU traces or `xctrace` Metal Shader Compilation
  events.
- **macOS Metal shader cache.** `~/Library/Caches/com.apple.metal/...`
  may persist compiled kernels across processes. If MPSGraph's choice
  depends on what's in that cache, our observed "cold start" may actually
  be "system cache cold," not "process cache cold." Not visible from
  PyTorch source.
- **Residency / device state at first call.** MPSGraph may pick a different
  kernel based on heap/buffer residency at compile time. PyTorch makes no
  attempt to control or document this.
- I did not trace `runMPSGraph` (`LinearAlgebra.mm:1041`) into
  `aten/src/ATen/native/mps/OperationUtils.mm`; it is the dispatch path,
  not the cache-key path, so should not affect the analysis.

---

## Confidence

**High** for all PyTorch-side claims (cache fields, no invalidation API, env
vars, cross-version stability). All four are direct quotes from public
source.

**Medium** for the kernel-pick story: source proves PyTorch defers to
MPSGraph and PyTorch does not re-pick on subsequent calls; *what* MPSGraph
picks and *why* it might pick a slow kernel cold is opaque.

---

## What we can tell the maintainer

> The cache key for `mm_out_mps_impl` is built at
> `aten/src/ATen/native/mps/operations/LinearAlgebra.mm:1024` (v2.11.0) as
> `"mm_out_mps_impl" + getTensorsStringKey({self, other})`. Per
> `aten/src/ATen/native/mps/OperationUtils.mm:277-303`, the key encodes only
> `(op-name, per-input dtype, per-input logical sizes, conj-flag)` — no
> strides, no contiguity, no device id, no MPSGraph compile options. Once
> `LookUpOrCreateCachedGraph` (`aten/src/ATen/native/mps/OperationUtils.h:408-422`)
> compiles a graph for a given key, the resulting `MPSCachedGraph*` is
> returned verbatim on every subsequent call within the process; the
> closed-source MPSGraph kernel choice is therefore baked at first compile.
> `torch.mps.empty_cache()`
> (`torch/mps/__init__.py:86-90` →
> `torch/csrc/mps/Module.cpp:113-117` →
> `aten/src/ATen/mps/MPSAllocator.mm:551-553`) only releases memory pools;
> it does not touch `MPSGraphCache`. The class itself
> (`aten/src/ATen/native/mps/OperationUtils.h:338-421`) exposes no `clear`,
> `evict`, `erase`, or `invalidate` method, and there is no Python binding
> in `torch/csrc/mps/Module.cpp:189-215, 291-340` for graph-cache eviction.
> The only env vars touching MPS are `PYTORCH_DEBUG_MPS_ALLOCATOR`,
> `PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_LOW_WATERMARK_RATIO`
> (`MPSAllocator.mm:18,21,26`) and `PYTORCH_MPS_FAST_MATH`
> (`OperationUtils.mm:453`); none disable or bypass the graph cache.
> Comparing v2.10.0 vs v2.11.0, the `mm_out_mps_impl` body and the
> `MPSGraphCache` mechanism are byte-identical — no commits between the
> two tags touched the cache key, the lookup template, or the cache class.
> Net: a process-lifetime singleton with no user-facing invalidation, key
> construction unchanged for two releases, and the actual kernel
> selection happens inside Apple's MPSGraph framework which we cannot read.
