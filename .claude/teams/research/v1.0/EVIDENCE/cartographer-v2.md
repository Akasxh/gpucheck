---
specialist: research-cartographer-v2
slug: v1.0
started: 2026-05-01T20:30:00Z
completed: 2026-05-01T21:15:00Z
binding_to: SYNTHESIS_v1.md, EVIDENCE/skeptic.md attack #1
tool_calls_count: 27
citations_count: 11
confidence: high (on tile constants extracted from MLX + Apple Feature Set Tables); medium (on PyTorch-MPS specific tiles, which are MPSGraph-internal and unpublished)
---

# Cartographer v2 — Apple-tile mapping for fuzzer extension (binding for skeptic attack #1)

## §0. Charter recap

Skeptic v1 attack #1 (SYNTHESIS_v1.md L19, EVIDENCE/skeptic.md L19-L35): the
fuzzer's "non-tile-aligned" priority was calibrated against NVIDIA tile sizes
(32/64/128). Apple's MPSGraph and MLX use different tile sizes / SIMD groupings,
which Apple does not fully publish. A non-tile-aligned shape against NVIDIA may
BE tile-aligned against Apple — so the fuzzer might miss MPS bugs by accident.

Charter (this turn): map what is known/inferable about MPS tile sizes from
**MLX source** (the only open-source production-quality Apple Silicon ML kernel
suite), **Apple Metal Feature Set Tables** (primary spec), and **llama.cpp
ggml-metal** (cross-validation), then recommend a tile-set extension for
gpucheck's MPS path.

## §1. Apple SIMD group size — the foundational constant

The first question is whether Apple's "warp-equivalent" coincides with NVIDIA's.
The answer: **YES, both are 32 threads per SIMD group, but for unrelated reasons,
and Apple does not formally guarantee this — it's an architectural fact of
GPU families Apple7+ that ships the M1+ generations**.

### Primary sources (NEW, not in Round 1)

**S1.** Apple Metal Feature Set Tables, retrieved 2026-05-01 from
[developer.apple.com/metal/Metal-Feature-Set-Tables.pdf](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
(published 2026-02-05, copyright 2014-2026):
- Page 2: M1-series → Apple7, M2-series → Apple8, M3 + M4-series → Apple9,
  M5-series → Apple10. All M-class run "Metal 3 & 4".
- Page 3-5: "SIMD-scoped reduction operations: Metal3, Apple7" — confirming
  SIMD-scoped ops are an Apple7+ guaranteed feature.
- Page 6 "GPU implementation limits by family": **Maximum threads per
  threadgroup = 1024** (Apple7-Apple10, all M-class). **Maximum total
  threadgroup memory allocation = 32 KB** (all M-class, footnote 5: "actual
  maximum by inspecting MTLComputePipelineState.maxTotalThreadsPerThreadgroup
  property at runtime"). **Threadgroup memory length alignment = 16 B**.
- Page 6: "Maximum threadgroup memory argument table = 31 entries" (compile-time
  cap, not load-bearing for tile sizes).

**Critical**: the Feature Set Tables PDF does NOT explicitly publish the SIMD
group size. Apple's contract is "query
`MTLComputePipelineState.threadExecutionWidth` at runtime" — this is the
[[threads_per_simdgroup]] value documented in MSL §5.8. The 32-thread observation
is derived from G13/G14/G15 reverse engineering and is consistent across every
shipping Apple Silicon GPU, but it is **not a contractual guarantee** from
Apple's docs.

**S2.** dougallj's reverse-engineered Apple G13 GPU Architecture Reference,
retrieved 2026-05-01 from
[dougallj.github.io/applegpu/docs.html](https://dougallj.github.io/applegpu/docs.html):
> "The G13 architecture has 32 threads per SIMD-group."
> "Each thread within a SIMD-group may be deactivated... a 32-bit execution
>  mask in register r0l."
> "Each SIMD-group has access to up to 128 general purpose registers."

This is the **exact width contract** that Apple does not publish but that
philipturner's metal-benchmarks repo and Asahi's Alyssa Rosenzweig
independently confirmed (cited below as S5).

**S3.** llama.cpp ggml-metal kernel, retrieved 2026-05-01:
[github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal):
> `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32`

llama.cpp's dispatch comments embed the assumption directly. Cross-validation:
the **whole production Apple Silicon LLM ecosystem assumes 32**.

### Cross-validation

| Source | SIMD width on Apple Silicon | Citation |
|---|---|---|
| Apple Feature Set Tables | not explicit; query `threadExecutionWidth` | S1 (page 6 footnote 5) |
| dougallj G13 RE | 32 (M1, A14) | S2 |
| ggml-metal | `N_SIMDWIDTH 32` (assumed) | S3 |
| MLX rms_norm/softmax | `constexpr int SIMD_SIZE = 32` | S6 below |
| philipturner metal-benchmarks | 32 (across M1-M2 Pro/Max) | S5 below |

**Implication for gpucheck**: 32 is a safe assumption; the fuzzer's existing
TILE 32 still applies on Apple Silicon. **The skeptic's concern was correct
about HIGHER tile sizes (64/128) but NOT about the base SIMD width**.

## §2. MLX matmul tile constants — the matmul-shaped fuzzer ground truth

The richest primary source for "what matmul tile sizes does Apple Silicon
prefer" is MLX's GEMM fragment & block parameters. These are **production
constants** that MLX ships against; they are not theoretical.

### S4. MLX MMA fragment size (the "Apple tensor core")

`mlx/backend/metal/kernels/steel/gemm/mma.h`, retrieved 2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h#L20-L40](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h):

```cpp
template <typename T, int kFragRows_, int kFragCols_>
struct BaseMMAFrag {
  static_assert(kFragRows_ == 8, "Only 8 x 8 fragment matrices are currently supported");
  static_assert(kFragCols_ == 8, "Only 8 x 8 fragment matrices are currently supported");
};

template <typename T>
struct BaseMMAFrag<T, 8, 8> {
  STEEL_CONST int kFragRows = 8;
  STEEL_CONST int kFragCols = 8;
  STEEL_CONST int kElemsPerFrag = (kFragRows * kFragCols) / 32;  // ← /32 = SIMD width
  ...
  typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;
```

★ **Critical Apple-specific tile constant**: **8x8 MMA fragment**. NVIDIA's
`mma.sync` HMMA is 16x16x16 (Volta/Turing/Ampere) or 16x8x16 (Ampere+ tensor
cores). Apple's `simdgroup_matrix<T,8,8>` is a fundamentally different tile
shape. **64 elements per 8x8 fragment / 32 SIMD lanes = 2 elements/lane** —
exactly half of NVIDIA's WMMA per-lane density.

llama.cpp uses the same 8x8 (`simdgroup_half8x8`, `simdgroup_bfloat8x8`,
`simdgroup_float8x8`):
```cpp
typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, ...>) mul_mm_t;
```
[github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal)

**Implication for gpucheck**: **8 is the load-bearing Apple tile constant**.
Currently NOT in `src/gpucheck/fuzzing/shapes.py:9` `TILE_SIZES = (32, 64, 128)`.

### S6. MLX GEMM kernel block-tile instantiations

`mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal`, retrieved
2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal#L21-L26](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal):

```cpp
#define instantiate_gemm_shapes_helper(iname, itype, oname, otype) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 2, 2) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 64, 16, 1, 2) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32, 32, 2, 2) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 64, 16, 1, 2) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 32, 32, 16, 2, 2) \
  instantiate_gemm_transpose_helper(iname, itype, oname, otype, 64, 32,  8, 4, 1)
```

The shipped (BM, BN, BK) block-tile combinations:
| BM | BN | BK | WM | WN | total threads |
|---|---|---|---|---|---|
| 64 | 64 | 16 | 2 | 2 | 4 simdgroups × 32 = 128 |
| 64 | 64 | 16 | 1 | 2 | 64 |
| 64 | 32 | 32 | 2 | 2 | 128 |
| 32 | 64 | 16 | 1 | 2 | 64 |
| 32 | 32 | 16 | 2 | 2 | 128 |
| 64 | 32 |  8 | 4 | 1 | 128 |

★ **Apple-specific tile constants observed in MLX matmul**: **{8, 16, 32, 64}**.
**128 is rare** (only appears as combined block via 64×2 / 32×4 in WM·WN×SIMD;
no MLX kernel ships BM=128 or BN=128 in the upstream tile set).

### S7. MLX GEMM dispatcher (the runtime tile selector)

`mlx/backend/metal/matmul.cpp`, retrieved 2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp#L86-L160](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp):

```cpp
#define GEMM_TPARAM_MACRO(devc)                              \
  if (devc == 'g' || devc == 'p') { /* Small device */       \
    if (out.dtype() == complex64) {                          \
      bm = 64; bn = 32; bk = 8;  wm = 4; wn = 1;             \
    } else if (!transpose_a && transpose_b) { /* nt */       \
      bm = 64; bn = 32; bk = 32; wm = 2; wn = 2;             \
    } else if (out.dtype() != float32) { /* half/bfloat */   \
      bm = 64; bn = 64; bk = 16; wm = 1; wn = 2;             \
    }                                                        \
  } else if (devc == 'd') { /* Large device */               \
    if ((size_t)batch_size_out * M * N >= 1ul << 20) { ...   \
      bm = 64; bn = 64; bk = 16; ...                         \
    }                                                        \
  } else { /* Medium device */                               \
    bm = 64; bn = 64; bk = 16; wm = 2; wn = 2;               \
  }
```

★ **The dispatcher's tile choices vary by hardware family AND dtype AND
transpose pattern**. Apple Silicon families considered: 'g' (presumably mobile
/ A-series), 'p' / 'd' / 's' / 'c' for desktop M-class. **The full set of
distinct tile values reachable through this dispatcher**:
- **bm ∈ {32, 64}** (note: never 128)
- **bn ∈ {8, 32, 64}**
- **bk ∈ {8, 16, 32, 64, 256, 512}** — note `bk = (K >= 8192 && K > (M + N)) ? 64 : 256;` for large devices in the NAX (Apple Neural Accelerator) path

For NAX/M5-series, the dispatcher overrides: `int bm = 128, bn = 128, bk = 512;`
then `bm = 64; wm = 2;` for s/c/d devices. **128 IS a real Apple-NAX tile**, but
gated behind tensor-core-like NAX hardware that only ships in M5 (Apple10).

### S8. MLX SDPA / FlashAttention tile

`mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal`,
retrieved 2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal#L14-L17](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal):

```cpp
#define instantiate_attn_shapes_helper(iname, itype, mname, mtype)  \
    instantiate_attn(iname, itype, 32, 16, 128, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  80, 4, 1, mname, mtype) \
    instantiate_attn(iname, itype, 32, 32,  64, 4, 1, mname, mtype)
```

Block(Q) × Block(K) × HeadDim instantiations:
- (BQ=32, BK=16, BD=128) — head_dim 128
- (BQ=32, BK=32, BD=80) — head_dim 80
- (BQ=32, BK=32, BD=64) — head_dim 64

★ **Head dimensions Apple ships fused FA for**: {64, 80, 96, 128, 256}
(see `scaled_dot_product_attention.cpp` instantiation list mentioned in
SYNTHESIS_v1 §6 sub-Q 6 — confirmed). Note **head_dim 80** is a peculiar Apple
optimization shape — it appears NOWHERE in NVIDIA's FlashAttention-2/3 fast
paths. (NVIDIA FA primarily ships for head_dim 32, 64, 128, 256.)

### S9. MLX SDPA-vector tile (small-Q path)

`mlx/backend/metal/kernels/sdpa_vector.h`, retrieved 2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h#L42-L43](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h):
```cpp
constexpr int BN = 32;  // batch of N
constexpr int BD = 32;  // depth
```

### S10. MLX softmax/rms_norm/layer_norm SIMD constant

`mlx/backend/metal/kernels/rms_norm.metal` line 22 and
`mlx/backend/metal/kernels/softmax.h` line 16:
```cpp
constexpr int SIMD_SIZE = 32;
```
Plus `mlx/backend/metal/kernels/defines.h` line 12:
```cpp
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;
```
And `SOFTMAX_N_READS = 4`, `RMS_N_READS = 4`. **4 is a small but recurrent
read-batch constant** — different from CUDA where this value tends to be
`16/T_per_thread`.

### S11. MLX implicit-GEMM conv2d tile

`mlx/backend/metal/conv.cpp`, retrieved 2026-05-01 from
[github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp) and the metal kernel
file `mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal#L42-L48`:

```cpp
#define instantiate_implicit_2d_blocks(name, itype)               \
    instantiate_implicit_2d_filter(name, itype, 32,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 64,  8, 16, 4, 1) \
    instantiate_implicit_2d_filter(name, itype, 32, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 32, 64, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 32, 16, 2, 2) \
    instantiate_implicit_2d_filter(name, itype, 64, 64, 16, 2, 2)
```

★ **Conv2d implicit-GEMM ships tiles {(32×8), (64×8), (32×32), (32×64),
(64×32), (64×64)}**. The dispatcher (conv.cpp `bn = 8` for `implicit_N <= 16`
case) **shows that the MLX team explicitly hand-tuned for `bn = 8` when output
channels are tiny** — an Apple-specific optimization. `bn = 8` is NOT in
gpucheck's TILE_SIZES.

### S5. Cross-validation: philipturner/metal-benchmarks

[github.com/philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks),
retrieved 2026-05-01:
> "4 schedulers, each dispatching one instruction from one simd (32 threads) per cycle."
> "MATMUL<8x8xF16>" and "MATMUL<8x8xF32>" benchmarks
> "ALU utilization maxes out at 24 simds/core"

Independently confirms **8x8 fragment** as the load-bearing matmul tile and
**32-thread SIMD** as the foundational unit on Apple Silicon GPUs M1/M1 Pro/Max
and M2 generation.

## §3. Synthesized Apple-tile constant set (master table)

Combining all primary sources, the **Apple-canonical tile constants** that gpucheck
must consider in addition to NVIDIA's {32, 64, 128}:

| Constant | Where it appears | Source |
|---|---|---|
| **8** | MMA fragment row/col; conv2d BN; small N case | S4, S11 |
| **16** | GEMM BK (most common); MMA fragment-internal; FA BK | S6, S8 |
| **32** | SIMD width; GEMM block; FA BQ/BK; reduction tile | S1-S6, S8-S10 |
| **64** | GEMM BM/BN dominant; FA head_dim; conv BM/BN | S6, S7, S8, S11 |
| **80** | FA head_dim Apple-only | S8 |
| **128** | NAX GEMM (M5+ only); FA head_dim | S7 (NAX), S8 |
| **256** | Large-device large-K BK; FA head_dim 256 | S7 |
| **4** | RMS_N_READS, SOFTMAX_N_READS (per-thread read batch) | S10 |
| **4096** | RMS_LOOPED_LIMIT (single-row vs looped boundary) | S10 |

## §4. Negative space — what Apple does NOT publish

Following The Cartographer's anomalies discipline:

1. **MPSGraph internal tile sizes are NOT public**. PyTorch MPS dispatches
   through `MetalPerformanceShadersGraph.framework`
   ([pytorch/CMakeLists.txt L line `-weak_framework MetalPerformanceShadersGraph`](https://github.com/pytorch/pytorch/blob/main/CMakeLists.txt)
   confirmed via context7 query 2026-05-01) which is closed-source. We can only
   infer MPS tile boundaries via MLX (proxy: MLX engineers know Apple HW deeply
   and ship for the same MSL substrate).

2. **The 32-thread SIMD width is not contractually guaranteed**. Apple's
   official position per Feature Set Tables footnote 5: "Check the actual
   maximum by inspecting the `MTLComputePipelineState.maxTotalThreadsPerThreadgroup`
   property at runtime." A future Apple GPU could in principle change this;
   no guarantee until [[threads_per_simdgroup]] is queried at compile/runtime.

3. **The MSL Spec PDF DOES enumerate `[[threads_per_simdgroup]]`** as an
   attribute available in compute kernels (Section 5.8.1 per philipturner repo
   reference), but does NOT pin its value to 32. This is the exact pattern
   from §2 above.

4. **No publicly documented BF16 tensor-core mapping** for Apple Silicon
   pre-M5. MLX maps BF16 onto FP32 simdgroup_matrix accumulators. NVIDIA
   Ampere+ has dedicated BF16 tensor cores. This is a pure correctness-floor
   difference, not a tile-size difference, but it is load-bearing for
   BF16 tolerance overlays (orthogonal to skeptic attack #1).

5. **Apple's NAX (Neural Accelerator) tile sizes ARE different**. The
   `is_nax_available()` branch in `scaled_dot_product_attention.cpp` (S7
   excerpt: `bm = 128, bn = 128, bk = 512`) shows M5+ has a different tile
   regime. **For v1.0, gpucheck's MPS path will run on M1/M2/M3/M4 — non-NAX
   tiles {8, 16, 32, 64} dominate; 128 only matters on M5+**.

## §5. Recommended fuzzer tile-set extension (binding for engineering team)

### Current state (gpucheck v0.1.0)

`src/gpucheck/fuzzing/shapes.py:9-16`:
```python
TILE_SIZES = (32, 64, 128)
PRIMES = (7, 13, 31, 127, 257)
POWER_OF_2_BOUNDARIES = (127, 128, 129, 255, 256, 257, 511, 512, 513)
LARGE_DIMS = (2048, 4096, 8192)
```

### Recommended MPS-path tile extension

```python
# Apple-specific tile boundaries; sourced from MLX matmul/conv/SDPA
# kernels (production-shipping tiles as of MLX main, 2026-05-01).
# See cartographer-v2.md §3 for citations.
TILE_SIZES_MPS = (8, 16, 32, 64, 128)  # add 8 and 16; keep 32/64/128
POWER_OF_2_BOUNDARIES_MPS = (
    7, 8, 9,        # 8x8 MMA fragment boundary (NEW for MPS)
    15, 16, 17,     # GEMM BK=16 boundary (NEW for MPS)
    31, 32, 33,     # SIMD width / GEMM BK=32
    63, 64, 65,     # GEMM BM/BN=64 dominant
    79, 80, 81,     # FA head_dim 80 (Apple-only) (NEW)
    127, 128, 129,  # already covered
    255, 256, 257,  # already covered
    511, 512, 513,  # already covered
)
```

**Rationale per added constant**:
- **8 (TILE)**: 8x8 simdgroup_matrix MMA fragment. Universal in MLX + ggml-metal
  matmul kernels (S4, llama.cpp). NOT in gpucheck's current set. Highest
  priority addition.
- **16 (TILE)**: dominant `BK` in MLX GEMM (S6 — appears in 4 of 6 instantiations)
  and conv (S11 — appears in all 6 instantiations). Also the MTL threadgroup
  memory length alignment (S1).
- **80 (boundary)**: FA head_dim Apple-only optimization (S8). Useful for
  catching head_dim-specific bugs that NVIDIA fuzzing wouldn't surface.
- **9, 17, 33, 65, 81 (just-over-tile)**: classic non-tile-aligned
  catastrophe-after-padding probes; analog of gpucheck's existing 129/257/513.

### What to keep unchanged

- 32, 64, 128 already in TILE_SIZES — these survive the MPS audit (32 is the
  SIMD width, 64 is dominant MLX block-tile, 128 is M5-NAX block-tile and FA
  head_dim 128).
- PRIMES (7, 13, 31, 127, 257) — these are device-independent
  non-power-of-2 / non-tile-aligned probes; they generate the kind of shape
  that found triton#9838 (n_cols=17 ≈ prime+10) and have NO Apple-specific
  re-calibration need.
- LARGE_DIMS (2048, 4096, 8192) — large-shape regime maps directly onto
  MLX's "large device large-K" path (`bk=64 if K>=8192`), and onto the
  RMS_LOOPED_LIMIT=4096 boundary in S10.

### What NOT to add (anti-recommendations)

- Do NOT add **256, 512** to TILE_SIZES_MPS as base tiles. They appear ONLY in
  the BK dimension for very-large-K dispatch paths (S7), and gpucheck's
  LARGE_DIMS=(2048, 4096, 8192) already exercises that regime.
- Do NOT add **head_dim 64, 96** to POWER_OF_2_BOUNDARIES_MPS. 64 is already
  there as 63/64/65; 96 is mid-tile (Apple ships FA at 96 but it's
  3*32 + 0 mod 32 — covered by existing tile reasoning).
- Do NOT chase NAX/M5 (BM=128 AppleNAX, BK=512) until v1.1 — gating tile
  selection on the M-generation is engineering complexity that doesn't ship
  for M1-M4 users.

## §6. Confidence

**High** on the matmul tile constants (BM/BN/BK combinations from MLX kernel
file are direct primary sources; the file is a production manifest, not a
guess). **High** on the SIMD width 32 (cross-validated S1+S2+S3+S5+S6).
**Medium** on PyTorch-MPS specifically — PyTorch dispatches through
MPSGraph.framework which is binary; we infer MPSGraph tile boundaries from
MLX as proxy. Skeptic attack #1's worry that "we don't know Apple's MPSGraph
tiles" remains literally true — **MLX is the closest open-source proxy and
these tiles are what production Apple Silicon ML workloads actually use**,
but we cannot verify MPSGraph's internal tile sizes match MLX's.

**Three primary citations NEW vs Round 1** (per charter requirement):
- S1: Apple Metal Feature Set Tables PDF (page-2 / page-6 / page-3 specific)
- S2: dougallj G13 GPU Architecture Reference (32-thread SIMD-group)
- S4: MLX `mma.h` — 8x8 MMA fragment (NOT in Round 1 cartographer.md, which
  only cited MLX as a high-level pattern)
- S5: philipturner/metal-benchmarks (cross-validation)
- S6-S11: MLX kernels at file+line precision (Round 1 cited the directory
  high-level only — these are NEW concrete tile constants)

This count is **6 NEW primary citations**, exceeding the charter requirement
of ≥3.

## §7. Cross-check against SYNTHESIS_v1 citation list

Round 1 SYNTHESIS_v1 §sub-Q 6 cited:
- `mlx/backend/metal/scaled_dot_product_attention.cpp` (file-level only)
- "MLX kernels: scaled_dot_product_attention, layer_norm, rms_norm, softmax,
  logsumexp, conv, rope, fft, etc."

This v2 cartographer report adds file:line precision and quotes specific
constants; no double-citation.

## §8. Cited primary sources (consolidated)

1. [developer.apple.com/metal/Metal-Feature-Set-Tables.pdf](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) — Apple's official Metal Feature Set Tables, retrieved 2026-05-01.
2. [dougallj.github.io/applegpu/docs.html](https://dougallj.github.io/applegpu/docs.html) — Apple G13 GPU Architecture Reference, retrieved 2026-05-01.
3. [github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal) — N_SIMDWIDTH=32 + simdgroup_half8x8 kernel_mul_mm instantiations, retrieved 2026-05-01.
4. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h) — 8x8 BaseMMAFrag, /32 SIMD divisor, retrieved 2026-05-01.
5. [github.com/philipturner/metal-benchmarks](https://github.com/philipturner/metal-benchmarks) — 32-thread SIMD, MATMUL<8x8> benchmarks, retrieved 2026-05-01.
6. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.metal) — full 6-tuple (BM,BN,BK,WM,WN) tile set, retrieved 2026-05-01.
7. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp) — GEMM_TPARAM_MACRO dispatcher and NAX BM=128/BK=512 path, retrieved 2026-05-01.
8. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.metal) — FA tile shapes (BQ=32, BK=16/32, BD=64/80/128), retrieved 2026-05-01.
9. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h) — BN=32, BD=32 vector SDPA, retrieved 2026-05-01.
10. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/rms_norm.metal) and [softmax.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h) — SIMD_SIZE=32, RMS_N_READS=4, retrieved 2026-05-01.
11. [github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/conv.cpp) and [steel_conv.metal](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/conv/kernels/steel_conv.metal) — implicit-GEMM conv2d tile set with bn=8 small-N case, retrieved 2026-05-01.
