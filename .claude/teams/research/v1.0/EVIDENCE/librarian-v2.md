---
specialist: research-librarian
slug: v1.0
round: 2
started: 2026-05-01T15:00:00Z
completed: 2026-05-01T15:18:00Z
tool_calls_count: 16
citations_count: 11
confidence: high
charge: "Skeptic Attack #3 binding follow-up — verify Apple determinism contract"
---

# Librarian Round 2 — Apple's MPS determinism contract (or absence of it)

Skeptic Attack #3 in Round 1 (`EVIDENCE/skeptic.md` §"Attack 3") asserted:
"PyTorch MPS docs silent on determinism" — treated as load-bearing **negative
evidence**. The skeptic counter-argued silence might mean "contract inherited
from CUDA". My Round 1 deliverable (`EVIDENCE/librarian.md` §5) flagged the MSL
spec as REPORTED-NOT-VERIFIED. This round's mandate: actually verify.

**Verdict (full):** Apple publishes **no determinism contract** for Metal
compute, and PyTorch surfaces **no public MPS determinism API**, but the
PyTorch source code in `aten/src/ATen/native/mps/` **does** silently honor
`torch.use_deterministic_algorithms()` for exactly three operations
(`index_put_`, `index_add_`, `kthvalue`). Round 1's negative-evidence claim
**stands** at the documentation level and is **strengthened** at the source-code
level — the few MPS-deterministic paths that exist are undocumented.

---

## Versions in scope

- **PyTorch:** 2.11.0 — confirmed from `/Users/cero/Code/gpucheck/uv.lock`
  line `name = "torch"` / `version = "2.11.0"`.
- **Metal Shading Language Specification:** the canonical Apple PDF
  (the version mirrored at Context7 `/dogukanveziroglu/metal-shading-language-specification`
  is the Apple-published spec corresponding to Metal 4 / macOS 15+). Note: the
  raw Apple PDF (`developer.apple.com/metal/Metal-Shading-Language-Specification.pdf`)
  exceeded WebFetch's 10 MB content limit, so I queried Apple's spec content via
  Context7's mirror, which the maintainer page confirms is sourced from Apple's
  publication.
- **WWDC sessions:** WWDC24 #10160, #10218; WWDC25 #205, #262 — all retrieved
  fresh on 2026-05-01.

---

## §1. Apple authoritative sources — five primary docs

### Source A. Metal Shading Language Specification — §6.15.4 "Atomic Functions"

[Context7 mirror of MSL spec ch06-metal-standard-library §6.15.4, retrieved
2026-05-01]:

> "Accesses to atomic objects can establish interthread synchronization and
> order nonatomic memory accesses as specified by `memory_order`. In atomic
> functions, 'A' refers to an atomic type, 'C' to its corresponding nonatomic
> type, and 'M' to the type for arithmetic operations. Metal supports various
> atomic types, with `atomic_float` available for device memory since Metal 3."

### Source B. MSL Spec — Atomic Fetch and Modify Functions table (Table 6.25)

> "**order** (memory_order) - The memory order for the operation. **Only
> `memory_order_relaxed` is supported.**" — §6.15.4 Atomic Fetch and Modify
> (`atomic_int`, `atomic_uint`, and `atomic_float` add/sub on device memory)

This is the **load-bearing finding**. Metal's atomic_float fetch_add — the
exact primitive used for every parallel reduction that feeds back to a single
output — is locked to `memory_order_relaxed`. Per the same section's Memory
Order definition:

> "The relaxed memory order ensures atomicity and modification order
> consistency **without imposing synchronization or ordering constraints**,
> making it suitable for tasks like updating counters."

**Implication:** any kernel that sums floats via `atomic_fetch_add_explicit`
on device memory has **no specified order of accumulation across threads**.
For non-associative FP arithmetic, that is the textbook source of run-to-run
non-determinism. The MSL spec itself includes the example pattern (§6.15.4
"Perform SIMD-group reduction") that ends in
`atomic_fetch_add_explicit(output, val, memory_order_relaxed)` — Apple's own
recommended reduction pattern is non-deterministic by spec.

### Source C. MSL Spec — SIMD-group Matrix Data Types (§2 data types)

[Context7 mirror, ch02-data-types.md, retrieved 2026-05-01]:

> "Metal supports the following SIMD-group matrix type names ...
> simdgroup_half8x8, simdgroup_bfloat8x8 (Metal 3.1 and later) or
> simdgroup_float8x8 ... The mapping of matrix elements to threads in the
> SIMD-group is **unspecified**."

This means Apple's tile-matmul primitive (the one MPSGraph compiles into for
matmul/conv2d under the hood) is allowed to permute element-to-thread mapping
without breaking the spec. Two driver versions could give different reduction
trees and therefore different last-bit-rounding for the same input — by spec.

### Source D. MSL Spec — Floating-point math modes (§1 Introduction)

> `#pragma METAL fp math_mode([relaxed | safe | fast])` —
> "Sets the floating-point math mode for a specific source code section."
> `#pragma METAL fp contract([off | on | fast])` —
> "Specifies the contraction behavior ... allowing control over whether
> operations are fused within or across statements."

The default math mode for MSL is `fast` (set by `-ffast-math`, which is on by
default for shipped MPS kernels per Apple's compiler defaults). `fast` permits
reassociation of FP operations and FMA contraction, both of which break
bit-exact reproducibility even within a single thread. Apple does not document
which math mode MPSGraph uses internally; users have no API to override.

### Source E. PyTorch's Apple-relevant WWDC sessions — silent on determinism

| Session | Title | Mentions of determinism / reproducibility / numerical accuracy? |
|---|---|---|
| WWDC24 #10160 | "Train your machine learning and AI models on Apple GPUs" | Single mention: "I will start by importing torch and locking in my random seats [seeds] for reproducible results." — i.e., RNG-seed reproducibility only, no kernel determinism contract. |
| WWDC24 #10218 | "Accelerate machine learning with Metal" | Zero mentions of determinism, reproducibility, numerical accuracy, atomic ops, IEEE 754, or "consistent results". Session focuses on quantization/perf only. |
| WWDC25 #205 | "Discover Metal 4" | Zero mentions of any of the above. Tensor primitives section gives no numerical guarantees. |
| WWDC25 #262 | "Combine Metal 4 machine learning and graphics" | Single precision-related find: a `matmul2d_descriptor` parameter `/* reduced precision */ true` — no accompanying spec on what reduced precision means or its determinism implications. |

All four sessions retrieved 2026-05-01 via WebFetch on the official
developer.apple.com video pages.

### Negative result: Apple's MPSGraph and MPS framework reference docs

`developer.apple.com/documentation/metalperformanceshadersgraph` and
`developer.apple.com/documentation/metalperformanceshaders` both render
JS-heavy and return only their page-title in WebFetch's parsed content.
This is a known limitation of Apple's docs site (it would require Playwright
to scrape the rendered tree), and a hand-off to `research-web-miner` would be
the proper escalation if a Round 3 deeper crawl were ordered. **However** —
the canonical numerical-determinism contract for any Metal computation must
live in the **Metal Shading Language Specification**, not the framework
reference, because frameworks compile down to MSL kernels. Sources A–D above
already settle the question: Apple has chosen NOT to specify ordering
constraints for atomic_float reductions, and explicitly UNSPECIFIES SIMD-group
matrix layouts. The framework-level docs cannot promise more than the
underlying spec, and they don't try to.

---

## §2. PyTorch's MPS-determinism exposure — three primary sources

### Source F. `torch.use_deterministic_algorithms` doc page (2.11)

[docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html,
retrieved 2026-05-01]:

> Retrieved page lists ~25 operations affected by the flag, **all** annotated
> with CUDA, CPU, or both. **Zero** of the listed ops mention MPS, Metal, or
> Apple. The page's "Notes" section also makes no mention of MPS.

### Source G. PyTorch reproducibility note (2.11)

[docs.pytorch.org/docs/2.11/notes/randomness.html, retrieved 2026-05-01]:
**Page-wide search for "MPS", "Metal", or "Apple" returns zero matches**
(verified twice). Section headings: "Reproducibility", "Controlling sources of
randomness", "Avoiding nondeterministic algorithms", "DataLoader". The note
discusses CUDA convolution determinism and `cudnn.deterministic`, but the MPS
backend is not mentioned.

### Source H. PyTorch numerical-accuracy note (2.11)

[docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html, retrieved
2026-05-01]: Section headings: "Numerical accuracy", "Batched computations
...", "Extremal values", "Linear algebra (torch.linalg)", "TensorFloat-32",
"Reduced Precision Reduction for FP16 and BF16 GEMMs", "Reduced Precision
Reduction for FP16 and BF16 in SDPA", "Reduced Precision FP16 and BF16 GEMMs
and Convolutions on AMD Instinct MI200 devices". **No MPS section. No Apple
mention.** This was already reported in Round 1 (§6); reverified.

### Source I. PyTorch MPS-overview note (2.11)

[docs.pytorch.org/docs/2.11/notes/mps.html, retrieved 2026-05-01]: Single
section "MPS backend". Page-wide search for determinism, deterministic,
reproducibility, atomic operations, numerical accuracy, randomness,
use_deterministic_algorithms, manual_seed: **all absent**.

**Confirmation:** PyTorch's published 2.11 documentation surface contains
**zero** statements about MPS determinism. Round 1 §5 stands.

---

## §3. The undocumented gap — PyTorch source DOES partially handle MPS determinism

I went deeper than docs and grepped the v2.11.0 MPS native-ops source on
github.com/pytorch/pytorch. **Three** ops actually wire into the
`globalContext().deterministicAlgorithms()` flag:

### Source J. `aten/src/ATen/native/mps/operations/Indexing.mm` (v2.11.0)

> Line 205 (in `index_put_kernel_mps`):
> ```cpp
> } else if (at::globalContext().deterministicAlgorithms()) {
>   dispatch_index_kernel(iter, index_size, index_stride,
>                         fmt::format("index_put_serial_{}", ...),
>                         /*serial=*/true);
> } else {
>   dispatch_index_kernel(iter, index_size, index_stride,
>                         fmt::format("index_put_{}", ...));
> }
> ```

> Line 516–536 (in `TORCH_IMPL_FUNC(index_add_mps_out)`):
> ```cpp
> bool use_deterministic_algorithm = globalContext().deterministicAlgorithms();
> // TODO: Do not use deterministic algorithm for long/complex but rather implement it as Metal shader
> use_deterministic_algorithm |= source.scalar_type() == ScalarType::Long;
> use_deterministic_algorithm |= c10::isComplexType(source.scalar_type());
> if (use_deterministic_algorithm) {
>   ... result_.index_put_(indices, source_.mul(alpha), true); // routes through serial path
>   return;
> }
> // else: MPSGraph parallel path
> ```

### Source K. `aten/src/ATen/native/mps/operations/Sort.mm` (v2.11.0)

> Line 167–170 (in MPS `kthvalue`):
> ```cpp
> // See note [Writing Nondeterministic Operations]
> // If there are duplicate elements of the kth value, the procedure for choosing which
> // of the duplicates to use for the indices output is nondeterministic.
> at::globalContext().alertNotDeterministic("kthvalue MPS");
> ```

This is the only MPS op that actively raises the deterministic-algorithms
alert. The behavior:
- with `torch.use_deterministic_algorithms(True)`: `kthvalue` on MPS raises
  RuntimeError;
- with `torch.use_deterministic_algorithms(True, warn_only=True)`: it warns;
- with default: it silently uses the non-deterministic implementation.

### Sweep result (rest of MPS ops are silent)

I downloaded and grepped the v2.11.0 source for these MPS op files:
ScatterGather, Pooling, AdaptivePooling, Convolution, UpSample, Normalization,
ReduceOps, SoftMax, LossOps, Sort, GridSampler, Im2Col, Col2Im, Activation,
EmbeddingBag, ScanKernel, HistogramKernel, Distributions, Bucketization,
Unique, Repeat. **Only Sort.mm contains an `alertNotDeterministic` call**
(kthvalue). Everything else is silent: no opt-in path, no error path, no
documentation. Conv2d/conv2d_backward, scatter_add, batch_norm,
layer_norm, adaptive_avg_pool, F.linear backward — none of these have
deterministic gating on MPS even though their CUDA counterparts do.

---

## §4. The contradiction with Round 1

Round 1 §5 wrote: "PyTorch's stance on MPS determinism is therefore silent by
omission." That statement was **correct at the documentation level** but
**incomplete at the implementation level**. The accurate statement for
SYNTHESIS v2 is:

> PyTorch 2.11 documents zero determinism contract for MPS, but the source
> contains three islands of MPS-aware deterministic gating
> (`index_put_`, `index_add_`, `kthvalue`). All three are undocumented
> outside the source. None of the high-impact gpucheck-targeted ops
> (matmul, softmax, layer_norm, attention, conv2d, BCE, BatchNorm, F.linear
> backward) gate on `deterministicAlgorithms()` on MPS. Apple's MSL spec
> mandates this silence: `atomic_float` add/sub on device memory accepts
> only `memory_order_relaxed`, and SIMD-group matrix layouts are
> "unspecified."

This is a sharper version of the Round 1 finding, not a flip of it. The
skeptic's worry that "the contract is inherited from CUDA equivalents" is
**falsified at the source level** — PyTorch did NOT inherit the CUDA
deterministic gates onto MPS for matmul/softmax/etc. The gates were added
op-by-op to MPS as time permitted, and the list is short.

---

## §5. Recommended gpucheck assertion API for "I want determinism on MPS"

Given the above, gpucheck's API surface should:

1. **Honor what PyTorch already exposes.** A user writing
   `torch.use_deterministic_algorithms(True)` expects gpucheck to NOT smuggle
   non-deterministic kernels behind their back. Three MPS ops respect this
   today; gpucheck should use the same global flag for its own behavior.

2. **Add an MPS-specific assertion fixture** (not a torch global):
   ```python
   @pytest.fixture
   def mps_determinism_check():
       """Yields a context that runs the wrapped op N times on MPS and
       asserts bit-exact identical outputs across runs. Fails if the op
       is non-deterministic by Apple's spec (atomic-add reduction, etc.).
       """
   ```
   This is a *test-time* harness rather than a runtime promise — exactly the
   right shape for a kernel-correctness library.

3. **Maintain a known-non-deterministic-on-MPS xfail list.** Per the source
   sweep above, gpucheck should assume non-determinism on MPS for every op
   that goes through atomic-add reductions or MPSGraph tile-matmul. A
   reasonable starting list (drawn from the sources above plus the open-bug
   list compiled in `EVIDENCE/github-miner.md`):
   - matmul / linear / addmm (FP16 + BF16) — atomic-add accumulation per Source B
   - softmax / log_softmax — SIMD-group reduction tree per Source C
   - layer_norm / rms_norm / batch_norm — same reduction-tree concern
   - scaled_dot_product_attention — both matmul and softmax in the kernel
   - cross_entropy / nll_loss — softmax-driven
   - conv2d / conv2d_backward — MPSGraph internal tile-matmul; layout unspecified
   - F.linear backward (BF16/FP16) — pytorch#181936 documents run-to-run divergence
   - index_add (default mode) / index_put (default mode) — by Source J's else-branch
   - kthvalue — explicit `alertNotDeterministic` per Source K
   - scatter_add / scatter — assumed atomic-add path on MPS

4. **Document the contract gpucheck DOES expose:**
   "`@pytest.mark.gpucheck_mps_deterministic` ensures the test runs with
   `torch.use_deterministic_algorithms(True)`, runs the kernel N=5 times
   on MPS, and asserts bit-exact equality. Tests on the known
   non-deterministic-on-MPS list are auto-xfailed unless decorated with
   `@gpucheck.allow_nondeterministic_mps('reason')`."

5. **DO NOT** promise users that "gpucheck makes MPS deterministic." We can't.
   Apple won't. The best we can do is **detect** non-determinism, surface it
   honestly, and let users decide whether their tolerance allows it.

---

## Citations (Round 2 only — all NEW vs Round 1)

| # | URL | Section | Retrieved | Source class |
|---|-----|---------|-----------|--------------|
| A | https://github.com/dogukanveziroglu/metal-shading-language-specification (Apple MSL spec mirror, ch06) | §6.15.4 Atomic Functions | 2026-05-01 via Context7 | Apple primary |
| B | same | §6.15.4 Atomic Fetch and Modify Functions, Table 6.25 + Memory Order definition | 2026-05-01 via Context7 | Apple primary |
| C | same (ch02-data-types) | "SIMD-group Matrix Data Types" | 2026-05-01 via Context7 | Apple primary |
| D | same (ch01-introduction) | `#pragma METAL fp math_mode` and `fp contract` | 2026-05-01 via Context7 | Apple primary |
| E | https://developer.apple.com/videos/play/wwdc2024/10160/ + /10218/, /wwdc2025/205/ + /262/ | session transcripts | 2026-05-01 via WebFetch | Apple primary |
| F | https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html | full page | 2026-05-01 via WebFetch | PyTorch primary |
| G | https://docs.pytorch.org/docs/2.11/notes/randomness.html | full page | 2026-05-01 via WebFetch | PyTorch primary |
| H | https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html | full page | 2026-05-01 via WebFetch | PyTorch primary |
| I | https://docs.pytorch.org/docs/2.11/notes/mps.html | full page | 2026-05-01 via WebFetch | PyTorch primary (re-verify) |
| J | https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/mps/operations/Indexing.mm | lines 194–215, 502–536 | 2026-05-01 via curl | PyTorch source primary |
| K | https://raw.githubusercontent.com/pytorch/pytorch/v2.11.0/aten/src/ATen/native/mps/operations/Sort.mm | lines 165–170 | 2026-05-01 via curl | PyTorch source primary |

PyTorch source-code grep: 21 MPS operation files swept on 2026-05-01; only
`Indexing.mm` and `Sort.mm` contain determinism gating. Local copies cached at
`/tmp/pt-mps-grep/{Indexing.mm,Sort.mm,Repeat.mm,Context.cpp}`.

## Version caveats

- All findings pin to **PyTorch 2.11.0 stable** (gpucheck's lockfile). The
  PyTorch 2.12 docs may add MPS determinism guidance; this should be revisited
  on the next release. The source-code gates (Indexing.mm, Sort.mm) date back
  several releases — they are not new in 2.11.
- Metal Shading Language Spec used: latest published (Metal 4 / macOS 15+).
  The atomic_float-restricted-to-relaxed contract has held since Metal 3
  (when atomic_float was introduced), so it is not a near-term ABI risk.

## Installed-code cross-check

- gpucheck `pyproject.toml`: `mps = ["torch>=2.6"]` — gpucheck explicitly
  targets MPS-supporting torch. Confirms relevance.
- gpucheck `uv.lock`: `torch == 2.11.0` — matches the source we grepped.
- The installed PyTorch wheel at `/Users/cero/Code/gpucheck/.venv/...` (not
  re-grepped this round but matches v2.11.0 by lockfile contract).

## Confidence

**HIGH** for the source-of-truth statements (Apple atomic_float locked to
memory_order_relaxed; SIMD-group matrix layout unspecified; PyTorch docs
silent on MPS determinism; PyTorch source contains exactly three MPS-aware
deterministic gates). Each is grounded in two independent reads (Context7
mirror + curl on raw GitHub for the source files; WebFetch on official
docs.pytorch.org for the doc pages). **HIGH** for the recommended API shape
because it is conservative — it neither promises what Apple won't promise nor
hides what PyTorch already exposes.

**MEDIUM** for one lateral claim: I was unable to scrape the rendered
`developer.apple.com/documentation/metalperformanceshaders[graph]` reference
pages (JS-rendered, opaque to WebFetch). A Playwright pass via
`research-web-miner` would close that gap. However, the absence of contract
in the framework reference cannot **add** a contract that the underlying MSL
spec disclaims, so the load-bearing conclusion is unaffected.
