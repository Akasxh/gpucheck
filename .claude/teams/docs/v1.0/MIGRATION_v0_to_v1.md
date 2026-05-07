# Migrating from gpucheck v0.1.0 → v1.0

> **Phase 1 status (this document):** placeholder skeleton with one
> section per public API expected to break or extend in v1.0. Concrete
> *before/after* code snippets and the final API shape come from
> Phase 3 once `<cwd>/.claude/teams/research/v1.0/SYNTHESIS.md` and
> `<cwd>/.claude/teams/engineering/v1.0/DIFF_LOG.md` exist. Each section
> below carries a `{{TODO Phase 3: ...}}` marker for the part that
> requires real diff data.

This guide is for users on `gpucheck >= 0.1, < 1.0` upgrading to
`gpucheck >= 1.0`. v1.0 introduces:

1. **Apple Silicon Metal Performance Shaders (MPS) backend** as the
   headline feature (research QUESTION sub-question 1).
2. **Stride / contiguity fuzzing** (was a documented gap in
   `CLAUDE.md` "Known Weaknesses").
3. **Thread-safe tolerance overrides** (replaces the `# NOT thread-safe`
   stack at `src/gpucheck/assertions/tolerances.py:26-28`).
4. **Release-bundle hygiene** (CHANGELOG, MIGRATION, CONTRIBUTING all
   land for the first time).

The version jump from `0.1.0` straight to `1.0.0rc1` is intentional:
v0 was an alpha PyPI publish marked
`Development Status :: 3 - Alpha` (`pyproject.toml:17`); v1 is the
first release-quality tag.

---

## Quick checklist

| If you currently … | You will need to … |
|---|---|
| Pin `gpucheck<1` | Read this entire file before bumping |
| Use `@devices("cuda:0")` only | No change required, but consider `@devices()` auto-detect to pick up MPS |
| Construct a `GPUInfo` directly in tests | **Check API shape** — see §1 |
| Use `@require_arch("Ampere", ...)` on a Mac | **Read §2** |
| Read `gpu.compute_capability` on Apple Silicon | **Read §2** |
| Wrap test code in `tolerance_context(...)` from `pytest-xdist` workers | **Now safe** (was racy in v0) — see §4 |
| Call `check_memory_leaks(fn)` on a Mac | **Now uses MPS allocator stats** — see §5 |
| Run `compute-sanitizer` via `run_with_sanitizer` on a Mac | **Still raises** — sanitizer is NVIDIA-only |
| Use `lookup_gpu_specs("M2 Max")` | **New** — works in v1 |

---

## 1. `GPUInfo` shape change

**Status:** API-shape decision pending engineering.

The `GPUInfo` dataclass at `src/gpucheck/arch/detection.py:104-122` is
NVIDIA-shaped:

```python
@dataclass(frozen=True, slots=True)
class GPUInfo:
    device_id: int
    name: str
    compute_capability: tuple[int, int]   # NVIDIA-only concept
    architecture: str                     # "Pascal" .. "Blackwell-Consumer"
    memory_total_mb: int
    memory_free_mb: int
    driver_version: str
    cuda_version: str                     # NVIDIA-only
    supports_fp16: bool
    supports_bf16: bool
    supports_fp8: bool
    supports_tf32: bool
    tensor_core_generation: int | None    # NVIDIA-only concept
    max_shared_memory_per_block: int
```

On Apple Silicon, `compute_capability`, `cuda_version`, and
`tensor_core_generation` are not meaningful.

**v1.0 options being weighed by engineering:**

- **Option A — extend in place.** Add a `backend: Literal["cuda","mps"]`
  field; allow `compute_capability=None` and `cuda_version=""` on MPS;
  add `chip: str` (e.g. `"M2 Max"`) and a Metal-specific
  `metal_family: int` field.
- **Option B — sibling dataclass.** Introduce
  `gpucheck.arch.MPSDevice` (or `AppleSiliconDevice`) with its own
  fields; have `detect_gpus()` return a `list[GPUInfo | MPSDevice]`.
- **Option C — protocol.** Introduce a `GPUDeviceLike` Protocol;
  `GPUInfo` and `MPSDevice` both satisfy it.

**v0 → v1 migration template (Option A):**

```python
# v0.1.0
gpu = detect_gpu()
if gpu and gpu.compute_capability >= (8, 0):
    enable_bf16_path()

# v1.0 — same code still works on CUDA hosts.
# On Apple Silicon hosts, gpu.compute_capability is None (or (0, 0)).
# Use gpu.backend == "mps" to gate Apple-specific paths.
gpu = detect_gpu()
if gpu and gpu.backend == "cuda" and gpu.compute_capability >= (8, 0):
    enable_bf16_path()
elif gpu and gpu.backend == "mps":
    enable_mps_path()
```

`{{TODO Phase 3: replace this section with the actual chosen option,
the actual final field list, and concrete before/after diffs from
DIFF_LOG.}}`

---

## 2. `@require_arch` and `@require_capability` semantics on non-CUDA

**Status:** API decision pending engineering.

In v0, `@require_arch("Ampere", "Hopper")` skips on any host whose
`GPUInfo.architecture` is not in the given list
(`src/gpucheck/arch/compatibility.py:58-92`). On Apple Silicon, every
test decorated this way will skip — which is correct but
inflexible.

`@require_capability(8, 0)` is meaningless on MPS because Apple GPUs
do not expose an SM number.

**v1.0 options being weighed by engineering:**

- **Option α — extend `require_arch`.** Accept Apple architecture
  names: `"Apple-Silicon"`, `"M1"`, `"M2"`, `"M3"`, `"M4"`, plus a
  catch-all `"Apple"`. The decorator becomes backend-agnostic.
- **Option β — new decorator.** Add `@require_backend("cuda" | "mps")`
  alongside the existing two.
- **Option γ — both.** Accept Apple names in `require_arch` *and*
  ship `@require_backend` for the common case.

**v0 → v1 migration template (Option γ):**

```python
# v0.1.0 — runs only on Ampere/Hopper CUDA GPUs
@require_arch("Ampere", "Hopper")
def test_bf16_matmul(): ...

# v1.0 — same code still skips on MPS hosts.
# To opt in to MPS:
@require_backend("cuda", "mps")
def test_matmul_any_backend(): ...

# To opt in to CUDA-Ampere+ OR any Apple Silicon:
@require_arch("Ampere", "Hopper", "M1", "M2", "M3", "M4")
def test_bf16_matmul(): ...
```

`{{TODO Phase 3: confirm the chosen decorator shape and replace.}}`

---

## 3. `@devices(...)` and `@parametrize_gpu(devices=...)`

In v0, `@devices()` with no arguments auto-detects CUDA devices only
(`src/gpucheck/decorators/devices.py:13-22`).

In v1, `@devices()` will additionally include `"mps"` when
`torch.backends.mps.is_available()` returns `True`.

**v0 behaviour (still supported):**

```python
@devices("cuda:0")
def test_kernel(device): ...
```

**New v1 behaviour:**

```python
# Auto-detect — picks up CUDA AND MPS
@devices()
def test_kernel(device): ...
# device parametrizes over ["cuda:0", "mps"] on a hybrid host

# Explicit MPS
@devices("mps")
def test_metal_kernel(device): ...

# Mix
@devices("cuda:0", "mps")
def test_both(device): ...
```

`_is_device_available` (`devices.py:25-43`) gains an `mps` branch
that calls `torch.backends.mps.is_available()` /
`torch.backends.mps.is_built()`. Tests for unavailable devices still
skip cleanly.

`{{TODO Phase 3: confirm whether `"all"` includes MPS, and whether
`@devices("all")` semantics changed.}}`

---

## 4. `tolerance_context()` is now thread-safe

`src/gpucheck/assertions/tolerances.py:26-28` currently warns:

```python
# Override stack (module-level). NOT thread-safe — each thread/worker
# should use its own process (pytest-xdist worker) for parallel test
# execution.
_tolerance_overrides: list[tuple[float, float]] = []
```

In v1.0 the override stack is rewritten as a `threading.local()` (or
`contextvars.ContextVar`) so:

- `pytest-xdist` workers (already isolated) continue to work.
- Concurrent threads inside a single worker no longer leak their
  override into other threads.

**No user-facing API change** — same `tolerance_context(atol, rtol)`
signature. The only observable difference is correctness on
multi-threaded test runs.

`{{TODO Phase 3: confirm whether `contextvars` or `threading.local`
was chosen — affects asyncio behaviour.}}`

---

## 5. `check_memory_leaks` and `memory_guard` on MPS

In v0:
- `src/gpucheck/sanitizers/memory.py:36-66` reads
  `torch.cuda.memory_stats()` first, falls back to pynvml.
- On a Mac, both fail, returning a zero-filled report — silent no-op.

In v1, the same functions add an MPS branch using
`torch.mps.current_allocated_memory()` and (where exposed)
`torch.mps.driver_allocated_memory()`. The `SanitizerMemoryReport`
shape is unchanged.

**v0 → v1 migration:** none required. Calls that returned
zero-filled reports on Mac will now return real numbers.

`{{TODO Phase 3: cite research SYNTHESIS sub-question 3 for the exact
torch.mps API surface relied on.}}`

---

## 6. Tolerance table extended for MPS

The default tolerances at `src/gpucheck/assertions/tolerances.py:12-24`
are calibrated against cuBLAS on Turing/Ampere. Apple Silicon's MPS
matmul kernels exhibit different rounding behaviour for FP16, BF16, and
FP32 (research QUESTION sub-question 7 quantifies this).

**v0 → v1 migration:** in most cases, none. `assert_close` continues to
auto-pick tolerances based on dtype. If you previously hard-coded
multipliers because v0 was too tight on MPS-evaluated tensors, you can
now drop the override.

If you need to opt out and force CUDA-style tolerances, use
`tolerance_context(atol=..., rtol=...)` to override.

`{{TODO Phase 3: insert the actual tolerance multipliers from research
SYNTHESIS sub-Q 7. Numbers must be sourced from measured bug data, not
folk wisdom (research QUESTION § Hard rules).}}`

---

## 7. Roofline `_KNOWN_SPECS` extended

`src/gpucheck/analysis/roofline.py:40-46` v0 ships only with NVIDIA
specs (A100, H100, RTX 4090, RTX 3090, V100). v1 adds Apple Silicon:

```python
# v1.0
_KNOWN_SPECS["M1"]         = GPUSpecs(...)  # filled from Apple's GPU spec
_KNOWN_SPECS["M2"]         = GPUSpecs(...)
_KNOWN_SPECS["M2 Pro"]     = GPUSpecs(...)
_KNOWN_SPECS["M2 Max"]     = GPUSpecs(...)
_KNOWN_SPECS["M3"]         = GPUSpecs(...)
_KNOWN_SPECS["M3 Pro"]     = GPUSpecs(...)
_KNOWN_SPECS["M3 Max"]     = GPUSpecs(...)
_KNOWN_SPECS["M4"]         = GPUSpecs(...)  # if applicable
```

**v0 → v1 migration:** none required. `lookup_gpu_specs(device_name)`
on an Apple host now returns a non-`None` value where it returned
`None` before.

`{{TODO Phase 3: fill in real numbers, sourced from Apple's published
SoC GPU compute and memory bandwidth specs. Cite each source.}}`

---

## 8. Examples / `examples/` directory

In v0, examples live in `examples/`, including
`examples/triton_layernorm_bug.py` and
`examples/triton_matmul_bug.py` (the upstream-bug reproducers cited at
`README.md:328-338`).

In v1, the examples directory gains:
- `{{TODO Phase 3: list of new MPS-validated examples added by
  engineering Track A.}}`
- Possibly an `examples/stride_fuzzing_example.py` for Track B.

The two existing Triton-bug reproducers continue to require a CUDA
host. Re-running them on MPS will skip cleanly.

---

## 9. Removed / deprecated APIs

**v1.0rc1 removes nothing from v0.1.0** — it is API-additive in every
case identified by Phase 1. If engineering Track A or C ends up
requiring a removal, it will be documented here in Phase 3.

`{{TODO Phase 3: list any removed/deprecated symbols once DIFF_LOG is
final.}}`

---

## 10. Stride / contiguity fuzzing — new public API

`{{TODO Phase 3: document the new `fuzz_strides()` (or equivalent) API
introduced by `feat/track-b-strides`. Reference DIFF_LOG. Show
`before` (manually testing `tensor.T`, `tensor[::2]`, `as_strided`) and
`after` (one decorator covers all three).}}`

The pre-existing caveat at `src/gpucheck/fuzzing/inputs.py:146-151`
("This function generates contiguous tensors only ... Consider testing
with `tensor.T`, `tensor[::2]`, or `tensor.as_strided(...)`
separately") will be removed once Track B lands.

---

## 11. `__version__` bump

```python
# v0.1.0:  src/gpucheck/__init__.py:8
__version__ = "0.1.0"

# v1.0.0rc1: src/gpucheck/__init__.py:8
__version__ = "1.0.0rc1"

# v1.0.0:    src/gpucheck/__init__.py:8
__version__ = "1.0.0"
```

`pyproject.toml:7` mirrors this.

---

## Why the version jump?

v0.1.0 was published as an alpha to PyPI to claim the package name and
shake out installer-side issues. v1.0 is the first version with:
- a published CHANGELOG, MIGRATION, and CONTRIBUTING guide,
- a hardware-validation matrix beyond a single SKU,
- a documented API stability commitment.

Following Semantic Versioning, the additions described above are not
themselves breaking; the version bump signals the **stability
commitment** rather than a wave of breaking changes.

---

## Cross-references

- `<cwd>/CHANGELOG.md` — the full per-version diff.
- `<cwd>/CONTRIBUTING.md` — how to file issues / send PRs.
- `<cwd>/.claude/teams/research/v1.0/SYNTHESIS.md` —
  the research substrate for §1, §2, §6 (when produced).
- `<cwd>/.claude/teams/engineering/v1.0/DIFF_LOG.md` — the engineering
  substrate for every `{{TODO Phase 3}}` marker (when produced).
