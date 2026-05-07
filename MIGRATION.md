# Migrating from gpucheck v0.1.0 → v1.0

This guide covers users on `gpucheck >= 0.1, < 1.0` upgrading to
`gpucheck >= 1.0`.

**TL;DR — v1.0 is API-additive.** Existing CUDA-only test suites continue
to work without modification. The new entry points are the
`gpucheck.backends` package, `@devices("mps")` / `@devices("all")`, the
`[tool.gpucheck.mps.xfail]` config block, and the `[mps]` install extra.
The 2× MPS tolerance multiplier is **PROVISIONAL** pending M-machine
calibration.

The version jump from `0.1.0` straight to `1.0.0rc1` is intentional:
v0 was an alpha PyPI publish (`Development Status :: 3 - Alpha`); v1 is
the first release-quality tag with a published CHANGELOG, MIGRATION,
CONTRIBUTING guide, and committed lockfile.

---

## Quick checklist

| If you currently … | You will need to … |
|---|---|
| Pin `gpucheck<1` | Read this file before bumping |
| Use `@devices("cuda:0")` only | No change required |
| Want MPS coverage | Add `pip install gpucheck[mps]` and use `@devices()` (auto) or `@devices("mps")` |
| Use `@devices("all")` | Behaviour expanded — now includes MPS on Apple Silicon |
| Construct a `GPUInfo` directly in tests | Note new `backend: str = "cuda"` field — see §1 |
| Call `gpucheck.arch.detection.detect_gpus()` directly | Still works; the v1 API surface is `gpucheck.backends.get_backend()` — see §2 |
| Wrap test code in `tolerance_context(...)` from `pytest-xdist` workers or threads | Now safe — see §4 |
| Have an op known-broken on MPS | Add it to `[tool.gpucheck.mps.xfail]` in `pyproject.toml` — see §5 |
| Want stride / contiguity coverage | Use `fuzz_strides()` or `parametrize_gpu(stride_categories=...)` — see §7 |
| Want reproducible installs | `uv sync --frozen` against the committed `uv.lock` — see §9 |

---

## 1. New `Backend` Protocol

v1.0 introduces a `runtime_checkable` `Backend` Protocol at
`src/gpucheck/backends/_protocol.py`, with two implementations:
`CUDABackend` (`backends/cuda.py`) and `MPSBackend` (`backends/mps.py`).

The Protocol is the **v1 public API for backend introspection**:

```python
from gpucheck import available_backends, get_backend

# v1.0 — list backends present on this host
print(available_backends())          # e.g. ("cuda",) or ("cuda", "mps") or ("mps",)

# Get a specific backend
cuda = get_backend("cuda")
mps  = get_backend("mps")            # raises if MPS not available

# Use the backend's deadlock-safe event timer
with mps.event_timer() as timer:
    run_kernel()
print(timer.elapsed_ms())
```

### Compatibility note

If you wrote against `gpucheck.arch.detection.detect_gpus()` directly,
**that still works** — the v1 `Backend` Protocol is a higher-level entry
point, not a replacement. `detect_gpus()` continues to return a list of
`GPUInfo`. New code should prefer `gpucheck.backends.get_backend()`.

---

## 2. `GPUInfo` shape extension

The `GPUInfo` dataclass in `src/gpucheck/arch/detection.py` gains a single
new field:

```python
@dataclass(frozen=True, slots=True)
class GPUInfo:
    device_id: int
    name: str
    compute_capability: tuple[int, int]
    architecture: str
    memory_total_mb: int
    memory_free_mb: int
    driver_version: str
    cuda_version: str
    supports_fp16: bool
    supports_bf16: bool
    supports_fp8: bool
    supports_tf32: bool
    tensor_core_generation: int | None
    max_shared_memory_per_block: int
    backend: str = "cuda"          # NEW in v1.0 — defaults to "cuda" for back-compat
```

**Migration:**

- All existing fields are unchanged. CUDA-only code is unaffected.
- `backend` defaults to `"cuda"` — code that constructs `GPUInfo(...)`
  without keyword arguments and relies on positional ordering of the
  trailing field is **not** broken because the new field is at the
  end with a default.
- On MPS hosts, `compute_capability`, `cuda_version`, and
  `tensor_core_generation` carry backend-appropriate sentinel values
  (e.g. `(0, 0)`, `""`, `None`). Code that branches on
  `gpu.compute_capability >= (8, 0)` should now also check
  `gpu.backend == "cuda"`:

```python
# v0.1.0
gpu = detect_gpu()
if gpu and gpu.compute_capability >= (8, 0):
    enable_bf16_path()

# v1.0 — same code still runs unchanged on CUDA hosts.
# To handle MPS too:
gpu = detect_gpu()
if gpu and gpu.backend == "cuda" and gpu.compute_capability >= (8, 0):
    enable_bf16_path_cuda()
elif gpu and gpu.backend == "mps":
    enable_mps_path()
```

---

## 3. `@devices("mps")` and `@devices("all")`

In v0, `@devices()` with no arguments auto-detected CUDA devices only.
In v1, **`@devices()` additionally includes `"mps"`** when
`torch.backends.mps.is_available()` returns `True`, and `@devices("all")`
resolves to the union of all detected devices.

```python
# v0 behaviour (still supported)
@devices("cuda:0")
def test_kernel(device): ...

# v1 — auto-detect picks up CUDA AND MPS
@devices()
def test_kernel(device): ...
# device parametrizes over ["cuda:0", "mps"] on a hybrid host

# Explicit MPS
@devices("mps")
def test_metal_kernel(device): ...

# Mix
@devices("cuda:0", "mps")
def test_both(device): ...

# All — explicit
@devices("all")
def test_everywhere(device): ...
```

Tests for unavailable devices skip cleanly via `_is_device_available`.

---

## 4. `tolerance_context()` is now thread-safe

In v0, `src/gpucheck/assertions/tolerances.py` used a module-level list
as the override stack with the inline comment `# NOT thread-safe`. In
v1, the override stack is a `contextvars.ContextVar`:

- `pytest-xdist` workers (already process-isolated) continue to work.
- Concurrent threads inside a single worker no longer leak overrides
  across threads.
- `asyncio` tasks are isolated per the standard `ContextVar` semantics.

**No user-facing API change** — the same `tolerance_context(atol, rtol)`
context manager works exactly as before. Prior code is unchanged.

```python
from gpucheck import tolerance_context

with tolerance_context(atol=1e-3, rtol=1e-3):
    assert_close(actual, expected)   # ← uses overridden tolerances
```

---

## 5. Per-kernel xfail registry

v1 introduces a curated registry of known-broken MPS kernels in
`pyproject.toml`:

```toml
[tool.gpucheck.mps.xfail]
ops = [
  "scaled_dot_product_attention.large",       # pytorch#179352
  "scaled_dot_product_attention.backward",    # pytorch#179294
  "layer_norm.backward.shape1",               # pytorch#173525
  "batch_norm.backward.channels_last",        # pytorch#175189
  "conv2d.large_channels",                    # pytorch#142836
  "conv2d.backward.channels_last_format",     # pytorch#174269
  "F.linear.backward.bf16_3d_nobias_m5",      # pytorch#181936
  "softmax.large_attention",                  # pytorch#96602
  "avg_pool2d.backward.channels_last",        # pytorch#175190
  "binary_ops.uint16_uint32_uint64",          # pytorch#176296
  "BCE_loss",                                 # pytorch#137001
  "matmul.backward.over_32K_elements",        # pytorch#177116
]
```

These 12 entries are drawn from the research SYNTHESIS top-impact open
MPS bugs (see `.claude/teams/research/v1.0/SYNTHESIS.md` §Sub-Q 2).

Public API:

```python
import gpucheck
gpucheck.is_mps_xfailed("softmax.large_attention")    # → True
gpucheck.mps_xfail_list()                             # → ("scaled_dot_product_attention.large", ...)
gpucheck.register_mps_xfail("my_custom.broken_op")    # programmatic registration
```

The xfail registry is a **living document**. Add entries as you discover
new broken kernels; remove entries as upstream issues close. Tolerance
multipliers cannot rescue silent-correctness or crash bugs — that is
what the registry is for.

---

## 6. PROVISIONAL 2× MPS tolerance multiplier

The default tolerances in `src/gpucheck/assertions/tolerances.py` are
calibrated against cuBLAS on Turing/Ampere. v1.0 ships with an MPS
overlay that applies a **2× per-dtype multiplier** for FP32, FP16, and
BF16 on top of the CUDA baseline:

| dtype | CUDA atol | CUDA rtol | MPS atol | MPS rtol | Multiplier |
|---|---|---|---|---|---|
| float32 | 1e-4 | 1e-4 | 2e-4 | 2e-4 | 2× |
| float16 | 1e-2 | 1e-2 | 2e-2 | 2e-2 | 2× |
| bfloat16 | 5e-2 | 5e-2 | 1e-1 | 1e-1 | 2× |
| float64 | 1e-10 | 1e-7 | unchanged | unchanged | 1× |

**This 2× multiplier is PROVISIONAL.** It is grounded in two arguments —
the FlashAttention precision-floor precedent and the absence of FP16
tensor cores on Apple Silicon — but it has not yet been calibrated
against P99 measured drift on Akash's M-machine. See
`.claude/teams/research/v1.0/SYNTHESIS.md` §Sub-Q 7 for the full
calibration plan.

If your kernel exhibits drift > 2× on MPS, the current guidance is to
**move the op to the xfail registry rather than further inflate the
multiplier**. The multiplier covers precision-floor noise; the registry
covers implementation bugs.

The number may change in v1.0.0 final after calibration. Users hard-coding
overlays on top of v1.0.0rc1 should track this CHANGELOG section.

**v0 → v1 migration:** none required for code that uses
`assert_close` defaults. If you previously hard-coded MPS-friendly
multipliers, you can drop them. To force CUDA-style tolerances on an
MPS tensor, use `tolerance_context(atol=..., rtol=...)`.

---

## 7. Stride / contiguity fuzzing — new public API

v1 adds a 7-category stride corpus at `src/gpucheck/fuzzing/strides.py`:

```python
from gpucheck.fuzzing import fuzz_strides, fuzz_strides_for_category, STRIDE_CATEGORIES

# Deterministic corpus across all categories
strides = fuzz_strides(shape=(8, 16, 32), dtype=torch.float32, seed=42)

# One specific category (snake_case is canonical; kebab-case aliases
# such as "broadcast-induced" are accepted with a DeprecationWarning).
broadcast = fuzz_strides_for_category(
    shape=(8, 16, 32), dtype=torch.float32, category="broadcast",
)

# Hypothesis property-based testing
from gpucheck.fuzzing import StrideStrategy
from hypothesis import given

@given(stride=StrideStrategy(shape=(8, 16, 32)))
def test_kernel_any_stride(stride): ...

# Wired into parametrize_gpu
from gpucheck import parametrize_gpu

@parametrize_gpu(
    dtypes=("float32",),
    shapes=((8, 16, 32),),
    devices=("cuda:0",),
    stride_categories=("row_major", "transpose", "broadcast"),
)
def test_my_kernel(dtype, shape, device, stride_category, stride): ...
```

The 7 categories (canonical snake_case): `row_major`, `column_major`,
`broadcast`, `transpose`, `slice`, `non_contig`, `gather`. The previous
kebab-case spellings (`row-major`, `broadcast-induced`,
`contiguous-after-clone`, `gather-induced`) are still accepted but emit
a `DeprecationWarning` and route to the snake_case form.

Pre-v1.0, the project documented this gap as "No stride/contiguity
fuzzing (only shapes and values)" in `CLAUDE.md` "Known Weaknesses".
That gap is closed.

---

## 8. Determinism sanitizer

v1 adds `gpucheck.sanitizers.determinism`:

```python
from gpucheck.sanitizers import assert_deterministic, requires_determinism, DeterminismError

# Function form — fix seeds, re-run `n` times, compare outputs.
# `*args` / `**kwargs` are forwarded to my_kernel.
# Default mode is byte-identical (torch.equal); pass atol=/rtol= to
# opt into tolerance-based determinism (the right contract for MPS).
assert_deterministic(my_kernel, x, y, n=2)
assert_deterministic(my_kernel, x, y, n=3, atol=1e-5)  # MPS-friendly

# Decorator form — gate a test on determinism
@requires_determinism(n=3, seed=0)        # byte-identical mode
def test_kernel_is_reproducible(): ...

@requires_determinism(n=3, atol=1e-5)     # tolerance mode (MPS)
def test_mps_kernel_is_quasi_deterministic(): ...
```

PyTorch's MPS docs are silent on determinism, and the empirical record
shows real run-to-run divergence (pytorch#181936, pytorch#170837,
pytorch#177116). gpucheck-MPS does **not** export a "deterministic
parity" guarantee. Use `assert_deterministic` to verify locally; do not
assume bit-exactness across runs.

---

## 9. `uv.lock` committed (DEP-1 mitigation)

v1.0 ships a committed `uv.lock`. CI installs from the lockfile via
`uv sync --frozen`. Downstream consumers benefit from supply-chain
reproducibility:

```bash
# Reproducible install
uv sync --frozen

# Equivalently, pin to the same dep set with pip:
uv export --no-dev > requirements.lock.txt
pip install -r requirements.lock.txt
```

The `pyproject.toml` floors continue to declare lower bounds; the
lockfile pins exact versions for the release.

---

## 10. New install extras

```bash
pip install gpucheck[mps]           # MPS backend; pins torch>=2.6
pip install gpucheck[apple]         # alias for [mps]
pip install gpucheck[hypothesis]    # property-based shape, tensor, stride strategies (no change)
```

The `[mps]` extra pins `torch>=2.6`, the floor at which
`torch.mps.synchronize()` is stable enough for the deadlock-safe
benchmark path (see CHANGELOG §Track A).

---

## 11. `__version__` bump

```python
# v0.1.0:  src/gpucheck/__init__.py
__version__ = "0.1.0"

# v1.0.0rc1: src/gpucheck/__init__.py
__version__ = "1.0.0rc1"
```

`pyproject.toml` `version` mirrors this.

---

## 12. Removed / deprecated APIs

**v1.0 removes nothing from v0.1.0.** It is API-additive in every case.

If a future release deprecates a symbol, it will appear in this section
with a removal target version.

---

## Cross-references

- [`CHANGELOG.md`](./CHANGELOG.md) — full per-version diff with citations.
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) — dev setup, PR process,
  conventional-commits convention.
- `.claude/teams/research/v1.0/SYNTHESIS.md` — research substrate for
  §1, §5, §6 (xfail registry, tolerance multipliers).
- `.claude/teams/engineering/v1.0/DIFF_LOG.md` — engineering substrate;
  one row per file change across all four v1.0 tracks.
- `.claude/teams/security/v1.0/FINDINGS.md` — security ledger
  (3 MEDIUM mitigated in v1.0: CFG-2, TM-E1, DEP-1).
