# gpucheck v1.0 — API Stability + DX Audit

**Persona:** api-design-dx-lead
**Charter:** depth-expansion #9 — grade every public symbol of `release/v1.0` on five axes (discoverability / type safety / ergonomics / error-message quality / deprecation safety) and identify the top 3 v1.1 fixes.
**Date:** 2026-05-01
**Scope:** `src/gpucheck/__init__.py` v1.0.0rc1 + module-level `__all__` lists.

Score scale: 0/5 (broken) → 5/5 (exemplary; do not change). Each entry cites a load-bearing line in the source.

---

## Surface-area inventory

`gpucheck.__init__._LAZY_MAP` exports **24** names. Every entry has both a `_LAZY_MAP` row AND an `if TYPE_CHECKING` import — the lazy import path and static type checking path are kept in sync (good). The `__all__` list contains the same 24 names + `__version__`.

A second tier of public symbols lives in submodules (`fuzz_strides`, `ShapeStrategy`, `StrideStrategy`, `requires_arch`, `requires_determinism`, `assert_deterministic`, `apply_mps_xfail_config`, `memory_guard`, `check_memory_leaks`, the analysis module) and is reachable via dotted imports but **not** re-exported from the top-level package. This split is the single largest source of DX smell — see §"Top 3 smells".

---

## 1. `assert_close(actual, expected, *, rtol, atol, k_dim, nan_equal, baseline_2x, msg)`

- **Discoverability (5/5):** Top-level export, in `__all__`, in `_LAZY_MAP`, in `if TYPE_CHECKING` block. `from gpucheck import assert_close` works the way users expect.
- **Type safety (3/5):** Signature uses `Any` for `actual`/`expected` because the function legitimately accepts torch.Tensor, numpy.ndarray, cupy.ndarray, and `__cuda_array_interface__` objects. Mypy strict passes, but there are **no `@overload`s** to narrow the return type or constrain mixed input types — a user passing a plain Python `list` only finds out at runtime via `_to_numpy`. `compute_tolerance` returns `tuple[float, float]` cleanly. **Fix:** add three `@overload`s for (Tensor, Tensor) / (ndarray, ndarray) / (Any, Any).
- **Ergonomics (5/5):** Zero-config defaults work; every knob is keyword-only; the `baseline_2x` flag is named after the FlashAttention paper convention; `k_dim=` reads like the user's intent ("inner dim of the matmul"). `close.py:120-139` docstring is accurate and complete.
- **Error message quality (5/5):** The failure path (`close.py:280-283`) prints `"Tensors are not close! (atol=1.00e-02, rtol=1.00e-02; override with atol=/rtol= or use k_dim=/baseline_2x=)"` — names the exact knobs to twiddle, the exact tolerance values, and the underlying mismatch report follows. NaN/Inf branches each print remediation hints (e.g. "Use nan_equal=True to allow matching NaN positions"). This is the gold standard the rest of the API should match.
- **Deprecation safety (4/5):** Keyword-only after `*` is the right call — adding new flags in v1.1 will be backwards-compatible. The one risk: `baseline_2x` is a boolean coupled to the FlashAttention semantic; if Apple/MPS demands a 4× variant we'd need `baseline_multiplier: float` and would have to keep `baseline_2x` as a deprecated alias. Recommend documenting the long-term plan now.

## 2. `compute_tolerance(dtype, *, k_dim, device_type)`

- **Discoverability (4/5):** Exported from `gpucheck`. Issue: there is a name collision — `arch.tensor_cores` defines a different `compute_tolerance` (CLAUDE.md §Known Issues #7). Users who do `from gpucheck.arch.tensor_cores import compute_tolerance` get a different function. Fix in v1.1: rename the arch-side one to `tensor_core_tolerance`.
- **Type safety (3/5):** `dtype: Any` is broad — accepts `torch.dtype`, `np.dtype`, and `str`. A `DtypeLike` Protocol or `TypeAlias` would document this without breaking callers. `device_type: str | None` would be better as `Literal["cuda", "mps", "cpu"] | None`.
- **Ergonomics (4/5):** Keyword-only `k_dim` reads well, `device_type` is awkwardly named — users will type `device=` first and get a TypeError. Recommend a `device=` keyword alias.
- **Error message quality (2/5):** Silently falls back to float32 tolerances on unknown dtype (`tolerances.py:100`). A user passing a typo'd `"flaot16"` gets the wrong tolerance with **no warning**. Recommend a `warnings.warn(stacklevel=2)` with `"unknown dtype 'flaot16'; falling back to float32 defaults"`.
- **Deprecation safety (4/5):** Returning `tuple[float, float]` is a structural commitment — adding a third value (e.g. `ulp_tolerance`) breaks tuple-unpacking callers. v1.1 should return a frozen dataclass or NamedTuple with field access; tuple-unpacking can be preserved via `__iter__`.

## 3. `tolerance_context(atol, rtol)`

- **Discoverability (5/5):** Top-level export.
- **Type safety (5/5):** Backed by `ContextVar` in `tolerances.py:51-53` — thread-safe and asyncio-task-safe. Fixes the "thread-safety issue in tolerance override stack" listed in CLAUDE.md gaps. The `Generator[None, None, None]` return type is correct for a `@contextmanager`.
- **Ergonomics (4/5):** Positional `atol, rtol` is correct, but the call site `with tolerance_context(1e-3, 1e-3):` reads ambiguously — which is which? Recommend keyword-only.
- **Error message quality (n/a):** No failure paths.
- **Deprecation safety (3/5):** Today the override is a single `(atol, rtol)` tuple replacing the dtype-aware lookup entirely. If v1.1 wants per-dtype overrides inside a context (e.g. `tolerance_context(float16=(1e-2, 1e-2), float32=(1e-4, 1e-4))`), the current positional contract becomes a constraint. Reserve `**dtype_overrides: tuple[float, float]` now.

## 4. `@dtypes`, `@shapes`, `@devices`

- **Discoverability (5/5):** All three top-level. Constants (`FLOAT_DTYPES`, `EDGE_SHAPES`, etc.) also exported.
- **Type safety (3/5):** `dtypes` accepts `DtypeArg = str | torch.dtype` — fine — but returns `Callable[..., Any]` because `pytest.mark.parametrize` is itself untyped. The user's typed test signature `def test_x(dtype: torch.dtype)` is **not enforced** by the decorator; mypy can't catch a typo. Fix in v1.1 (non-breaking): `ParamSpec` on a generic decorator wrapper.
- **Ergonomics (5/5):** `@dtypes("float16", "float32")` reads exactly like the user's intent. Lazy `_DtypeGroup` (`dtypes.py:79-95`) means `from gpucheck import FLOAT_DTYPES` doesn't import torch — verified zero-cost-import principle holds.
- **Error message quality (3/5):** A bad dtype string like `@dtypes("flaot16")` raises `AttributeError` at collection ("module 'torch' has no attribute 'flaot16'") — pytest surfaces it but the message points to torch internals, not the gpucheck call site. Wrap and re-raise with `"unknown dtype 'flaot16' passed to @dtypes; valid: float16, float32, ..."`.
- **Deprecation safety (4/5):** `@dtypes(*FLOAT_DTYPES)` and `@dtypes("float16")` shapes are stable. Adding a `device=` kwarg (e.g. for cross-product) would force callers using `**kwargs` to break — but no real callers use that pattern. Safe.

## 5. `@parametrize_gpu`

- **Discoverability (5/5):** Top-level export.
- **Type safety (3/5):** Same `Callable[..., Any]` return as the others. The `skip` callback is dynamically dispatched as 3-arg or 4-arg via `try/except TypeError` (`parametrize.py:135-139`) — this is brittle. Recommend `Callable[[Any, Shape, str], bool] | Callable[[Any, Shape, str, str], bool]` with explicit overloads.
- **Ergonomics (4/5):** Keyword-only with sensible defaults. The `stride_categories=(...)` opt-in is well-named. The mutation of test signature (gains `stride_category` param when `stride_categories` is set) is implicit — users discover it from the docstring or via test failure. Recommend a separate `parametrize_gpu_with_strides()` or split into two decorators.
- **Error message quality (4/5):** `parametrize.py:96-101` raises `ValueError` listing valid stride categories — actionable. Skip predicates emit `"filtered by skip predicate"` — could be more specific.
- **Deprecation safety (3/5):** The signature-mutation behavior is a footgun. Locking the contract now ("parametrize_gpu adds parameters in this order: dtype, shape, device, optionally stride_category") makes future additions tricky.

## 6. `gpu_benchmark`, `memory_tracker`, `gpu_device` (fixtures)

- **Discoverability (4/5):** Listed in `__all__`, registered via plugin. But they are pytest fixtures, not importable callables — `from gpucheck import gpu_benchmark` returns the fixture function, not a runner. Confusing. Document this in the docstring, or better, expose `gpu_benchmark_runner` as the importable.
- **Type safety (4/5):** `BenchmarkResult` is a frozen `@dataclass(slots=True)` — exemplary. `GPUDevice`, `MemoryReport`, `MemorySnapshot` similarly tight. `_BenchmarkRunner.__call__` has full keyword-only signature with `int | None` defaults.
- **Ergonomics (5/5):** `result = gpu_benchmark(my_kernel, x)` reads beautifully. `result.median`, `result.p95`, `result.std` — every field named after the statistic.
- **Error message quality (4/5):** `pytest.skip("No GPU (CUDA or MPS) available for benchmarking")` and the import-error message `"gpu_benchmark requires PyTorch for accurate GPU timing. Install it with: pip install torch"` are both actionable. The MPS `flush_l2=True` warning is correctly tagged `UserWarning` with stacklevel.
- **Deprecation safety (3/5):** `BenchmarkResult` is frozen — adding a field requires bumping minor version (frozen dataclasses don't support `__init_subclass__` extension cleanly). Use `dataclass(frozen=True, kw_only=True)` and add fields with defaults to preserve construction calls.

## 7. `fuzz_shapes`, `fuzz_strides`, `ShapeStrategy`, `StrideStrategy`

- **Discoverability (2/5):** `fuzz_shapes` is exported top-level; `fuzz_strides`, `ShapeStrategy`, `StrideStrategy`, and `STRIDE_CATEGORIES` are **only** reachable via `gpucheck.fuzzing.*`. The CLAUDE.md "Known Issues" #1 and #2 flag exactly this. **Critical fix for v1.1:** add to `_LAZY_MAP`.
- **Type safety (3/5):** `fuzz_strides_for_category(...)` returns `Any` because torch is optional. A `TYPE_CHECKING` `torch.Tensor` annotation would sharpen this without breaking lazy import.
- **Ergonomics (4/5):** `fuzz_shapes(ndim=2, n=50)` reads well. The `ShapeStrategy.__new__` pattern that returns a Hypothesis `SearchStrategy` instead of an instance of the class is **clever but surprising** (`shapes.py:200-207`) — `isinstance(s, ShapeStrategy)` returns False. Type-checkers may also complain. Document or refactor to a factory function.
- **Error message quality (4/5):** `RuntimeError("ShapeStrategy requires hypothesis: pip install gpucheck[hypothesis]")` is exemplary — names the exact pip command. `fuzz_shapes` validates `min_size <= max_size` (`shapes.py:130-131`). `fuzz_strides_for_category` lists valid categories on bad input (`strides.py:198-202`).
- **Deprecation safety (3/5):** The factory-via-`__new__` pattern is hard to evolve — switching to a real class breaks `isinstance` callers (none exist today, but type-checkers' inference will). Lock it to a function in v1.1: `def shape_strategy(...) -> SearchStrategy:`.

## 8. `@requires_arch`, `@requires_determinism`, `assert_deterministic`

- **Discoverability (1/5):** **None of these are top-level exports.** `requires_arch` lives at `gpucheck.arch.compatibility.require_arch` (note: also misnamed — singular `require` vs the docs' `requires`). `assert_deterministic` is at `gpucheck.sanitizers.determinism`. CLAUDE.md "Improvement Priorities" #1-3 flagged exactly this gap. Top-level `from gpucheck import requires_arch` does not work today.
- **Type safety (4/5):** `Callable[[Callable[..., Any]], Callable[..., Any]]` is correct but loose; `ParamSpec` would preserve the wrapped signature. `DeterminismError` subclasses `AssertionError` cleanly.
- **Ergonomics (4/5):** `@require_arch("Ampere", "Hopper")` reads well; arch alias normalization (`compatibility.py:67-74`) handles "Blackwell" → "Blackwell-DC"/"Blackwell-Consumer" correctly. `assert_deterministic(fn, n=3, seed=0)` is intuitive.
- **Error message quality (5/5):** `DeterminismError` (`determinism.py:127-133`) is best-in-class — names the run number, n, seed, and offers three remediations (tolerance_context, MPS xfail, "best-effort determinism per SYNTHESIS §4"). `require_arch`'s skip message names the detected arch + SM tag.
- **Deprecation safety (3/5):** Naming inconsistency (`require_arch` exists, prompt asks for `requires_arch`) is itself a v1.0 contract risk. Pick one and add the other as a `DeprecationWarning` alias before users codify it.

## 9. `Backend`, `get_backend`, `available_backends` (new in v1.0)

- **Discoverability (5/5):** All three top-level. Listed in `_LAZY_MAP`, `__all__`, and `if TYPE_CHECKING`. The new-in-v1.0 status is documented in `backends/__init__.py:1-26`.
- **Type safety (5/5):** `Backend` is a `@runtime_checkable Protocol` with named methods (`backends/_protocol.py:35-73`). `EventTimer` likewise. Structural typing means user code can type-check against `Backend` without depending on the concrete `CUDABackend`/`MPSBackend`. This is the cleanest piece of v1.0 typing.
- **Ergonomics (4/5):** `get_backend("cuda")` reads well; `available_backends()` returns ordered list. Minor: passing the string `"CUDA"` works (`.lower()` normalization at `backends/__init__.py:69`) but isn't documented.
- **Error message quality (4/5):** `RuntimeError("Backend 'cuda' is not available on this system (missing torch, missing hardware, or driver issue)")` is good — three diagnoses listed. Could be sharper if it told the user *which* of the three. `ValueError(f"Unknown backend {name!r}; expected 'cuda' or 'mps'")` is exemplary.
- **Deprecation safety (5/5):** Protocol-based design is the most evolution-friendly choice in the codebase. Adding a method to `Backend` is technically breaking for third-party implementations, but realistically no one ships a custom backend in 1.0. Adding a method with a default implementation can be done via mixin in v1.1.

## 10. `is_mps_xfailed`, `register_mps_xfail`, `apply_mps_xfail_config`, `mps_xfail_list`

- **Discoverability (4/5):** All four exported top-level. Good.
- **Type safety (4/5):** Set-based registry, `op_name: str` exact-match. `apply_mps_xfail_config(config: dict[str, Any])` is loose — a `TypedDict` for the gpucheck pyproject schema would catch typos.
- **Ergonomics (3/5):** Four functions for what could be one `MPSXfailRegistry` object: `register`, `is_xfailed`, `list`, `apply_from_config`. The flat-function design works for v1.0 but doesn't compose (e.g. testing two different xfail sets in one process requires `reset` + `register` mutation, not parallel registries). Module-level mutable state (`_mps_xfail_set: set[str]`) is the same hazard as the old tolerance stack.
- **Error message quality (3/5):** Exact-string match silently returns False on a typo — `is_mps_xfailed("scaled_dot_product_attention.larg")` returns False with no warning even though `large` is in the registry. Recommend a fuzzy-match warning when no match is found and a similar entry exists.
- **Deprecation safety (3/5):** Module-level set is the same anti-pattern that the `tolerance_overrides` ContextVar fixed. Migrating to a `ContextVar[frozenset[str]]` would be source-compatible but allocator-different.

---

## Average score per axis

| Axis                | Avg | Notes                                                                 |
|---------------------|-----|-----------------------------------------------------------------------|
| Discoverability     | 3.8 | Pulled down by fuzzing, sanitizers, arch decorators not top-level     |
| Type safety         | 3.7 | Loose `Any`s on tensor inputs; no `@overload`s; factory-via-`__new__` |
| Ergonomics          | 4.3 | Strong overall; keyword-only patterns + sensible defaults             |
| Error message quality | 3.8 | `assert_close` and `DeterminismError` are gold; `compute_tolerance` silently swallows typos |
| Deprecation safety  | 3.6 | Frozen tuples + factory-via-`__new__` + module-level state are evolution-blockers |

**Overall: 3.8/5.** v1.0 ships solid; v1.1 has clear, additive headroom.

---

## Top 3 API smells to fix in v1.1 (without breaking v1.0)

### Smell 1 — Hidden public surface

`fuzz_strides`, `StrideStrategy`, `ShapeStrategy`, `STRIDE_CATEGORIES`, `requires_arch`/`require_arch`, `requires_determinism`, `assert_deterministic`, `memory_guard`, `check_memory_leaks`, `apply_mps_xfail_config` (some), `tolerances_from_config`, `mps_xfail_from_config`, the entire `analysis/` and `reporting/` modules — all reachable only via dotted imports. The `__init__.py` exports 24 names but the actually-public surface is closer to 40. Users either don't find the symbol or develop a habit of `from gpucheck.fuzzing.strides import ...` which then ossifies the implementation path. **Fix:** add to `_LAZY_MAP` + `__all__` + `if TYPE_CHECKING`. Pure addition, zero break risk.

### Smell 2 — `Any` everywhere on the tensor surface

`assert_close(actual: Any, expected: Any)`, `compute_tolerance(dtype: Any)`, `fuzz_strides_for_category(...) -> Any`, `_to_numpy(tensor: Any)`. Justified by lazy torch import and multi-backend support, but it leaves users without IDE autocomplete and lets bugs through (e.g. passing a `list` → unhelpful numpy traceback). **Fix:** introduce a `TensorLike` Protocol with `.detach`, `.shape`, `.dtype`, `.device` (structural; no runtime cost), add `@overload`s for the common (Tensor, Tensor) and (ndarray, ndarray) call patterns, and `Literal["cuda", "mps", "cpu"]` for `device_type`. Backwards compatible — only sharpens the inference.

### Smell 3 — Module-level mutable state in tolerance/xfail registries

`_mps_xfail_set: set[str]` in `tolerances.py:57` and `_config_overrides: dict` at line 161 are module-level mutable singletons. The tolerance *override stack* was correctly migrated to `ContextVar` (CLAUDE.md "Known Weaknesses"); the xfail set and config overlay weren't. They leak across pytest sessions in `pytest --forked` setups, can't be tested in parallel, and have no thread isolation. **Fix:** mirror the ContextVar pattern from `_tolerance_overrides`. Backwards-compatible — `register_mps_xfail` and `apply_config_tolerances` keep their exact signatures; only the underlying storage moves.

---

## Top 3 strengths to preserve

1. **`assert_close` failure messages** — `close.py:280-283` literally tells the user "override with atol=/rtol= or use k_dim=/baseline_2x=". This is the highest-quality DX surface in the project; copy the pattern to every other failure path.
2. **Lazy import discipline** — `_LAZY_MAP` + `if TYPE_CHECKING` + `_DtypeGroup` lazy resolution + `_torch_mod()` helpers throughout. `import gpucheck` triggers zero torch/pynvml/hypothesis imports. Verified by inspection of every `__init__.py`. Do not regress this in v1.1.
3. **Backend Protocol** — `backends/_protocol.py` is the cleanest typing surface in the codebase: `@runtime_checkable Protocol` with named methods, structural conformance, evolution-friendly. Use this as the template for v1.1 `Reporter`/`Sanitizer` abstractions if they ever need a public contract.

---

## Closing recommendation

v1.0 is **shippable** as-is. The audit identified zero hard blockers — every smell is fixable additively. Recommend the three v1.1 fixes above land in a single PR within ~6 weeks of v1.0 release; they unlock cleaner downstream UX without forcing any user code change.
