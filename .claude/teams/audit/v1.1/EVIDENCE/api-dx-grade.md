# gpucheck v1.0 Public API DX Grade

**Auditor:** api-design-dx-lead persona
**Branch:** release/v1.0 (commit a9a9d44)
**Scope:** every public symbol reachable via `import gpucheck` or `from gpucheck.<sub> import ...`
**Axes:** Discoverability / Type safety / Ergonomics / Error-message quality / Deprecation safety (0–5 each)

Sources of truth:
- `src/gpucheck/__init__.py` — top-level lazy map + `__all__`
- `src/gpucheck/<sub>/__init__.py` — submodule re-exports
- The actual symbol implementations cited per row.

A score of `3` is "ships-quality, room to improve". `5` is "I would point to this in a talk." `0–1` flags an active wart.

---

## 1. `assert_close` — the headline

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | In `__init__._LAZY_MAP`, in `__all__`, in `TYPE_CHECKING` block, doc'd as the headline. `gpucheck.<TAB>` reveals it. |
| Type safety | 2 | Signature is `actual: Any, expected: Any` with no overloads. mypy users get no narrowing for `torch.Tensor` vs `np.ndarray` vs `cupy.ndarray`. Missing `TypeAlias`/`TensorLike` Protocol. The internal `_to_numpy` already enumerates the supported shapes — that information should surface as a `Protocol`. |
| Ergonomics | 4 | `assert_close(actual, expected, rtol=..., atol=..., k_dim=..., baseline_2x=True)` reads better than `torch.testing.assert_close` (which doesn't know about matmul k_dim) and `np.testing.assert_allclose` (which has no nan_equal default). Loses one point for `baseline_2x` being a magic boolean — `tolerance="flash_attention_2x"` would self-document. |
| Error-message quality | 4 | Failure message includes the actual atol/rtol values **and** the override hint: `"override with atol=/rtol= or use k_dim=/baseline_2x="`. NaN failure message tells you exactly which kwarg unblocks it. Loses one point because `format_mismatch_report` output isn't shown in the error itself for shape-mismatch (just the shapes); a histogram would help. Example: `"Tensors are not close! (atol=1.00e-02, rtol=1.00e-02; override with atol=/rtol= or use k_dim=/baseline_2x=)"` is excellent. |
| Deprecation safety | 4 | All knobs are keyword-only (`*` separator). New flags can be added without breaking callers. Risk: `baseline_2x: bool` will be hard to remove if we ever promote it to a string enum. |

**Avg: 3.8** — strong headline; type safety is the soft spot.

---

## 2. `compute_tolerance`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 4 | Exported, lazy-mapped, in `__all__`. `tensor_cores.compute_tolerance` shadows the name internally — collision risk noted in AGENT.md but doesn't yet hit users. |
| Type safety | 3 | `dtype: Any` is too permissive — it silently falls back to float32 for typos. A `Literal["float16","float32",...]` union or a `DType` Protocol would catch `compute_tolerance("flaot16")` at lint time. Return is correctly `tuple[float, float]`. |
| Ergonomics | 4 | `compute_tolerance(torch.float16, k_dim=1024, device_type="mps")` reads cleanly. Keyword-only k_dim and device_type are correct. `device_type: str` is a stringly-typed enum candidate. |
| Error-message quality | 1 | Silently returns float32 defaults for unknown dtypes. No warning, no error. A test that typos `"flaot16"` will pass with the wrong tolerance and the user will never know. **Top fix candidate.** |
| Deprecation safety | 4 | Keyword-only kwargs make adding new ones safe. |

**Avg: 3.2** — silent fallback on unknown dtype is the bug.

---

## 3. `tolerance_context`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 4 | In `__all__`. Naming is consistent with Python's stdlib `decimal.localcontext()`. |
| Type safety | 4 | `(atol: float, rtol: float) -> Generator[None, None, None]`. Backed by `ContextVar` so it's correctly typed for asyncio/threads (recent fix). |
| Ergonomics | 3 | `with tolerance_context(atol=1e-3, rtol=1e-3):` is clear, but the override is *absolute* — it ignores dtype, k_dim, and MPS multipliers entirely (see `tolerances.py:91-93`: `if overrides: return overrides[-1]`). A user expecting "double the defaults" gets a fixed scalar. **Should be `tolerance_context(scale=2.0)` or `tolerance_context(atol_factor=, rtol_factor=)`.** |
| Error-message quality | 2 | No errors emitted by the context manager itself; surprising-override behavior produces silent test passes. No log line saying "atol overridden globally → 1e-3". |
| Deprecation safety | 3 | Adding a third positional `scale=` kwarg is non-breaking; making it positional-only would break callers using `atol=`/`rtol=`. |

**Avg: 3.2** — the absolute-override semantics are a footgun.

---

## 4. `@dtypes`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | `gpucheck.dtypes`, plus `FLOAT_DTYPES`/`HALF_DTYPES`/`ALL_DTYPES`/`FP8_DTYPES` constants in `__all__`. Pytest plugin convention. |
| Type safety | 3 | `dtype_args: DtypeArg = str | torch.dtype`. At collection time strings stay as strings (smart, avoids torch import). Missing `Literal` for known dtype names — would catch typos. |
| Ergonomics | 5 | `@dtypes("float16", "float32")` and `@dtypes(*FLOAT_DTYPES)` both read perfectly. Better than `pytest.mark.parametrize("dtype", [...])`. |
| Error-message quality | 3 | A typo'd dtype string hits `_resolve_dtype` at test execution and crashes with `AttributeError: module 'torch' has no attribute 'flaot16'` — not actionable. Should validate at decoration with a helpful "did you mean float16?". |
| Deprecation safety | 5 | `*args` accepts anything; no positional/keyword conflicts; predefined groups are tuples and additive. |

**Avg: 4.2** — strongest decorator; only weakness is typo handling.

---

## 5. `@shapes`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | All four groups (`SMALL_SHAPES`/`MEDIUM_SHAPES`/`LARGE_SHAPES`/`EDGE_SHAPES`) re-exported. |
| Type safety | 4 | `Shape = tuple[int, ...]` alias is clean. Could be tighter — `tuple[PositiveInt, ...]` to ban `(-1, 128)` early, though that's a `pydantic`-grade ask. |
| Ergonomics | 5 | `@shapes((128, 128), (256, 256))` reads exactly as intended. `_shape_id` produces clean `"128x256"` test IDs (better than pytest's default `(128, 128)0`). |
| Error-message quality | 3 | No validation — `@shapes(128)` (forgot the tuple) would pass through and fail later. `@shapes((-1, 128))` produces a confusing tensor allocation error downstream. |
| Deprecation safety | 5 | `*args` of tuples; new kwargs are non-breaking. |

**Avg: 4.4** — one of the cleanest pieces of API in the project.

---

## 6. `@devices`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | Top-level export. The "all" sentinel is documented. |
| Type safety | 2 | `*device_args: str` — accepts any string. No `Literal["cuda", "cuda:0", "mps", "cpu", "all"]`. A typo like `"cudo:0"` becomes a skipped test silently (because `_is_device_available` returns False). **Same silent-failure pattern as `compute_tolerance`.** |
| Ergonomics | 4 | `@devices("cuda:0", "mps")`, `@devices()` for auto, `@devices("all")` — three intuitive modes. The "no args = auto" overload is slightly magical. |
| Error-message quality | 2 | A typo silently skips with `"device cudo:0 not available"` — looks like a hardware issue, not a typo. **Top fix candidate: validate device strings against a known set, fall back to `torch.device(...)` only after explicit allow-list miss.** |
| Deprecation safety | 4 | Adding new sentinels (`"rocm"`, `"xpu"`) is additive. |

**Avg: 3.4** — the silent typo skip is the wart.

---

## 7. `@parametrize_gpu`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | The "do everything" decorator users will reach for. Top-level export. |
| Type safety | 3 | `dtypes: Sequence[DtypeArg]`, `shapes: Sequence[Shape]`, `devices: Sequence[str] \| None`, `skip: SkipFilter`. The `skip` callback signature changes (3 args vs 4 args) based on `stride_categories` — typed as `Callable[..., bool] \| None` because the real type can't be expressed without `Protocol` overloads. mypy strict won't catch a 3-arg skip used with 4-arg parametrize. |
| Ergonomics | 4 | The all-keyword API and explicit cartesian product are great. The overloaded signature when `stride_categories` is set (test gains a fourth fixture parameter) is implicit — discoverable only via docstring. |
| Error-message quality | 4 | Validates `stride_categories` eagerly at decoration time with a useful error: `"Unknown stride categories: ['col_major']; expected from ['broadcast', ...]"`. This is exactly the actionable pattern the rest of the API should adopt. |
| Deprecation safety | 3 | `stride_categories` was added post-v1.0 as keyword-only — fine. But the test signature change (3 → 4 fixtures) is implicit; if we ever add another optional axis, signatures balloon. |

**Avg: 3.8** — well-designed; signature-overload limitation is real.

---

## 8. `gpu_benchmark` (fixture)

| Axis | Score | Note |
|---|---|---|
| Discoverability | 4 | Registered via pytest entry point + re-exported. `gpu_benchmark` (snake_case fixture, vs `BenchmarkResult` PascalCase type) is on-convention. |
| Type safety | 4 | `__call__` is fully typed including the `KernelCallable` Protocol. `BenchmarkResult` is a frozen slotted dataclass — perfect. The `flush_l2: bool \| None` triple-state (None=use-runner-default) is slightly clunky; a sentinel `Default` enum would be cleaner. |
| Ergonomics | 4 | `result = gpu_benchmark(my_kernel, x, warmup=20, rounds=200)` reads cleanly. `result.median`, `result.p95` etc. are obvious. Loses a point because there's no `result < 1.0` shorthand — users compare `result.median < 1.0`. A `__lt__` overload (compare by median) would feel pythonic; but is also confusing. Status quo defensible. |
| Error-message quality | 3 | `pytest.skip("No GPU (CUDA or MPS) available for benchmarking")` is good. The "all samples removed as outliers" warning is helpful. But: a user calling `gpu_benchmark(broken_kernel)` where the kernel raises gets a raw exception — would benefit from a "benchmark wrapper context: ran 3/100 rounds before exception". |
| Deprecation safety | 4 | All call kwargs are keyword-only or positional-after-fn. Can add new axes (e.g. `cooldown_ms=`) safely. |

**Avg: 3.8** — solid fixture; minor polish opportunities.

---

## 9. `memory_tracker` (fixture)

| Axis | Score | Note |
|---|---|---|
| Discoverability | 4 | Re-exported from `fixtures/__init__.py` lazy map but **not** in top-level `gpucheck.__init__._LAZY_MAP`. Users have to know to do `from gpucheck.fixtures import memory_tracker`. **Top-level discovery gap.** |
| Type safety | 4 | `MemoryTracker.start/stop/report` are typed; `MemoryReport` and `MemorySnapshot` are frozen slotted dataclasses. |
| Ergonomics | 3 | The fixture **auto-starts** in the fixture body, but if the user calls `tracker.stop()` themselves, the report is delivered, otherwise the teardown does it and only emits a `RuntimeWarning`. That dual mode is surprising — a user who forgets `.stop()` gets a passing test plus a warning instead of a hard failure. Compared to `pytest-benchmark`'s `benchmark.pedantic(...)` API, ours feels half-finished. |
| Error-message quality | 2 | Leak detected → `RuntimeWarning("GPU memory leak detected: 1.5MB not freed")`. Warnings can be silenced; a leak should produce a *test failure*, not a warning. |
| Deprecation safety | 3 | `MemoryTracker.__init__(device_id, leak_threshold)` uses positional args — promoting to keyword-only is breaking. |

**Avg: 3.2** — discoverability and silent-warning behaviour are the issues.

---

## 10. `gpu_device` (fixture)

| Axis | Score | Note |
|---|---|---|
| Discoverability | 3 | Like `memory_tracker`, missing from top-level lazy map. Available as a fixture name to pytest, but `gpucheck.gpu_device` raises AttributeError. |
| Type safety | 5 | Returns a `GPUDevice` (frozen slotted dataclass) with explicit fields. `compute_capability: tuple[int, int]` is precise. |
| Ergonomics | 4 | `def test_x(gpu_device): assert gpu_device.compute_capability >= (8, 0)` is exactly what users want. |
| Error-message quality | 4 | `pytest.skip("No GPU available")` is fine. Could include "(checked pynvml + torch.cuda)" so users know what's expected. |
| Deprecation safety | 5 | Adding fields to the frozen dataclass is a soft-break only for code matching by `__match_args__`; positional unpack is undocumented. |

**Avg: 4.2** — clean type and clean fixture.

---

## 11. `fuzz_shapes`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | Top-level export. |
| Type safety | 4 | `(ndim: int = 2, *, min_size: int, max_size: int, n: int, seed: int \| None)` — fully typed. Returns `list[tuple[int, ...]]`. |
| Ergonomics | 5 | `fuzz_shapes(ndim=2, n=50, seed=42)` reads great. The priority categorization in the docstring (degenerate > non-tile-aligned > prime > pow2 > large > mixed) is documentation as design. |
| Error-message quality | 5 | Validates `min_size > max_size` and `ndim < 0` with explicit messages including the offending values. Reference quality. |
| Deprecation safety | 5 | All kwargs keyword-only; additive. |

**Avg: 4.8** — best-in-class. Pin this as the template.

---

## 12. `fuzz_strides` / `fuzz_strides_for_category`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 2 | **Not** in top-level `__init__`. Live at `gpucheck.fuzzing.fuzz_strides`. AGENT.md flags it. |
| Type safety | 3 | `dtype: Any` (because torch isn't import-time available); `device: str = "cpu"` is stringly typed. Returns `list[tuple[str, Any]]` — that `Any` should be `torch.Tensor` under TYPE_CHECKING. |
| Ergonomics | 3 | The two-function split (`fuzz_strides` returns the corpus, `fuzz_strides_for_category` returns one) is correct but the names are too similar — calling the wrong one is easy. Compare to `fuzz_shapes` which has one entry point. |
| Error-message quality | 5 | `"Unknown stride category 'col_major'; expected one of [...]"` — sorted suggestion, actionable. |
| Deprecation safety | 4 | Categories tuple is exported; adding categories is additive (existing category-keyed code keeps working). |

**Avg: 3.4** — top-level discoverability is the v1.1 fix.

---

## 13. `ShapeStrategy`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 2 | Re-exported from `gpucheck.fuzzing` but **not** from top-level `gpucheck`. Users searching `gpucheck.<TAB>` won't find it. AGENT.md flags this. |
| Type safety | 2 | `__new__` returns `Any` because Hypothesis `SearchStrategy` isn't always importable. Class-as-factory pattern (`__new__` returning a non-`Self`) breaks `isinstance(s, ShapeStrategy)` and confuses mypy. |
| Ergonomics | 4 | `@given(shape=ShapeStrategy(ndim=2, max_size=512))` reads as intended. The "look like a class, behave like a strategy factory" pattern is clever but unusual. |
| Error-message quality | 4 | Missing-hypothesis raises `RuntimeError("ShapeStrategy requires hypothesis: pip install gpucheck[hypothesis]")`. Excellent — names the extra. |
| Deprecation safety | 2 | The `__new__`-returns-strategy pattern locks us in: we can never make `ShapeStrategy` actually behave as a class without breaking callers who treat the result as a `SearchStrategy`. |

**Avg: 2.8** — the cleverness is hurting us. Consider adding a `shape_strategy(...)` function as the canonical name.

---

## 14. `StrideStrategy`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 2 | Same problem as `ShapeStrategy` — only via `gpucheck.fuzzing`. |
| Type safety | 2 | Same `__new__` factory pattern; same mypy/`isinstance` issues. |
| Ergonomics | 4 | `@given(t=StrideStrategy(shape=(64,64), dtype=torch.float32))` reads cleanly. Hypothesis shrinks toward `row_major`. |
| Error-message quality | 4 | Missing-hypothesis message points at the extra. |
| Deprecation safety | 2 | Inherits the `__new__` lock-in. |

**Avg: 2.8** — same fixes as `ShapeStrategy`.

---

## 15. `@requires_arch` (compatibility module)

| Axis | Score | Note |
|---|---|---|
| Discoverability | 2 | Lives at `gpucheck.arch.compatibility.require_arch` — note **`require_arch`** (singular `require`) not `requires_arch`. The audit spec says `@requires_arch`; the actual code uses `require_arch`. **Naming inconsistency vs `requires_determinism`** — one says `requires`, the other says `require`. |
| Type safety | 3 | `*archs: str` — stringly typed. A typo silently skips ("requires architecture Foo, but found Ada"). |
| Ergonomics | 4 | `@require_arch("Ampere", "Hopper")` reads well. Case-insensitive match + alias expansion (`"Blackwell"` → DC + Consumer) is thoughtful. |
| Error-message quality | 4 | Skip reason includes the SM tag and detected arch: `"Requires architecture Hopper, but found Ada (SM89)"`. Good. |
| Deprecation safety | 2 | Renaming `require_arch` → `requires_arch` for consistency with `requires_determinism` is breaking. We're stuck with the inconsistency unless we add the alias and deprecate. |

**Avg: 3.0** — the `require` vs `requires` split is the real wart.

---

## 16. `@requires_determinism` / `assert_deterministic`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 2 | Not in top-level `__init__._LAZY_MAP`. Reachable via `gpucheck.sanitizers.requires_determinism`. The AGENT.md lists this as a known gap. |
| Type safety | 4 | `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)` is fully typed. `requires_determinism()` returns `Callable[[Callable], Callable]` — proper decorator type. |
| Ergonomics | 5 | `@requires_determinism(n=5, seed=42)` and `assert_deterministic(my_fn, x)` both read well. The dual API (decorator + function) covers both styles. |
| Error-message quality | 5 | The `DeterminismError` message is exemplary: includes run number, n, seed, mentions MPS best-effort caveat, and points at *three remediations* (`tolerance_context`, MPS xfail, accepting precision floor). This is what every error message should look like. |
| Deprecation safety | 4 | Keyword-only kwargs; additive. |

**Avg: 4.0** — top-tier error message; only loss is top-level discoverability.

---

## 17. `Backend`, `get_backend`, `available_backends`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 5 | All three in top-level `__all__`. |
| Type safety | 5 | `Backend` is a `@runtime_checkable Protocol` — structural typing done right. `EventTimer` is a separate Protocol. Methods all explicitly typed. This is the model for the rest of the codebase. |
| Ergonomics | 4 | `cuda = get_backend("cuda")` and `for b in available_backends(): ...` are both natural. `get_backend("cuda")` raises if unavailable; `available_backends()` filters — appropriate split. The `name: str` parameter could be `Literal["cuda", "mps"]` to give IDE autocomplete. |
| Error-message quality | 5 | `RuntimeError("Backend 'cuda' is not available on this system (missing torch, missing hardware, or driver issue)")` enumerates the three causes. `ValueError("Unknown backend 'foo'; expected 'cuda' or 'mps'")` lists the valid values. |
| Deprecation safety | 5 | Adding a new backend is a Protocol implementation, not a signature change. Pristine extensibility. |

**Avg: 4.8** — the **gold-standard** API in the codebase. Use as the template for v1.1 refactors.

---

## 18. MPS xfail trio: `is_mps_xfailed`, `register_mps_xfail`, `apply_mps_xfail_config`

| Axis | Score | Note |
|---|---|---|
| Discoverability | 4 | `is_mps_xfailed` and `register_mps_xfail` in top-level `__all__`. **`apply_mps_xfail_config` is in `gpucheck.assertions.__all__` but missing from the top-level `__init__._LAZY_MAP`** — inconsistent. |
| Type safety | 4 | `is_mps_xfailed(op_name: str) -> bool` and `register_mps_xfail(*ops: str) -> None` are clean. `apply_mps_xfail_config(config: dict[str, Any])` — that `Any` should be a `TypedDict` for the pyproject section (`MPSConfigSection`). |
| Ergonomics | 3 | `if is_mps_xfailed("scaled_dot_product_attention.large"): pytest.xfail(...)` requires the test author to do the dispatch. A `@xfail_on_mps("op.subcategory")` decorator wrapping the boilerplate would be more pytest-idiomatic. The current API exposes plumbing where a marker would be the right altitude. |
| Error-message quality | 2 | No errors emitted by `is_mps_xfailed` (returns False for unknown). `register_mps_xfail("typo")` silently registers the typo. **No way for the user to typo-check their pyproject xfail list against an op registry.** |
| Deprecation safety | 4 | All three functions take strings; additive evolution OK. The `_mps_xfail_set` module global is private. |

**Avg: 3.4** — mid-tier; the missing `@xfail_on_mps` decorator is the v1.1 ergonomics fix.

---

## Summary Table (axis means rounded to 1 decimal)

| Symbol | Disc | Type | Ergo | Err | Depr | Avg |
|---|---|---|---|---|---|---|
| `assert_close` | 5 | 2 | 4 | 4 | 4 | 3.8 |
| `compute_tolerance` | 4 | 3 | 4 | 1 | 4 | 3.2 |
| `tolerance_context` | 4 | 4 | 3 | 2 | 3 | 3.2 |
| `@dtypes` | 5 | 3 | 5 | 3 | 5 | 4.2 |
| `@shapes` | 5 | 4 | 5 | 3 | 5 | 4.4 |
| `@devices` | 5 | 2 | 4 | 2 | 4 | 3.4 |
| `@parametrize_gpu` | 5 | 3 | 4 | 4 | 3 | 3.8 |
| `gpu_benchmark` | 4 | 4 | 4 | 3 | 4 | 3.8 |
| `memory_tracker` | 4 | 4 | 3 | 2 | 3 | 3.2 |
| `gpu_device` | 3 | 5 | 4 | 4 | 5 | 4.2 |
| `fuzz_shapes` | 5 | 4 | 5 | 5 | 5 | 4.8 |
| `fuzz_strides*` | 2 | 3 | 3 | 5 | 4 | 3.4 |
| `ShapeStrategy` | 2 | 2 | 4 | 4 | 2 | 2.8 |
| `StrideStrategy` | 2 | 2 | 4 | 4 | 2 | 2.8 |
| `@require_arch` | 2 | 3 | 4 | 4 | 2 | 3.0 |
| determinism | 2 | 4 | 5 | 5 | 4 | 4.0 |
| Backend trio | 5 | 5 | 4 | 5 | 5 | 4.8 |
| MPS xfail trio | 4 | 4 | 3 | 2 | 4 | 3.4 |

**Per-axis means across 18 symbols:**

- Discoverability: **3.7**
- Type safety: **3.4**
- Ergonomics: **4.0**
- Error-message quality: **3.4**
- Deprecation safety: **3.8**

Lowest weakest-link: type safety + error-message quality tied at 3.4. Strongest: ergonomics 4.0.

---

## Top 3 v1.1 Fixes (additive only, no v1.0 breakage)

### Fix 1 — Validate dtype + device strings; loud failure on typo
**Affects:** `compute_tolerance`, `@devices`, `register_mps_xfail`, `@dtypes` (decoration time).
**Why:** four of the five worst error-message scores trace to the same root: stringly-typed inputs + silent fallback. A typo'd `"flaot16"` returns float32 tolerances; `"cudo:0"` skips silently; `register_mps_xfail("layernor")` registers a typo into the pyproject contract.
**How without breaking v1.0:** add `strict: bool = False` kwarg defaulting to current behavior; change default to `True` in v1.2 with a `DeprecationWarning` in v1.1 when fallback fires. Emit `DidYouMeanError` style messages: `"Unknown dtype 'flaot16'; did you mean 'float16'? Valid: float16, float32, ..."`.

### Fix 2 — Promote `memory_tracker`, `gpu_device`, `fuzz_strides`, `ShapeStrategy`, `StrideStrategy`, `requires_determinism`, `assert_deterministic` to top-level `_LAZY_MAP`
**Why:** `gpucheck.<TAB>` is the discoverability test; today seven public symbols fail it. AGENT.md flags this. Pure additive change.
**How:** extend `_LAZY_MAP` and `__all__` in `__init__.py`; mirror in `TYPE_CHECKING` imports. Zero risk.

### Fix 3 — Add `tolerance_context(scale=...)` overload and `@xfail_on_mps(op_name)` decorator
**Why:** the two ergonomic gaps where today's API exposes plumbing instead of intent. `tolerance_context(scale=2.0)` matches the FlashAttention `baseline_2x` pattern but as a context. `@xfail_on_mps("op.subcategory")` wraps the `is_mps_xfailed` + `pytest.xfail` boilerplate.
**How:** new keyword-only param `scale: float | None = None` to `tolerance_context` (mutually exclusive with `atol`/`rtol`); new decorator in `gpucheck.assertions`. Pure addition.

---

## Top 3 Strengths to Preserve

1. **The Backend Protocol pair (`Backend`, `EventTimer`).** `@runtime_checkable` Protocol with explicit method types, error messages enumerating failure modes, and trivial extensibility for ROCm/XPU. Use this as the template when refactoring `compute_tolerance` / `_to_numpy` to use a `TensorLike` Protocol.

2. **The `DeterminismError` message.** Explicit run number, seed, n, *and* three named remediations (`tolerance_context`, MPS xfail config, precision-floor acceptance). Every error message in v1.1 should be benchmarked against this one.

3. **The `fuzz_shapes` design.** One entry point, keyword-only knobs, priority-ordered category docstring (degenerate > non-tile-aligned > prime > pow2 > large > mixed), eager validation with actionable messages. The shape of every future fuzzer (`fuzz_dtypes`, `fuzz_devices`) should mirror this.

---

## Candidate API to Deprecate

**`baseline_2x: bool` keyword on `assert_close`.** It's a magic boolean that hardcodes the FlashAttention 2x convention; it can't express 1.5x or other scales; and it conflicts subtly with `atol=`/`rtol=` overrides (the code path branches on whether the user set both). Replace with `tolerance_scale: float | None = None` (or `tolerance_profile: Literal["default", "flash_attention", ...]`). Keep `baseline_2x` for v1.1 with a `DeprecationWarning`; remove in v1.2. Net win: documents intent, allows non-2x scales, removes a special-case branch in `assert_close`'s tolerance computation.
