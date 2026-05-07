# gpucheck v1.1-prep — File-by-File Structural Audit

Repo: `/Users/cero/Code/gpucheck` (branch `release/v1.0`, tip `82b853e`)
Files audited: 41 `.py` under `src/gpucheck/`
Auditor charter: 8-point checklist per file (sections omitted when nothing to flag).

Citation form: `relative/path:line` (paths relative to `src/gpucheck/`).

---

## `__init__.py`

1. **Public API:** `__version__`, `assert_close`, `compute_tolerance`, `tolerance_context`, `is_mps_xfailed`, `mps_xfail_list`, `register_mps_xfail`, `dtypes`, `shapes`, `devices`, `parametrize_gpu`, `fuzz_shapes`, `GPUInfo`, `detect_gpu`, `gpu_available`, `gpu_count`, `BenchmarkResult`, `GPUDevice`, `available_backends`, `get_backend`, `Backend`, `FLOAT_DTYPES`, `HALF_DTYPES`, `ALL_DTYPES`, `FP8_DTYPES`, `SMALL_SHAPES`, `MEDIUM_SHAPES`, `LARGE_SHAPES`, `EDGE_SHAPES`. (`__init__.py:75-105`)
2. **Dead code:** `apply_mps_xfail_config` and `reset_mps_xfail` are exported by `assertions/__init__.py:22-25` but **not** in `_LAZY_MAP` here, while `register_mps_xfail` / `mps_xfail_list` / `is_mps_xfailed` are. Asymmetric coverage. (`__init__.py:11-40` vs `assertions/__init__.py:16-25`)
4. **Stale docstrings:** Module docstring is one line; no per-symbol docs (acceptable for a lazy re-export module).
6. **Naming:** `_LAZY_MAP` is shouty-snake (correct); all keys are public symbols.

## `plugin.py`

1. **Public API:** `pytest_addoption`, `pytest_configure`, `pytest_collection_modifyitems`, `pytest_terminal_summary`, `gpu_benchmark` (fixture), `gpu_device` (fixture), `memory_tracker` (fixture). (`plugin.py:25,46,91,107,127,137,189`)
2. **Dead code:** `_lazy_detect_gpus` and the `_gpu_available` / `_gpu_count` shims (`plugin.py:10-22`) duplicate `arch.gpu_available` / `arch.gpu_count` (`arch/__init__.py:10-17`). Two implementations of the same logic.
3. **Type smells:** `_lazy_detect_gpus() -> list[Any]` (`plugin.py:10`) — the underlying `detect_gpus()` returns `list[GPUInfo]`. Loss of type info. `pytest_terminal_summary(terminalreporter: Any, ...)` (`plugin.py:107`) — could be typed as `TerminalReporter`. Both fixtures return `Any` (`plugin.py:128,138,190`).
4. **Stale docstrings:** `pytest_addoption`, `pytest_configure`, `pytest_collection_modifyitems`, `pytest_terminal_summary` have no docstring at all (`plugin.py:25,46,91,107`).
5. **Boundary errors:** `_load_pyproject_config` swallows **all** exceptions silently (`plugin.py:86-88`). Fine intentionally, but a debug-level log entry would help diagnostics. The `import torch` inside `gpu_device` (`plugin.py:151`) is unguarded — if the user passes `--gpu-device cuda:1` and torch isn't installed, this raises `ImportError` instead of `pytest.skip`.
6. **Naming:** `gpu_count` shadowed as local variable inside `pytest_collection_modifyitems` (`plugin.py:95`) — the local shadows the module-level `_gpu_count` shim. Minor confusion risk.
8. **Lines to remove:** None.

## `assertions/__init__.py`

1. **Public API:** `assert_close`, `compute_tolerance`, `tolerance_context`, `is_mps_xfailed`, `mps_xfail_list`, `apply_mps_xfail_config`, `register_mps_xfail`, `reset_mps_xfail`. (`assertions/__init__.py:16-25`)
6. **Naming:** `apply_mps_xfail_config` and `reset_mps_xfail` are exported here but missing from top-level `gpucheck.__init__._LAZY_MAP`. Either add them or document them as internal-only.

## `assertions/close.py`

1. **Public API:** `assert_close`. (`assertions/close.py:109`)
2. **Dead code:** `_to_numpy` has a fallback torch-via-dlpack branch (`close.py:53-64`) duplicating logic from lines 28-38 — only reachable if `cupy` ImportErrors AND the object has `__cuda_array_interface__`. Tightly scoped, but worth a comment.
3. **Type smells:** `_to_numpy(tensor: Any) -> npt.NDArray[Any]` (`close.py:22`) — the `Any` is unavoidable for tensor-like polymorphism. `_resolve_dtype` returns `Any` (`close.py:76`). `assert_close` parameters typed as `Any` (`close.py:110-111`) — acceptable for the polymorphic API but worth documenting accepted protocols.
4. **Stale docstrings:** `_resolve_dtype` (`close.py:76-86`), `_is_float_dtype` (`close.py:70-73`), `_to_numpy` (`close.py:22-23`) lack full param/return docs (private, acceptable).
5. **Boundary errors:** `_to_numpy` swallows `ImportError` for cupy and `(ImportError, RuntimeError)` for the torch fallback (`close.py:50,64`). The bare `np.asarray(tensor)` at line 67 will surface a meaningful `TypeError` for unsupported objects — fine.
7. **Lazy-import compliance — VIOLATION:** `import torch as _torch` at module load (`close.py:13-19`). Even though wrapped in `try/except`, it triggers torch import at the moment `gpucheck.assertions` is imported. This contradicts CLAUDE.md "torch/pynvml never imported at collection time" and is the only top-level torch import in the entire `src/`. Fix: replicate the `_torch_mod()` pattern from `fuzzing/inputs.py:19`.

## `assertions/reporting.py`

1. **Public API:** `format_mismatch_report`. (`reporting.py:17`) (`_error_histogram` is private but useful — consider exposing.)
2. **Dead code:** `_safe_import_numpy` (`reporting.py:11-14`) is a one-liner wrapper for `import numpy as np`; numpy is **already** a non-optional dep (used unconditionally at the top of `assertions/close.py:7`). Wrapper can be inlined or removed.
3. **Type smells:** Both functions return `str` correctly. `Any` for histograms is fine.
4. **Stale docstrings:** `_safe_import_numpy` has no docstring (private, ok).
6. **Naming:** `_safe_import_numpy` implies a try/except that doesn't exist — name is misleading.

## `assertions/tolerances.py`

1. **Public API:** `compute_tolerance`, `tolerance_context`, `tolerances_from_config`, `apply_config_tolerances`, `reset_config_tolerances`, `mps_xfail_from_config`, `apply_mps_xfail_config`, `reset_mps_xfail`, `register_mps_xfail`, `is_mps_xfailed`, `mps_xfail_list`. (`tolerances.py:70,114,138,164,175,184,208,217,222,227,238`)
2. **Dead code:** `_config_overrides` is forward-referenced at line 97 but defined at line 161 — works because of function lookup at call time, but reading order is awkward.
3. **Type smells:** `_normalize_dtype_name(dtype: Any) -> str` (`tolerances.py:60`) — `Any` justified by polymorphic dtype input.
6. **Naming:** Two parallel registries with similar names — `_config_overrides` (line 161) and `_DEFAULT_TOLERANCES` (line 13). Naming OK but the override-shadowing precedence logic at lines 97-100 is non-obvious; deserves a comment.
8. **Lines to remove:** None.

## `decorators/__init__.py`

1. **Public API:** `dtypes`, `shapes`, `devices`, `parametrize_gpu`, `FLOAT_DTYPES`, `HALF_DTYPES`, `ALL_DTYPES`, `FP8_DTYPES`, `SMALL_SHAPES`, `MEDIUM_SHAPES`, `LARGE_SHAPES`, `EDGE_SHAPES`. (`decorators/__init__.py:22-35`)

## `decorators/devices.py`

1. **Public API:** `devices`. (`decorators/devices.py:79`)
3. **Type smells:** `devices(*device_args: str) -> Callable[..., Any]` (`devices.py:79`) — return type is opaque; `_MarkDecorator` from pytest would be more precise.
5. **Boundary errors:** `_is_device_available` swallows `(ImportError, RuntimeError, ValueError)` (`devices.py:66`) — the `ValueError` is implicit-only via `int()` parse failure on line 57; adding `KeyError` would be safer if `device.split(":")[1]` fails on edge inputs (it can't, splits never throw KeyError, ok).
7. **Lazy-import compliance:** `import torch` is inside each helper (`devices.py:16,32,49`). Compliant.

## `decorators/dtypes.py`

1. **Public API:** `dtypes`, `FLOAT_DTYPES`, `HALF_DTYPES`, `ALL_DTYPES`, `FP8_DTYPES`. (`dtypes.py:108,98-101`)
2. **Dead code:** `_DtypeGroup.__len__` (`dtypes.py:91-92`) returns `len(self._names)` — fine, but unused outside the class. (Probably internal use.)
3. **Type smells:** `_resolve_dtype(d: DtypeArg) -> Any` (`dtypes.py:37`) — return type forced to `Any` because torch is lazy. Justifiable.
4. **Stale docstrings:** `_DtypeGroup.__iter__/__len__/__repr__` have no docs (private, ok).
6. **Naming:** Public groups (`HALF_DTYPES` etc.) and underscore-suffixed name lists (`HALF_DTYPES_NAMES`) coexist (`dtypes.py:70-77,98-101`). Not exposed in `__all__` so likely fine, but the `_NAMES` constants feel internal.
7. **Lazy-import compliance:** `import torch` only inside `_resolve_dtype` (`dtypes.py:40`). Compliant.

## `decorators/parametrize.py`

1. **Public API:** `parametrize_gpu`. (`parametrize.py:35`)
2. **Dead code:** `marks: list[Any] = []` (`parametrize.py:111`) is shadowed by `marks = []` (`parametrize.py:132`) without an explicit type annotation in the second branch. Working but inconsistent.
3. **Type smells:** `SkipFilter = Callable[..., bool] | None` (`parametrize.py:16`) — the `...` defeats type-checking on the predicate's signature, and the runtime tries 4-arg first then falls back to 3-arg via `TypeError` at lines 137-139 — fragile (a 3-arg `skip` that **internally** raises `TypeError` for unrelated reasons would be silently downgraded to 4-arg failure).
6. **Naming:** Function name shadows decorator's keyword arg `dtypes`/`shapes`/`devices` — these decorators (`parametrize.py:37-39`) shadow the imported decorator names of the same identifiers from `decorators.dtypes` etc. Not a real bug because they're params, but readability suffers.

## `decorators/shapes.py`

1. **Public API:** `shapes`, `SMALL_SHAPES`, `MEDIUM_SHAPES`, `LARGE_SHAPES`, `EDGE_SHAPES`. (`shapes.py:18-47,59`)
4. **Stale docstrings:** `_shape_id` private, ok.

## `fixtures/__init__.py`

1. **Public API:** `BenchmarkResult`, `GPUDevice`, `MemoryReport`, `MemorySnapshot`, `MemoryTracker`, `gpu_benchmark`, `gpu_device`, `memory_tracker`. (`fixtures/__init__.py:28-37`)
2. **Dead code:** `gpu_benchmark`, `gpu_device`, `memory_tracker` are exported but they are **also** registered as fixtures in `plugin.py:127,137,189`. The `plugin.py` versions are what pytest actually picks up; the entries in `_LAZY_MAP` here (`fixtures/__init__.py:10,12,16`) are unreachable for fixture purposes (only useful for direct API access).

## `fixtures/benchmark.py`

1. **Public API:** `BenchmarkResult`, `KernelCallable`, `gpu_benchmark` (fixture). (`benchmark.py:13,39,330`) `_BenchmarkRunner` is private (line 124).
2. **Dead code:** Module-level `_torch_cache` (`benchmark.py:92`) is private but exposes `_get_torch()` for hot-loop reuse — used in `_flush_l2_cache` (line 108) and `__post_init__` (line 139). Both `_run_cuda` and `_run_mps` re-import torch at line 259/302 instead of using `_get_torch()` — inconsistent.
3. **Type smells:** `KernelCallable.__call__(*args: Any, **kwargs: Any) -> Any` (`benchmark.py:42`) — by definition unbounded; acceptable for a Protocol.  `_torch_cache: Any = None` (`benchmark.py:92`).
5. **Boundary errors:** `_get_l2_cache_size` swallows `(ImportError, RuntimeError, OSError)` (`benchmark.py:87`); the inner `(AttributeError, pynvml.NVMLError)` is correct. Solid.
6. **Naming:** Both `gpu_benchmark` (fixture, `benchmark.py:330`) and `gpu_benchmark` (the `plugin.py` fixture at line 127) define the same fixture name — pytest fixture override behaviour depends on registration order (`plugin.py` wins). Confusing duplication; pick one.
7. **Lazy-import compliance:** All torch imports lazy. Compliant.

## `fixtures/gpu.py`

1. **Public API:** `GPUDevice`, `detect_gpu`, `gpu_device` (fixture). (`gpu.py:17,116,139`)
2. **Dead code:** `_detect_gpu_torch` and `_detect_gpu_pynvml` (`gpu.py:43,90`) are duplicated logic vs `arch/detection.py::_detect_via_pynvml`/`_detect_via_torch` (`detection.py:133,206`) — two parallel detection stacks. The `arch` version returns full `GPUInfo`; this one returns the simpler `GPUDevice`. Pick one source of truth.
4. **Stale docstrings:** Helpers are minimally documented.
5. **Boundary errors:** `_cleanup_gpu` warns on RuntimeError (`gpu.py:135-136`) — appropriate.
7. **Lazy-import compliance:** All torch/pynvml imports lazy. Compliant.

## `fixtures/profiler.py`

1. **Public API:** `MemorySnapshot`, `MemoryReport`, `MemoryTracker`, `memory_tracker` (fixture). (`profiler.py:15,28,131,183`)
2. **Dead code:** `_MemoryTracker = MemoryTracker` (`profiler.py:180`) backward-compat alias — commented as such, but not exported anywhere. Drop or document why it must stay.
6. **Naming:** `MemoryReport` here clashes with `sanitizers.SanitizerMemoryReport` (alias `MemoryReport` at `sanitizers/__init__.py:14`). Two different `MemoryReport` symbols at different import paths. Likely confusing.
7. **Lazy-import compliance:** All torch/pynvml imports lazy. Compliant.

## `fuzzing/__init__.py`

1. **Public API:** `fuzz_shapes`, `random_inputs`, `edge_inputs`, `mixed_inputs`, `ShapeStrategy`, `gpu_shapes`, `gpu_tensors`, `fuzz_strides`, `fuzz_strides_for_category`, `StrideStrategy`, `STRIDE_CATEGORIES`. (`fuzzing/__init__.py:33-45`)
2. **Dead code:** Mixed strategy: `inputs`, `shapes`, `strides` are eagerly imported (lines 8-17) but `strategies` (which guards on hypothesis) is in `_LAZY_MAP` (lines 19-22). The eager import of `strides` re-imports `torch` lazily — but importing `inputs` at package init triggers the torch fallback path even though `_TORCH_IMPORTED` only flips on first call. OK but inconsistent: most fuzzing helpers are eager, only the hypothesis-dependent ones are lazy.

## `fuzzing/inputs.py`

1. **Public API:** `random_inputs`, `edge_inputs`, `mixed_inputs`. (`inputs.py:75,126,222`)
2. **Dead code:** `_FP8_TYPES_CACHE` is module-level mutable state (`inputs.py:54,67`); fine, just flagged as a global.
3. **Type smells:** `_torch_mod() -> Any` (`inputs.py:19`); `random_inputs(... custom_fn: Callable[..., Any] | None = None)` (`inputs.py:82`) — the `...` loses callback signature.
5. **Boundary errors:** `contextlib.suppress(RuntimeError, OverflowError)` at lines 201, 257 — correct narrow scope.
7. **Lazy-import compliance:** All torch via `_torch_mod()`. Compliant.

## `fuzzing/shapes.py`

1. **Public API:** `fuzz_shapes`, `ShapeStrategy`, `TILE_SIZES`, `PRIMES`, `POWER_OF_2_BOUNDARIES`, `LARGE_DIMS`. (`shapes.py:97,182,10,12,14,16`)
2. **Dead code:** Constant `LARGE_DIMS` defined at line 16 — only used in `_large_shapes` (line 72). OK but unexposed in `fuzzing/__init__.py`.
3. **Type smells:** `ShapeStrategy.__new__(...) -> Any` (`shapes.py:200`) — Hypothesis's `SearchStrategy` is parametric. `Any` is acceptable but could be `st.SearchStrategy[tuple[int, ...]]` under TYPE_CHECKING.
6. **Naming:** `ShapeStrategy` is a class with `__new__` returning a SearchStrategy — looks like a class but acts as a factory. Documented (`shapes.py:182-198`) but unusual.

## `fuzzing/strategies.py`

1. **Public API:** `gpu_shapes`, `gpu_tensors`. (`strategies.py:54,102`)
3. **Type smells:** `gpu_shapes(...) -> Any`, `gpu_tensors(...) -> Any` (`strategies.py:61,112`). Hypothesis SearchStrategy parametric type would be more informative.
6. **Naming:** `gpu_shapes` here vs `fuzz_shapes` in `shapes.py` — both produce shape generators but with different signatures (one Hypothesis, one batch). Naming is OK because of `gpu_*` prefix convention.
7. **Lazy-import compliance:** Hypothesis lazy via `_check_hypothesis()`; torch lazy via `_torch_mod()`. Compliant.

## `fuzzing/strides.py`

1. **Public API:** `CATEGORIES` (re-exported as `STRIDE_CATEGORIES`), `fuzz_strides`, `fuzz_strides_for_category`, `StrideStrategy`. (`strides.py:33,185,211,260`)
3. **Type smells:** All helpers `(_row_major, _column_major, ...)` typed `(... ) -> Any` (`strides.py:55,62,77,94,112,129,154`). Justified by torch.Tensor returning lazily.
6. **Naming:** `_slice` (`strides.py:112`) shadows builtin `slice`. Used only as helper key; minor.
7. **Lazy-import compliance:** torch lazy via `_torch_mod()`. Compliant.

## `sanitizers/__init__.py`

1. **Public API:** `MemoryReport` (alias), `SanitizerMemoryReport`, `SanitizerReport`, `check_memory_leaks`, `memory_guard`, `run_with_sanitizer`, `assert_deterministic`, `requires_determinism`, `DeterminismError`. (`sanitizers/__init__.py:16-26`)
6. **Naming:** Backward-compat alias `MemoryReport = SanitizerMemoryReport` (`sanitizers/__init__.py:14`) collides with `fixtures.MemoryReport` (a different class entirely). Two `MemoryReport` types in the project.

## `sanitizers/determinism.py`

1. **Public API:** `assert_deterministic`, `requires_determinism`, `DeterminismError`. (`determinism.py:85,137,35`)
3. **Type smells:** `assert_deterministic` returns `Any` (`determinism.py:91`) — appropriate for pass-through return.
5. **Boundary errors:** `_seed_all` swallows `ImportError` widely (`determinism.py:46,59`) but does not catch `RuntimeError` from `torch.cuda.is_available()` (rare but possible on misconfigured drivers). Minor.

## `sanitizers/memory.py`

1. **Public API:** `SanitizerMemoryReport`, `check_memory_leaks`, `memory_guard`, `_MutableReport` (private, but appears as the *yielded* type via `Generator[_MutableReport, ...]` at line 142 — leaks an underscore-prefixed type into the public signature). (`memory.py:14,83,141,216`)
3. **Type smells:** `memory_guard()` is annotated to yield `_MutableReport` (`memory.py:142`) — exposing a private type in a public function's annotation. Either rename (drop underscore) or wrap.
4. **Stale docstrings:** `_MutableReport` is undocumented as a public-facing object even though it's what `memory_guard` yields (`memory.py:216`).
5. **Boundary errors:** `_get_pynvml_memory` catches `(ImportError, RuntimeError, OSError)` (`memory.py:65`) but a freshly-failed `pynvml.NVMLError` is not in the tuple. Look at line 53 — `pynvml` not imported yet so subclass check fails. Likely benign because pynvml exceptions inherit from `Exception` so we'd hit the OSError catch sometimes — fragile.
8. **Lines to remove:** Two near-identical torch-vs-pynvml branches (`memory.py:102-138` and `memory.py:167-207`) — refactor opportunity, not removal.

## `sanitizers/race.py`

1. **Public API:** `SanitizerTool`, `SanitizerError`, `SanitizerReport`, `run_with_sanitizer`. (`race.py:18,33,43,176`)
3. **Type smells:** Tight typing throughout. `Any` on `extra_args` would be more precise as `Sequence[str]` but `list[str] | None` is fine.
5. **Boundary errors:** `os.unlink(script_path)` (`race.py:257`) is not in a `try` — if cleanup fails (race / permission), traceback masks original exception. Wrap in `contextlib.suppress(OSError)`.
6. **Naming:** Local `warnings` shadows the imported `warnings` module (`race.py:117` shadows `race.py:11`). Confusing — rename local to `warning_lines`.

## `arch/__init__.py`

1. **Public API:** `GPUInfo`, `detect_gpu`, `detect_gpus`, `gpu_available`, `gpu_count`, `require_arch`, `require_capability`, `supports_tensor_cores`, `warn_tensor_core_fallback`. (`arch/__init__.py:26-36`)
2. **Dead code:** `gpu_available`, `gpu_count`, `detect_gpu` here (`arch/__init__.py:10-23`) duplicate `plugin.py:17-22` shims. Pick one.

## `arch/compatibility.py`

1. **Public API:** `SM_ARCH_MAP`, `SM_ARCH_MAP_DETAILED`, `require_arch`, `require_capability`, `check_compatibility`. (`compatibility.py:20,32,58,95,147`)
2. **Dead code:** `_ARCH_PARENT_SM` (`compatibility.py:47`) is only used inside `check_compatibility` — could be local. `_sm_tag_to_cc` (`compatibility.py:201`) only used in `check_compatibility`.
3. **Type smells:** `require_arch(*archs: str) -> Callable[..., Any]` and `require_capability` (`compatibility.py:58,95`) — `Callable[..., Any]` is too loose; `Callable[[Callable[..., T]], Callable[..., T]]` would express decorator-of-decorator more accurately.
6. **Naming:** Two parallel arch maps — `SM_ARCH_MAP` (`compatibility.py:20`) and `arch.detection.SM_TO_ARCH` (`detection.py:14`). Same data, different keys (`"SM80"` vs `(8, 0)`). Single source of truth would reduce drift risk.

## `arch/detection.py`

1. **Public API:** `SM_TO_ARCH`, `GPUInfo`, `detect_gpus`. (`detection.py:14,104,268`)
2. **Dead code:** `_FP16_MIN_CC`, `_BF16_MIN_CC`, `_FP8_MIN_CC`, `_TF32_MIN_CC` (`detection.py:31-34`) — used internally only; OK.
3. **Type smells:** `_default_shared_memory(cc: tuple[int, int]) -> int` is OK. The `Exception` catches at lines 157, 229 are too broad — specific exceptions preferred per CLAUDE.md "no bare except, specific exceptions only".
5. **Boundary errors:** Bare `except Exception` at `detection.py:157` (cuda version detection) and `detection.py:229` (mem_get_info) — violates project standard.
7. **Lazy-import compliance:** `import pynvml` (line 136), `import torch` (line 209) — both lazy. Compliant.

## `arch/tensor_cores.py`

1. **Public API:** `supports_tensor_cores`, `compute_tolerance`, `warn_tensor_core_fallback`. (`tensor_cores.py:58,96,169`)
2. **Dead code:** Imports `_DEFAULT_TOLERANCES` from `assertions.tolerances` as `_CANONICAL_TOLERANCES` (`tensor_cores.py:9-11`) — dipping into private symbols across modules. Either expose officially or duplicate.
3. **Type smells:** `compute_tolerance(dtype: str, k_dim: int, gpu_info: GPUInfo | None) -> tuple[float, float]` (`tensor_cores.py:96`) — name **collides** with the more general `assertions.tolerances.compute_tolerance` (`tolerances.py:70`). Two different `compute_tolerance` functions with different signatures. **HIGH-IMPACT** naming collision.
6. **Naming:** `compute_tolerance` collision (see #3).
7. **Lazy-import compliance:** `import os` (`tensor_cores.py:178`) at function scope — odd choice (stdlib). `import torch` lazy (line 192). Compliant.

## `analysis/__init__.py`

1. **Public API:** `GPUSpecs`, `RooflinePoint`, `classify_bottleneck`, `compute_roofline`, `compute_roofline_point`, `lookup_gpu_specs`, `render_roofline_ascii`, `BottleneckAnalysis`, `auto_classify_bottleneck`, `RegressionReport`, `RegressionResult`, `detect_regression`, `e_divisive_single`, `format_regression_table`, `load_baseline`, `mann_whitney_u`, `save_baseline`. (`analysis/__init__.py:8-29,40-61`)

## `analysis/bottleneck.py`

1. **Public API:** `BottleneckAnalysis`, `auto_classify_bottleneck`. (`bottleneck.py:18,115`)
2. **Dead code:** None.
3. **Type smells:** Tight.
5. **Boundary errors:** `_sync_gpu` only handles `ImportError` (`bottleneck.py:48`); a `RuntimeError` from `torch.cuda.synchronize()` (e.g. CUDA driver fault) propagates. May or may not be desired.
7. **Lazy-import compliance:** torch lazy. Compliant.

## `analysis/regression.py`

1. **Public API:** `RegressionReport`, `RegressionResult` (alias), `mann_whitney_u`, `e_divisive_single`, `detect_regression`, `save_baseline`, `load_baseline`, `format_regression_table`. (`regression.py:28,42,50,140,219,322,344,364`)
2. **Dead code:** `_median` (`regression.py:204`) is defined here AND in `roofline.py:310` — two near-identical helpers in sibling modules. Consolidate.
3. **Type smells:** `_collect_metadata() -> dict[str, str]` (`regression.py:305`) types fine.
5. **Boundary errors:** `save_baseline` swallows `(json.JSONDecodeError, OSError)` on read (`regression.py:334`) but does NOT catch errors on `p.write_text` (line 341) — a permission error there would propagate. Probably correct.
6. **Naming:** Two `_median` defs (this file and `roofline.py`). DRY.

## `analysis/roofline.py`

1. **Public API:** `Bottleneck` (Literal), `GPUSpecs`, `RooflinePoint`, `lookup_gpu_specs`, `compute_roofline`, `compute_roofline_point`, `classify_bottleneck`, `render_roofline_ascii`. (`roofline.py:16,24,58,49,90,147,164,213`)
2. **Dead code:** `_median` (`roofline.py:310`) duplicates `regression.py:204`.
6. **Naming:** `_median` duplication.

## `backends/__init__.py`

1. **Public API:** `Backend`, `EventTimer`, `available_backends`, `get_backend`. (`backends/__init__.py:33,64`)
5. **Boundary errors:** `available_backends` swallows `ImportError` from each backend constructor (`backends/__init__.py:48,58`) — appropriate.

## `backends/_protocol.py`

1. **Public API:** `Backend`, `EventTimer` (private module path, but exported via `backends/__init__.py`). (`_protocol.py:18,36`)
6. **Naming:** Filename is `_protocol.py` but symbols inside are public — leading-underscore module is an established convention to mark "import via parent package, not this path". Documented (`_protocol.py:2-5`). OK.

## `backends/cuda.py`

1. **Public API:** `CUDABackend`. (`cuda.py:45`)
2. **Dead code:** `_CUDAEventTimer.elapsed_ms` is set inside `event_timer` `finally:` block (`cuda.py:80`); fine.
3. **Type smells:** `_torch() -> Any` (`cuda.py:30`) — pattern repeated. OK.
5. **Boundary errors:** `mem_stats` returns zeros on `RuntimeError` (`cuda.py:86-87`) — acceptable.
7. **Lazy-import compliance:** Compliant (`_torch()` accessor).

## `backends/mps.py`

1. **Public API:** `MPSBackend`. (`mps.py:74`)
2. **Dead code:** `_FLUSH_L2_WARNED` global (`mps.py:60`) — a one-shot warning latch, used in `flush_l2` (line 167). OK.
3. **Type smells:** Multiple bare `except Exception` at lines 99, 137, 141, 148, 191 — violates CLAUDE.md "no bare except, specific exceptions only".
5. **Boundary errors:** `arch_info` falls back to zero memory on `Exception` (`mps.py:191`); `mem_stats` similarly (lines 137-148). Specific exception types preferred.
8. **Lines to remove:** None.

## `reporting/__init__.py`

1. **Public API:** `ConsoleReporter`, `JSONReporter`, `emit_github_annotations`, `write_junit_xml`, `generate_pr_comment`, `HTMLReporter`. (`reporting/__init__.py:8-15,26-33`)

## `reporting/ci.py`

1. **Public API:** `emit_github_annotations`, `write_junit_xml`, `generate_pr_comment`. (`ci.py:28,57,115`)
3. **Type smells:** `generate_pr_comment(comparison: dict[str, Any], ...) -> str` (`ci.py:115`) — `dict[str, Any]` for the comparison shape; would be cleaner as a TypedDict matching `JSONReporter.compare_runs` output.
5. **Boundary errors:** `write_junit_xml` does no try/except around `tree.write` (`ci.py:102`) or the `open(path, "a")` append (line 104). Permission/disk errors will propagate to test-runner; debatable but reasonable.
8. **Lines to remove:** Empty conditional comment block at `ci.py:141-143` (extra blank line between two structurally identical `f"... ms"` builds).

## `reporting/console.py`

1. **Public API:** `TestResult`, `BenchmarkEntry`, `MemoryEntry`, `ConsoleReporter`. (`console.py:18,30,52,70`)
2. **Dead code:** Imports `os`, `sys` lazily inside `__init__` (`console.py:76-77`) — could be top-level (stdlib, free).
3. **Type smells:** `ConsoleReporter.__init__(... file: Any = None)` (`console.py:74`) — should be `IO[str] | None`.
5. **Boundary errors:** None obvious.
7. **Lazy-import compliance:** Top-of-file `from rich.console import Console` (`console.py:9-12`) — rich is a non-optional dependency per pyproject (verifiable elsewhere). OK.

## `reporting/html.py`

1. **Public API:** `HTMLReporter`. (`html.py:45`)
3. **Type smells:** `_esc(value: Any) -> str` (`html.py:86`); `comparison: dict[str, Any] | None` (`html.py:49`) — TypedDict candidate.
5. **Boundary errors:** `Path(self.json_path).read_text(...)` (`html.py:55`) and `out.write_text(...)` (line 77) have no try/except. A missing input file or unwritable output will raise; debatable but should at least be documented in the docstring.

## `reporting/json.py`

1. **Public API:** `RunRecord`, `JSONReporter`. (`json.py:16,26`)
2. **Dead code:** `RunRecord` is exposed at module level but is **not** in `reporting/__init__.py:_LAZY_MAP`. Either add or document as internal.
3. **Type smells:** `set_gpu_info(info: dict[str, Any])` (`json.py:37`) — TypedDict candidate.
5. **Boundary errors:** `compare_runs` reads two JSON files with `json.loads(Path(...).read_text(...))` (`json.py:102-103`) and no try/except. A malformed or missing file raises raw exception.

---

## Summary statistics

| Module | Files | Public symbols | Lazy-import OK | Notes |
|---|---|---|---|---|
| `assertions/` | 4 | 11 | **Partial** (close.py top-level torch) | Largest violation |
| `decorators/` | 5 | 14 | OK | Clean |
| `fixtures/` | 4 | 10 | OK | Duplicate fixtures vs plugin.py |
| `fuzzing/` | 5 | 14 | OK | Mixed eager/lazy import policy |
| `sanitizers/` | 4 | 11 | OK | Underscore-typed yields, naming clashes |
| `arch/` | 4 | 13 | OK | Two `compute_tolerance` collision |
| `analysis/` | 4 | 17 | OK | Duplicate `_median` |
| `backends/` | 4 | 4 | OK | Bare excepts in mps.py |
| `reporting/` | 5 | 9 | OK (rich is required dep) | TypedDict opportunities |
| `__init__.py` + `plugin.py` | 2 | 28 + 7 fixtures/hooks | OK | duplicate gpu_available shim |

---

## Top 10 highest-impact findings (correctness-risk × frequency)

1. **`assertions/close.py:13-19` — top-level `import torch as _torch`.** Only top-level torch import in `src/`. Imported every time anyone imports `gpucheck.assertions`, defeating the project's lazy-import claim. **Fix:** move into `_torch_mod()` accessor.

2. **`arch/tensor_cores.py:96` vs `assertions/tolerances.py:70` — two functions named `compute_tolerance` with different signatures.** Different parameter sets, different return semantics, different scaling. Importing one expecting the other is a footgun. **Fix:** rename arch one to `compute_tolerance_arch_aware` or fold into tolerances.py.

3. **`sanitizers/__init__.py:14` aliases `MemoryReport = SanitizerMemoryReport`, while `fixtures/profiler.py:28` also defines a class named `MemoryReport`.** Two `MemoryReport` types at different paths in the same package.

4. **`fixtures/gpu.py:43,90` duplicates `arch/detection.py:133,206` GPU detection.** Two parallel detection stacks — one returning `GPUDevice`, one returning `GPUInfo`. Drift risk (different default fallbacks, different exception handling).

5. **`plugin.py:17-22,94-95` shims `_gpu_available`/`_gpu_count` duplicating `arch/__init__.py:10-17`.** Three implementations of "is a GPU available?" across modules.

6. **`decorators/parametrize.py:135-139` — `try: skip(4-arg) except TypeError: skip(3-arg)`.** Catching `TypeError` to detect signature length is fragile — a 3-arg `skip` whose body raises `TypeError` for unrelated reasons would be silently downgraded.

7. **`arch/detection.py:157,229` — bare `except Exception`.** Violates CLAUDE.md "no bare except, specific exceptions only". `backends/mps.py:99,137,141,148,191` has the same problem in 5 places — much more frequent.

8. **`sanitizers/memory.py:142,216` — `memory_guard` yields a private `_MutableReport`.** Underscore-typed value escaping into a public function signature; users either rely on a private symbol or get type-checker noise.

9. **`fixtures/__init__.py:10,12,16` — registers `gpu_benchmark`/`gpu_device`/`memory_tracker` in `_LAZY_MAP` while `plugin.py:127,137,189` also defines them as fixtures.** Pytest finds the `plugin.py` versions; the `_LAZY_MAP` entries are unreachable as fixtures (only useful as direct API). Pick one.

10. **`analysis/regression.py:204` and `analysis/roofline.py:310` — duplicated `_median` helper in sibling modules.** DRY violation; one place would catch any future bug fix.
