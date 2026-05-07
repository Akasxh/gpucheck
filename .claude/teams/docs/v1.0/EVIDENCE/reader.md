# EVIDENCE — docs-reader

**Persona:** docs-reader (Phase B inner, FM-3.3 — accuracy ground truth).
**Method:** Read every public symbol in `src/gpucheck/{assertions,decorators,
fixtures,fuzzing,sanitizers,arch,analysis,reporting}` plus `plugin.py`. Record
docstring presence, completeness, and stale-vs-source references. All cites use
absolute paths and 1-based line numbers (matches Read output).

## Module-by-module public-symbol inventory

Legend: **OK** (has docstring, complete) · **THIN** (1-line, undocumented
parameters) · **MISSING** (no docstring) · **STALE** (docstring exists but
contradicts code or v1.0 reality).

### `src/gpucheck/assertions/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `assert_close` | `close.py:109` | **OK** | Has Numpy-style docstring. No mention of GPU fast-path or MPS. |
| `_to_numpy` | `close.py:22` | OK (private) | — |
| `_is_float_dtype` | `close.py:70` | OK (private) | — |
| `_resolve_dtype` | `close.py:76` | OK (private) | — |
| `compute_tolerance` (assertions) | `tolerances.py:41` | **OK** | k_dim semantics documented. |
| `tolerance_context` | `tolerances.py:73` | **OK** | — |
| `tolerances_from_config` | `tolerances.py:91` | **OK** | — |
| `apply_config_tolerances` | `tolerances.py:117` | **OK** | — |
| `reset_config_tolerances` | `tolerances.py:128` | **THIN** | One-liner; OK for trivial fn. |
| `_DEFAULT_TOLERANCES` (re-imported by `arch/tensor_cores.py:10`) | `tolerances.py:12` | OK (comment) | Calibrated against cuBLAS Turing/Ampere — **no MPS row**. |
| `format_mismatch_report` | `reporting.py:17` | **OK** | — |
| `_error_histogram` | `reporting.py:113` | OK (private) | — |
| Re-exports | `__init__.py:5-8` | **THIN** | Module docstring fine; per-symbol __all__ documented elsewhere. |

### `src/gpucheck/decorators/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `dtypes` | `dtypes.py:108` | **OK** | Lazy resolution explained. |
| `FLOAT_DTYPES`, `HALF_DTYPES`, `ALL_DTYPES`, `FP8_DTYPES` | `dtypes.py:98-101` | **MISSING** | Module-level group constants exposed in public API have **no per-symbol docstring**. README §3 (`README.md:150`) treats them as documented; only the `dtypes()` decorator is. |
| `_DtypeGroup` | `dtypes.py:79` | **THIN** | Class docstring is 1 line. |
| `shapes` | `shapes.py:59` | **OK** | — |
| `SMALL_SHAPES`, `MEDIUM_SHAPES`, `LARGE_SHAPES`, `EDGE_SHAPES` | `shapes.py:18-47` | **MISSING** | Same gap: README treats as public, no docstring. |
| `devices` | `devices.py:55` | **OK** but **STALE** | Docstring talks only about CUDA devices ("auto-detects all available CUDA devices", `devices.py:58`). v1.0 will need MPS-aware detection. The `_is_device_available` helper (`devices.py:25`) only special-cases `cpu` and `cuda` — MPS will silently fall through. |
| `parametrize_gpu` | `parametrize.py:28` | **OK** | But `devices=None` auto-detects only CUDA (`parametrize.py:58-60`). Stale once MPS lands. |

### `src/gpucheck/fixtures/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `BenchmarkResult` | `benchmark.py:13` | **OK** | One-line class docstring; fields self-explanatory. |
| `KernelCallable` | `benchmark.py:39` | **OK** | — |
| `_BenchmarkRunner` | `benchmark.py:124` | **THIN** | Class is 1 line; `__call__` (`benchmark.py:147`) is OK. |
| `_BenchmarkRunner.__call__` | `benchmark.py:147` | **OK** but **STALE** | Says "CUDA events for accurate GPU timing" (`benchmark.py:156`). v1.0 needs torch.mps.Event branch and an MPS path documented. |
| `gpu_benchmark` (fixture in `benchmark.py:247`) | `benchmark.py:247` | **OK** but **STALE** | Same CUDA-only language. |
| `_get_l2_cache_size` | `benchmark.py:70` | OK (private) | NVML-only — no MPS equivalent documented. |
| `_flush_l2_cache` | `benchmark.py:105` | OK (private) | CUDA only; should branch on backend in v1.0. |
| `GPUDevice` | `gpu.py:17` | **OK** | But `compute_capability: tuple[int, int]` is CUDA-specific — Apple GPUs do not have an SM number. v1.0 needs an alternative (e.g. `Optional[tuple]` or `chip: str`). |
| `detect_gpu` (fixture-side) | `gpu.py:116` | **OK** but **STALE** | "Auto-detect a GPU, preferring pynvml (lighter) over torch" — neither backend covers MPS. |
| `_detect_gpu_pynvml`, `_detect_gpu_torch` | `gpu.py:43, 90` | OK (private) | Both NVIDIA-only. |
| `gpu_device` (fixture) | `gpu.py:140` | **OK** but **STALE** | "skip if none available" — needs to branch on MPS too. |
| `MemorySnapshot` / `MemoryReport` | `profiler.py:15, 28` | **OK** | One-liners on dataclasses. |
| `MemoryTracker` | `profiler.py:131` | **THIN** | Class has 1-line docstring; `start()` and `report` property have **MISSING** docstrings. |
| `MemoryTracker.start` | `profiler.py:142` | **MISSING** | — |
| `MemoryTracker.stop` | `profiler.py:146` | **MISSING** | — |
| `memory_tracker` (fixture) | `profiler.py:184` | **OK** | — |

### `src/gpucheck/fuzzing/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `fuzz_shapes` | `shapes.py:97` | **OK** | — |
| `ShapeStrategy` | `shapes.py:182` | **OK** | — |
| `TILE_SIZES`, `PRIMES`, `POWER_OF_2_BOUNDARIES`, `LARGE_DIMS` | `shapes.py:10-16` | **MISSING** | Public per `__all__` (`shapes.py:235-242`); no per-constant doc. |
| `random_inputs` | `inputs.py:75` | **OK** | — |
| `edge_inputs` | `inputs.py:126` | **OK** | Already includes a stride/contiguity caveat (`inputs.py:146-151`) — good baseline for v1.0 stride-fuzzing. |
| `mixed_inputs` | `inputs.py:222` | **OK** | — |
| `gpu_shapes` | `strategies.py:54` | **OK** | — |
| `gpu_tensors` | `strategies.py:102` | **OK** | — |

### `src/gpucheck/sanitizers/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `SanitizerMemoryReport` | `memory.py:14` | **OK** | — |
| `check_memory_leaks` | `memory.py:83` | **OK** | "Uses torch.cuda.memory_stats when available" — CUDA-only path; no MPS branch documented. |
| `memory_guard` | `memory.py:141` | **OK** | Same CUDA assumption. |
| `_MutableReport` | `memory.py:216` | **THIN** | Class doc is 1 line; properties undocumented. |
| `_MutableReport.to_report` | `memory.py:252` | **MISSING** | — |
| `SanitizerError` | `race.py:21` | **OK** | — |
| `SanitizerReport` | `race.py:31` | **OK** | — |
| `run_with_sanitizer` | `race.py:129` | **OK** | NVIDIA-only by design (`compute-sanitizer`). Should mention "no MPS equivalent; use Metal validation layer instead" in v1.0. |

### `src/gpucheck/arch/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `GPUInfo` | `detection.py:104` | **OK** but **STALE** | Dataclass has `supports_fp16/bf16/fp8/tf32` and `tensor_core_generation` — all NVIDIA-specific. No MPS-equivalent fields, no `backend: Literal["cuda","mps"]`. |
| `SM_TO_ARCH`, `_TENSOR_CORE_GEN`, `_FP16_MIN_CC`, etc. | `detection.py:14-48` | OK (constants) | Pure CUDA. |
| `_resolve_arch`, `_tensor_core_gen` | `detection.py:51, 78` | OK (private) | — |
| `detect_gpus` | `detection.py:259` | **OK** | "pynvml … falls back to torch.cuda" — explicitly CUDA-bound. |
| `gpu_available`, `gpu_count`, `detect_gpu` | `__init__.py:10, 15, 20` | **OK** | All return-only-CUDA in v0; v1.0 must include MPS. |
| `require_arch` | `compatibility.py:58` | **OK** | Architecture names are CUDA marketing names (Volta..Blackwell). v1.0 should accept "Apple-Silicon" / "M1" / "M2" / "MPS" or be split across decorators. |
| `require_capability` | `compatibility.py:95` | **OK** | (major, minor) is meaningless on MPS. |
| `check_compatibility`, `_KNOWN_INCOMPATIBILITIES` | `compatibility.py:124, 147` | **OK** | CUDA-only. |
| `supports_tensor_cores` | `tensor_cores.py:58` | **OK** | NVIDIA-only by design. v1.0 should add a parallel `supports_metal_tensors()` or note that `tensor_core_generation is None` is the universal "no" answer. |
| `compute_tolerance` (arch variant) | `tensor_cores.py:96` | **OK** but **STALE** | Architecture switch only handles `volta`, `hopper`, etc. — no MPS branch. |
| `warn_tensor_core_fallback` | `tensor_cores.py:169` | **OK** | NVIDIA env vars only. |

### `src/gpucheck/analysis/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `RegressionReport` | `regression.py:28` | **OK** | — |
| `mann_whitney_u` | `regression.py:50` | **OK** | — |
| `e_divisive_single` | `regression.py:140` | **OK** | — |
| `_normal_cdf`, `_median`, `_cohens_d`, `_mean_abs_diff_*` | `regression.py:106, 126, 131, 186, 204` | OK (private) | — |
| `detect_regression` | `regression.py:219` | **OK** | — |
| `save_baseline`, `load_baseline` | `regression.py:322, 344` | **OK** | — |
| `format_regression_table` | `regression.py:364` | **OK** | — |
| `GPUSpecs` | `roofline.py:24` | **OK** | `_KNOWN_SPECS` table at `roofline.py:40` is **STALE for v1.0**: only A100/H100/4090/3090/V100. No M1/M2/M3/M4 entries. |
| `lookup_gpu_specs` | `roofline.py:49` | **OK** | Will return None for any Apple GPU in v0. |
| `RooflinePoint` | `roofline.py:58` | **OK** | — |
| `compute_roofline` | `roofline.py:90` | **OK** | — |
| `compute_roofline_point` (legacy) | `roofline.py:147` | **OK** | — |
| `classify_bottleneck` | `roofline.py:164` | **OK** | — |
| `render_roofline_ascii` | `roofline.py:213` | **OK** | — |
| `BottleneckAnalysis` | `bottleneck.py:18` | **OK** | — |
| `auto_classify_bottleneck` | `bottleneck.py:115` | **OK** | — |

### `src/gpucheck/reporting/`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `TestResult` | `console.py:18` | **OK** | — |
| `BenchmarkEntry` | `console.py:30` | **OK** | — |
| `MemoryEntry` | `console.py:52` | **OK** | — |
| `ConsoleReporter` | `console.py:70` | **OK** | — |
| `ConsoleReporter.gpu_info_panel` | `console.py:93` | **OK** | — |
| `ConsoleReporter.test_summary` | `console.py:110` | **OK** | — |
| `ConsoleReporter.benchmark_table` | `console.py:141` | **OK** | — |
| `ConsoleReporter.memory_summary` | `console.py:165` | **OK** | — |
| `ConsoleReporter.error_detail` | `console.py:188` | **OK** | — |
| `RunRecord` | `json.py:15` | **THIN** | Single-line; dataclass fields are self-describing. |
| `JSONReporter` | `json.py:26` | **THIN** | Class doc 1 line; methods documented. |
| `JSONReporter.set_gpu_info`, `add_test_result`, `add_benchmark`, `add_memory` | `json.py:37-64` | **MISSING** | Four public mutator methods, **none** has a docstring. |
| `JSONReporter.flush` | `json.py:70` | **OK** | — |
| `JSONReporter.compare_runs` | `json.py:91` | **OK** | — |
| `emit_github_annotations` | `ci.py:28` | **OK** | — |
| `write_junit_xml` | `ci.py:57` | **OK** | — |
| `generate_pr_comment` | `ci.py:115` | **OK** | — |

### `src/gpucheck/plugin.py`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| `pytest_addoption` | `plugin.py:25` | **MISSING** | pytest hook — no docstring. Acceptable for hooks but flag. |
| `pytest_configure` | `plugin.py:46` | **MISSING** | Same. |
| `pytest_collection_modifyitems` | `plugin.py:52` | **MISSING** | Same. |
| `pytest_terminal_summary` | `plugin.py:68` | **MISSING** | Same. Output is GPU summary at session end — should describe schema (Device, CUDA, Compute, Memory, GPUs). On MPS box this prints "No GPU detected" silently. |
| `gpu_benchmark` (plugin fixture) | `plugin.py:88` | **OK** | One-liner. |
| `gpu_device` (plugin fixture) | `plugin.py:98` | **OK** but **STALE** | Uses `import torch; torch.cuda.is_available()` (`plugin.py:114`) — no MPS branch. |
| `memory_tracker` (plugin fixture) | `plugin.py:150` | **OK** | — |

### `src/gpucheck/__init__.py`

| Symbol | File:line | Status | Notes |
|---|---|---|---|
| Module docstring | `__init__.py:1` | **OK** | "pytest plugin for GPU kernel testing." |
| `__version__` | `__init__.py:8` | **STALE** | Pinned at `"0.1.0"`. Per pyproject.toml:7 also `0.1.0`. Phase 3 must bump to `1.0.0` (or `1.0.0rc1`) when DIFF_LOG lands. |
| `_LAZY_MAP` | `__init__.py:11` | OK (private) | — |
| `__getattr__` | `__init__.py:37` | **MISSING** | Public mechanism, no docstring. Briefly explain "Lazy attribute access for top-level public API". |

## Aggregate counts

| Status | Count |
|---|---|
| MISSING docstrings on public symbols | **18** |
| THIN docstrings worth expanding | **7** |
| STALE — CUDA-only language that contradicts v1.0 MPS pivot | **14** |
| OK | rest (~80) |

**MISSING set** (full list, all public per `__all__` or pytest entry-point):
1. `FLOAT_DTYPES` `dtypes.py:99`
2. `HALF_DTYPES` `dtypes.py:98`
3. `ALL_DTYPES` `dtypes.py:100`
4. `FP8_DTYPES` `dtypes.py:101`
5. `SMALL_SHAPES` `shapes.py:18`
6. `MEDIUM_SHAPES` `shapes.py:25`
7. `LARGE_SHAPES` `shapes.py:32`
8. `EDGE_SHAPES` `shapes.py:39`
9. `TILE_SIZES` `fuzzing/shapes.py:10`
10. `PRIMES` `fuzzing/shapes.py:12`
11. `POWER_OF_2_BOUNDARIES` `fuzzing/shapes.py:14`
12. `LARGE_DIMS` `fuzzing/shapes.py:16`
13. `MemoryTracker.start` `fixtures/profiler.py:142`
14. `MemoryTracker.stop` `fixtures/profiler.py:146`
15. `_MutableReport.to_report` `sanitizers/memory.py:252`
16. `JSONReporter.set_gpu_info` / `add_test_result` / `add_benchmark` / `add_memory` `reporting/json.py:37-64` (count as 4)
17. `gpucheck.__getattr__` `__init__.py:37`
18. `pytest_addoption` / `pytest_configure` / `pytest_collection_modifyitems` / `pytest_terminal_summary` (count as 4 — hooks, conventionally OK but flagged)

Combined: 4 dtype groups + 4 shape groups + 4 fuzzing constants + 2 MemoryTracker
methods + 1 _MutableReport.to_report + 4 JSONReporter mutators + 1
gpucheck.__getattr__ + 4 pytest hooks = **24 truly missing**.

If pytest hook 4 are excluded (acceptable per pytest convention), **20 missing
docstrings on user-facing public symbols**.

## Stale-claim seeds for AUDIT.md

1. `assertions/close.py:163-176` — GPU fast-path checks
   `actual.device.type == "cuda"`. **Must extend to `"mps"`**.
2. `assertions/tolerances.py:12-24` — Tolerance table calibrated against
   "cuBLAS matmul on Turing/Ampere GPUs" — **no MPS calibration row**.
3. `arch/detection.py:14-48` — All architecture maps are NVIDIA SM tags.
4. `arch/detection.py:281-282` — Warning text "Install pynvml or torch
   for GPU support" — should mention MPS path too.
5. `arch/tensor_cores.py:138-166` — `_arch_adjust` only branches on
   `volta`, `hopper`, `ada`, etc. — no `apple`/`mps`/`m1`/`m2`/`m3` case.
6. `decorators/devices.py:25-43` — `_is_device_available` only handles
   `cpu` and `cuda*`. MPS falls through to "let torch figure it out".
7. `fixtures/benchmark.py:156` / `247-255` — Docstrings advertise CUDA
   events as the timing primitive.
8. `fixtures/gpu.py:117` / `141` — "preferring pynvml" / "skip if none
   available" — both NVIDIA-bound.
9. `plugin.py:68-83` — Terminal summary prints CUDA fields ("CUDA",
   "Compute") that are meaningless on MPS.
10. `analysis/roofline.py:40-46` — `_KNOWN_SPECS` lacks Apple GPU entries.
11. `sanitizers/memory.py:36, 90` — `torch.cuda.memory_stats()` only.
12. `sanitizers/race.py:129` — `compute-sanitizer` is NVIDIA-only.
