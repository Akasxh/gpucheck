# AUDIT.md — gpucheck v1.0 Phase 1 Documentation Audit

**Owner:** docs-lead (acting as docs-detector + docs-reader + docs-planner
+ docs-reviewer + docs-skeptic during Phase 1).
**Scope:** README.md, CLAUDE.md, inline docstrings across `src/gpucheck/`,
top-level governance docs.
**Phase 1 mandate:** *audit only* — no source files modified outside
`.claude/teams/docs/v1.0/`.
**Cross-team status (read-only):** research SYNTHESIS.md not yet
produced; engineering DIFF_LOG.md not yet produced.

All citations use absolute file paths and 1-based line numbers as
emitted by `Read`.

---

## A. Stale claims in README.md and CLAUDE.md

Each row: severity / file:line / current text / required change.

### A.1 CUDA-only language (P0 — blocks v1.0 release)

| # | File:line | Current claim | Required v1.0 change |
|---|---|---|---|
| 1 | `/Users/cero/Code/gpucheck/README.md:10` | "You write a CUDA kernel, eyeball `torch.allclose`..." | "You write a CUDA or Metal kernel..." |
| 2 | `/Users/cero/Code/gpucheck/README.md:12` | "tested gpucheck against Triton tutorials and PyTorch CUDA ops" | Add MPS coverage line once research/engineering produce data |
| 3 | `/Users/cero/Code/gpucheck/README.md:23` | `@devices("cuda:0")` | Add `@devices("mps")` example or use `@devices()` auto-detect |
| 4 | `/Users/cero/Code/gpucheck/README.md:53` | "Every code block here has been tested on a real NVIDIA GeForce GTX 1650 ..." | Dual-validate on Apple Silicon + GTX 1650, OR split sections per backend |
| 5 | `/Users/cero/Code/gpucheck/README.md:55-82` | `detect_gpu()` sample shows GTX 1650 output literally | Add parallel MPS output (`MPSDevice` or extended `GPUInfo`) |
| 6 | `/Users/cero/Code/gpucheck/README.md:97` | `device="cuda"` in test sample | Show `cuda` and `mps` |
| 7 | `/Users/cero/Code/gpucheck/README.md:122` | `@devices("cuda:0")` | Same |
| 8 | `/Users/cero/Code/gpucheck/README.md:140-148` | `parametrize_gpu(devices=("cuda:0",))` | Same |
| 9 | `/Users/cero/Code/gpucheck/README.md:177` | "CUDA events, which are more accurate than wall-clock timing" | "On CUDA: CUDA events. On MPS: `torch.mps.Event` (PyTorch 2.x). On CPU: `time.perf_counter`." |
| 10 | `/Users/cero/Code/gpucheck/README.md:251` | "uses `torch.cuda.memory_stats()` ... falls back to `pynvml`" | Add MPS branch using `torch.mps.current_allocated_memory()` per research QUESTION sub-question 3 |
| 11 | `/Users/cero/Code/gpucheck/README.md:273` | "Supported architectures: Volta (SM70) ... Blackwell (SM120)" | Note CUDA-only scope; add Apple-Silicon list (M1/M2/M3/M4 family) |
| 12 | `/Users/cero/Code/gpucheck/README.md:295` | `GPUSpecs(peak_flops=3.5e12, peak_bandwidth=128e9) # GTX 1650` | (P2 nit) GTX 1650 FP32 peak is ≈ 2.98 TFLOP/s; correct the illustrative number |
| 13 | `/Users/cero/Code/gpucheck/README.md:344-346` | "Hardware tested: NVIDIA GeForce GTX 1650 (Turing, SM75, 4GB GDDR6, no tensor cores)" | Add Apple Silicon row (chip, OS, PyTorch+MPS version) |
| 14 | `/Users/cero/Code/gpucheck/README.md:367` | "AMD ROCm and Intel XPU are not supported yet ... ROCm support is planned" | "Apple Silicon supported via MPS as of v1.0; AMD ROCm and Intel XPU planned" |

### A.2 GTX 1650-only validation (P0)

| # | File:line | Current claim | Required v1.0 change |
|---|---|---|---|
| 15 | `/Users/cero/Code/gpucheck/README.md:53` | "tested on a real NVIDIA GeForce GTX 1650" | Add Apple Silicon row; tag CUDA-validated vs MPS-validated examples |
| 16 | `/Users/cero/Code/gpucheck/README.md:79-82` | Hard-coded GTX 1650 output | Add MPS-equivalent block |
| 17 | `/Users/cero/Code/gpucheck/README.md:188-190` | "On our GTX 1650, a 256x256 float32 matmul takes about 0.055ms" | Either qualify as illustrative or add MPS counterpart |
| 18 | `/Users/cero/Code/gpucheck/README.md:269-271` | "On our GTX 1650 (Turing, SM75) ..." | Keep as architecture-specific anecdote, but add MPS analogue |
| 19 | `/Users/cero/Code/gpucheck/README.md:362-364` | "Architecture detection verified for: Turing (SM75) ..." | Append MPS verification once Track-A lands |

### A.3 "no MPS" / "weakness" caveats becoming reality (P0/P1)

| # | File:line | Current claim | Required v1.0 change |
|---|---|---|---|
| 20 | `/Users/cero/Code/gpucheck/README.md:367` | "AMD ROCm and Intel XPU are not supported yet" (implicitly excludes MPS) | Replace with explicit MPS-supported, ROCm/XPU-planned phrasing |
| 21 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — bullet "No stride/contiguity fuzzing (only shapes and values)" | gap | Phase 3: move to "Strengths" once Track B (`feat/track-b-strides`) merges per `SESSION_PRECHECK.md:29` |
| 22 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No AMD ROCm or Intel XPU support" | gap | Reword to MPS-supported, ROCm/XPU-planned |
| 23 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "Thread-safety issue in tolerance override stack" | gap | Phase 3: remove once Track C (`feat/track-c-thread-safety`) merges. Source reference: `src/gpucheck/assertions/tolerances.py:26-28` (current "NOT thread-safe" comment) |
| 24 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "Memory leak detection uses process-level metrics (imprecise)" | gap | Phase 3: re-evaluate once MPS path lands; `sanitizers/memory.py:54-66` shows pynvml fallback |

### A.4 Missing dashboard / determinism / stride-fuzzing mentions (P1)

| # | File:line | Current claim | Required v1.0 change |
|---|---|---|---|
| 25 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No HTML/dashboard reporting" | gap | Phase 3: status TBD; engineering Track D (`feat/track-d-bundle`) may include dashboard. If not, document explicitly. |
| 26 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No determinism testing support" | gap | Research QUESTION sub-question 4 covers determinism. Phase 3 docs must reflect whatever Track decisions land. |
| 27 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No CUDA graph testing support" | gap | Document as still-not-supported in v1.0 unless engineering picks it up. |
| 28 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No multi-GPU communication testing (NCCL)" | gap | Same — document explicitly. |
| 29 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No gradient/backward pass testing" | gap | Same. |
| 30 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "Reporting module (console, json, ci) has zero test coverage" | gap | Verify against test runs once testing-team output is available. |
| 31 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Known Weaknesses & Gaps" — "No changelog, no contributing guide, no migration docs" | gap | This Phase 1 closes that gap by drafting all three. |

### A.5 Internal inconsistencies (independent of v1.0 — discovered by docs-skeptic)

| # | File:line | Bug | Required fix |
|---|---|---|---|
| 32 | `/Users/cero/Code/gpucheck/README.md:12` says "8 real bugs"; `/Users/cero/Code/gpucheck/README.md:328-334` lists only 5 in table | Lead-paragraph claim of "8 real bugs in Triton/PyTorch" is unbacked by visible README content | Either expand table to 8 rows with provenance for each bug, or reword the lead to match the table |
| 33 | `/Users/cero/Code/gpucheck/CLAUDE.md` "Git Conventions" claims "conventional commits (type(scope): description)" | Last 25 commits actually use `[ Type ] :` bracket style (verified via `git log --oneline -25`); 0/25 match strict Conventional Commits 1.0 | Either standardize on Conventional Commits going forward (breaking) or document the **actual** bracket convention; CONTRIBUTING_DRAFT.md documents what to do. |
| 34 | `/Users/cero/Code/gpucheck/__init__.py:8` `__version__ = "0.1.0"` | Stale for v1.0 tag | Phase 3 bump to 1.0.0rc1 (or 1.0.0 — see CHANGELOG_DRAFT.md) |
| 35 | `/Users/cero/Code/gpucheck/pyproject.toml:7` `version = "0.1.0"` | Stale | Same — keep aligned with `__init__.py` |

---

## B. Inline docstring gaps in `src/gpucheck/`

Source: `EVIDENCE/reader.md`, refined by `EVIDENCE/skeptic.md`.

### B.1 Missing public-symbol docstrings (19 — P0)

| # | Symbol | File:line | Reason it's public |
|---|---|---|---|
| 1 | `FLOAT_DTYPES` | `/Users/cero/Code/gpucheck/src/gpucheck/decorators/dtypes.py:99` | Re-exported via `gpucheck.__init__:19` and listed in `README.md:150` |
| 2 | `HALF_DTYPES` | `dtypes.py:98` | Same |
| 3 | `ALL_DTYPES` | `dtypes.py:100` | Same |
| 4 | `FP8_DTYPES` | `dtypes.py:101` | Same |
| 5 | `SMALL_SHAPES` | `/Users/cero/Code/gpucheck/src/gpucheck/decorators/shapes.py:18` | Re-exported via `gpucheck.__init__:23` |
| 6 | `MEDIUM_SHAPES` | `shapes.py:25` | Same |
| 7 | `LARGE_SHAPES` | `shapes.py:32` | Same |
| 8 | `EDGE_SHAPES` | `shapes.py:39` | Same |
| 9 | `TILE_SIZES` | `/Users/cero/Code/gpucheck/src/gpucheck/fuzzing/shapes.py:10` | Listed in `__all__` at `shapes.py:235-242` |
| 10 | `PRIMES` | `fuzzing/shapes.py:12` | Same |
| 11 | `POWER_OF_2_BOUNDARIES` | `fuzzing/shapes.py:14` | Same |
| 12 | `LARGE_DIMS` | `fuzzing/shapes.py:16` | Same |
| 13 | `MemoryTracker.start` | `/Users/cero/Code/gpucheck/src/gpucheck/fixtures/profiler.py:142` | Public method on a public class |
| 14 | `MemoryTracker.stop` | `fixtures/profiler.py:146` | Same — returns `MemoryReport` shown in README:240 |
| 15 | `_MutableReport.to_report` | `/Users/cero/Code/gpucheck/src/gpucheck/sanitizers/memory.py:252` | Caller observes via `memory_guard()` yield |
| 16 | `JSONReporter.set_gpu_info` | `/Users/cero/Code/gpucheck/src/gpucheck/reporting/json.py:37` | Public per `reporting/__init__.py:9` |
| 17 | `JSONReporter.add_test_result` | `reporting/json.py:40` | Same |
| 18 | `JSONReporter.add_benchmark` | `reporting/json.py:48` | Same |
| 19 | `JSONReporter.add_memory` | `reporting/json.py:58` | Same |

### B.2 Pytest hooks (4 — convention allows undocumented; flagged P2)

| # | Symbol | File:line |
|---|---|---|
| 20 | `pytest_addoption` | `/Users/cero/Code/gpucheck/src/gpucheck/plugin.py:25` |
| 21 | `pytest_configure` | `plugin.py:46` |
| 22 | `pytest_collection_modifyitems` | `plugin.py:52` |
| 23 | `pytest_terminal_summary` | `plugin.py:68` |

### B.3 Stale type/text in existing docstrings — CUDA-only language inside docstrings (14 — P0/P1)

These are docstrings that exist but require widening for v1.0 MPS:

| # | Symbol | File:line | Stale phrase |
|---|---|---|---|
| 1 | `assertions/close.py` GPU fast-path branch | `close.py:163-176` | Code checks `actual.device.type == "cuda"` only; surrounding docstring mentions only CUDA fast-path |
| 2 | `assertions/tolerances.py` `_DEFAULT_TOLERANCES` comment | `tolerances.py:13-16` | "Calibrated against cuBLAS matmul on Turing/Ampere GPUs." |
| 3 | `decorators/devices.devices` | `decorators/devices.py:55-77` | "auto-detects all available CUDA devices" |
| 4 | `fixtures/benchmark._BenchmarkRunner.__call__` | `fixtures/benchmark.py:156` | "CUDA events for accurate GPU timing" |
| 5 | `fixtures/benchmark.gpu_benchmark` | `fixtures/benchmark.py:247-255` | Same |
| 6 | `fixtures/gpu.detect_gpu` | `fixtures/gpu.py:117` | "Auto-detect a GPU, preferring pynvml (lighter) over torch" — pynvml is NVIDIA-only |
| 7 | `fixtures/gpu.gpu_device` (fixture) | `fixtures/gpu.py:140-148` | "skip if none available" — needs to recognize MPS |
| 8 | `arch/detection.detect_gpus` | `arch/detection.py:259-282` | "Uses pynvml as the primary backend ... Falls back to torch.cuda" |
| 9 | `arch/detection.GPUInfo` dataclass shape | `arch/detection.py:104-122` | NVIDIA-shaped fields (`compute_capability`, `tensor_core_generation`, `cuda_version`) |
| 10 | `arch/compatibility.require_arch` | `arch/compatibility.py:58-92` | Architecture names limited to NVIDIA marketing tags |
| 11 | `arch/compatibility.require_capability` | `arch/compatibility.py:95-120` | (major, minor) is NVIDIA-only |
| 12 | `arch/tensor_cores.compute_tolerance` | `arch/tensor_cores.py:96-135` | `_arch_adjust` only branches on `volta`/`hopper`/`ada` |
| 13 | `analysis/roofline._KNOWN_SPECS` | `analysis/roofline.py:40-46` | No Apple Silicon entries |
| 14 | `sanitizers/race.run_with_sanitizer` | `sanitizers/race.py:129-151` | NVIDIA-only by design (compute-sanitizer); should explicitly say so |

### B.4 THIN docstrings worth expanding (7 — P2)

`assertions/__init__.py:1`, `decorators/dtypes._DtypeGroup:79`,
`fixtures/benchmark._BenchmarkRunner:124`, `fixtures/profiler.MemoryTracker:131`,
`reporting/json.RunRecord:15`, `reporting/json.JSONReporter:26`,
`sanitizers/memory._MutableReport:216`.

### B.5 Aggregate doc gaps

- **Truly missing public docstrings:** **19**
- **Stale CUDA-only inside docstrings:** **14**
- **THIN docstrings:** **7**
- **Pytest hooks (convention allows skip):** 4

---

## C. Missing top-level governance docs

| # | File | Required? | Phase 1 deliverable |
|---|---|---|---|
| 1 | `/Users/cero/Code/gpucheck/CHANGELOG.md` | **YES — release-blocker** | `CHANGELOG_DRAFT.md` (Keep-a-Changelog `[Unreleased]` + `[1.0.0rc1]` placeholders, Phase 3 fills entries from DIFF_LOG) |
| 2 | `/Users/cero/Code/gpucheck/CONTRIBUTING.md` | **YES — release-quality** | `CONTRIBUTING_DRAFT.md` (full structure: dev setup, test layout, PR process, conventional commits decision, expert system pointer, SECURITY pointer) |
| 3 | `/Users/cero/Code/gpucheck/MIGRATION.md` (or `MIGRATION_v0_to_v1.md`) | **YES — release-blocker** | `MIGRATION_v0_to_v1.md` (per-public-API placeholder) |
| 4 | `/Users/cero/Code/gpucheck/CODE_OF_CONDUCT.md` | recommended | **DEFER** — recommend Contributor Covenant 2.1 verbatim, not drafted in Phase 1 |
| 5 | `/Users/cero/Code/gpucheck/SECURITY.md` | recommended | **DEFER to security team**, cite their workspace `.claude/teams/security/v1.0/` |

---

## D. Items Phase 1 explicitly defers to Phase 3

1. Source-file edits — adding the 19 missing docstrings.
2. Filling `CHANGELOG_DRAFT.md` with concrete entries from DIFF_LOG.
3. Filling `MIGRATION_v0_to_v1.md` with concrete API-shape changes
   from research SYNTHESIS + engineering DIFF_LOG.
4. Re-running every `examples/` snippet under both CUDA and MPS.
5. Adding tolerance-table rows for MPS dtypes (depends on research
   sub-question 7).
6. Mermaid architecture diagram for the README.
7. CODE_OF_CONDUCT.md adoption.
8. SECURITY.md (security-team handoff).

---

## E. Cross-team handback notes

- **To research-lead:** Phase 1 docs audit assumes the SYNTHESIS will
  cover (per QUESTION):
  - tolerance multipliers for MPS per-dtype (sub-Q 7),
  - top-10 open `module: mps` PyTorch issues (sub-Q 2),
  - canonical `torch.mps.*` API surface (sub-Q 3).
  Phase 3 docs cite SYNTHESIS for every numeric tolerance and bug
  reference.

- **To engineering-lead:** Phase 3 docs read DIFF_LOG.md to fill
  CHANGELOG `[1.0.0rc1]` and MIGRATION sections. Engineering must
  decide before docs Phase 3:
  - **What is the MPS device-type string?** (`"mps"` is conventional;
    `decorators/devices.py:25-43` will need a third branch.)
  - **Does `GPUInfo` get extended with a `backend: Literal["cuda","mps"]`
    field, or is a separate `MPSDevice`/`AppleSiliconDevice` dataclass
    introduced?** Either choice is documentable; docs needs to know
    which.
  - **Does `@require_arch("Apple-Silicon")` work, or is there a new
    `@require_backend("mps")` decorator?**
  - **Does `pip install gpucheck[mps]` exist as a new extra?**

- **To testing-lead:** docs Phase 3 needs a final validated
  test-count number (today: "120 unit + 235 GPU integration + 53
  example = 408 total") to update `README.md:354-358`.

- **To security-team:** docs Phase 3 expects a `SECURITY.md` from
  security-team's session; docs will only link to it.
