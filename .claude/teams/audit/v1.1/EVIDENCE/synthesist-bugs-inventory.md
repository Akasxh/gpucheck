# Synthesist — gpucheck v1.1 Bugs Inventory

Cross-cut of 8 Wave-1 audits at `/Users/cero/Code/gpucheck/.claude/teams/audit/v1.1/`.
Inputs (binding): `SUMMARIES/{api-dx-grade,security-postmerge,archaeologist-debt,detector-files,mutator-survivors,tracer-runtime,docs-tester,empiricist-mac-bench}.summary.md` and corresponding `EVIDENCE/*.md`.

Schema per issue: `source / severity / impact / ease / file:line / fix sketch / conflicts`.

Severity rubric: CRITICAL = data-loss / wrong-answer / RCE; HIGH = silent-fail with user impact; MEDIUM = ergonomic or correctness on minority paths; LOW = polish, defense-in-depth.
Ease rubric: TRIVIAL = one-line; EASY = single-file; MEDIUM = multi-file; HARD = architectural.

---

## ISS-01 — `compute_tolerance` silently returns float32 defaults for unknown dtypes
- **source**: api-dx-grade #2 (`compute_tolerance` row, "Error-message quality 1")
- **severity**: HIGH
- **impact**: api-dx-grade quotes — `"Silently returns float32 defaults for unknown dtypes. No warning, no error. A test that typos 'flaot16' will pass with the wrong tolerance and the user will never know."` Highest-impact silent-fallback in the API surface; user gets green tests with wrong tolerances.
- **ease**: EASY
- **file:line**: `src/gpucheck/assertions/tolerances.py:70` (`compute_tolerance`); also implicated at `assertions/tolerances.py:60` `_normalize_dtype_name(dtype: Any) -> str`
- **fix sketch**:
  - Add `strict: bool = False` kwarg defaulting to current behavior (api-dx-grade Fix 1).
  - On unknown dtype with `strict=False`, emit `DeprecationWarning` with did-you-mean.
  - Flip default in v1.2; in v1.1 emit warning whenever fallback fires.
- **conflicts/contradictions with**: ISS-04 (same root pattern; both should land together)

## ISS-02 — `@devices` typo silently skips test instead of erroring
- **source**: api-dx-grade #6 (`@devices`, "Error-message quality 2")
- **severity**: HIGH
- **impact**: api-dx-grade quotes — `"A typo silently skips with 'device cudo:0 not available' — looks like a hardware issue, not a typo. **Top fix candidate: validate device strings against a known set, fall back to torch.device(...) only after explicit allow-list miss.**"` Direct test-coverage hole: typo'd device string yields a passing test with zero assertions executed.
- **ease**: EASY
- **file:line**: `src/gpucheck/decorators/devices.py:79` (`devices(*device_args: str)`); `_is_device_available` (`devices.py:66`)
- **fix sketch**:
  - Validate against an allowlist `{"cuda", "cuda:N", "mps", "cpu", "all"}` at decoration time; error on miss.
  - Add `strict: bool = False` per ISS-01 pattern.
- **conflicts/contradictions with**: ISS-01, ISS-04

## ISS-03 — `register_mps_xfail` registers typo'd op names silently into pyproject contract
- **source**: api-dx-grade #18 ("MPS xfail trio", Error-message quality 2)
- **severity**: HIGH
- **impact**: api-dx-grade quotes — `"register_mps_xfail('typo') silently registers the typo. **No way for the user to typo-check their pyproject xfail list against an op registry.**"` Long-term: pyproject.toml accumulates dead xfails that don't match real op names; a real failure stays unmasked because the xfail string was wrong.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/assertions/tolerances.py:227-240` (xfail registry); `is_mps_xfailed`
- **fix sketch**:
  - Maintain a known-op registry (or accept any `<op>.<subcategory>` shape with format-validation).
  - `register_mps_xfail` warns on names not matching a sentinel set.
  - Add `apply_mps_xfail_config` typo report via `pytest_terminal_summary`.
- **conflicts/contradictions with**: none

## ISS-04 — `@dtypes` typo causes opaque crash at execution time, not decoration
- **source**: api-dx-grade #4 (`@dtypes`, Error-message quality 3)
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"A typo'd dtype string hits _resolve_dtype at test execution and crashes with AttributeError: module 'torch' has no attribute 'flaot16' — not actionable. Should validate at decoration with a helpful 'did you mean float16?'."` Late failure with confusing torch-internals message instead of actionable lint.
- **ease**: EASY
- **file:line**: `src/gpucheck/decorators/dtypes.py:108` (`dtypes`), `dtypes.py:37` (`_resolve_dtype`)
- **fix sketch**:
  - Eagerly validate against `_DEFAULT_TOLERANCES` keys (or a canonical dtype-name set) at decoration.
  - Did-you-mean message via `difflib.get_close_matches`.
- **conflicts/contradictions with**: ISS-01, ISS-02

## ISS-05 — `tolerance_context` absolute-override semantics bypass MPS overlay (silent test pass)
- **source**: api-dx-grade #3 (`tolerance_context`, Ergonomics 3); tracer-runtime finding 5
- **severity**: HIGH
- **impact**: api-dx-grade quotes — `"the override is *absolute* — it ignores dtype, k_dim, and MPS multipliers entirely (see tolerances.py:91-93: 'if overrides: return overrides[-1]'). A user expecting 'double the defaults' gets a fixed scalar."` Tracer-runtime confirms — `"active tolerance_context(7.7e-9, 8.8e-9) returns those values verbatim even with device_type='mps', ignoring 2× multiplier. ... user with tight tolerance gets no MPS slack."` Two independent audits converge on the same trap.
- **ease**: EASY
- **file:line**: `src/gpucheck/assertions/tolerances.py:91-93` (override bypass) and `tolerances.py:114` (`tolerance_context`)
- **fix sketch**:
  - Add `tolerance_context(scale=2.0)` mode that multiplies resolved (atol, rtol) instead of overriding.
  - Keep absolute-override mode but emit one-time warning when `device_type=="mps"` AND override active AND multiplier would have fired.
- **conflicts/contradictions with**: none (api-dx-grade and tracer-runtime converge)

## ISS-06 — `assert_close` parameters typed as `Any`, no `TensorLike` Protocol
- **source**: api-dx-grade #1 (`assert_close`, Type safety 2)
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"Signature is actual: Any, expected: Any with no overloads. mypy users get no narrowing for torch.Tensor vs np.ndarray vs cupy.ndarray. Missing TypeAlias/TensorLike Protocol. The internal _to_numpy already enumerates the supported shapes — that information should surface as a Protocol."` Headline API gives mypy users zero help.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/assertions/close.py:109` (`assert_close`); `close.py:22` (`_to_numpy`)
- **fix sketch**:
  - Define `TensorLike` Protocol with `.detach()`, `.cpu()`, `.numpy()`, `__cuda_array_interface__`, and `__array__`.
  - Use under `TYPE_CHECKING` to avoid runtime cost.
- **conflicts/contradictions with**: none

## ISS-07 — `baseline_2x: bool` is a magic boolean blocking non-2x scales
- **source**: api-dx-grade "Candidate API to Deprecate"
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"It's a magic boolean that hardcodes the FlashAttention 2x convention; it can't express 1.5x or other scales; and it conflicts subtly with atol=/rtol= overrides (the code path branches on whether the user set both). Replace with tolerance_scale: float | None = None"` Locks the API into one tolerance profile and creates branching that mutator-survivors confirms is poorly tested.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/assertions/close.py:109` (signature); `close.py:152-160` (branch); `close.py:170-171` (eff_atol/rtol multiplier)
- **fix sketch**:
  - Add `tolerance_scale: float | None = None` keyword-only param.
  - Keep `baseline_2x` with `DeprecationWarning` in v1.1; remove in v1.2.
- **conflicts/contradictions with**: none

## ISS-08 — `assertions/close.py` top-level `import torch as _torch` defeats lazy-import contract
- **source**: detector-files Top-3 #1 (highest-impact single-file finding)
- **severity**: HIGH
- **impact**: detector-files quotes — `"top-level import torch as _torch. Only top-level torch import in src/. **Defeats the lazy-import contract** advertised in CLAUDE.md. Major."` Imports torch at the moment any user does `import gpucheck.assertions` — undermines a documented "import-time zero-cost" design pillar.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/assertions/close.py:13-19`
- **fix sketch**:
  - Replicate the `_torch_mod()` accessor pattern from `fuzzing/inputs.py:19`.
  - Replace usages of `_torch` with `_torch_mod()` calls.
- **conflicts/contradictions with**: none

## ISS-09 — Two `compute_tolerance` functions with different signatures (naming collision)
- **source**: detector-files Top-3 #2; Top-10 #2
- **severity**: HIGH
- **impact**: detector-files quotes — `"compute_tolerance(dtype: str, k_dim: int, gpu_info: GPUInfo | None) -> tuple[float, float] (tensor_cores.py:96) — name **collides** with the more general assertions.tolerances.compute_tolerance (tolerances.py:70). Two different compute_tolerance functions with different signatures. **HIGH-IMPACT** naming collision."` Importing the wrong one is a footgun; cross-module type assumptions silently break.
- **ease**: EASY
- **file:line**: `src/gpucheck/arch/tensor_cores.py:96`; `src/gpucheck/assertions/tolerances.py:70`
- **fix sketch**:
  - Rename `arch.tensor_cores.compute_tolerance` to `compute_tolerance_arch_aware` or fold into `tolerances.py` as a private helper.
  - Re-export from `arch/__init__.py` if any external caller depended on it.
- **conflicts/contradictions with**: none

## ISS-10 — Two `MemoryReport` types at different paths in same package
- **source**: detector-files Top-3 #2; Top-10 #3
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"sanitizers/__init__.py:14 aliases MemoryReport = SanitizerMemoryReport, while fixtures/profiler.py:28 also defines a class named MemoryReport. Two MemoryReport types at different paths in the same package."` IDE autocomplete will dispatch to the wrong one; isinstance checks across modules will fail.
- **ease**: EASY
- **file:line**: `src/gpucheck/sanitizers/__init__.py:14`; `src/gpucheck/fixtures/profiler.py:28`
- **fix sketch**:
  - Rename the alias to `SanitizerMemoryReport` only (drop the `MemoryReport` alias) OR rename `fixtures/profiler.py:28` to `FixtureMemoryReport`.
  - Update `_LAZY_MAP` and tests.
- **conflicts/contradictions with**: none

## ISS-11 — Triplicated GPU-detection logic across modules (drift risk)
- **source**: detector-files Top-3 #1 ("Duplicated logic"); Top-10 #4, #5
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"GPU detection 3× (fixtures/gpu.py:43,90 / arch/detection.py:133,206 / plugin.py:10-22). _median twice (analysis/regression.py:204 + analysis/roofline.py:310). Three gpu_available shims. Drift risk on every fix."` Every CUDA/MPS/pynvml fix has to land in three places.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/fixtures/gpu.py:43,90`; `src/gpucheck/arch/detection.py:133,206`; `src/gpucheck/plugin.py:10-22`
- **fix sketch**:
  - Make `arch/detection.py` the single source of truth; have fixtures and plugin shim to it.
  - Collapse three `gpu_available` shims to one re-exported symbol.
- **conflicts/contradictions with**: ISS-12 (same DRY bucket)

## ISS-12 — Duplicated `_median` helper in sibling analysis modules
- **source**: detector-files Top-3 #1; Top-10 #10
- **severity**: LOW
- **impact**: detector-files quotes — `"_median twice (analysis/regression.py:204 + analysis/roofline.py:310). Drift risk on every fix."` Bug-fix lag if one is patched without the other.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/analysis/regression.py:204`; `src/gpucheck/analysis/roofline.py:310`
- **fix sketch**:
  - Move `_median` to a shared `analysis/_stats.py`; import from both.
- **conflicts/contradictions with**: ISS-11

## ISS-13 — Bare `except Exception` violates project standard (7 sites)
- **source**: detector-files Top-3 #3; Top-10 #7
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"arch/detection.py:157,229 + 5 places in backends/mps.py (lines 99,137,141,148,191) — directly violates CLAUDE.md 'no bare except' rule."` Swallows real errors (driver faults, KeyboardInterrupt subclasses) and impedes diagnosis.
- **ease**: EASY
- **file:line**: `src/gpucheck/arch/detection.py:157,229`; `src/gpucheck/backends/mps.py:99,137,141,148,191`
- **fix sketch**:
  - Replace each with specific exception tuple appropriate to the call site (likely `(ImportError, RuntimeError, OSError)` per existing patterns).
- **conflicts/contradictions with**: none

## ISS-14 — `decorators/parametrize.py` uses TypeError to detect callback arity (fragile)
- **source**: detector-files Top-3 #2; Top-10 #6
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"uses try/except TypeError to detect callback arity. A 3-arg skip predicate that internally raises TypeError gets silently downgraded."` False-positive arity detection silently swallows real bugs in user-supplied skip predicates.
- **ease**: EASY
- **file:line**: `src/gpucheck/decorators/parametrize.py:135-139`
- **fix sketch**:
  - Use `inspect.signature` to count required params before calling.
  - Catch `TypeError` only on the call itself, not as an arity probe.
- **conflicts/contradictions with**: none

## ISS-15 — `memory_guard` yields private `_MutableReport` (underscore type leaks into public API)
- **source**: detector-files Top-3 #3; Top-10 #8
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"memory_guard (public) yields a _MutableReport (private). Underscore-prefixed type leaks into public API."` Users either rely on a private symbol or get type-checker noise.
- **ease**: EASY
- **file:line**: `src/gpucheck/sanitizers/memory.py:142,216`
- **fix sketch**:
  - Rename `_MutableReport` to `MutableMemoryReport` and re-export.
  - Or wrap the yielded object in a public `MemoryGuardSession` dataclass.
- **conflicts/contradictions with**: none

## ISS-16 — `_load_pyproject_config` swallows all exceptions silently
- **source**: security-postmerge PM-2; detector-files plugin.py boundary error
- **severity**: LOW
- **impact**: security-postmerge quotes — `"bare except Exception: pass on plugin.py:86-88 silently absorbs any error including MemoryError and TOML-parser exceptions — masking misconfiguration from the user."` Users with malformed pyproject get no signal.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/plugin.py:86-88`
- **fix sketch**:
  - Narrow to `(OSError, tomllib.TOMLDecodeError)` and emit a `RuntimeWarning` per security-postmerge fix sketch.
- **conflicts/contradictions with**: none

## ISS-17 — HTML reporter `class="{klass}"` and `style="background:{bg}"` interpolated raw (defense-in-depth gap)
- **source**: security-postmerge PM-5
- **severity**: LOW
- **impact**: security-postmerge quotes — `"reporting/html.py escapes attacker-tainted strings but interpolates class='{klass}' (lines 171/218/243) and style='background:{bg}' (line 93) raw. Not exploitable today (whitelisted constants); one wrong line in v1.1 → stored-XSS in CI artifacts."` Future-proofing.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/reporting/html.py:90-96` (_pill bg), `:171,218,243` (klass)
- **fix sketch**:
  - Wrap `klass` and `bg` in `_esc()` defensively.
  - Add a hostile-name regression test that feeds `name="</td><script>alert(1)</script>"` and asserts no `<script>` in output.
  - Add `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">`.
- **conflicts/contradictions with**: none

## ISS-18 — `_to_numpy` slow-path crashes on torch <2.1 with stride-fuzzed tensors (no `.contiguous()`)
- **source**: security-postmerge PM-4; mutator-survivors note (close.py slow path)
- **severity**: MEDIUM
- **impact**: security-postmerge quotes — `"Track-B's stride helpers produce non-contiguous and stride-0 tensors. assertions/close.py:_to_numpy calls .numpy() without .contiguous(). Availability bug on torch <2.1 (RuntimeError). One-line fix."` New Track-B stride fuzzer creates inputs the slow path can't accept on older torch.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/assertions/close.py:22-67` (`_to_numpy`)
- **fix sketch**:
  - Force contiguity: `t = tensor.detach().cpu().contiguous()`.
- **conflicts/contradictions with**: none

## ISS-19 — `flush_l2=True` warning fires every benchmark call on MPS (no gate)
- **source**: tracer-runtime finding 2; H2
- **severity**: LOW
- **impact**: tracer-runtime quotes — `"flush_l2=True warning fires every call in the fixture (no gate), unlike MPSBackend.flush_l2 which gates on _FLUSH_L2_WARNED. UX bug."` 100 benchmark calls in one session → 100 duplicate warnings. Minor UX noise.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/fixtures/benchmark.py:304-310` (no gate); compare `backends/mps.py:166-176`
- **fix sketch**:
  - Add module-level `_FIXTURE_FLUSH_L2_WARNED = False` latch (or share `_FLUSH_L2_WARNED` from MPSBackend).
- **conflicts/contradictions with**: ISS-21 (same architectural root)

## ISS-20 — L2 flush silently absent on MPS (no replacement / approximation offered)
- **source**: tracer-runtime finding 3; H3; empiricist-mac-bench process notes
- **severity**: MEDIUM
- **impact**: tracer-runtime quotes — `"L2 flush on MPS is silently absent — warning fires, no replacement (e.g., 16MB buffer fill) offered. MPS benchmarks of small kernels are systematically optimistic vs CUDA."` Users comparing MPS to CUDA benchmarks get inflated MPS numbers because L2 stays hot.
- **ease**: EASY
- **file:line**: `src/gpucheck/fixtures/benchmark.py:304-310`; `_run_mps:283-327`
- **fix sketch**:
  - Add a `torch.empty((10_000_000,), device='mps').fill_(0.0); torch.mps.synchronize()` step on `do_flush=True` (per tracer-runtime suggestion).
  - Document the approximation in the warning and in benchmark docs.
- **conflicts/contradictions with**: none

## ISS-21 — Fixture `_run_mps` is a standalone duplicate of `MPSBackend.event_timer` (architectural drift)
- **source**: tracer-runtime finding 1; tracer-runtime hidden-cost #4
- **severity**: MEDIUM
- **impact**: tracer-runtime quotes — `"fixtures/benchmark.py _run_mps is a standalone duplicate of MPSBackend.event_timer. Backend Protocol is NOT used by the fixture. v1.1 unification candidate."` Bug fixes to one path don't propagate to the other; the carefully designed Backend Protocol is bypassed by gpucheck's own fixture.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/fixtures/benchmark.py:283-327` (`_run_mps`); compare `src/gpucheck/backends/mps.py:111-129` (`MPSBackend.event_timer`)
- **fix sketch**:
  - Refactor `_run_mps` to call `backends.get_backend("mps").event_timer()` per round.
  - Add a regression test that ensures the two paths produce equivalent timing.
- **conflicts/contradictions with**: ISS-19, ISS-20 (same code site)

## ISS-22 — Fast-path `torch.allclose` is a 1.77 ms sunk cost on guaranteed-mismatch slow paths
- **source**: tracer-runtime finding 6; finding "Slow-path tax"
- **severity**: LOW
- **impact**: tracer-runtime quotes — `"Slow-path tax: 1.77 ms of dead work — when torch.allclose returns False on MPS, that 1.77 ms is wasted before numpy fallback. For guaranteed-mismatch tests this is pure overhead."` Performance-only on a path most users hit only on test failures.
- **ease**: HARD
- **file:line**: `src/gpucheck/assertions/close.py:185` (fast-path), `:192-201` (slow path)
- **fix sketch**:
  - Investigate whether a cheaper "must mismatch" predicate (e.g. random sub-sample, max-abs-diff scalar) avoids the full reduce.
  - Or skip fast-path for explicit `nan_equal=False` + heuristic on dtype.
  - Likely defer; not a correctness issue.
- **conflicts/contradictions with**: none

## ISS-23 — `format_mismatch_report` is the slow-path bottleneck (~16 ms) on MPS
- **source**: tracer-runtime "Slowest step overall"
- **severity**: LOW
- **impact**: tracer-runtime quotes — `"format_mismatch_report at assertions/reporting.py — ~16 ms for rich panel + histogram on 1M-element mismatch."` Test failure UX is fine but the report dominates measured time on assertion failures.
- **ease**: MEDIUM
- **file:line**: `src/gpucheck/assertions/reporting.py:17` (`format_mismatch_report`)
- **fix sketch**:
  - Lazy-render Rich panel only when stdout is a TTY (or when explicitly requested).
  - Cache histogram bin edges across repeated assert_close calls.
- **conflicts/contradictions with**: none

## ISS-24 — REFUTED claim: silent fp64 downcast on MPS (linguist-v3 finding refuted by tracer)
- **source**: tracer-runtime finding 4 ("REFUTED linguist-v3 silent fp64 downcast"); H4
- **severity**: N/A (this is a contradiction, not a bug to fix in v1.1)
- **impact**: tracer-runtime quotes — `"On torch 2.11 every fp64-on-MPS path tested raises TypeError: Cannot convert a MPS Tensor to float64.... **Fail-loud, not silent.** May be different on torch <2.11. **This is a v1.1 plan reconciliation item — linguist-v3's 'silent-downcast catcher API' is not needed if torch 2.11 already fails loud.**"` This contradicts a prior research-round-3 finding; flag for plan reconciliation.
- **ease**: N/A
- **file:line**: N/A
- **fix sketch**:
  - Drop linguist-v3 silent-downcast catcher API from v1.1 plan.
  - Optionally add a note in MPS docs: "fp64 on MPS is a hard TypeError; if you need fp64, fall back to CPU".
- **conflicts/contradictions with**: linguist-v3 (refuted)

## ISS-25 — `assertions/reporting.py` has 83 surviving mutants (38% of total) — substring-match TEST_BUGs
- **source**: mutator-survivors highest-survivor file; "Top-3 highest-leverage new tests" #1
- **severity**: HIGH
- **impact**: mutator-survivors quotes — `"src/gpucheck/assertions/reporting.py — 83 of 221 (38%). Has only 3 substring-only tests for format_mismatch_report; zero tests for _error_histogram."` Reporting module's behavior is essentially unverified — any regression to numeric content of failure reports passes CI.
- **ease**: EASY
- **file:line**: `src/gpucheck/assertions/reporting.py:17` (`format_mismatch_report`); `_error_histogram` private
- **fix sketch**:
  - Implement `test_report_contains_correct_max_error_value` per mutator-survivors recommendation (kills ~30 mutants).
  - Replace `assert "X" in report` with equality on row content.
  - Add ANSI-style assertion (`"\x1b[" in rpt`).
- **conflicts/contradictions with**: none

## ISS-26 — Tolerance config loader (`tolerances_from_config`, `apply_config_tolerances`) has zero tests
- **source**: mutator-survivors #2 ("Top-3 highest-leverage new tests"); IDs 50, 74-90, 92
- **severity**: HIGH
- **impact**: mutator-survivors quotes — `"covers entirely untested config loader → kills ~17 tolerances.py mutants"` (#2 of top-3 leverage tests). End-to-end pyproject → `compute_tolerance` round-trip is unverified; a malformed config can silently no-op.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/assertions/tolerances.py:138` (`tolerances_from_config`), `:164` (`apply_config_tolerances`), `:175` (`reset_config_tolerances`)
- **fix sketch**:
  - Add `tests/test_tolerances_config.py` with `test_tolerances_from_config_parses_overrides` and `test_apply_config_tolerances_round_trip` (mutator-survivors gives the exact code).
- **conflicts/contradictions with**: none

## ISS-27 — `_DEFAULT_TOLERANCES` table values not pinned in tests (tautological dict iteration)
- **source**: mutator-survivors #3 ("Top-3 highest-leverage new tests"); IDs 1-3, 13-21
- **severity**: MEDIUM
- **impact**: mutator-survivors quotes — `"@pytest.mark.parametrize over _DEFAULT_TOLERANCES.items() with hard-coded expected pairs (not the dict itself) → kills ~12 dict-value mutants currently shielded by tautological iteration"`. Any tolerance-table value mutation passes CI today.
- **ease**: TRIVIAL
- **file:line**: `tests/test_tolerances.py` (existing); pin against `src/gpucheck/assertions/tolerances.py:13-26`
- **fix sketch**:
  - `@pytest.mark.parametrize("name,expected", list(_DEFAULT_TOLERANCES.items()))` with hard-coded expected pairs (mutator-survivors gives exact snippet).
- **conflicts/contradictions with**: none

## ISS-28 — k_dim=1 boundary uncovered (`max(k_dim, 1)`, `k_dim > 0` mutations survive)
- **source**: mutator-survivors IDs 55, 58
- **severity**: LOW
- **impact**: mutator-survivors quotes — `"With > 1, k_dim=1 would skip scaling (currently sqrt(1/128)≈0.088×). Fix: assert compute_tolerance('float32', k_dim=1) ≠ base."` Boundary case in tolerance scaling is unverified.
- **ease**: TRIVIAL
- **file:line**: `tests/test_tolerances.py`; covers `src/gpucheck/assertions/tolerances.py:101-103`
- **fix sketch**:
  - Add `test_kdim_one_still_scales` per mutator-survivors recommendation.
- **conflicts/contradictions with**: none

## ISS-29 — MIGRATION.md `fuzz_strides` / `fuzz_strides_for_category` signature wrong
- **source**: docs-tester M-B8 (top-3 docs to fix #1)
- **severity**: HIGH
- **impact**: docs-tester quotes — `"M-B8 — fuzz_strides / fuzz_strides_for_category wrong signature in example"`. Real signature is `fuzz_strides(shape, dtype, *, n=None, device='cpu', seed=None, categories=None)`. Docs call it as `fuzz_strides(shape=(8,16,32), n=20, seed=42)` — `TypeError: missing 1 required positional argument: 'dtype'`. Every reader copy-pasting fails.
- **ease**: TRIVIAL
- **file:line**: `MIGRATION.md` §7 (L264-290)
- **fix sketch**:
  - Replace with `fuzz_strides(shape=(8,16,32), dtype=torch.float32, n=20, seed=42)`.
  - Fix `fuzz_strides_for_category` arg order.
- **conflicts/contradictions with**: none

## ISS-30 — MIGRATION.md `assert_deterministic` wrong kwargs (`args=`, `runs=`, `atol=`)
- **source**: docs-tester M-B9
- **severity**: HIGH
- **impact**: docs-tester quotes — `"M-B9 — assert_deterministic wrong kwargs"`. Real signature is `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)`. Docs say `args=(x,y), runs=2, atol=0.0` — those forward into `fn` as kwargs and explode.
- **ease**: TRIVIAL
- **file:line**: `MIGRATION.md` §8 (L305-314)
- **fix sketch**:
  - Replace with `assert_deterministic(my_kernel, x, y, n=2, seed=0)`.
- **conflicts/contradictions with**: none

## ISS-31 — MIGRATION.md `available_backends()` return type wrong + comment placement
- **source**: docs-tester M-B1
- **severity**: HIGH
- **impact**: docs-tester quotes — `"M-B1 — return type + comment placement wrong"`. Real return is `list[Backend]`; docs claim `("cuda",)` / `("cuda","mps")` (string tuple). Comment "raises if MPS not available" is on the `mps` line but `get_backend("cuda")` raises first on a CUDA-less host.
- **ease**: TRIVIAL
- **file:line**: `MIGRATION.md` §1 (L45-59)
- **fix sketch**:
  - Update prose to show `list[Backend]` real shape, or add `available_backend_names()` helper.
  - Fix comment placement; wrap example in try/except.
- **conflicts/contradictions with**: none

## ISS-32 — MIGRATION.md xfail-registry runtime gotcha not documented
- **source**: docs-tester M-B7
- **severity**: MEDIUM
- **impact**: docs-tester quotes — `"M-B7 — xfail registry empty outside pytest (gotcha not documented)"`. `is_mps_xfailed("softmax.large_attention")` returns False outside pytest because registry only loads in `pytest_configure`. Reader running snippet in REPL gets confusing False results.
- **ease**: TRIVIAL
- **file:line**: `MIGRATION.md` §5 (L210-215)
- **fix sketch**:
  - Add explicit `# only valid inside pytest_configure` note.
  - Add a `gpucheck.register_mps_xfail("softmax.large_attention")` line so the snippet is self-contained.
- **conflicts/contradictions with**: none

## ISS-33 — README §6 wrong fence language: `python` block contains shell command
- **source**: docs-tester R-B2
- **severity**: HIGH
- **impact**: docs-tester quotes — `"R-B2 — wrong fence lang"`. Block fenced as ` ```python ` but contents are `python -c "..."` (shell). Copy-paste into a `.py` file is a SyntaxError.
- **ease**: TRIVIAL
- **file:line**: `README.md` L66-76
- **fix sketch**:
  - Change fence to ` ```bash ` or strip `python -c "` wrapper.
- **conflicts/contradictions with**: none

## ISS-34 — README §9 wrong claimed numeric output (`+12.0%` / `d=4.21` vs actual `+11.7%` / `d=7.48`)
- **source**: docs-tester R-B16
- **severity**: HIGH
- **impact**: docs-tester quotes — `"R-B16 — wrong numeric output +12.0% / d=4.21 (actual +11.7% / d=7.48) in regression-detector example"`. README claims regression detector output that doesn't match the shipped code; readers will assume a bug.
- **ease**: TRIVIAL
- **file:line**: `README.md` L282-291
- **fix sketch**:
  - Update prose to actual `+11.7%` / `d=7.48`, or add `# doctest: +SKIP`.
- **conflicts/contradictions with**: none

## ISS-35 — README + CONTRIBUTING claim `pytest tests/gpu_integration/` auto-skips; on MPS hosts → 52 failures
- **source**: docs-tester R-B21 / T-B5 (top-3 docs to fix #3)
- **severity**: HIGH
- **impact**: docs-tester quotes — `"pytest tests/gpu_integration/ claims auto-skip without GPU; on MPS hosts this **FAILS 52 tests**"`. The integration tests are hard-coded to CUDA; the auto-skip claim is wrong on Apple Silicon.
- **ease**: EASY
- **file:line**: `README.md` L452-454; `CONTRIBUTING.md` L78-87; covers `tests/gpu_integration/*`
- **fix sketch**:
  - Either: gate `gpu_integration/` collection on `torch.cuda.is_available()` only (not the union), OR
  - Update doc to "skipped on hosts without **CUDA**" (more accurate to current code).
- **conflicts/contradictions with**: none

## ISS-36 — README §4/§6/§7 hard-code `device="cuda"` despite advertising MPS as first-class
- **source**: docs-tester R-B6, R-B8, R-B12, R-B13 (top-3 docs to fix #2 partial)
- **severity**: MEDIUM
- **impact**: docs-tester quotes — `"R-B8 / R-B12 / R-B13 — hard-coded CUDA on a CUDA-less host"`. README "Step by step usage guide" walk-through fails on every Mac dev box even though MPS is advertised as a supported backend.
- **ease**: EASY
- **file:line**: `README.md` §4-§7 (L159-251)
- **fix sketch**:
  - Inject a tiny helper `device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")`.
  - Or add `# doctest: +SKIP` and a "this snippet requires CUDA" callout.
- **conflicts/contradictions with**: none

## ISS-37 — `src/gpucheck/__init__.py` has zero `Examples:` doctests
- **source**: docs-tester I-B1; nice-to-have #10
- **severity**: LOW
- **impact**: docs-tester quotes — `"src/gpucheck/__init__.py has zero Examples: docstrings."` `python -m doctest src/gpucheck/__init__.py` finds no blocks.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/__init__.py:1-105`
- **fix sketch**:
  - Add a smoke doctest: `>>> import gpucheck; gpucheck.__version__`.
- **conflicts/contradictions with**: none

## ISS-38 — Top-level `_LAZY_MAP` missing 7 public symbols (discoverability gap)
- **source**: api-dx-grade Fix 2 ("top-3 v1.1 fixes"); detector-files `__init__.py` finding
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"Promote 7 hidden symbols to top-level _LAZY_MAP — memory_tracker, gpu_device, fuzz_strides, ShapeStrategy, StrideStrategy, requires_determinism, assert_deterministic."` `gpucheck.<TAB>` is the discoverability test; today seven public symbols fail it.
- **ease**: EASY
- **file:line**: `src/gpucheck/__init__.py:75-105` (`_LAZY_MAP`, `__all__`, `TYPE_CHECKING`)
- **fix sketch**:
  - Extend `_LAZY_MAP` and `__all__` for the 7 symbols.
  - Mirror in `TYPE_CHECKING` imports.
- **conflicts/contradictions with**: none

## ISS-39 — `apply_mps_xfail_config` and `reset_mps_xfail` exported by submodule but missing from top-level `_LAZY_MAP`
- **source**: detector-files `__init__.py:14-40`; `assertions/__init__.py:22-25`
- **severity**: LOW
- **impact**: detector-files quotes — `"apply_mps_xfail_config and reset_mps_xfail are exported by assertions/__init__.py:22-25 but **not** in _LAZY_MAP here, while register_mps_xfail / mps_xfail_list / is_mps_xfailed are. Asymmetric coverage."` Inconsistent surface; users have to remember which path each symbol lives at.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/__init__.py:11-40`
- **fix sketch**:
  - Add the two symbols to `_LAZY_MAP` and `__all__` (sub-task of ISS-38).
- **conflicts/contradictions with**: ISS-38 (same fix lane)

## ISS-40 — `ShapeStrategy` / `StrideStrategy` `__new__`-as-factory blocks isinstance/mypy
- **source**: api-dx-grade #13, #14 (lowest score: 2.8 each)
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"__new__ returns Any because Hypothesis SearchStrategy isn't always importable. Class-as-factory pattern (__new__ returning a non-Self) breaks isinstance(s, ShapeStrategy) and confuses mypy."` Locks the API: cannot make `ShapeStrategy` actually behave as a class without breaking callers.
- **ease**: HARD
- **file:line**: `src/gpucheck/fuzzing/shapes.py:182-200` (`ShapeStrategy.__new__`); `src/gpucheck/fuzzing/strides.py:260` (`StrideStrategy`)
- **fix sketch**:
  - Add canonical `shape_strategy(...)` and `stride_strategy(...)` factory functions.
  - Keep `ShapeStrategy`/`StrideStrategy` aliases with `DeprecationWarning`; remove in v2.0.
- **conflicts/contradictions with**: none

## ISS-41 — `@require_arch` vs `@requires_determinism` naming inconsistency
- **source**: api-dx-grade #15 ("require_arch" Discoverability 2)
- **severity**: LOW
- **impact**: api-dx-grade quotes — `"actual code uses require_arch... naming inconsistency vs requires_determinism — one says requires, the other says require."` Cognitive overhead; users will mis-spell which decorator is which.
- **ease**: EASY
- **file:line**: `src/gpucheck/arch/compatibility.py:58` (`require_arch`); `src/gpucheck/sanitizers/determinism.py:137` (`requires_determinism`)
- **fix sketch**:
  - Add alias `requires_arch = require_arch` and document `requires_arch` as canonical.
  - `DeprecationWarning` on `require_arch` in v1.2.
- **conflicts/contradictions with**: none

## ISS-42 — Two parallel arch maps (`SM_ARCH_MAP` vs `SM_TO_ARCH`) — drift risk
- **source**: detector-files `arch/compatibility.py` Naming
- **severity**: LOW
- **impact**: detector-files quotes — `"SM_ARCH_MAP (compatibility.py:20) and arch.detection.SM_TO_ARCH (detection.py:14). Same data, different keys ('SM80' vs (8, 0)). Single source of truth would reduce drift risk."` New SM levels (e.g. SM120 Blackwell variants) need updating in two places.
- **ease**: EASY
- **file:line**: `src/gpucheck/arch/compatibility.py:20`; `src/gpucheck/arch/detection.py:14`
- **fix sketch**:
  - Pick one canonical (likely `SM_TO_ARCH` keyed by `(int, int)`), derive the other via `_cc_to_sm_tag`.
- **conflicts/contradictions with**: none

## ISS-43 — `fixtures/__init__.py` registers fixtures in `_LAZY_MAP` while `plugin.py` also registers them
- **source**: detector-files `fixtures/__init__.py:10,12,16` Top-10 #9
- **severity**: LOW
- **impact**: detector-files quotes — `"plugin.py versions are what pytest actually picks up; the entries in _LAZY_MAP here are unreachable for fixture purposes."` Two registrations; pytest fixture override depends on registration order.
- **ease**: EASY
- **file:line**: `src/gpucheck/fixtures/__init__.py:10,12,16`; `src/gpucheck/plugin.py:127,137,189`
- **fix sketch**:
  - Drop `_LAZY_MAP` fixture entries; keep dataclasses (`BenchmarkResult`, `GPUDevice`, `MemoryReport`).
  - Document that fixtures are registered solely in `plugin.py`.
- **conflicts/contradictions with**: none

## ISS-44 — `memory_tracker` leak detected → `RuntimeWarning`, not a test failure
- **source**: api-dx-grade #9 (`memory_tracker`, Error-message quality 2)
- **severity**: MEDIUM
- **impact**: api-dx-grade quotes — `"Leak detected → RuntimeWarning('GPU memory leak detected: 1.5MB not freed'). Warnings can be silenced; a leak should produce a *test failure*, not a warning."` Memory leaks are exactly the bugs a sanitizer should fail on.
- **ease**: EASY
- **file:line**: `src/gpucheck/sanitizers/memory.py:142` (`memory_guard`); `src/gpucheck/fixtures/profiler.py:131` (`MemoryTracker`)
- **fix sketch**:
  - Add `fail_on_leak: bool = False` kwarg in v1.1 (warn-only by default).
  - Promote to default `True` in v1.2 with `DeprecationWarning`.
- **conflicts/contradictions with**: ISS-15 (memory_guard already flagged for cleanup)

## ISS-45 — `_safe_import_numpy` is a misleading wrapper for non-optional dep
- **source**: detector-files `assertions/reporting.py:11-14`
- **severity**: LOW
- **impact**: detector-files quotes — `"_safe_import_numpy is a one-liner wrapper for import numpy as np; numpy is **already** a non-optional dep... Wrapper can be inlined or removed."` Name implies try/except that doesn't exist.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/assertions/reporting.py:11-14`
- **fix sketch**:
  - Inline: `import numpy as np` at top of module.
- **conflicts/contradictions with**: none

## ISS-46 — `arch.tensor_cores` imports `_DEFAULT_TOLERANCES` as `_CANONICAL_TOLERANCES` (private cross-module dip)
- **source**: detector-files `arch/tensor_cores.py:9-11`
- **severity**: LOW
- **impact**: detector-files quotes — `"Imports _DEFAULT_TOLERANCES from assertions.tolerances as _CANONICAL_TOLERANCES — dipping into private symbols across modules. Either expose officially or duplicate."` Coupling on a private name; renaming `_DEFAULT_TOLERANCES` would silently break `arch`.
- **ease**: EASY
- **file:line**: `src/gpucheck/arch/tensor_cores.py:9-11`; `src/gpucheck/assertions/tolerances.py:13`
- **fix sketch**:
  - Promote `_DEFAULT_TOLERANCES` to public `DEFAULT_TOLERANCES` (or add a function `get_default_tolerances()`).
  - Re-export from `assertions/__init__.py`.
- **conflicts/contradictions with**: ISS-09 (same compute_tolerance refactor lane)

## ISS-47 — `sanitizers/race.py:257` `os.unlink(script_path)` not in try (cleanup error masks original)
- **source**: detector-files `sanitizers/race.py:257`
- **severity**: LOW
- **impact**: detector-files quotes — `"os.unlink(script_path) is not in a try — if cleanup fails (race / permission), traceback masks original exception."` Diagnostic hazard during a sanitizer run already in failure mode.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/sanitizers/race.py:257`
- **fix sketch**:
  - Wrap in `contextlib.suppress(OSError)`.
- **conflicts/contradictions with**: none

## ISS-48 — `sanitizers/race.py:117` local `warnings` shadows imported `warnings` module
- **source**: detector-files `sanitizers/race.py:117`
- **severity**: LOW
- **impact**: detector-files quotes — `"Local warnings shadows the imported warnings module. Confusing — rename local to warning_lines."`
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/sanitizers/race.py:117`
- **fix sketch**:
  - Rename local to `warning_lines`.
- **conflicts/contradictions with**: none

## ISS-49 — `plugin.py` `import torch` inside `gpu_device` unguarded (raises ImportError instead of pytest.skip)
- **source**: detector-files `plugin.py:151` Boundary errors
- **severity**: MEDIUM
- **impact**: detector-files quotes — `"The import torch inside gpu_device (plugin.py:151) is unguarded — if the user passes --gpu-device cuda:1 and torch isn't installed, this raises ImportError instead of pytest.skip."` Test-collection failure rather than graceful skip on a torch-less environment.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/plugin.py:151`
- **fix sketch**:
  - Wrap in try/except, call `pytest.skip("torch not installed")` on ImportError.
- **conflicts/contradictions with**: none

## ISS-50 — Reporting module had stale "zero test coverage" claim in CLAUDE.md
- **source**: archaeologist-debt #4 (CLAUDE.md "Known Weaknesses" stale)
- **severity**: LOW (doc hygiene)
- **impact**: archaeologist-debt quotes — `"CLAUDE.md still says 'Reporting module (console, json, ci) has zero test coverage' — but tests/test_reporting_console.py, tests/test_reporting_json.py, tests/test_reporting_ci.py, tests/test_reporting_html.py all exist at HEAD... CLAUDE.md 'Known Weaknesses' is STALE — Cannot trust as backlog of record."` Backlog-of-record is wrong; v1.1 audit is being done against an unreliable map.
- **ease**: EASY
- **file:line**: `CLAUDE.md` "Known Weaknesses & Gaps" section
- **fix sketch**:
  - Mechanically split into "Active backlog" (tracked in issues) and "Resolved in v1.0" (tracked in CHANGELOG.md) per archaeologist-debt #4.
  - Drop items now tested: reporting coverage, MPS, strides, thread-safety.
- **conflicts/contradictions with**: none

## ISS-51 — GPU CI gate permanently disabled (commit `22780ae` moved tests to `tests/gpu_integration/`)
- **source**: archaeologist-debt #1 (top-priority debt item)
- **severity**: HIGH (CI/process)
- **impact**: archaeologist-debt quotes — `"22780ae moved GPU-dependent tests under tests/gpu_integration/ and silently exempted them from GitHub Actions. As of HEAD, ~6 integration test files (the deepest correctness-verifiers, e.g. test_arch_detection_gtx1650.py, test_benchmark_accuracy.py, test_decorator_combinations.py) never run automatically. Every 'GPU bug' found post-22780ae is a manual-run discovery."` Real correctness verifiers are documentation, not verification.
- **ease**: HARD (CI architecture)
- **file:line**: `tests/gpu_integration/` (suite); `.github/workflows/*.yml` (CI config)
- **fix sketch**:
  - Add a self-hosted-GPU runner gate, OR
  - Add Lambda-Labs / Modal CI job that runs gpu_integration on real CUDA + Apple Silicon hosts.
  - Or document clearly that integration tests are manual.
- **conflicts/contradictions with**: ISS-35 (fixing the doc "auto-skip" claim is the lighter-weight version of this)

## ISS-52 — Tolerance numerics never settled — CUDA path and MPS path diverge with no unifying scaling law
- **source**: archaeologist-debt #2 (top-priority debt item)
- **severity**: HIGH
- **impact**: archaeologist-debt quotes — `"Five separate commits (9352672, 8d8c894, f7f84eb, 6562f31, 24035aa's +108-line MPS recalibration) recalibrated tolerances.py. The MPS table in 24035aa was *added* alongside the CUDA table without a unifying abstraction; git blame shows ≈50/50 line ownership between the two eras."` No single scaling law; future GPU adds will face combinatorial table maintenance.
- **ease**: HARD
- **file:line**: `src/gpucheck/assertions/tolerances.py:13-43` (default tables + MPS multipliers)
- **fix sketch**:
  - Add a property test asserting CUDA vs MPS tolerances satisfy the same scaling law (atol ∝ sqrt(k/128)).
  - Refactor toward a `Backend.tolerance_overlay()` Protocol method (extends ISS-21).
- **conflicts/contradictions with**: none

## ISS-53 — `28d808e` ("mypy strict") is a -287-line net commit (likely silent revert of speculative type hints)
- **source**: archaeologist-debt #3 (Disguised reverts)
- **severity**: LOW (audit hygiene)
- **impact**: archaeologist-debt quotes — `"28d808e [Fix] : mypy strict mode errors is **-287 line net commit across 86 files** — silent rollback of speculative type hints"`. Type signatures across the codebase are weaker than the commit message implies; warrants a structural re-read before adding new typed APIs.
- **ease**: MEDIUM
- **file:line**: 86 files affected; broad audit needed
- **fix sketch**:
  - Run `mypy --strict src/` and capture remaining `Any`-returns.
  - Specifically re-strict the protocols around `assert_close` and `compute_tolerance` (covers ISS-06).
- **conflicts/contradictions with**: ISS-06

## ISS-54 — Worktree zoo: ~50 stale `worktrees/fuzz-*` refs in reflog
- **source**: archaeologist-debt #5 (operational debt)
- **severity**: LOW (operations)
- **impact**: archaeologist-debt quotes — `"~50 stale worktree refs, all pointing at 82b853e... bloat .git/refs/worktrees/. Should be pruned via git worktree prune"`. Pure ops cleanup.
- **ease**: TRIVIAL
- **file:line**: `.git/refs/worktrees/` (operational, not source)
- **fix sketch**:
  - `git worktree prune`.
- **conflicts/contradictions with**: none

## ISS-55 — Conventional-commits compliance is 100% post-v1.0 but tooling-incompatible (commit `02507da` packs two scopes)
- **source**: archaeologist-debt #5
- **severity**: LOW
- **impact**: archaeologist-debt quotes — `"02507da's subject feat(reporting+sanitizers): packs two scopes into one — most commitlint configs (@commitlint/config-conventional) reject + in scope. The 30 legacy bracket commits will trip release-please/semantic-release."` Future automation will choke.
- **ease**: EASY
- **file:line**: `CONTRIBUTING.md`; CI hooks
- **fix sketch**:
  - Install `commitlint` as a hooked gate matching the policy in CONTRIBUTING.md.
  - Document `since v1.0.0rc1` start point for changelog automation.
- **conflicts/contradictions with**: none

## ISS-56 — UPSTREAM: MPS matmul 1024³ fp32 is 4× slower than expected vs MLX and CPU AMX
- **source**: empiricist-mac-bench Three signal items #3
- **severity**: HIGH (for users; not gpucheck's bug)
- **impact**: empiricist quotes — `"MPS matmul 1024³ fp32 is 4× slower than expected (3.29 ms, 653 GFLOPs) vs MLX (1.34 ms, 1.6 TFLOPs) and CPU AMX (1.08 ms). fp16/bf16 at same shape are 3× faster than fp32. **Likely an MPSGraph fp32 GEMM dispatch / kernel-pick anomaly. WORTH FILING UPSTREAM.**"` Real PyTorch MPS performance bug; gpucheck users on Apple Silicon get misleadingly slow fp32 matmul numbers.
- **ease**: N/A (file upstream issue at github.com/pytorch/pytorch)
- **file:line**: External — PyTorch MPS GEMM dispatch
- **fix sketch**:
  - File a PyTorch issue with empiricist-mac-bench's repro (`/tmp/mac_bench-mps_isolate.py`).
  - In gpucheck v1.1, document the anomaly in MPS docs ("known fp32 GEMM dispatch issue at 1024³ — use fp16/bf16 or upgrade torch").
- **conflicts/contradictions with**: none

## ISS-57 — UPSTREAM: PyTorch CPU has no half-precision GEMM on Apple Silicon (>1s/iter at 1024³)
- **source**: empiricist-mac-bench Three signal items #1
- **severity**: MEDIUM (for users; not gpucheck's bug)
- **impact**: empiricist quotes — `"PyTorch CPU has no half-precision GEMM on Apple Silicon — fp16/bf16 matmul ≥1024³ takes >1 s/iter. fp16 conv2d throws on CPU. (Affects fallback expectations.)"` gpucheck's CPU-fallback expectations don't match reality on Apple Silicon for half-precision.
- **ease**: EASY (gpucheck doc) — N/A for upstream PyTorch fix
- **file:line**: gpucheck docs (MPS fallback section); upstream is PyTorch CPU GEMM
- **fix sketch**:
  - Document in gpucheck MPS docs: "fp16/bf16 CPU fallback on Apple Silicon is unusably slow (≥1 s/iter at 1024³) — keep fp16/bf16 on MPS or skip with `pytest.skip`".
  - Optionally file a PyTorch issue if not already known.
- **conflicts/contradictions with**: none

## ISS-58 — MPS conv2d N4_64_128 has 80-135% CV with default 3 warmup iterations
- **source**: empiricist-mac-bench Three signal items #2
- **severity**: MEDIUM
- **impact**: empiricist quotes — `"MPS conv2d N4_64_128_128x128_3x3 has 80-135% CV — first iter ~6ms, settles to ~1ms. v1.1 docs should recommend **5 warmups for conv2d on MPS** (current default 3)."` Users running default warmup get wildly variable conv2d benchmarks; statistical regression detection becomes unreliable.
- **ease**: TRIVIAL
- **file:line**: `src/gpucheck/fixtures/benchmark.py` (default warmup); benchmark docs
- **fix sketch**:
  - Bump default `warmup` from 3 to 5 on MPS, OR
  - Detect conv2d in fixture call and bump warmup automatically, OR
  - Document the recommendation prominently.
- **conflicts/contradictions with**: ISS-19, ISS-20 (same fixture)

## ISS-59 — Empiricist's lambda-factory pattern timed allocation, not kernel (caught + fixed mid-run)
- **source**: empiricist-mac-bench Process notes
- **severity**: N/A (resolved during audit; informational)
- **impact**: empiricist quotes — `"Original lambda-factory pattern timed only input allocation, not kernel — inflated MPS by ~25×. Caught via 4096³ sanity probe; fixed mid-run; final numbers post-fix."` Same trap is easy to fall into in user code; gpucheck's docs should warn.
- **ease**: EASY
- **file:line**: gpucheck benchmark docs
- **fix sketch**:
  - Add a "common pitfalls" doc section showing the lambda-factory anti-pattern with side-by-side correct version.
- **conflicts/contradictions with**: none

---

## Cross-audit contradictions

### C1 — Silent fp64 downcast on MPS (linguist-v3 vs tracer-runtime)
- **Audits involved**: linguist-v3 (R3, prior round) **vs** tracer-runtime (W1)
- **Claim under contention**: linguist-v3 hypothesized that fp64 tensors on MPS silently downcast to fp32, motivating a "silent-downcast catcher API" in v1.1.
- **Tracer-runtime verdict**: **REFUTED**. On torch 2.11, `torch.tensor(0.5, device='mps', dtype=torch.float64)`, `cpu_fp64.to('mps')`, `set_default_dtype(torch.float64) + torch.tensor(0.5).to('mps')`, and `mps_tensor.double()` all raise `TypeError: Cannot convert a MPS Tensor to float64...` at construction time.
- **Real or scope-mismatch?**: Possibly scope — linguist-v3 may have been examining torch < 2.11 where the failure mode could have been silent. Tracer-runtime explicitly notes: `"May be different on torch <2.11."`
- **Recommended resolution**:
  - Drop the silent-downcast catcher API from v1.1 plan (it's not needed on torch 2.11, the project's stated minimum).
  - If torch < 2.11 is in the support matrix, run a fast probe on torch 2.7 / 2.6 to confirm fail-loud behavior; if it ever was silent, add a note to MPS docs and pin minimum torch version.

### C2 — CLAUDE.md "Known Weaknesses" vs reality (multiple audits converge)
- **Audits involved**: archaeologist-debt #4, detector-files (entire reporting/ section), api-dx-grade (multiple)
- **Claim under contention**: CLAUDE.md says "Reporting module (console, json, ci) has zero test coverage", "No GPU CI", "No stride/contiguity fuzzing", "Thread-safety issue in tolerance override stack" — implying these are **active gaps**.
- **Verdict**: archaeologist-debt confirms several were **delivered** in v1.0 tracks (`02507da` reporting, `4ede763` strides, `5ddd26e` thread-safety) but never removed from the doc. detector-files independently observes the test files exist. The "No GPU CI" claim **is** still true (per ISS-51), so the doc is partially stale.
- **Real or scope-mismatch?**: Real — CLAUDE.md is being used as a backlog of record, but tracks landed without doc updates.
- **Recommended resolution**: Mechanically split CLAUDE.md "Known Weaknesses" into "Active backlog" (tracked in GitHub Issues) and "Resolved in v1.0" (tracked in CHANGELOG.md). See ISS-50.

### C3 — `flush_l2=True` on MPS: the warning fires (good?) but the work doesn't (bad?)
- **Audits involved**: tracer-runtime finding 2 (warning fires every call) vs tracer-runtime finding 3 (L2 flush silently absent)
- **Claim under contention**: Same code site is "good UX" (warns user the request is ignored) and "bad UX" (no replacement provided, benchmarks are systematically optimistic).
- **Verdict**: Both findings are valid and complementary, not contradictory. Tracer flagged both as v1.1 fix candidates.
- **Real or scope-mismatch?**: No real disagreement; complementary findings.
- **Recommended resolution**: Treat as ISS-19 (gate the warning) + ISS-20 (offer a buffer-fill approximation). Land together; cross-reference in commit message.

### C4 — Memory leak: warning vs failure (api-dx-grade vs design intent)
- **Audits involved**: api-dx-grade #9 (`memory_tracker` Error-message quality 2)
- **Claim under contention**: api-dx-grade says leak should be a **failure**, not a warning. The current design produces `RuntimeWarning` (which can be silenced).
- **Verdict**: One audit's recommendation; no other audit directly disagrees, but archaeologist-debt notes memory.py was patched 6 times without architectural fix — the warning behavior was a deliberate choice for "imprecise process-level metrics" (per CLAUDE.md).
- **Real or scope-mismatch?**: Scope — the warning behavior is justified by "imprecise process-level metrics" but api-dx-grade is right that it surprises users.
- **Recommended resolution**: ISS-44 — add `fail_on_leak: bool = False` kwarg in v1.1 (keeps current behavior by default), promote to True in v1.2.

### C5 — README MPS-as-first-class vs README CUDA-only snippets
- **Audits involved**: docs-tester R-B6/R-B8/R-B12/R-B13 (CUDA hard-coded in walk-through) vs README §"Strengths" claim (MPS first-class)
- **Claim under contention**: README simultaneously advertises MPS as first-class and shows examples that crash on Apple Silicon.
- **Verdict**: Real — internal inconsistency.
- **Recommended resolution**: ISS-36 — add device auto-detect helper in snippets, OR add "this snippet requires CUDA" callouts.

---

## Impact × ease 2D matrix

### Quadrant 1 — High impact × Easy (FIX FIRST in v1.1)

| ID | Title | Severity | Ease |
|---|---|---|---|
| ISS-01 | `compute_tolerance` silent fp32 fallback | HIGH | EASY |
| ISS-02 | `@devices` typo silently skips | HIGH | EASY |
| ISS-05 | `tolerance_context` bypasses MPS overlay | HIGH | EASY |
| ISS-08 | `assertions/close.py` top-level torch import | HIGH | TRIVIAL |
| ISS-09 | Two `compute_tolerance` naming collision | HIGH | EASY |
| ISS-18 | `_to_numpy` no `.contiguous()` (PM-4) | MEDIUM | TRIVIAL |
| ISS-25 | `reporting.py` 83 surviving mutants | HIGH | EASY |
| ISS-26 | Tolerance config loader untested | HIGH | TRIVIAL |
| ISS-29 | MIGRATION.md `fuzz_strides` wrong sig | HIGH | TRIVIAL |
| ISS-30 | MIGRATION.md `assert_deterministic` wrong kwargs | HIGH | TRIVIAL |
| ISS-31 | MIGRATION.md `available_backends` return type | HIGH | TRIVIAL |
| ISS-33 | README §6 wrong fence language | HIGH | TRIVIAL |
| ISS-34 | README §9 wrong numeric output | HIGH | TRIVIAL |
| ISS-35 | gpu_integration auto-skip claim wrong on MPS | HIGH | EASY |
| ISS-38 | 7 public symbols missing from `_LAZY_MAP` | MEDIUM | EASY |
| ISS-50 | CLAUDE.md "Known Weaknesses" stale | LOW | EASY |

### Quadrant 2 — High impact × Hard (PLAN — multi-PR or architectural)

| ID | Title | Severity | Ease |
|---|---|---|---|
| ISS-40 | `ShapeStrategy`/`StrideStrategy` `__new__` factory | MEDIUM | HARD |
| ISS-51 | GPU CI gate permanently disabled | HIGH | HARD |
| ISS-52 | Tolerance numerics never settled (CUDA/MPS divergence) | HIGH | HARD |
| ISS-22 | Fast-path 1.77 ms sunk cost | LOW | HARD |

### Quadrant 3 — Low impact × Easy (CLEANUP — bundle in v1.1 polish PR)

| ID | Title | Severity | Ease |
|---|---|---|---|
| ISS-12 | Duplicated `_median` helper | LOW | TRIVIAL |
| ISS-16 | `_load_pyproject_config` swallows all exceptions | LOW | TRIVIAL |
| ISS-17 | HTML reporter `class`/`bg` raw interpolation | LOW | TRIVIAL |
| ISS-19 | `flush_l2` warning fires every call | LOW | TRIVIAL |
| ISS-28 | k_dim=1 boundary uncovered | LOW | TRIVIAL |
| ISS-32 | xfail registry runtime gotcha undocumented | MEDIUM | TRIVIAL |
| ISS-37 | `__init__.py` no `Examples:` doctests | LOW | TRIVIAL |
| ISS-39 | `apply_mps_xfail_config` missing from `_LAZY_MAP` | LOW | TRIVIAL |
| ISS-41 | `require_arch` vs `requires_determinism` naming | LOW | EASY |
| ISS-42 | `SM_ARCH_MAP` vs `SM_TO_ARCH` duplication | LOW | EASY |
| ISS-43 | Fixtures registered in two places | LOW | EASY |
| ISS-45 | `_safe_import_numpy` misleading wrapper | LOW | TRIVIAL |
| ISS-46 | `tensor_cores` imports private `_DEFAULT_TOLERANCES` | LOW | EASY |
| ISS-47 | `race.py:257` `os.unlink` not in try | LOW | TRIVIAL |
| ISS-48 | `race.py:117` local `warnings` shadow | LOW | TRIVIAL |
| ISS-49 | `plugin.py:151` unguarded torch import | MEDIUM | TRIVIAL |
| ISS-54 | Worktree zoo (50 stale refs) | LOW | TRIVIAL |
| ISS-55 | Commitlint not enforced | LOW | EASY |
| ISS-58 | conv2d default warmup too low on MPS | MEDIUM | TRIVIAL |
| ISS-59 | Lambda-factory anti-pattern undocumented | N/A | EASY |

### Quadrant 4 — Low impact × Hard (DEFER beyond v1.1)

| ID | Title | Severity | Ease |
|---|---|---|---|
| ISS-23 | `format_mismatch_report` 16 ms slow path | LOW | MEDIUM |
| ISS-53 | `28d808e` -287-line silent revert | LOW | MEDIUM |

### Mid-tier (medium impact × medium-easy)

| ID | Title | Severity | Ease |
|---|---|---|---|
| ISS-03 | `register_mps_xfail` typo silent | HIGH | MEDIUM |
| ISS-04 | `@dtypes` typo opaque crash | MEDIUM | EASY |
| ISS-06 | `assert_close` no TensorLike Protocol | MEDIUM | MEDIUM |
| ISS-07 | `baseline_2x` magic boolean | MEDIUM | MEDIUM |
| ISS-10 | Two `MemoryReport` types | MEDIUM | EASY |
| ISS-11 | Triplicated GPU-detection logic | MEDIUM | MEDIUM |
| ISS-13 | 7 bare `except Exception` | MEDIUM | EASY |
| ISS-14 | `parametrize.py` TypeError arity probe | MEDIUM | EASY |
| ISS-15 | `memory_guard` private `_MutableReport` | MEDIUM | EASY |
| ISS-20 | L2 flush silently absent on MPS | MEDIUM | EASY |
| ISS-21 | `_run_mps` duplicates `MPSBackend.event_timer` | MEDIUM | MEDIUM |
| ISS-27 | `_DEFAULT_TOLERANCES` table not pinned | MEDIUM | TRIVIAL |
| ISS-36 | README CUDA-hard-coded snippets | MEDIUM | EASY |
| ISS-44 | Memory leak warning vs failure | MEDIUM | EASY |
| ISS-57 | PyTorch CPU half-precision GEMM gap (gpucheck doc only) | MEDIUM | EASY |

(N/A: ISS-24 is a refuted hypothesis — no fix; ISS-56 is upstream.)

---

## Mac/Metal-specific cluster (v1.1 Mac-focused track)

The following issues are **Mac/MPS-only or Mac-amplified**. v1.1 should track them as a parallel "Mac/Metal" lane separate from cross-platform cleanup.

| ID | Title | Mac-specific? |
|---|---|---|
| ISS-05 | `tolerance_context` bypasses MPS overlay | YES — only the MPS overlay is bypassed; CUDA path has no overlay |
| ISS-19 | `flush_l2` warning fires every call on MPS | YES — MPS-only fixture path |
| ISS-20 | L2 flush silently absent on MPS | YES — Apple Silicon has no L2-flush primitive |
| ISS-21 | `_run_mps` duplicates `MPSBackend.event_timer` | YES — MPS-only code path |
| ISS-22 | Fast-path 1.77 ms sunk cost | Partially — measured on MPS, applies to CUDA but in shorter timescales |
| ISS-23 | `format_mismatch_report` 16 ms slow path on MPS | Partially — magnitude measured on MPS |
| ISS-24 | Silent fp64 downcast on MPS (REFUTED) | YES |
| ISS-32 | xfail registry runtime gotcha | MPS-coded section but framework-general |
| ISS-35 | `gpu_integration` auto-skip claim wrong on MPS hosts | YES — failure mode is MPS-specific (CUDA-less hosts pass; MPS-having hosts fail 52 tests) |
| ISS-36 | README §4-§7 hard-code `device="cuda"` | YES — affects every Mac dev box |
| ISS-52 | Tolerance numerics CUDA/MPS divergence | YES — root cause is MPS overlay added without unifying scaling law |
| ISS-56 | UPSTREAM: MPS matmul 1024³ fp32 anomaly | YES — pure Apple Silicon |
| ISS-57 | UPSTREAM: PyTorch CPU half-precision GEMM gap on Apple Silicon | YES — Apple Silicon CPU |
| ISS-58 | MPS conv2d 80-135% CV at default warmup=3 | YES |

**Mac-track v1.1 minimum**: ISS-05, ISS-19, ISS-20, ISS-21, ISS-32, ISS-35, ISS-36, ISS-58.

---

## Fileable upstream candidates (NOT gpucheck bugs)

These were found by gpucheck audits but the bugs live in PyTorch itself. v1.1 should file these upstream and link from gpucheck docs.

| ID | Title | Where to file |
|---|---|---|
| ISS-56 | MPS matmul 1024³ fp32 is 4× slower than expected (3.29 ms / 653 GFLOPs vs MLX 1.34 ms / 1.6 TFLOPs vs CPU AMX 1.08 ms). fp16/bf16 at same shape are 3× faster than fp32 — likely MPSGraph fp32 GEMM dispatch / kernel-pick anomaly. | github.com/pytorch/pytorch (MPS label). Repro at empiricist-mac-bench `/tmp/mac_bench-mps_isolate.py` |
| ISS-57 | PyTorch CPU has no half-precision GEMM on Apple Silicon — fp16/bf16 matmul ≥1024³ takes >1 s/iter; fp16 conv2d throws `RuntimeError`. (May already be tracked upstream; verify before filing.) | github.com/pytorch/pytorch (CPU + Apple Silicon labels) |

**Total fileable upstream**: 2 (both from empiricist-mac-bench).

---

## LOG entry

(Will be appended to `LOG.md` separately; record here for traceability.)

`<ts> synthesist: 59 issues across 8 audits, 5 contradictions documented (1 resolved REFUTED, 1 partial-stale-doc, 1 complementary-not-contradictory, 1 scope-mismatch, 1 internal-inconsistency), 2 upstream-fileable, 14 Mac-specific in cluster.`
