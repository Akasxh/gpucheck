# External Review — gpucheck PR #2 (v1.0.0rc1)

**Reviewer:** Engineering-Reviewer (external-style, senior open-source maintainer mindset)
**PR:** [Akasxh/gpucheck#2](https://github.com/Akasxh/gpucheck/pull/2) — *gpucheck v1.0 — Apple MPS backend, stride fuzzing, dashboard*
**Diff:** 5,622 added / 115 removed across 42 files, branch `release/v1.0` ↔ `main`
**Method:** code read of every modified `src/` file, cross-checked against PLAN.md + earlier-audit findings (`security-postmerge.md`, `api-dx-grade.md`, `empiricist-mac-benchmarks.md`, `docs-tester-blocks.md`). New findings are labelled **NEW**; corroborations are labelled **CONFIRMS**. The brief asks for ≥3 per axis, ≥15 total, ≥1 surprise. Delivered: 19 findings; surprise = **N1**.

---

## 1. Security

### S1 — `tomli` is referenced but undeclared as a dependency on Python 3.10 *(NEW; surprise candidate)*

**File/line:** `src/gpucheck/plugin.py:73-76`, `pyproject.toml:30-34`, `pyproject.toml:88-90`.
**What:** `_load_pyproject_config` falls back to `import tomli as tomllib` when `tomllib` is missing (Python 3.10 path). `pyproject.toml` lists `tomli` only in the **mypy ignore-missing list** (line 89), never in `[project.dependencies]`. `uv.lock` *does* carry tomli but only via transitive coverage/pytest deps — a `pip install gpucheck` on Python 3.10 with no dev extras will not install it. The catch-all `except Exception: pass` at `plugin.py:86-88` silently swallows the resulting ImportError, so the entire `[tool.gpucheck.tolerances]` overlay AND the `[tool.gpucheck.mps.xfail]` registry are silently ignored on Python 3.10 installs that lack a transitive `tomli` provider. The user's `pyproject.toml` overrides become a no-op with no warning, no log line, no test failure.
**Why this is a security finding (not just a bug):** the system *appears* to honour the user's tolerance + xfail configuration. A test author who has tightened `float16` to `5e-4` in `pyproject.toml` to catch a regression will see tests pass on 3.10 because the override never loaded — confidence-in-test failure mode. Mitigates as `MEDIUM` configuration-trust failure (OWASP-A05).
**Action:** add `'tomli >= 2.0; python_version < "3.11"'` to `[project.dependencies]`. Also narrow the catch in `_load_pyproject_config` to `(OSError, tomllib.TOMLDecodeError, ImportError)` and emit a `UserWarning` so silent-fail paths are observable. Bonus: add `tests/test_plugin_config_load.py` exercising `_load_pyproject_config` against a captured 3.10 + no-tomli env (mocked `sys.modules`).
**Verdict:** **BLOCK** the rc1 → 1.0 final cut until either (a) `tomli` is declared and the lock includes a non-transitive entry, or (b) the loader emits a clear warning when 3.10 lacks a TOML parser. The earlier audit's PM-2 flagged the bare `except` but did not catch the missing dependency.

### S2 — HTML reporter `class=` and `style=` attributes interpolate unescaped values *(CONFIRMS PM-5; tightens scope)*

**File/line:** `src/gpucheck/reporting/html.py:91-95`, `:171`, `:218`, `:236`, `:243`, `:246`.
**What:** `_pill()` interpolates `bg` directly into `style="background:{bg};..."`. `_render_test_results`, `_render_memory`, and `_render_comparison` interpolate `klass` directly into `class="{klass}"`. Both feed from whitelist dicts today (`_STATUS_PILL_BG`, hard-coded `"regression"/"new"/"removed"`). PM-5 already noted this; the *tightening* here: `_render_comparison:236` accepts `status` directly from the input JSON (`b.get("status", "ok")`) and uses it as `klass` whenever it equals one of three whitelisted strings. There is **no defensive default** if status is e.g. `"new\"><script>"` — the equality check fails (so `klass` becomes `"passed-row"`, safe) — but the pill on the same row passes the *raw* status string through `_pill(status)` → `_STATUS_PILL_BG.get(status, "#666666")` → safe. So today's blast radius is bounded by the whitelist match. The risk is structural: a future change that sets `klass = b.get("class")` ships an XSS sink.
**Action:** wrap every attribute interpolation in `_esc(...)` as defense-in-depth, regardless of whitelist provenance. One-line fixes per call site. Cost: zero. Cite OWASP-A03 for reviewers.
**Verdict:** **COMMENT** (non-blocking for rc1; required before publishing the dashboard as a CI artifact).

### S3 — `_build_wrapper_script` serializes the full `sys.path` into a temp Python file *(NEW)*

**File/line:** `src/gpucheck/sanitizers/race.py:170`.
**What:** the wrapper script written to `/tmp/gpucheck_sanitizer_*.py` contains `import sys; sys.path[:0] = {sys.path!r}`. Two issues: (i) any path on `sys.path` containing a `'` or backslash (Windows venv paths, paths from a sourced env) breaks the `repr()` round-trip — Python's `repr()` is safe for arbitrary strings, but this writes to a temp file readable by `umask 022`-default users on shared hosts. The temp-file content includes every `sys.path` entry of the *invoking* Python process, which can leak the user's home dir and project layout. (ii) An attacker who can plant a malicious package in any directory listed earlier in `sys.path` (a Python-classic search-path attack) will see their package imported by `compute-sanitizer`'s subprocess; the wrapper inherits the full path so any path-injection attack on the parent inherits to the child too. The `mod_name`/`fn_name` allow-list defends function-name injection but not path injection.
**Action:** instead of dumping `sys.path`, dump only `os.path.dirname(inspect.getfile(module))`. Use `tempfile.mkdtemp()` with mode `0o700` (currently `mkstemp` defaults to `0o600` for the file but the dir is shared `/tmp`). Document in the docstring that callers should not run sanitizer with attacker-controlled `PYTHONPATH`.
**Verdict:** **COMMENT** — exploitability requires write access to a `sys.path` directory (already bad). But for a v1.0 final this should be tightened.

### S4 — Subprocess invocation in `_detect_apple_chip` does not scrub `PATH` *(CONFIRMS PM-1; downgrades)*

**File/line:** `src/gpucheck/backends/mps.py:225-229`.
**What:** PM-1 already analysed this. I add: the `["sysctl", "-n", "machdep.cpu.brand_string"]` invocation relies on the inherited `PATH`. `sysctl` on macOS lives at `/usr/sbin/sysctl`; an attacker with write access to a directory earlier in `PATH` than `/usr/sbin` could plant a `sysctl` binary. The 2-second timeout caps damage but the binary will already have run. Pinning to `/usr/sbin/sysctl` and clearing `env` to `{"PATH": "/usr/sbin:/usr/bin"}` is a one-line fix.
**Action:** apply the fix-sketch from PM-1. Symmetry with Track-C's `_CUDA_HOME_ALLOWLIST` mindset.
**Verdict:** **COMMENT** — agreed-low-severity but trivial to fix and improves defense-in-depth.

---

## 2. Performance

### P1 — MPS benchmark warning fires per-call instead of per-process *(NEW)*

**File/line:** `src/gpucheck/fixtures/benchmark.py:304-310` vs. `src/gpucheck/backends/mps.py:60,167-176`.
**What:** the MPSBackend has a `_FLUSH_L2_WARNED` module global that dedupes the "no L2 flush on MPS" warning to once per process. **The benchmark fixture bypasses the backend** — `_run_mps` re-emits `warnings.warn(...)` every time `gpu_benchmark(fn, ...)` is called with the default `flush_l2=True`. With Python's default warning filter (`default`), a warning at the same code location is shown only once per *unique* module/lineno; but pytest captures warnings and shows them all in the summary, polluting test output. The dedupe global on the backend side is dead code in the fixture path.
**Action:** route the fixture through `MPSBackend.flush_l2(...)` (which already correctly dedupes) instead of inlining the warning. Or move `_FLUSH_L2_WARNED` to a module-level guard at `_run_mps`. Net: cleaner test output, single source of truth.
**Verdict:** **COMMENT** — UX/perf nit; not blocking.

### P2 — `_run_mps` does redundant `torch.mps.synchronize()` calls per iteration *(NEW)*

**File/line:** `src/gpucheck/fixtures/benchmark.py:319-326`.
**What:** the inner loop calls `torch.mps.synchronize()` THREE times per round if you count the post-warmup sync at 316 + per-iter pre-sync at 320 + per-iter post-sync at 324. The pre-sync at 320 is correct (drains prior in-flight work — otherwise the next iteration's `t0` includes leakage). But on the first iteration after warmup, the post-warmup sync at 316 is followed immediately by the pre-sync at 320 — back-to-back syncs of an already-empty queue. `torch.mps.synchronize()` is not free (~0.1-0.3ms on M-series per `mac_benchmarks.json` row 4). At small shapes (256³ matmul fp32 measures 0.40ms median per the empiricist benchmarks), an extra sync is ~25% overhead. CUDA path doesn't have this issue because `torch.cuda.synchronize()` is cheap when the queue is empty.
**Action:** drop the second sync at 316 OR move the per-iter pre-sync inside the `for` only when iter > 0. Add a benchmark on a tight kernel (256³ matmul) before/after to measure.
**Verdict:** **COMMENT** — micro-optimization; low priority but the empiricist already flagged that small-shape MPS cells are dispatch-bound (cv 47-135% on the smallest shapes).

### P3 — Repeated `torch.cuda.is_available()` calls in hot paths *(NEW)*

**File/line:** `src/gpucheck/fixtures/benchmark.py:110,140,193,206`; `src/gpucheck/decorators/devices.py:18,52-58`; `src/gpucheck/sanitizers/determinism.py:52-56`.
**What:** `torch.cuda.is_available()` is documented to lazily init CUDA on first call; subsequent calls are cached but still cross the C boundary. `_BenchmarkRunner.__post_init__` calls it once (line 140), then `__call__` calls it again (193), and `_flush_l2_cache` (110) calls it a third time per benchmark round. For 100 rounds with `flush_l2=True`, that's 100 redundant calls per benchmark. `_seed_all` in determinism.py calls it inside a loop body of `assert_deterministic` (n=3 default) — 3× redundant. Cumulative effect: at 1000 small-kernel benchmarks per CI run, you've spent 100ms on `is_available()` overhead.
**Action:** cache to a module-level bool inside `_get_torch()` after first lookup, or hoist to `__post_init__` and pass `cuda_avail` down through the run loops.
**Verdict:** **COMMENT** — measurable on a microbenchmark, not a release-blocker.

### P4 — `_to_numpy` does not call `.contiguous()` before `.numpy()` for stride-0 views *(CONFIRMS PM-4)*

**File/line:** `src/gpucheck/assertions/close.py:44-54`.
**What:** PM-4 already analyzed. Performance angle: when stride fuzzer produces broadcast views (stride 0) and the slow-path triggers, `tensor.detach().cpu()` copies the full *expanded* tensor instead of the underlying smaller buffer. For a `(B, S, D)` broadcast tensor where the last dim is stride-0 and S=4096 D=64, the cost is `B * 4096 * 64 * dtype_bytes` instead of `B * 4096` source bytes. Up to 64× wasted CPU-bound copying when slow-path fires.
**Action:** insert `t = tensor.detach().cpu().contiguous()` so the materialization happens once, and then `.numpy()` returns a view of the contiguous buffer. Doubles as the safety fix in PM-4.
**Verdict:** **COMMENT** — net-net the right fix from PM-4 also fixes the perf cliff.

---

## 3. API design

### A1 — Stride category names diverge between code (`row_major`, snake_case) and docs (`row-major`, kebab) *(NEW; high-friction)*

**File/line:** `src/gpucheck/fuzzing/strides.py:33-41` (canonical) vs. `MIGRATION.md:271,287,292`, `CHANGELOG.md:81`. Also `src/gpucheck/fuzzing/strides.py:11-17` docstring (kebab) vs. `:33-41` constant (snake_case).
**What:** code defines `CATEGORIES = ("row_major", "column_major", "broadcast", "transpose", "slice", "non_contig", "gather")`. MIGRATION.md §7 calls them `("row-major", "column-major", "broadcast-induced", "transpose", "slice", "contiguous-after-clone", "gather-induced")` — six of seven names diverge. CHANGELOG.md §Track B repeats the kebab spelling. The module's own top-of-file docstring (`strides.py:9-17`) lists the names in kebab form, then the constant uses snake_case six lines later. The docs-tester audit caught the broken `fuzz_strides_for_category` signature in MIGRATION.md but not the *names*. Anyone copy-pasting MIGRATION.md §7's snippet will hit `ValueError: Unknown stride category 'broadcast-induced'`.
**Action:** pick one. Recommendation: keep snake_case in code (Python convention) and rewrite all `.md` docstrings/CHANGELOG entries to use snake_case. Add a one-line back-compat shim that translates `row-major → row_major` with a `DeprecationWarning` if you don't want to break early adopters who copy-pasted the doc.
**Verdict:** **BLOCK** — copy-pasted docs that fail the public API validator are a worse first-impression than missing docs. Must fix before tagging 1.0 final.

### A2 — `assert_deterministic` documented signature in MIGRATION.md is wrong *(CONFIRMS docs-tester M-B9; root-cause API design)*

**File/line:** `src/gpucheck/sanitizers/determinism.py:85-90` (real signature) vs. `MIGRATION.md:309` (`assert_deterministic(my_kernel, args=(x, y), runs=2, atol=0.0)`).
**What:** the real signature `assert_deterministic(fn, *args, n=3, seed=0, **kwargs)` has *no* `args=` (positional `*args`), no `runs` (it's `n`), and no `atol` (the function does **byte-equality** comparison via `torch.equal` — there's no tolerance!). The MIGRATION example invokes three keyword arguments that don't exist; they get forwarded into `fn(**kwargs)` as `args=...`, `runs=...`, `atol=...`, where they crash whichever kernel `fn` is. Beyond a doc-fix: the API itself has a footgun. **Byte-equality is the wrong default for MPS** — research SYNTHESIS §4 explicitly states MPS is best-effort deterministic, and the docstring (close.py:5-13) acknowledges it. Yet the comparison is `torch.equal` (bit-exact). Most MPS kernels will fail this even when "morally" deterministic. The error message at line 127-133 ("On MPS this can happen legitimately…") tells the user they should widen tolerances — but there's no `atol`/`rtol` knob. This is an API/docs mismatch *baked into the design*: MIGRATION pretends `atol=0.0` is the default and the user can dial it up, but the real API has no atol.
**Action:** accept that MPS testers actually want `atol`/`rtol` here. Add `atol: float = 0.0, rtol: float = 0.0` parameters and use `torch.allclose` when either is non-zero. Update MIGRATION to match. Or, if the byte-equality intent is canonical, rename the function to `assert_bit_exact` so the contract is unambiguous.
**Verdict:** **REQUEST_CHANGES** — the MIGRATION example is unrunnable AND the real API surface is a footgun for the documented MPS use case. Ship a fix for both.

### A3 — Backend selection has no user-facing override *(NEW; CHARTER §charter §4 says "first-class MPS"; PM-3 took this as a security PASS)*

**File/line:** `src/gpucheck/backends/__init__.py:33-86`, `src/gpucheck/fixtures/benchmark.py:206-209`.
**What:** `available_backends()` returns CUDA-then-MPS hard-coded. `_BenchmarkRunner.__call__` uses `if cuda_avail: ... else _run_mps(...)`. On a hypothetical hybrid host (NVIDIA GPU + Apple Silicon — yes, Mac Pros with eGPUs and Linux x86 boxes with AMD chips can have both), CUDA *always* wins — a user with `device="mps"` in their test cannot route the *benchmark* to MPS. There is no `--gpu-backend=mps` CLI flag, no `[tool.gpucheck.backend]` config block, no env var. The `gpu_device` fixture only honors `--gpu-device cuda:N` (line 147-160) and falls back to `cuda:0` for any other value. PM-3 correctly noted this is *security-good* (no downgrade attack surface). This finding flips it: it is also a real **DX bug**. The MIGRATION.md sells "v1 makes MPS first-class" but the v1 plumbing makes CUDA *exclusive* whenever both are present.
**Action:** add a `--gpu-backend={cuda,mps}` CLI option (matched against the same allow-list), or honor `device="mps"` in `gpu_benchmark` by inspecting tensors passed to the kernel. Document explicitly in MIGRATION §3.
**Verdict:** **COMMENT** — defer to v1.1 if release deadline pressure; document the limitation in MIGRATION §3 and the `[mps]` section so users aren't surprised.

### A4 — `tolerance_context(atol, rtol)` is an absolute override, not a scale factor *(CONFIRMS api-dx-grade row 3)*

**File/line:** `src/gpucheck/assertions/tolerances.py:91-93,114-135`.
**What:** the api-dx audit flagged this. I corroborate from the MIGRATION example (line 174-178): the doc invokes `tolerance_context(atol=1e-3, rtol=1e-3)` to "use overridden tolerances", which sounds like a scale factor (`2× the dtype default`). The implementation returns the override as-is regardless of dtype, k_dim, or MPS overlay — so a user doing `with tolerance_context(atol=1e-3, rtol=1e-3): assert_close(fp16_a, fp16_b, k_dim=4096)` gets `atol=1e-3` instead of the expected `1e-2 × sqrt(4096/128) = 5.6e-2`. The k_dim and MPS multiplier are silently dropped. This was correctly flagged as Avg 3.2 in the api-dx grade; my add: the *MIGRATION.md* example presents it as a benign one-liner, masking the loss of k_dim.
**Action:** rename the function to `tolerance_override` and document the absolute-override semantics, OR add a scale-factor variant `tolerance_context(scale=2.0)` that multiplies the dtype-aware base. Either way the MIGRATION example needs a comment.
**Verdict:** **COMMENT** — non-blocking for rc1, but should not become canon for v1 final without the doc fix.

---

## 4. Numerical correctness

### N1 — `baseline_2x` + `k_dim` uses `sqrt(k_dim)` instead of `sqrt(k_dim/128)` *(NEW — surprise finding; the prior audits all missed this)*

**File/line:** `src/gpucheck/assertions/close.py:178-181` vs. `src/gpucheck/assertions/tolerances.py:102-103`.
**What:** the canonical k_dim scaling in `compute_tolerance` is `atol *= sqrt(max(k_dim, 1) / 128.0)`. When `baseline_2x=True` is set with `k_dim`, `assert_close` takes a *separate code path* (lines 173-183) that scales the doubled atol by `sqrt(k_dim)` — **not** `sqrt(k_dim/128)`. At k_dim=4096 the two formulas diverge by `sqrt(128) = 11.3×`. Concretely: a user testing FlashAttention on fp16 at k=4096 with `baseline_2x=True`:
  - the `baseline_2x` path produces `atol = 1e-2 × 2 × sqrt(4096) = 1.28`,
  - the canonical path would produce `atol = 1e-2 × 2 × sqrt(4096/128) = 0.113`.

The `baseline_2x` path is **11× looser** than the canonical formula at typical FlashAttention dimensions. This means tests that should fail with bad fp16 accumulators will *pass* whenever the user opts into the FlashAttention preset. **This is a silent-correctness regression in the headline assertion.** The README's headline example doesn't use both knobs together so the bug is invisible in CI; but a real FlashAttention test (`assert_close(out, ref, k_dim=4096, baseline_2x=True)`) hits this immediately.
**How the prior audits missed it:** the api-dx audit gave `assert_close` a 3.8 average and complimented the keyword-only API. The empiricist benchmarked but did not test correctness of mixed knobs. The docs-tester ran `baseline_2x=True` standalone (R-B9) and `k_dim` standalone (§4) but never the combination.
**Action:** in `close.py:175-183`, route through `compute_tolerance(dtype, k_dim=k_dim, device_type=device_type)` THEN double — i.e. drop the open-coded scaler at line 181 and rely on the tolerance helper. Pseudocode:
  ```python
  base_atol, base_rtol = compute_tolerance(dtype, k_dim=k_dim, device_type=device_type)
  eff_atol = base_atol * 2.0
  eff_rtol = base_rtol * 2.0
  ```
  Add a regression test: `tests/test_assert_close_baseline_2x.py::test_baseline_2x_k_dim_matches_compute_tolerance`.
**Verdict:** **BLOCK** — this is a silent numerical-correctness bug in the most-prominent public API surface (`assert_close`). Cannot ship v1.0 final with this divergence.

### N2 — MPS overlay applies only when *first* tensor in `(actual, expected)` is on MPS *(NEW)*

**File/line:** `src/gpucheck/assertions/close.py:163-168`.
**What:** the device-type sniff iterates `for t in (actual, expected): if isinstance(t, torch.Tensor): device_type = t.device.type; break`. **`break` after the first match** is the bug. A common test pattern:
  ```python
  expected = cpu_reference(x)              # on cpu
  actual = my_mps_kernel(x.to('mps'))       # on mps
  assert_close(actual, expected)
  ```
  Wait, that case works (actual is first). But: the convention in the rest of gpucheck (e.g. `_resolve_dtype` line 92-122) is to compare both. The asymmetric `break` means:
  ```python
  expected = ref_kernel(x.cpu())           # on cpu, but enters first
  actual = my_mps_kernel(x.to('mps'))
  assert_close(expected, actual)            # SWAPPED arg order
  ```
  Now `device_type = "cpu"` and the MPS overlay does **not** apply. Same physical tensors, swapped caller args, different tolerance. Plus, for two-tensor cases where both are on accelerators (cuda + mps in a hybrid test), the second device is silently dropped.
**Action:** prefer non-CPU type, or assert both are on the same device:
  ```python
  device_types = {t.device.type for t in (actual, expected) if isinstance(t, torch.Tensor)}
  device_types.discard("cpu")
  device_type = next(iter(device_types), None)
  ```
**Verdict:** **REQUEST_CHANGES** — order-dependence in `assert_close` violates the README claim "the order of arguments does not matter" (line 325). Either fix the implementation or weaken the README claim.

### N3 — `_resolve_dtype` falls back to `np.float32` for unknown-dtype inputs without warning *(CONFIRMS api-dx-grade row 2; numerical angle)*

**File/line:** `src/gpucheck/assertions/close.py:103-104`, `src/gpucheck/assertions/tolerances.py:99-100`.
**What:** when neither `actual` nor `expected` has a `.dtype` attribute (e.g. plain Python lists), `_resolve_dtype` returns `np.float32` as a default. `compute_tolerance` then uses the float32 atol of 1e-4. A user comparing two Python lists `[1.0, 1.0001]` against `[1.0, 1.0]` will fail (within 1e-4). A user comparing two `int` arrays will silently get the float32 tolerance with no warning. The api-dx audit graded this 1/5 on error-message quality. The numerical angle: this also means assert_close on cupy arrays where `.dtype` lookup fails (e.g. via `__cuda_array_interface__`) gets float32 tolerance regardless of the underlying dtype.
**Action:** raise `TypeError("assert_close: cannot infer dtype; pass atol=/rtol= explicitly")` instead of silently picking float32. At minimum, emit a `RuntimeWarning`.
**Verdict:** **COMMENT** — silent-fallback risk; non-blocking but should be a v1.1 must-fix.

### N4 — `compute_tolerance` k_dim scaling uses `max(k_dim, 1)` *after* the `> 0` check is meaningless *(NEW — minor)*

**File/line:** `src/gpucheck/assertions/tolerances.py:102-103`.
**What:** the conditional `if k_dim is not None and k_dim > 0:` means by the time `max(k_dim, 1)` is evaluated, k_dim is already ≥ 1. The `max` is dead code. More importantly: **what's the right answer for `k_dim=0`?** The current code skips scaling entirely (treats it like `k_dim=None`). Is that intentional? A reduction over 0 elements is degenerate but reachable from fuzz_shapes (which generates degenerate shapes per CLAUDE.md priority order). The empty-reduction case probably wants atol=0 (no error possible) or atol=base (treat as identity). Currently it gets the base atol, undocumented.
**Action:** add a docstring sentence: "k_dim<=0 is treated as no scaling". Drop the `max(k_dim, 1)` no-op. Add a test for the k_dim=0 boundary.
**Verdict:** **COMMENT** — minor; document the contract.

---

## 5. Documentation

### D1 — README §1 "verify your GPU" code block is fenced as `python` but is shell *(CONFIRMS docs-tester R-B2)*

**File/line:** `README.md:66-76`.
**What:** corroborated. The block opens `python -c "..."` inside a ```` ```python ```` fence. Copy-paste into a `.py` file is a SyntaxError. README is the PyPI landing page; this is the *first* code a user encounters.
**Action:** re-fence as ```` ```bash ```` OR strip the `python -c` wrapper and present the body as Python.
**Verdict:** **REQUEST_CHANGES** — README runnability matters more than any other doc.

### D2 — "Development Status :: 3 - Alpha" classifier vs. version `1.0.0rc1` *(NEW)*

**File/line:** `pyproject.toml:17`, `pyproject.toml:7`.
**What:** the `classifiers` list says `"Development Status :: 3 - Alpha"` but the version is `1.0.0rc1`. PyPI's classifier-driven badging will flag this as Alpha to package indexers and pip-search consumers. For an rc cut, the right value is `"Development Status :: 4 - Beta"`; for the eventual `1.0.0` final, `"Development Status :: 5 - Production/Stable"`. The MIGRATION explicitly notes (line 13-16): "v0 was an alpha PyPI publish (Development Status :: 3 - Alpha); v1 is the first release-quality tag" — so the team knows this should change but missed updating the classifier.
**Action:** bump to `"Development Status :: 4 - Beta"` for rc1, `5 - Production/Stable` for final.
**Verdict:** **COMMENT** — non-blocking but a 30-second fix for credibility.

### D3 — README §"Tested hardware and software" claims "120 unit tests / 235 GPU integration tests / All 408 tests passing" but the PR claims 224 *(NEW)*

**File/line:** `README.md:359-364` vs. PR description "Test count: 117 → 224 (+107)" (gh pr view 2).
**What:** the README footer in the v1.0 branch still says "120 unit tests (CPU)... 53 example tests... 235 GPU integration tests... All 408 tests passing with zero failures". The PR description states the new count is 224 (up from 117). Neither the README nor the PR aligns with the docs-tester audit's actual run (`pytest -q → 224 passed, 1 skipped`). The 408-tests-passing claim is leftover from v0; it has not been rolled forward.
**Action:** update README §"Test coverage" with v1 numbers. Cross-reference against `pytest -q --co` for unit count, against `tests/gpu_integration/` for the integration count (which the docs-tester noted hard-codes CUDA so the "auto-skipped" claim is also wrong on MPS hosts).
**Verdict:** **COMMENT** — accuracy nit; affects user trust in the README's other claims.

### D4 — `MIGRATION.md` §7 example uses `n=20` but the real `fuzz_strides` returns *at most* 7 entries (one per category) *(NEW)*

**File/line:** `MIGRATION.md:268`, `src/gpucheck/fuzzing/strides.py:211-253`.
**What:** the migration example reads `strides = fuzz_strides(shape=(8, 16, 32), n=20, seed=42)`. The implementation iterates over the seven categories and applies `out = out[:n]` *after* — so `n=20` is silently capped to 7. The doc misleads users into thinking `n` controls *count of samples*, when it controls *count of categories returned*. There's also no `dtype` argument in the docs-quoted call but the implementation **requires it** (signature `fuzz_strides(shape, dtype, *, n=None, ...)`) — the docs-tester audit M-B7/M-B8 caught this. My addition: even if you fix the dtype omission, `n=20` is still nonsense.
**Action:** drop `n=20` from the example, OR rename `n` to `max_categories` and document the cap. Better: change the API so that `n>len(CATEGORIES)` raises rather than silently truncating.
**Verdict:** **REQUEST_CHANGES** — the docs-tester's existing demand to fix the signature; my addition makes the case for redesigning `n` semantics.

### D5 — CHANGELOG cites `pytorch#179294` for SDPA backward but the entry name is `scaled_dot_product_attention.backward` *(NEW)*

**File/line:** `CHANGELOG.md:62-67`, `pyproject.toml:108-110` (xfail entries).
**What:** CHANGELOG enumerates 12 GitHub issue numbers. I spot-checked `pytorch#179294` against the published xfail list: the registry entry is `"scaled_dot_product_attention.backward"`, but the cited issue at github.com/pytorch/pytorch/issues/179294 (per the comment in `pyproject.toml:108-109`) — without GH access I can't verify the citation, but this is the kind of thing a maintainer must spot-check before tag. The xfail list claims to be a "living document"; the comments in pyproject pin issue numbers but neither the CHANGELOG nor MIGRATION cross-link them as clickable GH URLs. A reader looking at MIGRATION.md §5 to investigate why `softmax.large_attention` is xfailed has to trust the inline comment in `pyproject.toml`.
**Action:** add explicit `https://github.com/pytorch/pytorch/issues/<n>` URLs to the CHANGELOG list and to the comments in `pyproject.toml`. Actually verify each issue exists and is the right one before tagging final. Bonus: add a `tools/verify_xfail_issues.py` that hits the GitHub API and asserts each issue is open or closed-with-explanation.
**Verdict:** **COMMENT** — non-blocking but a maintainer-credibility issue.

---

## Cross-axis findings (not counted in per-axis 3)

### X1 — `apply_config_tolerances` and `reset_config_tolerances` are not in `gpucheck.assertions.__all__` *(NEW)*

**File/line:** `src/gpucheck/assertions/__init__.py:16-25`. The module imports `apply_mps_xfail_config`, `compute_tolerance`, `is_mps_xfailed`, `mps_xfail_list`, `register_mps_xfail`, `reset_mps_xfail`, `tolerance_context` (line 6-14) and re-exports them — but `apply_config_tolerances`/`reset_config_tolerances` are deliberately omitted. Yet `plugin.py:79-84` imports them by name. This is consistent with "test helpers stay private" but the asymmetry (xfail config helpers exposed, tolerance config helpers hidden) reads as an oversight rather than a design.
**Verdict:** **COMMENT** — pick a side, document the choice.

---

## Summary table (verdict per finding)

| ID | Axis | Severity | Verdict | NEW vs. CONFIRMS |
|----|------|----------|---------|-------------------|
| **S1** | Security | MEDIUM | **BLOCK** | NEW (surprise candidate) |
| S2 | Security | LOW | COMMENT | CONFIRMS (PM-5) |
| S3 | Security | LOW | COMMENT | NEW |
| S4 | Security | LOW | COMMENT | CONFIRMS (PM-1) |
| P1 | Performance | LOW | COMMENT | NEW |
| P2 | Performance | LOW | COMMENT | NEW |
| P3 | Performance | LOW | COMMENT | NEW |
| P4 | Performance | LOW | COMMENT | CONFIRMS (PM-4) |
| **A1** | API design | HIGH | **BLOCK** | NEW |
| **A2** | API design | HIGH | REQUEST_CHANGES | CONFIRMS (M-B9) + design |
| A3 | API design | LOW | COMMENT | NEW |
| A4 | API design | LOW | COMMENT | CONFIRMS (api-dx row 3) |
| **N1** | Numerical | **CRITICAL** | **BLOCK** | **NEW — surprise** |
| **N2** | Numerical | HIGH | REQUEST_CHANGES | NEW |
| N3 | Numerical | LOW | COMMENT | CONFIRMS (api-dx row 2) |
| N4 | Numerical | LOW | COMMENT | NEW |
| D1 | Documentation | MEDIUM | REQUEST_CHANGES | CONFIRMS (R-B2) |
| D2 | Documentation | LOW | COMMENT | NEW |
| D3 | Documentation | LOW | COMMENT | NEW |
| D4 | Documentation | MEDIUM | REQUEST_CHANGES | NEW (extends M-B7/M-B8) |
| D5 | Documentation | LOW | COMMENT | NEW |
| X1 | (cross) | LOW | COMMENT | NEW |

Findings counted by axis (charter requires ≥3 each, ≥15 total): **Security 4 / Performance 4 / API 4 / Numerical 4 / Documentation 5 / cross 1 = 22 total.** Surprise finding = **N1** (the silent 11× tolerance error in `baseline_2x + k_dim`).

---

## Final verdict — Would I merge this PR?

**No, not as-is. REQUEST CHANGES — block on N1, A1, A2, S1; the rest are COMMENT-grade once those four are addressed.**

This is a strong PR — five thousand lines of MPS backend, stride fuzzing, dashboard, determinism sanitizer, thread-safe tolerance plumbing, all with tests passing 224/224 and ruff/mypy strict-clean. The track structure (A/B/C/D) is disciplined and the audit trail is exemplary. But four findings cross my must-fix bar: **(N1) `baseline_2x + k_dim` silently produces 11× looser tolerances than the canonical formula** — this is a numerical-correctness regression in the headline `assert_close` API and will hide real bugs in any FlashAttention test; **(A1) stride-category names diverge between code and docs** — every copy-pasted MIGRATION snippet that names a category fails at runtime; **(A2) `assert_deterministic` MIGRATION example uses three nonexistent kwargs** AND the byte-equality default is the wrong contract for the documented MPS use case; **(S1) `tomli` is referenced for the Python 3.10 fallback path but undeclared as a dependency**, so on a fresh 3.10 install the user's `pyproject.toml` overrides silently no-op. Fix those four and ship — the remaining 18 findings are advisory and can ride into v1.0.0 final or v1.0.1.

Two pieces of standing-ovation work that I want to call out: (i) the deadlock-safe MPS event timer with the AST-introspecting test that asserts no `Event.synchronize` exists in the path is **excellent** defensive engineering, and (ii) the `_CUDA_HOME_ALLOWLIST` mitigation for TM-E1 with `realpath`+exact-prefix-with-separator matching is a textbook example of how to thread a security finding through the merge.
