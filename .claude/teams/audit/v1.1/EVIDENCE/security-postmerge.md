# EVIDENCE — security-postmerge (security-architecture-reviewer, v1.1 prep)

**Persona:** security-architecture-reviewer
**Branch:** release/v1.0 @ 82b853e (Tracks A+B+C+D merged, lint clean)
**Method:** 3-phase (structural map → reasoning over merged-only diff → verification by re-reading the committed code).
**Diff scoped to:** `git diff origin/main..release/v1.0 -- src/gpucheck/` (1,780 added / 79 removed).
**Prior baseline (binding):** `.claude/teams/security/v1.0/FINDINGS.md` — 0 BLOCKER, 3 MEDIUM (CFG-2, TM-E1, DEP-1) all addressed in merge; 13 LOW; 5 N-prefix design-stage MPS items.

---

## Trust boundaries inherited and the 5 cross-track questions

| # | Question (from charter) | Real entry point in merged code | Verdict |
|---|---|---|---|
| 1 | Track A MPS subprocess + Track C `CUDA_HOME` allowlist | `backends/mps.py:225` `subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], ...)` — single argv, no shell, no env, no user input. Does NOT route through `_find_compute_sanitizer` so the allowlist is irrelevant here. | PASS (see PM-1) |
| 2 | xfail config-load + Track D CI permissions | `plugin.py:53,66-78` `_load_pyproject_config(config.rootpath)` reads `pyproject.toml` from the pytest rootpath; xfail entries are stringified into a `set`; no `eval`, no path expansion off the data, no subprocess. | PASS (see PM-2) |
| 3 | Backend-selection runtime override (env / CLI) | `backends/__init__.py:33-86` selects strictly by `name.lower() in {"cuda","mps"}`; `available_backends()` is hard-coded CUDA→MPS; **no env var, no config knob, no CLI flag** downgrades to a less-secure fallback. | PASS (see PM-3) |
| 4 | Stride fuzzing × `assert_close` MPS overlay | `fuzzing/strides.py:77-91` (`_broadcast`, stride-0 expand), `assertions/close.py:174-189` (GPU fast-path), `:22-67` (`_to_numpy` slow-path) | ADVISORY — see PM-4 |
| 5 | HTML reporter XSS handling | `reporting/html.py:86-87,99-129,157-252` | ADVISORY — see PM-5 |

---

## Findings

### PM-1 — MPS subprocess does NOT inherit the Track-C allowlist (and does not need to)

- **Severity:** LOW (informational; downgrade-or-keep at scribe's discretion)
- **Category:** OWASP-A03 (Injection), defense-in-depth
- **Location:** `src/gpucheck/backends/mps.py:217-232`
- **Exploitability:** Local, requires write to system PATH or hostile `sysctl`. No realistic vector.
- **Confidence:** HIGH
- **Description:** Track-A's only subprocess in the MPS backend is `_detect_apple_chip()` which calls `["sysctl", "-n", "machdep.cpu.brand_string"]`. This invocation:
  - uses an absolute-name argv list (no shell),
  - takes no user input,
  - has a 2-second timeout,
  - is unrelated to compute-sanitizer / `CUDA_HOME`.
  Therefore Track-C's `_CUDA_HOME_ALLOWLIST` (`sanitizers/race.py:26-30`) does not apply, and **no new validation gap was introduced** by the merge. The MPS subprocess deliberately avoids `xcrun metal` (research findings N1/N2 — Apple shader-compiler path traversal / preprocessor abuse), as called out in the comment block on `mps.py:222-223`.
- **Evidence:**
  ```python
  out = subprocess.check_output(
      ["sysctl", "-n", "machdep.cpu.brand_string"],
      text=True,
      timeout=2.0,
  )
  ```
  No env scrubbing (parent env inherited), but `sysctl` does not honor any env-driven path resolution that would matter here.
- **Fix sketch:** none required. If the team wants symmetry with Track-C, pin to `/usr/sbin/sysctl` (the macOS canonical path) and pass `env={"PATH": "/usr/sbin:/usr/bin"}`.
- **Verdict:** **PASS**

---

### PM-2 — `pyproject.toml` xfail config loader is safe under hostile input, but is over-broad in exception swallowing

- **Severity:** LOW
- **Category:** OWASP-A05 (Security Misconfiguration) / robustness
- **Location:** `src/gpucheck/plugin.py:56-88`, `src/gpucheck/assertions/tolerances.py:184-214`
- **Exploitability:** Local, requires attacker control of the project's `pyproject.toml`. This is equivalent to controlling the test harness — not a realistic privilege escalation.
- **Confidence:** HIGH
- **Description:** A malicious `pyproject.toml` cannot achieve code execution via the xfail loader:
  1. `tomllib.load(f)` is a pure-data parser; no `__reduce__`-style RCE surface.
  2. `mps_xfail_from_config` extracts `ops` only if it is a `list`, then coerces each entry to `str`.
  3. The set is consulted by `is_mps_xfailed(op_name) -> bool` (string equality only).
  4. There is **no path traversal** — the loader joins `Path(rootpath) / "pyproject.toml"` and rejects on `is_file()` failure; it never reads attacker-supplied paths.
  Worst-case effect: an attacker who controls the file can suppress test failures via inflated xfail lists (denial of *testing*). That requires repo write access already.

  The cosmetic concern is the bare `except Exception: pass` on `plugin.py:86-88`. It silently absorbs any error including `MemoryError` and TOML-parser exceptions — masking misconfiguration from the user. Reasonable for a "best-effort config load" but documented in the docstring.
- **Evidence:**
  ```python
  # plugin.py:73-85
  try:
      import tomllib
  except ModuleNotFoundError:
      import tomli as tomllib  # type: ignore[no-redef,unused-ignore]
  with pyproject.open("rb") as f:
      data = tomllib.load(f)
  ...
  apply_config_tolerances(data)
  apply_mps_xfail_config(data)
  ```
  ```python
  # tolerances.py:199-205
  section = config.get("tool", {}).get("gpucheck", {}).get("mps", {}).get("xfail")
  ...
  ops = section.get("ops")
  if not isinstance(ops, list):
      return None
  return {str(o) for o in ops}
  ```
- **Fix sketch:** narrow the catch and log:
  ```python
  except (OSError, tomllib.TOMLDecodeError) as exc:
      import warnings; warnings.warn(f"gpucheck: pyproject.toml load failed: {exc}", RuntimeWarning, stacklevel=2)
  ```
- **Verdict:** **PASS** (LOW informational note retained for v1.1 cleanup)

---

### PM-3 — Backend selection has no user-controllable downgrade path (priority is hard-coded)

- **Severity:** LOW
- **Category:** OWASP-A05 (Security Misconfiguration) / supply-chain
- **Location:** `src/gpucheck/backends/__init__.py:33-86`
- **Exploitability:** N/A (no override surface)
- **Confidence:** HIGH
- **Description:** `available_backends()` returns CUDA-then-MPS in a **hard-coded order**. `get_backend(name)` accepts only the literals `"cuda"` and `"mps"` (rejects everything else with `ValueError`). There is **no environment variable, no `pytest_addoption` flag, no `[tool.gpucheck.backend]` config block** that would let a hostile environment force selection of a less-secure or attacker-controlled backend. I grep'd `os.environ` / `getenv` in `src/gpucheck/backends/`, `plugin.py`, and `decorators/` — zero hits.
  The one user-facing input that touches device choice is `--gpu-device` (`plugin.py:27-31`), parsed in `gpu_device` fixture (`plugin.py:147-186`) and validated against `torch.cuda.device_count()` — no MPS code path, no shell-out, fails closed with `pytest.fail` on bad format.
- **Evidence:**
  ```python
  # backends/__init__.py:33-61
  def available_backends() -> list[Backend]:
      backends: list[Backend] = []
      try:
          from gpucheck.backends.cuda import CUDABackend
          cuda = CUDABackend()
          if cuda.is_available():
              backends.append(cuda)
      except ImportError: pass
      try:
          from gpucheck.backends.mps import MPSBackend
          mps = MPSBackend()
          if mps.is_available():
              backends.append(mps)
      except ImportError: pass
      return backends
  ```
- **Fix sketch:** none required. If v1.1 ever introduces an env-driven backend override (e.g. `GPUCHECK_BACKEND`), it must be allowlisted exactly as `name_lower in {"cuda","mps"}` does today.
- **Verdict:** **PASS**

---

### PM-4 — `assert_close` slow-path can crash on stride-0 / non-contiguous fuzzed tensors (availability bug, not RCE)

- **Severity:** LOW
- **Category:** Robustness / fuzz-input handling (not OWASP)
- **Location:** `src/gpucheck/assertions/close.py:22-67` (`_to_numpy`), `:174-189` (GPU fast-path), `src/gpucheck/fuzzing/strides.py:77-91` (`_broadcast`)
- **Exploitability:** Local, the attacker is the test author. Outcome is `RuntimeError`, not memory disclosure.
- **Confidence:** HIGH
- **Description:** Track-B's stride fuzzer can produce tensors with **stride 0** (`_broadcast` via `tensor.expand(shape)` at `strides.py:91`) and other non-contiguous layouts (`_transpose`, `_slice`, `_non_contig`). When the stride fuzzer is paired with `assert_close`:
  - **GPU fast-path (CUDA or MPS)** — `torch.allclose(actual, expected, ...)` is stride-aware and handles expanded views correctly. **Safe.** No buffer over-read on the device side; PyTorch's stride machinery rejects out-of-bounds reads at the dispatcher.
  - **Slow path (CPU numpy)** — `_to_numpy` calls `tensor.detach().cpu().numpy()` (close.py:32-37). On torch >=2.1, calling `.numpy()` on a non-contiguous view succeeds (numpy makes a strided wrapper) but on stride-0 expanded views the resulting numpy array shares storage at stride 0; `np.abs(actual_f64 - expected_f64)` then **broadcasts correctly** because numpy honors stride 0 too. No buffer over-read.
  - **Edge case** (the actual bug): on torch <2.1, `.numpy()` raises `RuntimeError: input is not contiguous` for some stride layouts. The current code does not call `.contiguous()` before `.numpy()`, so the slow path can crash. This is an **availability bug**, not a memory-safety bug — numpy never reads past the storage size; PyTorch's stride/storage offset checks gate that.
- **Evidence:** the `_to_numpy` helper has no `.contiguous()` call before `.numpy()`:
  ```python
  # close.py:28-38
  if hasattr(tensor, "detach"):
      t = tensor.detach().cpu()
      if t.is_floating_point():
          if t.dtype.itemsize >= 8:
              return t.double().numpy()
          if t.dtype.itemsize >= 4:
              return t.numpy()
          return t.float().numpy()
      return t.numpy()
  ```
- **Fix sketch:** force contiguity on the CPU side to make the slow-path uniformly safe:
  ```python
  t = tensor.detach().cpu().contiguous()  # tolerates stride-0 / transposed views
  ```
  No security impact, just better UX with the new stride fuzzer.
- **Verdict:** **ADVISORY** (NEW post-merge — this surface didn't exist before Track B added stride fuzzing; the v1.0 audit predates the merge)

---

### PM-5 — HTML reporter XSS surface: escaping is mostly correct, with two cosmetic gaps and one tainted `class=` attribute path

- **Severity:** LOW
- **Category:** OWASP-A03 (Injection) / Stored-XSS class
- **Location:** `src/gpucheck/reporting/html.py` — `_esc` (line 86-87), `_pill` (90-96), `_render_test_results` (157-180), `_render_benchmarks` (183-206), `_render_memory` (209-226), `_render_comparison` (229-252).
- **Exploitability:** Local; attacker must control test names, error messages, or kernel names that flow into `results.json` and then into the HTML dashboard. Realistic when CI publishes the dashboard as an artifact.
- **Confidence:** MEDIUM
- **Description:** The `_esc()` helper uses `html.escape(value, quote=True)` — strong, escapes `& < > " '`. **Almost every** dynamic value reaches `_esc()` before insertion. Counts:
  - **Test/benchmark names:** `_esc(b.get("name", ""))`, `_esc(r.get("name", ""))` — safe.
  - **Error messages:** `_esc(msg)` inside `<pre>` — safe.
  - **GPU info keys+values:** `_esc(k)` / `_esc(v)` — safe.
  - **Status pill text:** `_esc(status.upper())` — safe.

  Three concerns:

  **(a) `class="{klass}"` is unescaped, but `klass` is whitelisted.** Lines 171, 218, 243 interpolate `klass` directly without `_esc`. `klass` is computed from a 2-or-3-element whitelist (`"regression"`, `"passed-row"`, `"new"`, `"removed"`) so today it cannot carry attacker bytes. Confidence on this rests on the whitelist being preserved. If a future commit ever sets `klass = r.get("class")` or similar, it becomes a stored-XSS via `class="><script>...`. Defense-in-depth says always wrap in `_esc`.

  **(b) `_pill()` style attribute interpolates `bg` raw.** Line 93: `style="background:{bg};...`. `bg` is from a static `_STATUS_PILL_BG` dict or fallback `#666666`. Safe today; if anyone later does `_STATUS_PILL_BG[user_input]`, the raw interpolation becomes a CSS-injection vector. Same defense-in-depth rule.

  **(c) `delta:+.1f%` formatter on line 245 will crash on non-numeric `delta_pct`.** Not security; UX.

  None of (a)/(b)/(c) is exploitable on the merged code. This is structural / future-proofing.
- **Evidence:**
  ```python
  # html.py:90-96
  def _pill(status: str) -> str:
      bg = _STATUS_PILL_BG.get(status, "#666666")
      return (
          f'<span class="pill" style="background:{bg};color:#fff;'   # bg unescaped
          f'padding:2px 8px;border-radius:9px;font-size:12px;">'
          f"{_esc(status.upper())}</span>"
      )

  # html.py:171
  f'<tr class="{klass}"><td>{_esc(r.get("name", ""))}</td>'           # klass unescaped (whitelisted)
  ```
- **Fix sketch:**
  1. Wrap `klass` and `bg` in `_esc()` defensively, even though both are whitelisted today:
     ```python
     return f'<span class="pill" style="background:{_esc(bg)};color:#fff;...'
     f'<tr class="{_esc(klass)}"><td>{_esc(r.get("name", ""))}</td>'
     ```
  2. Add a unit test that feeds `_render_test_results` a dict with `name="</td><script>alert(1)</script>"` and asserts `<script>` does **not** appear in the output (regression guard).
  3. Add a `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline'` `<meta>` tag so even if a future regression introduces script injection, the browser won't execute it.
- **Verdict:** **ADVISORY** (NEW post-merge — Track-D introduced the HTML reporter; the v1.0 audit only saw JSON+CI reporters)

---

## Cross-track interactions inspected and cleared

- **A↔C subprocess inheritance** — verified MPS subprocess does not exec `compute-sanitizer` and does not consume `CUDA_HOME`. PM-1.
- **A↔D config-loading TOCTOU / RCE** — verified `tomllib.load` + string-set construction has no code-execution surface. PM-2.
- **A↔D backend-priority downgrade** — verified no env/CLI/config knob exists for backend coercion. PM-3.
- **A↔B fuzzed-stride buffer over-read** — verified PyTorch stride machinery + numpy stride-aware ops both refuse out-of-bounds. Only availability bug remains (PM-4).
- **D↔* HTML XSS** — `html.escape(quote=True)` covers all attacker-tainted strings; remaining concerns are defense-in-depth. PM-5.

## What I did NOT find (and looked for)

- No `eval` / `exec` / `compile` introduced in the merge.
- No `pickle.load` / `marshal.loads` introduced in the merge.
- No new `subprocess` shell=True; the only new subprocess is `mps.py:225` (sysctl, hardened argv).
- No new env-var trust without validation (the existing `NVIDIA_TF32_OVERRIDE` finding from v1.0 LOW list is unchanged).
- No new path-traversal sinks; `_load_pyproject_config` joins via `pathlib` and gates on `is_file()`.
- No regression of the v1.0 `CUDA_HOME` allowlist — `sanitizers/race.py:62-96` is unchanged structurally and still uses `os.path.realpath` + prefix-with-separator check.

---

## Confidence

- PM-1, PM-2, PM-3: HIGH (verified by reading every site).
- PM-4: HIGH (availability bug confirmed by reading `_to_numpy`; security non-impact reasoned from PyTorch stride invariants).
- PM-5: MEDIUM — I did not run a hostile-string test through `HTMLReporter.render`. The reasoning relies on `html.escape(quote=True)` semantics and the static-whitelist invariant on `klass`/`bg`, both of which are easy to verify by inspection but easy to break in v1.1.

## Tally (post-merge, NEW only)

| Severity | Count |
|---|---|
| BLOCKER / CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW (NEW post-merge) | 2 (PM-4, PM-5) |
| PASS | 3 (PM-1, PM-2, PM-3) |

The v1.0 baseline's 3 MEDIUM items (CFG-2, TM-E1, DEP-1) are all addressed by merged commits 02507da, race.py:21-109, and a60e3a5 respectively. No regressions.

**Overall post-merge verdict from this reviewer: ADVISORY** — clean to merge for v1.0; PM-4 and PM-5 should be addressed in v1.1.
