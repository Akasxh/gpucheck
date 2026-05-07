# architecture-reviewer — gpucheck v1.0

## Phase 1 — structural map

```
gpucheck (pytest plugin)
├── plugin.py                pytest hooks (markers, fixtures, terminal summary)
├── assertions/              pure: numeric helpers, dtype tolerances
├── decorators/              pure: parametrize wrappers
├── fixtures/                interacts with torch.cuda + pynvml
├── arch/                    reads pynvml / torch + 1 env var (NVIDIA_TF32_OVERRIDE)
├── analysis/                reads/writes JSON baselines (untrusted at rest? — see below)
├── reporting/               console/JSON/CI (XML, GitHub annotations)
├── fuzzing/                 hypothesis strategies, no I/O
├── sanitizers/
│   ├── memory.py            torch.cuda + pynvml, no subprocess
│   └── race.py              **subprocess to compute-sanitizer** ← privilege boundary
```

## Phase 2 — trust-boundary analysis

### Boundaries
1. **PyPI install boundary** — gpucheck installs into the user's
   pytest environment. No post-install scripts in `pyproject.toml` (good;
   hatchling backend, no `setup.py`). Optional extras pull torch,
   cupy-cuda12x, triton, hypothesis — all mature / signed.
2. **Test code boundary** — gpucheck APIs are called *by user test code*.
   This means the threat actor model is "compromised test author" or
   "test corpus from a malicious source", not "remote attacker". The
   plugin runs with the same privileges as `pytest`.
3. **Subprocess boundary** — `sanitizers/race.py` shells out to a
   binary located via `shutil.which` *or* an env-var-resolved path. The
   subprocess inherits the parent env, including `LD_PRELOAD`,
   `DYLD_INSERT_LIBRARIES`, `PYTHONPATH`. The MPS work will add a second
   such boundary (`xcrun metal`).
4. **File-system boundary** — JSON baselines for regression analysis
   (`analysis/regression.py`) and reports (`reporting/json.py`,
   `reporting/ci.py`) live on disk and may be read on a later run.

### Defense-in-depth review
| Layer | Status | Notes |
|---|---|---|
| Input validation at API | partial | race.py:117-120 validates `__module__` / `__name__` as identifiers — good. No size cap on `extra_args`. |
| Subprocess argv hygiene | good | argv list, no `shell=True` anywhere in the repo. |
| Env-var hygiene | weak | Subprocess inherits the full parent env (race.py:183 does not pass `env=...`). MPS implementer must not repeat. |
| Tempfile hygiene | adequate | `tempfile.mkstemp` — but parent dir is `/tmp` (world-traversable on macOS / Linux). |
| Error redaction | good | reports use Rich console; no obvious PII leakage. |
| JSON deserialization | weak | Baselines are loaded with `json.loads` then iterated; no schema check. A malformed baseline can raise inside `regression.py` rather than fail safely. |

### Component coupling
- `sanitizers/race.py` and (proposed) `sanitizers/mps.py` will share
  the same subprocess pattern. **Recommendation**: factor out a single
  `_run_external_tool(binary, argv, *, allowlisted_roots, env_passthrough)`
  helper in `sanitizers/_subprocess.py` so both call sites get the same
  hardening for free.
- Reporting and analysis modules share JSON I/O. **Recommendation**:
  centralize a `_load_baseline(path) -> Baseline` with a typed-dict /
  pydantic-style schema check.

### Data-flow map (untrusted → sensitive op)
```
user test code
   ├──(API call)── gpucheck.assert_close                — pure, no boundary crossed
   ├──(API call)── gpucheck.run_with_sanitizer(fn)
   │                 ├── inspect.getmodule(fn).__name__ ── validated as identifier ✓
   │                 ├── tempfile.mkstemp + write Python source
   │                 └── subprocess.run([compute-sanitizer, …, sys.executable, script]) ── env-var path-trusted ✗ (TM-E1)
   ├──(JSON)──── load baseline.json ── no schema, no size cap ⚠
   └──(env)───── NVIDIA_TF32_OVERRIDE ── only read, never written ✓
```

## Phase 3 — verification & findings

### AR-1 — No factored helper for external tool invocation
- **Severity**: LOW (advisory; design hardening)
- **Location**: `src/gpucheck/sanitizers/race.py` (entire file)
- **Description**: The subprocess + temp-file + env-var-resolution
  pattern in race.py will be copy-pasted into the upcoming
  `sanitizers/mps.py`. Without a shared helper, hardening fixes (such
  as TM-E1) will need to be applied twice and may drift.
- **Confidence**: HIGH
- **Remediation**: extract a `_run_external_tool(...)` in
  `src/gpucheck/sanitizers/_subprocess.py`; both race.py and mps.py call
  it. Helper enforces: (a) absolute resolved path, (b) allowlisted
  install roots, (c) explicit `env=` to drop `LD_PRELOAD` /
  `DYLD_INSERT_LIBRARIES`, (d) cwd to a tmpdir, (e) stdin=DEVNULL.

### AR-2 — JSON baselines accepted without schema validation
- **Severity**: LOW
- **Location**: `src/gpucheck/analysis/regression.py:333,350` ;
  `src/gpucheck/reporting/json.py:102-103`
- **Description**: A baseline JSON read from disk is fed straight into
  the analysis code. If a CI artifact is replaced (e.g., main-branch
  baseline overwritten by a PR) with a structurally different file,
  the plugin raises `KeyError` / `TypeError` mid-run rather than a
  clean error. There is no integrity check (no signature, no checksum).
- **Confidence**: HIGH
- **Remediation**: add a `_validate_baseline(d: Any) -> Baseline`
  function that asserts presence and types of required keys, with a
  size cap (e.g., 10 MiB) before parsing. Optionally accept a
  `baseline.sha256` companion file.

### AR-3 — Subprocess inherits full parent env
- **Severity**: LOW (becomes MEDIUM if combined with TM-E1)
- **Location**: `src/gpucheck/sanitizers/race.py:183-188`
- **Description**: `subprocess.run(cmd, …)` (no `env=`). The child
  inherits `LD_PRELOAD`, `DYLD_INSERT_LIBRARIES`, `PYTHONPATH`, and
  any user env. For a test runner this is conventional; the concern
  is that user-controlled env vars then influence
  `compute-sanitizer`'s loader. Defense-in-depth fix: pass an explicit
  minimal env.
- **Confidence**: MEDIUM (low likelihood, easy fix)
- **Remediation**:
```python
env = {k: v for k, v in os.environ.items()
       if k in {"PATH", "HOME", "USER", "TMPDIR", "CUDA_HOME",
                "CUDA_PATH", "CUDA_VISIBLE_DEVICES"}}
subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
```

## Architectural note on the MPS expansion
The repository's strength is its lazy-import discipline (CLAUDE.md key
design decision: "torch/pynvml never imported at collection time").
**This same discipline must extend to MPS deps**: `import objc`,
`import Metal`, `import mlx` must live behind a function-local guard,
not at module top level. Otherwise, importing `gpucheck.sanitizers`
on Linux CI will fail to load.
