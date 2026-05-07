# threat-modeler — MPS backend + existing subprocess

## Phase 1 — attack-surface enumeration

### Existing surfaces (in repo today)
| # | Surface | Location |
|---|---|---|
| E1 | `subprocess.run` of `compute-sanitizer` with user-importable function | `src/gpucheck/sanitizers/race.py:183` |
| E2 | Trust of `CUDA_HOME`/`CUDA_PATH` env vars to locate binary | `src/gpucheck/sanitizers/race.py:56-59` |
| E3 | `tempfile.mkstemp` writes Python source executed by subprocess | `src/gpucheck/sanitizers/race.py:165-168` |
| E4 | `sys.path[:0] = {sys.path!r}` echoed into a child script | `src/gpucheck/sanitizers/race.py:122-126` |
| E5 | JSON baselines deserialised without schema validation | `src/gpucheck/analysis/regression.py:333,350`, `src/gpucheck/reporting/json.py:102-103` |
| E6 | Trust of `NVIDIA_TF32_OVERRIDE` env var | `src/gpucheck/arch/tensor_cores.py:183` |

### NEW surfaces (MPS — not yet implemented)
| # | Surface | Threat type |
|---|---|---|
| N1 | `subprocess.run([... "xcrun", "metal", ...])` compiling shader source | Command/path injection, RCE via untrusted shader |
| N2 | `subprocess.run([... "xcrun", "metallib", ...])` linking `.air` to `.metallib` | Path injection, output-path traversal |
| N3 | `task_info(mach_task_self(), TASK_VM_INFO, ...)` via ctypes | Privilege confusion (process-level vs container-level) |
| N4 | `[mps]` / `[apple]` extras pulling pyobjc / mlx | Supply chain (typosquat, dep confusion) |
| N5 | MPS-dispatch sanitizer hook (proposed): wraps `torch.mps.<op>` to record args | TOCTOU between record & dispatch, race in pre/post hooks |

## Phase 2 — STRIDE analysis

Framework: **STRIDE** (this is a traditional Python library, not an
agentic AI codebase).

Trust boundaries:
- **Untrusted**: pytest test functions written by gpucheck *consumers*
  (they may be malicious or compromised — e.g., a transitive test fixture
  pulled from a typosquat package).
- **Trusted-with-care**: env vars `CUDA_HOME`, `CUDA_PATH`,
  `NVIDIA_TF32_OVERRIDE`, future `METAL_HOME`, `XCRUN`,
  `GPUCHECK_METAL_FLAGS`.
- **Trusted**: gpucheck source itself, the host's `xcrun` / `metal` /
  `metallib` toolchain (assumes attacker has not already replaced them).

### N1 — `xcrun metal` subprocess

| STRIDE | Vector | Likelihood | Impact |
|---|---|---|---|
| **T**ampering | Test author calls `compile_metal(shader_src)` with attacker-influenced content (e.g., shader source read from a fuzzed JSON corpus). Metal compiler is a full Clang frontend with `#include`, `#pragma`, `#import`. A `#include "/etc/passwd"` style payload won't RCE on its own, but a `#pragma clang load_plugin "../../mal.dylib"` *can* if the toolchain accepts it. | LOW | RCE |
| **E**lev. of Priv. | If the gpucheck process runs in CI with elevated permissions (Apple Developer signing identity, keychain access), an injected shader's `__attribute__((constructor))` translation step can run on the build host. | LOW–MED | CI compromise |
| **S**poofing | Path injection on `xcrun`: if `_find_xcrun()` follows `XCRUN` env var (mirroring `_find_compute_sanitizer` at race.py:56), an attacker who controls the env can substitute a shim binary. | MED | local RCE |
| **D**oS | Hostile shader can `#include` a recursive header chain or use macro expansion bombs to exhaust compiler memory. | MED | hangs CI |

#### Recommended mitigation pattern
```python
# RECOMMENDED skeleton for src/gpucheck/sanitizers/mps.py
import shutil, subprocess, tempfile, pathlib, os

_ALLOWED_FLAGS = frozenset({"-O0", "-O1", "-O2", "-O3", "-std=metal2.4",
                            "-std=metal3.0", "-Werror"})

def _find_xcrun() -> str:
    # 1. Resolve via shutil.which only — DO NOT trust XCRUN env var.
    path = shutil.which("xcrun")
    if path is None:
        raise FileNotFoundError("xcrun not on PATH")
    # 2. Reject anything outside the developer dir.
    real = os.path.realpath(path)
    if not real.startswith("/usr/bin/") and not real.startswith(
        "/Applications/Xcode.app/"
    ):
        raise PermissionError(f"xcrun resolved to untrusted path: {real}")
    return real

def compile_metal(source: str, *, extra_flags: list[str] | None = None) -> bytes:
    if not isinstance(source, str):
        raise TypeError("metal source must be str")
    # 3. Hard cap on size to defeat compiler-bomb DoS.
    if len(source) > 1_000_000:
        raise ValueError("shader source > 1 MiB")
    # 4. Reject preprocessor escape hatches.
    forbidden = ("#include", "#import", "#pragma clang load_plugin",
                 "__attribute__((constructor))")
    for tok in forbidden:
        if tok in source:
            raise ValueError(f"forbidden token in shader source: {tok}")
    # 5. Whitelist flags.
    flags = list(extra_flags or [])
    for f in flags:
        if f not in _ALLOWED_FLAGS:
            raise ValueError(f"flag not allowlisted: {f!r}")
    xcrun = _find_xcrun()
    with tempfile.TemporaryDirectory(prefix="gpucheck_metal_") as td:
        src_path = pathlib.Path(td) / "k.metal"
        air_path = pathlib.Path(td) / "k.air"
        src_path.write_text(source, encoding="utf-8")
        # 6. argv list, no shell=True; absolute, controlled paths.
        subprocess.run(
            [xcrun, "metal", "-c", *flags, str(src_path), "-o", str(air_path)],
            check=True, capture_output=True, timeout=30, env={"PATH": "/usr/bin"},
        )
        return air_path.read_bytes()
```

### N2 — `xcrun metallib`
Mostly identical to N1. Additional concern: the *output path*. If the
gpucheck API exposes `metallib_path: str` to the test author, an attacker
can pass `../../home/user/.ssh/authorized_keys` and overwrite arbitrary
files. **Mitigation**: never accept an output path from the caller —
always write to a `TemporaryDirectory()` and return bytes (as in the
skeleton above).

### N3 — `task_info` mach syscall

Vector: gpucheck's existing CUDA memory tracker reads
`torch.cuda.memory_stats()`, which is process-scoped. The MPS replacement
proposed via mach `task_info(TASK_VM_INFO)` returns **process-wide RSS,
not GPU-specific bytes**. Two attack-relevant consequences:

| STRIDE | Vector | Likelihood | Impact |
|---|---|---|---|
| **I**nfo. Disclosure | `task_info` reveals total resident memory across *all* threads of the test process, including other tests' data. Memory-leak assertions become a side channel that leaks sibling-test memory pressure. | LOW | side-channel |
| **R**epudiation | Because `task_info` numbers are reported as "GPU memory used", a sanitizer report can be deflected ("the leak is in another test, not my kernel"). | MED | false-positive reports |

mach `task_info` itself is **not a privilege boundary** when called on
`mach_task_self()` — the process can always read its own task info, no
entitlement needed. So no AppArmor/sandbox-escape risk. The real risk
is *misrepresentation* of the data. **Mitigation**: label the metric
honestly (`process_rss_bytes`, not `gpu_memory_bytes`) and document that
MPS does not provide per-allocation accounting comparable to
`torch.cuda.memory_stats()`.

### N4 — `[mps]` / `[apple]` extras supply chain

Realistic candidate deps:
- `pyobjc-core`, `pyobjc-framework-Metal`, `pyobjc-framework-MetalKit`
  — all maintained by ronaldoussoren on PyPI (verified human, MIT).
- `mlx` — Apple's official package, MIT-licensed, published from
  `ml-explore/mlx`.

Threats:

| STRIDE | Vector | Likelihood |
|---|---|---|
| **S**poofing (typosquat) | An attacker publishes `pyobj-framework-Metal` or `pyobjc-Metal-framework` (similar string distance). pyproject.toml typo at merge time pulls the malicious package. | MED |
| **T**ampering (dep confusion) | If an internal mirror is later configured, attacker uploads `mlx` to the mirror's index with a higher version. | LOW |
| **T**ampering (post-install) | pyobjc subpackages historically used `setup.py` with native-extension build steps that *do* execute code at install time. | HIGH for pyobjc, KNOWN-GOOD upstream |

**Mitigation**:
1. Pin extras with both lower and upper bounds:
   `pyobjc-framework-Metal>=10.0,<12.0`.
2. Ship a `uv.lock` or `requirements.lock` in the repo and CI-verify hashes.
3. Add `--require-hashes` install to the MPS-CI job.

### N5 — MPS-dispatch sanitizer hook

Proposed design (inferred from charter wording): wrap each
`torch.mps.<op>` so that the sanitizer can record args before dispatch
and verify state after. This creates two race windows.

| STRIDE | Vector | Likelihood | Impact |
|---|---|---|---|
| **T**ampering / TOCTOU | Hook reads tensor metadata (shape, stride, dtype, ptr) at *time of check*, then the kernel runs at *time of use*. Another thread can mutate the tensor (via numpy view, shared memory, or another MPS stream) between the two. | HIGH (asyncio-style code) | corrupted reports, missed bugs |
| **R**ace | If hooks register themselves with `torch.mps.set_pre_op_hook`-like API and aren't thread-safe, two pytest workers (`-n auto`) can interleave hook installation and removal. | MED | sanitizer no-ops silently |
| **E**lev. of Priv. (sanitizer bypass) | If MPS provides a "dispatch reorder" path (e.g., command-buffer commit with explicit ordering), an attacker test can deliberately reorder ops so the post-hook sees a state that masks the bug. | LOW (requires hostile test) | false negatives |

**Mitigation**:
1. Hooks must hold a per-tensor read lock (or copy metadata into an
   immutable record) for the full check-dispatch-verify window.
2. Hook registration must be process-global and guarded by a
   `threading.RLock`; document that pytest-xdist needs
   `--dist loadfile` if MPS sanitizer is enabled.
3. Force a `mps.synchronize()` between record-args and dispatch, and
   another between dispatch and verify-state. This kills the dispatch
   reorder vector.

## Existing surface findings (must remediate even without MPS)

### TM-E1 — Path-injection via `XCRUN`-style env-var trust
- **Severity**: MEDIUM
- **Location**: `src/gpucheck/sanitizers/race.py:56-59`
- **Vector**: `CUDA_HOME` / `CUDA_PATH` are read from `os.environ`,
  joined to `bin/compute-sanitizer`, and executed *without verifying
  that the resolved path lives under a trusted prefix*. An attacker
  who can set env vars in the user's shell (or who controls a `.env`
  loader the user runs before pytest) can swap the binary.
- **Confidence**: HIGH
- **Exploitability**: Local, Authenticated (attacker needs env-write).
- **Verification**: Confirmed by reading lines 56–62; only `os.access(…, os.X_OK)` is checked, not the path's prefix.
- **Remediation**:
```python
# VULNERABLE
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
if cuda_home:
    candidate = os.path.join(cuda_home, "bin", "compute-sanitizer")
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate

# FIXED
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH", "")
if cuda_home:
    candidate = os.path.realpath(os.path.join(cuda_home, "bin", "compute-sanitizer"))
    # Refuse anything outside conventional CUDA install roots.
    allowed_roots = ("/usr/local/cuda", "/opt/nvidia/cuda", "/opt/cuda",
                     os.path.realpath(cuda_home))
    if (any(candidate.startswith(r + os.sep) or candidate == r
            for r in allowed_roots)
        and os.path.isfile(candidate)
        and os.access(candidate, os.X_OK)):
        return candidate
```

### TM-E2 — `sys.path` snapshot leaked into child script
- **Severity**: LOW
- **Location**: `src/gpucheck/sanitizers/race.py:122-126`
- **Vector**: `f"import sys; sys.path[:0] = {sys.path!r}\n"` embeds the
  current `sys.path` (which may contain absolute paths to the user's
  workspace, including `/private/var/folders/...` tmp dirs and CWD-style
  `''`) into a Python source file written to disk. The file is
  short-lived but readable by other users on the host while it exists
  (default umask). This is *information disclosure* of project layout
  and absolute paths, not RCE — the script content is constructed from
  the validated module/function name (`isidentifier()` check is in
  place at line 117–120, well done).
- **Confidence**: HIGH
- **Exploitability**: Local, Unauthenticated (any local user during the
  ~hundreds-of-ms tmp file lifetime).
- **Remediation**: write the temp file to a `tempfile.TemporaryDirectory()`
  with `mode=0o700` rather than `tempfile.mkstemp` which uses world-
  readable default mode `0o600` on the file but the *parent dir* is
  often `/tmp` (world-traversable). Or better: pass the wrapper script
  via `python -c …` argv — no on-disk artifact at all.

## Phase 3 — verification

| Threat | Concrete code path? | Existing mitigation? | Net status |
|---|---|---|---|
| TM-E1 path injection | yes (race.py:56-59) | only `isfile + X_OK` | **MEDIUM** |
| TM-E2 sys.path disclosure | yes (race.py:122) | tmpfile mode | **LOW** |
| N1 metal RCE | no (code TBD) | n/a | **ADVISORY** |
| N2 metallib path traversal | no (code TBD) | n/a | **ADVISORY** |
| N3 task_info mislabeling | no (code TBD) | n/a | **ADVISORY** |
| N4 supply chain | no (extras TBD) | n/a | **ADVISORY** |
| N5 sanitizer TOCTOU | no (code TBD) | n/a | **ADVISORY** |

**No BLOCKER**. Two real findings on existing code (TM-E1, TM-E2). Five
ADVISORY-level design constraints for the upcoming MPS work.
