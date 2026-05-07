# THREAT MODEL — gpucheck v1.0 + MPS backend (release/v1.0 @ a9a9d44)

Framework: **STRIDE** (gpucheck is a traditional Python library, not
agentic AI). Out of scope: ASTRIDE / MITRE ATLAS.

## 1. System overview
gpucheck is a pytest plugin distributed via PyPI, providing GPU-kernel
test assertions, decorators, fixtures, fuzzing, sanitisers, and
reporters. Today it targets NVIDIA via `pynvml` + `torch.cuda` +
`compute-sanitizer`. The v1.0 release adds an Apple Silicon (MPS)
backend with: (a) Metal shader compilation via `xcrun metal/metallib`,
(b) memory accounting via mach `task_info`, (c) new `[mps]` and
`[apple]` extras, (d) an MPS-dispatch sanitiser hook.

## 2. Actors
| Actor | Trust | Channel |
|---|---|---|
| End user (test author) | Self-trusted (runs their own pytest) | Imports gpucheck APIs from test code |
| CI runner / GitHub Actions | Configurable trust | `pip install -e ".[dev]"` + `pytest` |
| PyPI mirror | External, mutable | Resolves transitive deps at install time |
| Apple toolchain (`xcrun`, `metal`, `metallib`) | Trusted (host-installed, signed) | subprocess argv |
| NVIDIA toolchain (`compute-sanitizer`) | Trusted | subprocess argv |
| Compromised transitive dep | Hostile | post-install hook, runtime import |
| Malicious test corpus | Hostile | strings/bytes passed to gpucheck APIs |

## 3. Data assets
| Asset | Sensitivity | Where |
|---|---|---|
| User shader source (Metal / CUDA) | LOW (shipped by test author) | `compile_metal(...)` parameter |
| Tensor data | LOW | in-memory, never written |
| Benchmark / regression baselines (JSON) | LOW (integrity matters, not confidentiality) | disk, `analysis/regression.py`, `reporting/json.py` |
| `GITHUB_TOKEN` in CI | MEDIUM | env var inside Actions runner |
| Apple developer signing identity (future) | HIGH | Keychain on dev machine; should never appear in gpucheck |
| `LD_PRELOAD` / `DYLD_INSERT_LIBRARIES` env | MEDIUM | inherited by subprocesses |

## 4. Trust boundaries

```
+-----------------------+    PyPI install       +-------------------------+
| End user laptop / CI  |---------------------->| gpucheck wheel + deps   |
+-----------+-----------+                       +------------+------------+
            |                                                 |
            | pytest collects                                  | imports
            v                                                 v
   +------------------------------+        +-----------------------------+
   | user's test code (untrusted) |------->| gpucheck (plugin + APIs)   |
   +------------+-----------------+        +--------------+--------------+
                                                          |
                       +----------------------------------+--------------+
                       | sanitizers (subprocess boundary)                 |
                       |   race.py  -> compute-sanitizer (CUDA)          |
                       |   mps.py   -> xcrun metal / metallib (NEW)      |
                       +-----------+---------------+----------------------+
                                   |               |
                            +------v----+    +-----v-----+
                            | NVIDIA tk |    | Apple tk  |
                            +-----------+    +-----------+
```

Boundary 1: **PyPI install**. Threat = supply chain (DEP-1, DEP-5).
Boundary 2: **User test code → gpucheck APIs**. Threat = malicious
input to `compile_metal`, `extra_args`, JSON baselines (TM-N1, AR-2,
OWASP-A03-1).
Boundary 3: **gpucheck → external tool subprocess**. Threat = path
injection (TM-E1), env-leak (AR-3), command-bomb DoS (TM-N1).
Boundary 4: **CI runner privileges**. Threat = `GITHUB_TOKEN` write
(CFG-2).

## 5. STRIDE matrix per surface

### Existing surfaces

| Surface | S | T | R | I | D | E |
|---|---|---|---|---|---|---|
| race.py subprocess | TM-E1 (env-var path swap) | OWASP-A03-1 (`extra_args`) | — | TM-E2 (sys.path leak in /tmp) | — | TM-E1 (escalates if env-write attacker had no shell) |
| JSON baselines | — | AR-2 (no schema/integrity) | — | — | — | — |
| GitHub Actions workflow | — | CFG-1 (mutable tag) | — | — | — | CFG-2 (write-all default) |

### NEW MPS surfaces (design-stage)

| Surface | S | T | R | I | D | E |
|---|---|---|---|---|---|---|
| N1 `xcrun metal` | path-swap on `xcrun` resolution | hostile shader source RCE / `#include` traversal | — | — | macro/header bomb | shader RCE → CI compromise |
| N2 `xcrun metallib` | inherits N1 | output-path traversal if path is caller-supplied | — | — | — | — |
| N3 mach `task_info` | — | — | — | mislabelled "GPU memory" leaks process-wide RSS | — | — |
| N4 `[mps]`/`[apple]` extras | typosquat; dep-confusion | post-install code in pyobjc | — | — | — | — |
| N5 MPS dispatch hook | — | TOCTOU between record-args and dispatch | — | — | — | sanitiser bypass via dispatch reorder |

## 6. Top-3 surface risks (by exploitability × impact)

1. **N1 — `xcrun metal` shader RCE** (ADVISORY).
   Real but design-stage. The recommended skeleton in
   `EVIDENCE/threat-modeler.md` blocks `#include`, `#pragma clang
   load_plugin`, `__attribute__((constructor))`, hard-caps source size
   to 1 MiB, allowlists flags, and pins `xcrun` to
   `/usr/bin/` or `/Applications/Xcode.app/`. Implementer must adopt
   that pattern verbatim or document deviations.

2. **CFG-2 — `permissions:` block missing on GitHub Actions**
   (MEDIUM, current code).
   `GITHUB_TOKEN` defaults to write-all on classic repos. Combined
   with CFG-4 / DEP-1 (CI installs unlocked deps fresh from PyPI on
   every run), a single compromised dev-time transitive dep can push
   to the repo. Single-line fix: `permissions: contents: read`.

3. **TM-E1 — Path injection via `CUDA_HOME` env var**
   (MEDIUM, current code).
   `_find_compute_sanitizer` (race.py:50–62) trusts any path under
   `CUDA_HOME` / `CUDA_PATH` provided it has `+x`. An attacker with
   env-write executes arbitrary code under the test runner. The MPS
   work must not replicate this pattern for `xcrun` (advisory N1
   covers this).

## 7. Mitigations summary

| ID | Severity | Owner | Status |
|---|---|---|---|
| TM-E1 | MEDIUM | sanitizers/race.py | open |
| TM-E2 | LOW | sanitizers/race.py | open |
| AR-1 (factor helper) | LOW (advisory) | sanitizers/_subprocess.py (new) | open |
| AR-2 (JSON schema) | LOW | analysis/regression.py + reporting/json.py | open |
| AR-3 (env passthrough) | LOW | sanitizers/race.py | open |
| OWASP-A03-1 | LOW | sanitizers/race.py | open |
| OWASP-A05-1 | LOW | arch/tensor_cores.py | open |
| CFG-1 (action SHA pin) | LOW | .github/workflows/ci.yml | open |
| CFG-2 (permissions) | MEDIUM | .github/workflows/ci.yml | open |
| CFG-4 (locked install) | LOW | .github/workflows/ci.yml | open |
| CFG-5 (.gitignore defenses) | LOW | .gitignore | open |
| DEP-1 (lock file) | MEDIUM | repo root | open |
| DEP-2..4 (floor bumps) | LOW | pyproject.toml | open |
| LIC-1 (license-files PEP 639) | LOW | pyproject.toml | open |
| N1..N5 | ADVISORY | mps backend (TBD) | not yet implemented |
