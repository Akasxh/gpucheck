# owasp-scanner — OWASP Top 10 (2025)

## Phase 1 — Tool
SAST tools (semgrep, bandit) **not installed**. Falling back to
LLM + grep pattern scan.

## Phase 2 — Per-category review

### A01 Broken Access Control
**N/A** — gpucheck is a library / pytest plugin. No HTTP routes, no
auth, no sessions. No findings.

### A02 Cryptographic Failures
Delegated to crypto-reviewer. See `crypto-reviewer.md`.

### A03 Injection
Grep evidence:
```
src/gpucheck/sanitizers/race.py:8:import subprocess
src/gpucheck/sanitizers/race.py:183:    result = subprocess.run(
```
Reviewed `subprocess.run` call manually:
- Argv is a `list`, not a single string — no shell metacharacters
  involved. ✓
- `shell=True` is **not** used anywhere in `src/`. ✓
- Wrapper-script content is built from `mod_name` + `fn_name` which
  are validated by `isidentifier()` (race.py:117-120). ✓
- `extra_args` (race.py:135, 179-180) is **not validated**: any list
  the user passes is forwarded straight into the argv. This is
  technically command-arg-injection (the user can pass
  `--export-name=$(rm -rf …)` or other compute-sanitizer flags that
  alter behaviour). The **caller is the test author**, so the threat
  model is limited — a test author who can pass `extra_args` can also
  just `os.system("…")` directly. Marking as **LOW** (defence-in-depth).

#### OWASP-A03-1 — Unvalidated `extra_args` forwarded to subprocess
- **Severity**: LOW
- **Category**: A03 Injection (command-arg)
- **Location**: `src/gpucheck/sanitizers/race.py:135,179-180`
- **Exploitability**: Local, Authenticated (test author).
- **Blast radius**: same as test process; not a privilege boundary.
- **Confidence**: MEDIUM
- **Description**: `extra_args: list[str] | None` is appended to the
  compute-sanitizer command line without an allowlist. A test author
  can pass flags that change sanitizer behaviour (e.g.,
  `--save report.xml` writing to an attacker-chosen path).
- **Remediation**:
```python
# VULNERABLE
if extra_args:
    cmd.extend(extra_args)
# FIXED
_ALLOWED_FLAGS = frozenset({
    "--check-api-memory-access", "--leak-check", "--track-stream-ordered-races",
    "--print-level", "--launch-skip", "--launch-count",
})
for arg in extra_args or ():
    head = arg.split("=", 1)[0]
    if head not in _ALLOWED_FLAGS:
        raise ValueError(f"sanitizer flag not allowlisted: {arg!r}")
    cmd.append(arg)
```
- **Verification notes**: Ran grep for `subprocess`, `os.system`,
  `popen`, `os.exec*` across `src/` — only the one site exists. No
  other injection vector in the current tree.

### A04 Insecure Design
- gpucheck has no rate limiting / lockout / CAPTCHA, but those are
  not applicable to a pytest plugin.
- No `pickle` / `yaml.load` (verified via grep, returns nothing). ✓
- `json.loads` on baseline files (regression.py:333, 350; json.py:102-103)
  is safe by itself but lacks schema validation — see AR-2.

### A05 Security Misconfiguration
- Plugin entry-point `gpucheck.plugin` registers via
  `pyproject.toml:49-50`. Standard pytest11 mechanism. ✓
- No debug-mode flags wired to env vars in production paths.
- `os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")` at
  `reporting/console.py:83` — informational only (controls Rich color
  detection). ✓

#### OWASP-A05-1 — `NVIDIA_TF32_OVERRIDE` env-var read without bounds check
- **Severity**: LOW
- **Category**: A05 Security Misconfiguration
- **Location**: `src/gpucheck/arch/tensor_cores.py:183`
- **Description**: Value is parsed as int (`int(tf32_override)`) and
  used to decide a compute path. An attacker who controls env can
  silently force a less-precise path, hiding a real numerical
  regression. Not a security vulnerability per se — a *correctness*
  side channel. Documenting only.
- **Remediation**: warn loudly when the env var is set; emit one log
  line per session.
- **Confidence**: HIGH

### A06 Vulnerable Components
Delegated to dependency-auditor. See `dependency-auditor.md`.

### A07 Identification & Authentication Failures
**N/A** — no auth in the library.

### A08 Software & Data Integrity
- `pyproject.toml` build backend = `hatchling`. No `setup.py`, no
  post-install scripts in the repo.
- CI uses `actions/checkout@v4` and `actions/setup-python@v5`. Both
  pinned to **major** tag, not SHA. **Pin to SHA** for supply-chain
  hygiene.
- JSON baselines: see AR-2 — accepted without integrity verification.

#### OWASP-A08-1 — GitHub Actions pinned to mutable tags
- **Severity**: LOW
- **Category**: A08 Software & Data Integrity
- **Location**: `.github/workflows/ci.yml:13,15,32,34`
- **Description**: `actions/checkout@v4` and `actions/setup-python@v5`
  are tags, not immutable SHAs. A compromise of those repos (or a
  forced re-tag) silently runs new code in CI. Industry best practice
  (e.g., GitHub Security Lab) is to pin third-party actions to a
  full SHA.
- **Confidence**: HIGH
- **Remediation**:
```yaml
# VULNERABLE
- uses: actions/checkout@v4
- uses: actions/setup-python@v5
# FIXED (use current SHAs at audit time)
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11   # v4.1.1
- uses: actions/setup-python@0a5c61591373683505ea898e09a3ea4f39ef2b9c # v5.0.0
```
- **Verification notes**: `actions/checkout` and `actions/setup-python`
  are first-party actions (low risk), but the rule applies generally;
  this is the pattern to enforce when third-party actions arrive.

### A09 Security Logging & Monitoring Failures
- No security-relevant events to log (library, no auth events).
- `logging.getLogger(__name__)` used in `arch/detection.py:11`. ✓
- No PII leaked to logs (verified by reading reporting/* files).

### A10 Server-Side Request Forgery
**N/A** — no outbound HTTP in the library.

## Phase 3 — verification

| Finding | False positive risk | Decision |
|---|---|---|
| OWASP-A03-1 (`extra_args`) | LOW (real, scope-limited) | **keep** — LOW |
| OWASP-A05-1 (`NVIDIA_TF32_OVERRIDE`) | MED (debatable severity) | **keep** — LOW |
| OWASP-A08-1 (Actions tag pinning) | LOW | **keep** — LOW |

No HIGH or CRITICAL OWASP findings. The repo's narrow scope (library,
no network, no auth) eliminates most of the Top 10 by construction.
