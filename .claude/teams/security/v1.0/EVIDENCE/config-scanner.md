# config-scanner — config files audit

## Phase 1 — Tool
checkov / trivy / hadolint **not installed**. Manual review.

## Phase 2 — config inventory

### Files surveyed
- `pyproject.toml` (project + tool configs)
- `.github/workflows/ci.yml` (GitHub Actions)
- `.gitignore`
- `CLAUDE.md` (instructions, not config)
- `LICENSE`
- No Dockerfiles, no `docker-compose.*`, no Terraform/CloudFormation/
  Pulumi, no Kubernetes manifests, no `.env*`.

### CI/CD audit — `.github/workflows/ci.yml`

```yaml
on:
  push:    branches: [main]
  pull_request:  branches: [main]
```

#### CFG-1 — Actions pinned to mutable major tags
- **Severity**: LOW (first-party actions; rule generalises to future
  third-party actions)
- **Category**: A08 Software & Data Integrity
- **Location**: `.github/workflows/ci.yml:13,15,32,34`
- **Description**: `actions/checkout@v4` and `actions/setup-python@v5`.
  Both are GitHub-owned but the **policy** of pinning to SHA is what
  protects the future state when a third-party action is added.
- **Confidence**: HIGH
- **Remediation**:
```yaml
- uses: actions/checkout@b4ffde65f46336ab88eb53be808477a3936bae11   # v4.1.1
- uses: actions/setup-python@0a5c61591373683505ea898e09a3ea4f39ef2b9c # v5.0.0
```

#### CFG-2 — `permissions:` block missing → defaults to `write-all`
- **Severity**: MEDIUM
- **Category**: A05 Security Misconfiguration
- **Location**: `.github/workflows/ci.yml:1-39` (file-wide)
- **Description**: With no top-level `permissions:` block and no
  per-job override, `GITHUB_TOKEN` is granted the *legacy default*
  permission set, which on classic repos means write access to
  contents, packages, pull requests, etc. Best practice (GitHub
  Security Lab, OpenSSF Scorecard) is **explicit minimum**.
- **Confidence**: HIGH
- **Exploitability**: Remote (any compromised dep in `[dev]` running
  during `pip install -e ".[dev]"` could exfiltrate or push using
  `$GITHUB_TOKEN`).
- **Blast radius**: arbitrary writes to the repo from a
  pull_request-triggered job.
- **Remediation**:
```yaml
# At top of .github/workflows/ci.yml, right after `on:`:
permissions:
  contents: read
```

#### CFG-3 — `pull_request` (not `pull_request_target`) — good
- gpucheck uses `pull_request:` (line 6), which runs on the **fork's**
  code in an unprivileged context. ✓ No action required.

#### CFG-4 — `pip install -e ".[dev]"` runs untrusted setup hooks
- **Severity**: LOW (defence-in-depth; hatchling has no setup hook,
  but transitive deps may)
- **Category**: A06 Vulnerable Components
- **Location**: `.github/workflows/ci.yml:18,36`
- **Description**: `pip install -e ".[dev]"` resolves transitive deps
  *fresh from PyPI* on every CI run because no lock file is
  committed (see DEP-1). Any of those transitive deps can run code at
  install time. With CFG-2 unfixed, that code runs with write access
  to the repo.
- **Confidence**: MEDIUM
- **Remediation**: combine with DEP-1 (commit `uv.lock`) and run
  `uv sync --frozen` instead.

### `.gitignore` audit
Contents (verified earlier):
```
__pycache__/   *.py[cod]   *$py.class   *.so   dist/   build/
*.egg-info/   *.egg   .pytest_cache/   .hypothesis/   .mypy_cache/
.ruff_cache/   .coverage   htmlcov/   *.log   .env   .venv/   venv/
.tox/   .nox/   *.ncu-rep   *.nsys-rep
```

- `.env` ignored ✓
- `.venv/`, `venv/` ignored ✓
- `*.log` ignored ✓
- `*.ncu-rep`, `*.nsys-rep` ignored — Nsight profile dumps may contain
  CUDA kernel source paths but no secrets ✓
- **Missing**: `*.pem`, `*.key`, `id_rsa`, `id_ed25519`, `.npmrc`,
  `.pypirc`, `.netrc`. These are unlikely to appear in this repo,
  but a defensive default block is cheap.

#### CFG-5 — `.gitignore` lacks key-file defensive entries
- **Severity**: LOW (informational)
- **Location**: `.gitignore:1-21`
- **Remediation**:
```
# Append:
*.pem
*.key
id_rsa
id_ed25519
.npmrc
.pypirc
.netrc
```

### `pyproject.toml` audit
- `requires-python = ">=3.10"` — current.
- `[tool.pytest.ini_options].addopts = "--ignore=tests/gpu_integration"`
  — fine.
- `[tool.ruff].extend-exclude = ["tests/gpu_integration"]` — fine.
- `[tool.mypy].strict = true` — good.
- `[tool.mypy.overrides]` whitelists missing imports for
  `pynvml/torch/triton/cupy/hypothesis`. Acceptable; these are
  optional deps.
- **No** `tool.uv` / `tool.poetry` lock-policy declared.

### MPS-related preconditions
When the implementer adds `[mps]` and `[apple]` extras, the
following must also land in `ci.yml`:
- A separate `mps-test` job gated on `runs-on: macos-14`
  (Apple Silicon).
- That job must inherit the same `permissions: contents: read` block.
- Use `XCODE_VERSION:` matrix or `xcode-select -p` check to fail-fast
  on missing dev tools rather than silent fallback.

## Phase 3 — verification

| Finding | Verified | Severity |
|---|---|---|
| CFG-1 mutable action tags | yes | LOW |
| CFG-2 missing permissions block | yes (re-read ci.yml) | **MEDIUM** |
| CFG-3 PR vs PR-target | yes | n/a (correct already) |
| CFG-4 unlocked deps in CI | yes | LOW |
| CFG-5 .gitignore defensive entries | yes | LOW |

## Output verdict
1 MEDIUM, 3 LOW. The MEDIUM is `permissions:` block missing from
the workflow.
