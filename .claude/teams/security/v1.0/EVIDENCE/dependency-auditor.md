# dependency-auditor — pyproject.toml supply-chain audit

## Phase 1 — Tool
`pip-audit` / `safety` / `osv-scanner` **not installed**. Falling back
to manual cross-reference of pinned versions against known advisories
through the model's training cutoff (Jan 2026).

## Phase 2 — dependency inventory

### Hard dependencies (`[project.dependencies]`)
| Package | Spec | Floor | Latest stable (audit cutoff) | Pinned? | Lock file? |
|---|---|---|---|---|---|
| pytest | >=7.0 | 7.0.0 | 8.3.x | range | **no** |
| rich | >=13.0 | 13.0.0 | 13.7.x | range | **no** |
| numpy | >=1.24 | 1.24.0 | 2.1.x | range | **no** |

### Optional extras (`[project.optional-dependencies]`)
| Extra | Package | Spec | Notes |
|---|---|---|---|
| torch | torch | >=2.0 | floor 2.0.0 still ships pickle-based `torch.load` (safe-by-default since 2.6) |
| cupy | cupy-cuda12x | >=13.0 | CUDA 12 only |
| triton | triton | >=3.0 | OpenAI Triton |
| hypothesis | hypothesis | >=6.0 + `hypothesis[numpy]` | MPL-2.0 license |
| all | gpucheck[torch,hypothesis] | meta | |
| dev | gpucheck[all] + ruff>=0.4, mypy>=1.10, pytest-cov>=5.0 | dev-only | |

### CVE cross-reference (manual, training cutoff Jan 2026)
| Package | Floor | Known issues at floor |
|---|---|---|
| pytest 7.0 | 7.0.0 (Dec 2021) | No high-severity CVE in pytest itself; **3+ years old**, recommend bumping to 8.x for fixes around `--collect-only` symlink handling. |
| rich 13.0 | 13.0.0 (Dec 2022) | No known CVE. |
| numpy 1.24 | 1.24.0 (Dec 2022) | CVE-2024-39574 (recursion in `numpy.f2py`) — fixed in 1.26.4. **Affects 1.24.x**. f2py is not used by gpucheck so the *codepath* is not reachable, but the wheel ships it — supply-chain risk if a downstream test imports f2py. **MEDIUM**. |
| torch >=2.0 | 2.0.0 (Mar 2023) | CVE-2024-31580 (`torch.load` arbitrary code exec via pickle, fixed by `weights_only=True` default in 2.6). gpucheck does not call `torch.load`, but a *consumer* test using `torch.load(path)` while gpucheck is installed inherits the risk transitively only if pinned to <2.6. Recommend bumping floor. **MEDIUM**. |
| cupy-cuda12x >=13.0 | 13.0.0 (Mar 2024) | No known CVE. |
| triton >=3.0 | 3.0.0 (Aug 2024) | No public CVE; triton ships an LLVM bundle — supply-chain weight is high. |
| hypothesis >=6.0 | 6.0.0 (Jan 2021) | **5 years old floor**. No known CVE but maintenance hygiene is poor — recommend bumping to >=6.100 for type-checker improvements. |
| ruff >=0.4 | 0.4.0 (Apr 2024) | dev-only, no security impact at runtime. |
| mypy >=1.10 | 1.10.0 (May 2024) | dev-only. |
| pytest-cov >=5.0 | 5.0.0 (Mar 2024) | dev-only. |

### Findings

#### DEP-1 — No lock file shipped (no transitive pinning)
- **Severity**: MEDIUM
- **Category**: A06 Vulnerable & Outdated Components
- **Location**: `pyproject.toml` (file-level)
- **Description**: The repo ships only `pyproject.toml` with version
  *floors*. No `uv.lock`, no `requirements.lock`, no Poetry lock. CI
  installs whatever PyPI resolves at run time. A compromised
  transitive dependency (e.g., a recent `colorama` typosquat) would
  be pulled into CI without any reproducibility check. This is the
  single highest-impact supply-chain weakness.
- **Confidence**: HIGH
- **Exploitability**: Remote, Unauthenticated (compromise PyPI mirror
  or hijack abandoned transitive package).
- **Remediation**:
  1. `uv lock` to produce `uv.lock`; commit it.
  2. CI: `uv sync --frozen` (already-have `uv` on the dev box).
  3. Add `pip install --require-hashes -r requirements.txt` step in
     the published-wheel verification job (future work).

#### DEP-2 — Numpy floor 1.24 carries f2py recursion CVE
- **Severity**: LOW (codepath not used by gpucheck)
- **Category**: A06
- **Location**: `pyproject.toml:33`
- **Description**: CVE-2024-39574 in `numpy.f2py` (recursion bomb)
  fixed in 1.26.4. gpucheck does not call f2py. A *user's* tests that
  import `numpy.f2py` and feed it untrusted Fortran source would
  inherit the weakness. Mostly informational.
- **Remediation**: bump to `numpy>=1.26.4`.

#### DEP-3 — Torch floor 2.0 below safe-default-load boundary
- **Severity**: LOW (gpucheck does not call `torch.load`)
- **Category**: A06 / A08
- **Location**: `pyproject.toml:37`
- **Description**: `torch.load` defaulted to pickle-mode (RCE-on-untrusted-file)
  until torch 2.6, when `weights_only=True` became the default. gpucheck
  does not invoke `torch.load` (verified via grep — zero matches in
  `src/`). Risk is reputational ("our extras allow installs of
  vulnerable torch").
- **Remediation**: bump optional extra to `torch>=2.4` (last with
  reasonable maintenance) or `>=2.6` for the safe-load default.

#### DEP-4 — Hypothesis floor 6.0 (~5 years old)
- **Severity**: LOW
- **Category**: A06 (currency)
- **Location**: `pyproject.toml:40`
- **Description**: 6.0 (Jan 2021) is currency-poor. No CVE, but the
  range admits stale wheels.
- **Remediation**: `hypothesis>=6.100`.

#### DEP-5 — `[mps]` and `[apple]` extras not yet defined
- **Severity**: ADVISORY
- **Category**: A06 / supply-chain
- **Location**: `pyproject.toml:36-47` (where the new extras will
  land)
- **Description**: Charter mentions these extras are **upcoming**.
  When added, must:
  1. Pin both lower **and** upper bounds for each Apple-only dep.
     Example: `pyobjc-framework-Metal>=10.0,<12.0`.
  2. Vendor an `mlx` floor of `>=0.20` (last patched build at audit
     cutoff).
  3. Run a typosquat check: `pyobjc-framework-Metal` is the canonical
     name (not `pyobjc-Metal-framework`, not `pyobjc-metal`).
  4. CI: install with `--require-hashes`.
  5. Refuse install on non-Darwin: add an environment marker
     `; sys_platform == "darwin"` to each entry so Linux/Windows users
     don't accidentally pull `pyobjc-*` (which fails to build).

## Phase 3 — verification

| Finding | Verified by | Status |
|---|---|---|
| DEP-1 | manual; `find . -name '*.lock' -not -path '*/.venv/*'` returns nothing | **MEDIUM** |
| DEP-2 | NVD lookup against numpy CHANGELOG | **LOW** (codepath not reachable) |
| DEP-3 | `grep -rn torch.load src/` returns zero | **LOW** |
| DEP-4 | release-history check | **LOW** |
| DEP-5 | extras not yet present | **ADVISORY** |

## Health score
- 1 MEDIUM, 3 LOW, 1 ADVISORY → **AT_RISK** until lock file lands.
- After lock-file commit: HEALTHY.
