# planner — dispatch calibration

## Inputs read
- AUDIT_CHARTER.md (full audit, MPS pre-implementation + baseline).
- pyproject.toml (3 hard deps, 4 optional extras, hatchling build).
- src/gpucheck/ tree (~5.4 kLOC Python, no compiled sources).
- .github/workflows/ci.yml (GitHub Actions, lint + test, no GPU).
- LICENSE (Apache-2.0).

## Tier confirmation
**Full audit** is correct. Triggering signals:
- New attack surface (MPS subprocess, mach syscall, sanitizer hook) →
  `threat-modeler` mandatory.
- New optional extras `[mps]` / `[apple]` → `dependency-auditor` and
  `license-auditor` mandatory (Apple-only deps may carry non-OSI
  licenses or supply-chain risk).
- Pre-existing sanitizers/race.py already shells out to
  `compute-sanitizer` → owasp-scanner / config-scanner mandatory to
  validate the existing pattern before MPS clones it.
- Repo is small (~5.4 kLOC) — full audit fits inside the 60-min deadline.

## Specialist dispatch plan

| Specialist | Tool fallback | Focus targets |
|---|---|---|
| threat-modeler | grep + manual | The 4 MPS surfaces + existing subprocess in `sanitizers/race.py` |
| architecture-reviewer | manual | trust boundaries: Python user code → subprocess → driver/kernel |
| owasp-scanner | grep | A03 (command/path injection in race.py), A05 (env var trust), A08 (CI/CD), A10 (n/a) |
| secrets-hunter | grep + `git log -p -S` | `.env`, README, examples/, full git history |
| dependency-auditor | manual cross-ref | every line of `[project.dependencies]` and `[project.optional-dependencies]` |
| license-auditor | manual | Apache-2.0 vs each pinned dep; flag GPL/AGPL/SSPL/ELv2 |
| config-scanner | manual | `.github/workflows/ci.yml`, `pyproject.toml`, `.gitignore` |
| crypto-reviewer | grep | known-weak primitives, RNG seeds, JSON-baseline integrity |

## Stack-specific guidance per specialist

### owasp-scanner
- Python: focus on `subprocess.run` argv construction (A03), `os.environ`
  trust (A05), `pickle`/`yaml.load` (none found, confirm), JSON parsing
  with untrusted input (A08).
- The plugin entry point `gpucheck.plugin` runs inside any pytest
  session — assume the *test runner* is not the threat actor; the
  threat actor is *test inputs* and *baseline JSON files* shipped with
  CI artifacts.

### secrets-hunter
- Run grep for AKIA / sk-ant / ghp_ / private key markers in current
  tree.
- `git log -p --all -S` for the same patterns. Repo is 5 commits deep,
  so history scan is cheap.
- Pay attention to `examples/` and `README.md` — common location for
  copy-pasted demo keys.

### dependency-auditor
- Hard pins: pytest>=7, rich>=13, numpy>=1.24 — all reputable, all use
  range pins (no upper bound). Check for known CVEs on those minimum
  versions.
- Optional: torch>=2.0 (CVE-2024-31580 PyTorch RCE in `torch.load` —
  applicable?), cupy-cuda12x>=13.0, triton>=3.0, hypothesis>=6.0.
- **No lock file present** (no `requirements.txt`, no `uv.lock`).
  Flag this as a supply-chain concern.

### license-auditor
- Project: Apache-2.0. Check each dep's license:
  - pytest = MIT, rich = MIT, numpy = BSD-3-Clause → all compatible.
  - torch = BSD-3-Clause, cupy = MIT, triton = MIT,
    hypothesis = MPL-2.0 → MPL-2.0 compatible with Apache-2.0 for
    aggregate distribution but obligations differ.
  - Future `[mps]` deps (pyobjc-framework-Metal) = MIT — compatible.
  - **Future `[apple]` MLX**: Apple's mlx is MIT (verify pre-merge).

### config-scanner
- ci.yml uses `actions/checkout@v4`, `actions/setup-python@v5`. Check
  whether these are pinned to a SHA or to a moving major tag.
- No Dockerfile, no IaC.
- `.gitignore` covers `.env`, `.venv`, `*.log` — looks correct.

### threat-modeler (the heart of this audit)
For each of the 4 MPS surfaces, write:
1. attack vector
2. preconditions
3. blast radius
4. existing/recommended mitigations
5. concrete remediation pattern in Python
Use STRIDE (traditional codebase, not agentic AI).

## Calibration checks
- If a finding cannot cite `file:line`, downgrade severity (the code
  doesn't exist yet for MPS surfaces — those are ADVISORY by definition,
  never BLOCKER).
- BLOCKER reserved for *exploitable existing code*. The bar is: a PoC
  in 10 lines or fewer, no auth bypass needed.

## Time budget
- 60 min hard. Specialists run in one big parallel write (this turn).
- Skeptic + evaluator each ~5 min.
- Synthesis 10 min.

## Recommendation
Proceed as planned. No specialist re-dispatch needed up front.
