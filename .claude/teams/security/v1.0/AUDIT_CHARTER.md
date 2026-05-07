# AUDIT CHARTER — gpucheck v1.0 / MPS-surface threat model + baseline

## Scope
1. **NEW SURFACES (threat-model only — code does not yet exist)** for the
   upcoming MPS backend:
   - Subprocess invocation of `xcrun metal` and `xcrun metallib` (Metal shader
     compilation).
   - GPU memory tracker reading `task_info` via mach syscalls.
   - New `[mps]` and `[apple]` extras in `pyproject.toml` pulling Apple-only
     deps.
   - New sanitizer that intercepts MPS dispatch (TOCTOU, dispatch reordering).
2. **BASELINE SCANS** on the existing release branch (release/v1.0,
   commit a9a9d44):
   - `src/gpucheck/` and `pyproject.toml` against the OWASP Top 10 (2025).
   - Secrets sweep across the full tree + git history.
   - Dependency / supply-chain audit on `pyproject.toml` pinned versions.
   - Config audit of `.yml` / `.toml` / `.json` files.

## Detected stack
- Language: Python (>=3.10), pure (no compiled native sources in repo).
- Build: hatchling.
- Package manager: pip / uv.
- Frameworks (optional extras): pytest>=7, rich>=13, numpy>=1.24,
  torch>=2.0, cupy-cuda12x>=13.0, triton>=3.0, hypothesis>=6.0.
- CI: GitHub Actions (`.github/workflows/ci.yml`).
- Project license: Apache-2.0.

## Tool availability
- Available: `git`, `grep`, `find`, `uv` (0.11.7).
- **Unavailable**: `gitleaks`, `trufflehog`, `semgrep`, `bandit`,
  `pip-audit`, `safety`, `checkov`, `trivy`, `hadolint`, `pip` itself
  (uv-only host).
- Specialists fall back to LLM + Grep pattern scanning. SECURITY_REPORT
  notes this in the metadata.

## Tier
**Full audit** — new attack surface (MPS) is being introduced, and a
baseline of the entire repo is requested. All 8 domain specialists are
dispatched. License-auditor included because Apple-only deps in `[mps]`
and `[apple]` extras may pull in non-OSI / proprietary code (e.g.,
`pyobjc-framework-Metal` is BSD, but supply-chain checks need explicit
sign-off).

## Specialists dispatched
- security-planner
- security-threat-modeler  (MPS surfaces 1–4 + STRIDE)
- security-architecture-reviewer
- security-owasp-scanner
- security-secrets-hunter  (full tree + git history)
- security-dependency-auditor
- security-license-auditor
- security-config-scanner
- security-crypto-reviewer

## Gates
- Round 2: security-skeptic
- Round 3: security-evaluator (5-dim rubric, PASS/FAIL)

## Constraints
- Hard deadline 60 min.
- BLOCKER findings pause Phase 2; must include exploitation evidence.
- Every finding cites `file:line`.
- Read-only audit.

## Important context
The MPS backend code is **not yet present** in the repository. The
threat-modeler treats surfaces 1–4 as *design-stage* threats and
emits ADVISORY-level guidance with concrete remediation patterns the
implementer must follow. The OWASP/secrets/deps/config specialists scan
only the existing tree.
