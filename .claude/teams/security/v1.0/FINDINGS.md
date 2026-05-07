# FINDINGS — gpucheck v1.0 Security Audit

**Verdict:** ADVISORY (Phase 2 cleared to proceed; no BLOCKER)
**Counts:** 0 CRITICAL · 0 HIGH · 3 MEDIUM · 13 LOW · 5 MPS design-stage
**Evaluator:** PASS (0.97 overall, no dimension < 0.50)
**Source:** `THREAT_MODEL.md` §7 + `EVIDENCE/skeptic.md` (severity ledger + verdict computation). This file mirrors the inline ledger that security-lead returned because the subagent harness blocked direct write of FINDINGS.md from within the dispatched persona.

## Per-finding ledger — existing code

| ID | Sev | File:line | Description | Verdict |
|---|---|---|---|---|
| TM-E1 | MEDIUM | `src/gpucheck/sanitizers/race.py:56-62` | Path injection via `CUDA_HOME`/`CUDA_PATH` env var; only `isfile + X_OK` checked, no prefix allowlist | ADVISORY |
| TM-E2 | LOW | `src/gpucheck/sanitizers/race.py:122-126` | `sys.path` snapshot leaked into world-traversable tmp file | ADVISORY |
| AR-1 | LOW | `src/gpucheck/sanitizers/race.py` (file) | No `_run_external_tool` helper consolidating subprocess invocation | ADVISORY |
| AR-2 | LOW | `src/gpucheck/analysis/regression.py:333,350`; `src/gpucheck/reporting/json.py:102-103` | JSON baselines lack schema validation / integrity checks | ADVISORY |
| AR-3 | LOW | `src/gpucheck/sanitizers/race.py:183-188` | Subprocess inherits full parent env (no env scrubbing) | ADVISORY |
| OWASP-A03-1 | LOW | `src/gpucheck/sanitizers/race.py:135,179-180` | Unvalidated `extra_args` passthrough to subprocess | ADVISORY |
| OWASP-A05-1 | LOW | `src/gpucheck/arch/tensor_cores.py:183` | `NVIDIA_TF32_OVERRIDE` env trust without validation | ADVISORY |
| CFG-1 | LOW | `.github/workflows/ci.yml:13,15,32,34` | GitHub Actions pinned to mutable tags (use SHA pins) | ADVISORY |
| **CFG-2** | **MEDIUM** | `.github/workflows/ci.yml:1-39` | **`permissions:` block missing** — `GITHUB_TOKEN` defaults to write-all | **ADVISORY** |
| CFG-4 | LOW | `.github/workflows/ci.yml:18,36` | CI installs unlocked deps (`pip install -e ".[dev]"`) | ADVISORY |
| CFG-5 | LOW | `.gitignore:1-21` | Lacks defensive entries (e.g. `.env`, `*.pem`, `dist/`, `*.egg-info/`) | ADVISORY |
| **DEP-1** | **MEDIUM** | `pyproject.toml` | **No lock file** — reproducibility + supply-chain attack surface | **ADVISORY** |
| DEP-2 | LOW | `pyproject.toml:33` | numpy floor 1.24 (f2py CVE applies) | ADVISORY |
| DEP-3 | LOW | `pyproject.toml:37` | torch floor 2.0 — predates safe-load default | ADVISORY |
| DEP-4 | LOW | `pyproject.toml:40` | hypothesis floor 6.0 (~5 yr stale) | ADVISORY |
| LIC-1 | LOW | `pyproject.toml:11` | PEP 639 `license-files` missing | ADVISORY |

## Per-finding ledger — MPS design-stage (no code yet)

| ID | Sev | Surface | Description | Verdict |
|---|---|---|---|---|
| **N1** | design | `xcrun metal` subprocess | Hostile shader triggers preprocessor traversal (`#include`, `#pragma clang load_plugin`, `__attribute__((constructor))`) or compiler-bomb DoS. Skeleton: forbid-token list + 1 MiB cap + flag allowlist + `xcrun` path pinned to `/usr/bin/` or `/Applications/Xcode.app/`. | ADVISORY |
| N2 | design | `xcrun metallib` subprocess | Path traversal via output filename | ADVISORY |
| N3 | design | mach `task_info` reads | Mislabel of unrelated `task_info` fields as "GPU memory" | ADVISORY |
| N4 | design | `[mps]` / `[apple]` extras | Supply chain (typosquat, dep confusion) on Apple-only deps | ADVISORY |
| N5 | design | MPS dispatch sanitizer | TOCTOU / race / dispatch-reorder bypass | ADVISORY |

## Domain PASS list

- **secrets-hunter** — clean working tree + clean git history across 5 commits
- **crypto-reviewer** — no crypto surface; only PRNG is `random.Random` for shape fuzzing (correct usage)
- **license-auditor** — Apache-2.0 + all-compatible deps

## Verdict computation

Per PROTOCOL.md §Verdict: 0 CRITICAL, 0 HIGH, 3 MEDIUM ⇒ **ADVISORY** (Phase 2 may proceed; engineering-lead must mitigate the 3 MEDIUMs in their respective tracks).

## Engineering hand-off

The 3 MEDIUMs that engineering-lead must address before Phase 3 merge:
1. **CFG-2** — add `permissions: contents: read` block to `.github/workflows/ci.yml` (one-line fix, can be a standalone commit on `release/v1.0`).
2. **TM-E1** — `os.path.realpath` + allowlist (`/usr/local/cuda`, `/opt/nvidia/cuda`, `/opt/cuda`) in `sanitizers/race.py:56-62`. Track-A or Track-C scope.
3. **DEP-1** — add `uv.lock` (or equivalent) committed; CI installs from lock; pyproject pins continue to declare floors. Track-D scope.

The 5 N-prefix MPS findings are **prescriptive** for Track A's MPS backend implementation: every recommended mitigation skeleton in `EVIDENCE/threat-modeler.md` must appear in the Track-A diff or be explicitly waived in CHARTER.md with rationale.
