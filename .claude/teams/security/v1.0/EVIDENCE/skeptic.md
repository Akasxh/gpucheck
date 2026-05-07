# skeptic — red-team of round 1 findings

Read all 8 specialist evidence files. Attacking claims, not the codebase.

## Findings to REMOVE (false positives)
None. Each finding cites a concrete file:line or — for ADVISORY MPS
items — labels itself ADVISORY because the code does not yet exist.
The threat-modeler correctly avoided BLOCKER labels for not-yet-written
code (charter rule).

## Findings to DOWNGRADE
| Finding | Specialist | Current | Recommended | Reason |
|---|---|---|---|---|
| TM-E1 path injection via `CUDA_HOME` | threat-modeler | MEDIUM | **MEDIUM (kept)** | Real, not a false positive. Threat actor needs env-write, which is a moderate bar. MEDIUM matches CVSS-aligned definition (security weakness with limited impact, difficult exploitation). |
| AR-3 inherited subprocess env | architecture-reviewer | LOW (with MEDIUM caveat) | **LOW (kept)** | Defence-in-depth, not exploitable on its own. |
| OWASP-A03-1 unvalidated `extra_args` | owasp-scanner | LOW | **LOW (kept)** | Test author can already do anything; this is hardening only. |

## Findings to UPGRADE
| Finding | Specialist | Current | Recommended | Reason |
|---|---|---|---|---|
| **CFG-2 `permissions:` block missing** | config-scanner | MEDIUM | **MEDIUM (kept)** | MEDIUM is correct. Default `GITHUB_TOKEN` permissions on classic repos let a hostile transitive dep push to the repo. Could argue for HIGH, but exploitation requires (a) a malicious dep entering the dev install set and (b) PR-trigger from an attacker; combined likelihood is moderate. Keeping MEDIUM. |
| **DEP-1 no lock file** | dependency-auditor | MEDIUM | **MEDIUM (kept)** | Same reasoning. Real supply-chain weakness, but no current evidence of compromise. |

## Duplicates / overlap
| Finding A | Finding B | Resolution |
|---|---|---|
| AR-1 (factor a `_run_external_tool` helper) | TM-E1 (path-prefix check on `CUDA_HOME`) | **Keep both.** They are complementary — AR-1 says *where* to put the fix, TM-E1 says *what* the fix must do. |
| AR-2 (no JSON schema validation) | (none — only AR speaks to it) | **Keep AR-2 alone.** |
| OWASP-A08-1 (Actions tag pinning) | CFG-1 (Actions tag pinning) | **Duplicate.** Keep CFG-1 (more detailed remediation YAML). Note OWASP-A08-1 as a cross-reference. |
| DEP-1 (no lock file) | CFG-4 (CI installs unlocked deps) | **Keep both.** DEP-1 is the root cause, CFG-4 is the CI manifestation. |

## Coverage gaps
| Gap | Impact | Recommendation |
|---|---|---|
| **No test of `gpucheck.plugin` hook security** — pytest plugin hooks run at collection time and can mutate session state. The OWASP scanner did not specifically check `plugin.py`. | LOW (pytest plugin hooks are sandboxed by pytest itself; no privilege boundary). | Spot-check `plugin.py` for any `os.system`/`subprocess` at module top level. *Done by lead while writing this:* `grep` confirms no module-top-level subprocess or env-mutation in `plugin.py`. **No new finding.** |
| **No fuzzer corpus check** — `examples/` and any committed fuzz seeds. | LOW | Verified `examples/` contains only `.py` demo scripts. No corpora committed. **No new finding.** |
| **No test isolation review** — pytest-xdist support not analysed. | LOW | The MPS sanitizer (N5) explicitly addresses this. Sufficient. |
| **No SBOM artefact** — license-auditor produced an inline SBOM stub but no machine-readable artefact (CycloneDX/SPDX). | LOW (compliance-only) | Tracked as backlog; not BLOCKER. |
| **`extras = "all"` does not include `triton` or `cupy`** | LOW (UX bug, not security) | Not a security finding. |

## Severity calibration cross-check
Counting post-skeptic:
- CRITICAL: **0**
- HIGH: **0**
- MEDIUM: **3** (TM-E1, CFG-2, DEP-1)
- LOW: **9** (TM-E2, AR-1, AR-2, AR-3, OWASP-A03-1, OWASP-A05-1, CFG-1=OWASP-A08-1 deduped → 1, CFG-4, CFG-5, DEP-2, DEP-3, DEP-4, LIC-1)
  → re-count: TM-E2(1), AR-1(2), AR-2(3), AR-3(4), OWASP-A03-1(5), OWASP-A05-1(6), CFG-1(7), CFG-4(8), CFG-5(9), DEP-2(10), DEP-3(11), DEP-4(12), LIC-1(13). **13 LOW.**
- ADVISORY: **5 MPS surfaces** (N1–N5) + DEP-5

## Verdict computation against PROTOCOL.md thresholds
```
BLOCKER = any CRITICAL OR ≥3 HIGH    →  0 + 0 → false
ADVISORY = any HIGH or MEDIUM AND not BLOCKER  →  3 MEDIUM → true
PASS = only LOW                       →  false
```
**Session verdict: ADVISORY.**

## Overall assessment
Quality is sufficient for a credible SECURITY_REPORT. Findings are
well-scoped, cite file:line, and propose concrete remediation. The
ADVISORY-level treatment of MPS surfaces (which do not yet exist in
code) is correct under the charter's BLOCKER rule (need exploitation
evidence on actual code).

**Recommendation**: proceed to evaluator gate. No re-dispatch needed.
