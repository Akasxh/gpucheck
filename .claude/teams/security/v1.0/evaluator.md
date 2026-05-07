# evaluator — 5-dimension rubric

Read: AUDIT_CHARTER.md, all 8 round-1 EVIDENCE files, skeptic.md,
THREAT_MODEL.md, FINDINGS.md.

## Dimension 1: Coverage (weight 0.25)
Charter mandated all 8 domain specialists. All 8 produced an evidence
file: planner, threat-modeler, architecture-reviewer, owasp-scanner,
secrets-hunter, dependency-auditor, license-auditor, config-scanner,
crypto-reviewer. ✓
- Threat modeler hit all 4 charter-named MPS surfaces (N1–N4) **plus
  the implied N5** (sanitizer hook).
- OWASP Top 10 categories all addressed (A01–A10) with explicit
  N/A reasons where appropriate.
- Secrets hunter scanned both working tree and full git history.
- Dep auditor manually cross-referenced each pin.
- License auditor produced an SBOM stub.

**Score: 1.00**

## Dimension 2: Accuracy (weight 0.25)
Skeptic flagged 0 false positives, 0 severity downgrades, 1
deduplication (OWASP-A08-1 ↔ CFG-1). Severity ratings cross-checked
against PROTOCOL.md CVSS-aligned definitions:
- 3 MEDIUMs are correctly calibrated (each is a "security weakness
  with limited impact or difficult exploitation").
- 13 LOWs each map to "best practice violation, minor concern,
  informational".
- ADVISORY MPS findings correctly avoid pretending to be exploitable
  on code that does not exist.

False-positive rate: 1 dedup / 22 raw findings ≈ 4.5%, well under the
20% threshold for full score.

**Score: 0.95** (-0.05 for the late-stage dedup that should have been
caught at write time, not at skeptic stage).

## Dimension 3: Actionability (weight 0.20)
Every MEDIUM finding has:
- file:line citation ✓
- a remediation block in correct language (Python / YAML / TOML) ✓

Every LOW finding has either a file:line or a file-level location.
The MPS ADVISORY items each include a complete code skeleton in the
correct target language.

Spot-check of one remediation correctness:
- TM-E1 fix: `os.path.realpath` + prefix allowlist — correct, would
  block the documented attack without breaking the legitimate
  `/usr/local/cuda/bin/compute-sanitizer` path.
- CFG-2 fix: `permissions: contents: read` — verified against GitHub
  Actions docs, this is the minimum read-only token.
- DEP-1 fix: `uv lock` + `uv sync --frozen` — correct given the
  detected toolchain (uv 0.11.7 is on the dev box).

**Score: 1.00**

## Dimension 4: Completeness (weight 0.15)
Skeptic identified 5 minor gaps; 3 were closed in-place (plugin.py
top-level subprocess check, examples/ corpus, test isolation handed
to N5). 2 remain documented as backlog (machine-readable SBOM,
extras="all" missing triton/cupy — UX bug, not security).

No obvious attack vector for a Python pytest plugin was missed:
subprocess (covered), env-var trust (covered), JSON deserialisation
(covered), pickle (verified absent), yaml.load (verified absent),
xml external entity (xml.etree.ElementTree used in reporting/ci.py:7
for emitting XML, not parsing — verified safe).

Architecture review is substantive: trust-boundary diagram, defense-
in-depth table, data-flow trace from user input to subprocess. Not
generic.

**Score: 0.95**

## Dimension 5: Report quality (weight 0.15)
- THREAT_MODEL.md: actor table, asset table, trust-boundary diagram,
  full STRIDE matrix per surface, top-3 prioritised risks. ✓
- FINDINGS.md: per-finding verdict ledger; verdict computation shown
  with the formula. ✓
- All evidence files follow the PROTOCOL schema (severity, location,
  exploitability, blast radius, confidence, remediation, verification).
- TURN_LOG appended with one row per dispatch + each gate transition.

Minor: SECURITY_REPORT.md as named by PROTOCOL.md is replaced by
THREAT_MODEL.md + FINDINGS.md in this engagement (charter explicitly
asked for those two file names). Acceptable per charter; flagging as
a stylistic deviation, not a defect.

**Score: 0.92**

## Overall score
```
0.25 * 1.00 + 0.25 * 0.95 + 0.20 * 1.00 + 0.15 * 0.95 + 0.15 * 0.92
= 0.250 + 0.2375 + 0.200 + 0.1425 + 0.138
= 0.968
```

## Verdict
**PASS** (overall ≥ 0.75 and no dimension below 0.50).

## Final session verdict (passed up to lead)
**ADVISORY** (3 MEDIUM, 13 LOW, 5 MPS-ADVISORY, 0 HIGH, 0 CRITICAL).
