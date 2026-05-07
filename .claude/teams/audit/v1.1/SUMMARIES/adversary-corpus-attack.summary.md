# adversary-corpus-attack summary (W3)

**Verdict: HEALTHY corpus, supports plan with 3 named re-source items. Confidence MEDIUM-HIGH.**

## Per-file verdicts (14 files)

- **STRONG-PRIMARY (12):** api-dx-grade, security-postmerge, archaeologist-debt, detector-files, docs-tester, empiricist, forge-schema, architect-CL, historian, cartographer, planner, synthesist
- **MIXED (2):** mutator-survivors (80% kill-rate is projection not measurement); tracer-runtime (probe scripts missing)
- **WEAK (0)**

## 6 independent spot-checks — all passed

- GitHub: LangGraph BaseStore, AutoGen Memory ABC, Letta function_sets/base.py — files exist with cited symbols
- arXiv: 2510.04618 (ACE, Zhang ICLR 2026), 2502.12110 (A-MEM, Xu NeurIPS 2025), 2504.19413 (Mem0) — match
- PyTorch issue 162872 — open, "MPS deadlock when calling Event.synchronize()", labels confirmed
- Source: two compute_tolerance fns, 7 bare-except sites, top-level torch import at close.py:14 — all confirmed

## Top 3 weakest citations

1. **tracer-runtime probe scripts** — `/tmp/trace_runtime.py` and `/tmp/trace_silent_downcast.py` do NOT exist on disk; cited line ranges unreproducible. By contrast, empiricist's `/tmp/mac_bench-*.py` are all present. Asymmetric reproducibility.
2. **mutator-survivors 80% kill-rate** — projected from 60-of-221 sampled mutants; no verification step until Phase B tests land. Synthesist + planner inherit the projection without flagging.
3. **security PM-4 "torch <2.1 raises RuntimeError"** — uncited version-boundary claim; no PR / changelog cited.

## Suppression check on synthesist

5 contradictions (C1-C5) named are reasonable; walked corpus for a suppressed 6th — none found. Mem0 vendor numbers were flagged by historian itself (defensible).

## Most likely gap to bite v1.1

**Missing tracer probes.** T-19 (eliminate `_run_mps` duplicate) cites tracer trace-2 as regression baseline; if executor refactors and gets a 100µs slowdown, no preserved artifact to compare. Recommend executor re-derives baseline before refactor.

## Citation-laundering walk

`planner T-22 → SUMMARIES/api-dx-grade.summary.md → EVIDENCE/api-dx-grade.md Fix 2 → src/gpucheck/__init__.py:75-105` — bottoms out at primary source. **Structural risk**: planner cites *summaries*, not *evidence*. Recommend executors verify against EVIDENCE/, not SUMMARIES/.
