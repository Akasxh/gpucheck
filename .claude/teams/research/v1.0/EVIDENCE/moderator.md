---
specialist: research-moderator
slug: v1.0
started: 2026-05-01T03:45:00Z
completed: 2026-05-01T03:45:30Z
tool_calls_count: 0
citations_count: 4
confidence: high
---

# Moderator — debate verdicts

Per protocol: the moderator dispatches **only** when the synthesist flags a
load-bearing contradiction. The synthesist (`EVIDENCE/synthesist.md` §2)
reports: "I find **no load-bearing contradiction** that requires a moderator
debate. The specialists converge on the same picture."

The synthesist did flag three SOFT tensions; per MEMORY.md lesson `REFRAME is
a valid moderator verdict — don't force winner-take-all on mis-posed debates`,
I evaluate whether each is genuinely a debate or a phrasing issue.

## Tension T1 — H1 "ship now with xfails" vs H2 "wait one cycle"

Per synthesist §2.T1: this is a risk-tolerance tradeoff, not a fact disagreement.

**Verdict: REFRAME.** This is not a debate over "what is true" but a debate
over "what is the right v1.0 decision given the facts". The facts are
established: 255 open MPS issues, 12 high-impact ones suitable for xfail,
fuzz playbook transfers, API surface stable. The reframe: ship gpucheck-MPS
in v1.0 in **observation-instrument-then-calibrate mode** — H1 modulated by H3
(MPS as correctness oracle). The xfail list is a living document.

I do not need to run a 3-round debate. The reframe IS the answer.

## Tension T2 — gpucheck's bug-finding record: 8 vs 2 externally-verified

Per synthesist §2.T2: this is a documentation-precision concern, not a research conclusion.

**Verdict: NOT_LOAD_BEARING.** v1.0 ship/no-ship doesn't depend on whether the
record is 8 or 2. The narrative for the engineering team is "we have a method
that found verifiable bugs in widely-used kernels; that method transfers to
MPS". 2 verifiable bugs is sufficient for that. SYNTHESIS will phrase it as
"8 bugs found, of which triton#9838 (open) and triton#9839 (closed) are filed
and externally-verified". No debate.

## Tension T3 — "Non-deterministic" has three meanings

Per synthesist §2.T3 + linguist §1: a phrasing concern, fully resolved by
consistent vocabulary in SYNTHESIS.

**Verdict: NOT_LOAD_BEARING.** No debate. SYNTHESIS uses linguist's A/B/C scheme:
A = run-to-run divergence (xfail), B = MPS-vs-CPU divergence (assert_close
catches with overlay), C = precision-floor drift (existing tolerances absorb
or 2× overlay does).

## §4. Skeptic vs synthesis — debate?

The skeptic's 6 attacks each absorb into careful labeling rather than flipping
the synthesis. None of them is a competing primary-source claim against the
synthesis. No moderator-grade contradiction exists.

**Verdict: NO MODERATOR DEBATE NEEDED.**

## §5. Adversary vs synthesis — debate?

The adversary affirms the corpus is healthy and finds no rejected sources. No
contradiction.

**Verdict: NO MODERATOR DEBATE NEEDED.**

## §6. Disposition

This file exists to document that the protocol's moderator gate ran and
returned no conditional debates. Per protocol, I record this affirmatively
rather than skipping the file.

If a future re-dispatch surfaces a load-bearing contradiction (e.g. an empirical
M-machine run shows the 2× tolerance is wrong by 10×), THAT is the moment for
a moderator debate.

## §7. Cross-references

- `EVIDENCE/synthesist.md` §2 (no contradiction flagged)
- `EVIDENCE/skeptic.md` §8 (passes)
- `EVIDENCE/adversary.md` §6 (passes)
- MEMORY.md lesson `REFRAME is a valid moderator verdict`

## Verdict

REFRAME / NOT_LOAD_BEARING for all three soft tensions. No 3-round debate
needed. SYNTHESIS may proceed.

## Confidence

High — the synthesist explicitly stated no load-bearing contradiction; the
skeptic and adversary both passed; the 3 soft tensions are reframes, not
factual disputes.
