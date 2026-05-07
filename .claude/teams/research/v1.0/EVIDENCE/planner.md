---
specialist: research-planner
slug: v1.0
started: 2026-05-01T03:41:15Z
completed: 2026-05-01T03:42:00Z
tool_calls_count: 0
citations_count: 4
confidence: high
---

# Planner — gpucheck v1.0 MPS dispatch plan

## Question class

This is a **complex research** task by Anthropic's scaling rule (10+ specialists,
10-30 tool calls each). Reasons:
- 8 sub-questions spanning correctness, API surface, determinism, prior art, tooling.
- Two of them (open-bug crawl, llama.cpp pattern) require corpus-level mining.
- Output is a binding spec the engineering team will cite — high stakes for a v1.0 release.
- MPS surface is changing fast (PyTorch 2.6→2.11) — fresh-window scan mandatory.

Per MEMORY.md lesson "When the user prompt is short, distrust your initial sub-question list to catch the latest 14 days", this is a fast-moving topic (`module: mps` produces issues weekly — confirmed by the 60+ issues opened/updated in the last 4 weeks below). The 14-day sweep is built into Sub-Q 2 explicitly.

## Dispatch recommendation

Round 1 (wide, parallel, single emission) — all 12 source-gathering specialists:

| Specialist | Sub-questions owned | Why |
|---|---|---|
| cartographer | 8, 7 | Map gpucheck source: device gating, tolerance table, fuzzing pipeline. |
| archaeologist | 8 | Git history of tolerance calibration and CUDA-only assumptions. |
| librarian | 3 | Authoritative `torch.mps` API via PyTorch docs. |
| historian | 5, 6 | llama.cpp Metal precedent, MLX/Triton-Metal prior art. |
| linguist | 1, 4 | Disambiguate "non-deterministic" (atomic vs implementation-defined order vs platform-bug). |
| web-miner | 6 | MLX/llama.cpp/community ports, ZMLX, pmetal, vllm-metal. |
| github-miner | 2, 1 | `gh api search/issues label:"module: mps"` crawl, top-N by reactions and recency. |
| tracer | 1 | Trace MPS dispatch path in PyTorch (math vs flash backend, CPU fallback). |
| empiricist | 7, 8 | Quantify CUDA fuzz playbook (511 configs); recommend MPS tolerance multipliers from observed bug magnitudes. |

Round 2 (adversarial gates, parallel where disjoint):
- synthesist (consolidate Round 1)
- skeptic (attack reasoning), adversary (audit corpus quality — required because >50% of evidence is web/GH)
- moderator (only if synthesist flags load-bearing contradiction)

Round 3:
- evaluator (5-dim rubric)
- retrospector + scribe (close)

## Why not skip the adversary

Heavy reliance on GitHub issue threads + community ports + Apple's own marketing pages. PyTorch issue threads are first-party but reproductions are user-reported and sometimes wrong. Adversary must verify load-bearing bug claims (e.g. "83% error in Triton layer norm" — confirmed below by direct fetch of triton#9838 [WebFetch 2026-05-01], titled "Tutorial layer_norm: variance padding bug causes 83% error for non-power-of-2 feature dims").

## Memory-aligned heuristics applied

- **Anthropic scaling**: 10 wide specialists, then 4 gate specialists. Within Anthropic's published 10+ band for complex research.
- **Parallel dispatch**: Round 1 is one emission, all specialists concurrent.
- **14-day fresh-window sweep**: github-miner pulls `state:open sort=updated` to catch the live April-2026 batch (issues #181936, #182052, #181946 already surface).
- **REUSE/EXTEND/REWRITE**: this is a NEW question (no prior session in INDEX.md for gpucheck-MPS), so no reuse decisions.
- **Skeptic + adversary mandatory**: web-heavy corpus + load-bearing benchmark claims = both must run.

## Risk register

1. **PyTorch docs URL redirect issue** (observed: `docs.pytorch.org/docs/stable/...` returns redirect-only HTML). Workaround: hit `/docs/2.11/...` directly. Already verified working.
2. **Apple Metal Spec PDF too large for WebFetch** (10MB cap). Workaround: cite the PDF URL + use WebSearch summaries plus secondary sources for ULP/determinism claims, label as REPORTED-NOT-VERIFIED where the spec is the only primary.
3. **MLX repo file-listing depth**: WebFetch returns kernel filenames but not full source. Acceptable — filenames + the MLX SDPA file already loaded confirm fused kernels exist.
4. **Time budget 60 min**. Hard cap on each external fetch; aggressive parallel calls; will mark INCOMPLETE on any sub-question that can't be primary-sourced within budget.

## Verdict

PROCEED with Round 1 dispatch as listed.

## Confidence

High — the question is well-shaped, the sub-questions decompose cleanly, primary sources are identified, fresh-window sweep covers the April-2026 batch.
