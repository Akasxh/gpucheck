# Research Question — gpucheck v1.0 MPS Backend

**Slug:** v1.0
**Date:** 2026-05-01
**Requested by:** session orchestrator (Phase 1)
**Downstream consumer:** engineering-lead (CHARTER.md will cite SYNTHESIS.md as binding spec)

## Primary question

Should gpucheck add an Apple-Silicon MPS backend as the headline v1.0 feature, and if so what are the load-bearing correctness, performance, and tooling facts that the implementation must respect?

## Sub-questions (each must be answered with ≥3 primary sources)

1. **MPS correctness baseline.** What is the current state of `torch.mps` correctness vs CUDA (as of 2026-05) for the kernels gpucheck targets:
   - matmul (FP32 / FP16 / BF16)
   - softmax / log_softmax
   - layer_norm / rms_norm / batch_norm / group_norm
   - scaled_dot_product_attention (and FlashAttention variants)
   - cross_entropy / nll_loss / kl_div
   - conv2d
   Cite primary issue threads, release notes, and any MPS test reports from PyTorch CI.

2. **Open MPS bugs.** Crawl `github.com/pytorch/pytorch` issues with the `module: mps` label. Categorize by (a) numerical drift, (b) crash/hang, (c) missing op, (d) determinism. Identify the top 10 highest-impact open bugs and for each note the affected kernel + dtype + recommended workaround.

3. **MPS API surface.** What does `torch.mps.*` provide as of the latest stable PyTorch? Specifically: `torch.mps.synchronize()`, `torch.mps.Event`, `torch.mps.empty_cache()`, `torch.mps.current_allocated_memory()`, `torch.mps.profiler.*`, `torch.backends.mps.is_available()`, `torch.backends.mps.is_built()`. Use **context7** for authoritative API docs.

4. **Determinism guarantees.** What does Apple's Metal Performance Shaders documentation say about determinism? What does PyTorch's MPS backend say? Are there ops with non-deterministic kernels (e.g. atomic-add reductions, segmented scatter)? Cite Apple Metal Shading Language spec and PyTorch determinism docs.

5. **llama.cpp Metal dispatch precedent.** How does llama.cpp dispatch CPU vs Metal? What's its correctness-validation pattern? How does it handle the "kernel is fast but slightly inaccurate" tradeoff? Reference the relevant `ggml-metal.m` and test fixtures.

6. **FlashAttention / Triton on Metal.** Status as of 2026-05: are there community ports of FlashAttention to Metal? Has Triton added a Metal backend? What's the gap between CUDA-Triton kernels and what would compile to MPS? Reference any MLX or Apple-internal projects.

7. **Tolerance defaults.** Given findings from (1) and (2), what tolerance multipliers should gpucheck apply for MPS vs the existing CUDA defaults? Map per-dtype (FP16, BF16, FP32) and per-op family. Source the multipliers from real bug data, not folk wisdom.

8. **gpucheck's existing CUDA bug-finding record.** Re-confirm the existing record: 8 real bugs in Triton/PyTorch found via 511 test configs, including triton#9838 (83% error in layer norm) and triton#9839 (FP16 drift in matmul). What was the *fuzzing strategy* that found those? Document so MPS can apply the same playbook.

## Required deliverables (paths relative to `~/Code/gpucheck/.claude/teams/research/v1.0/`)

- `EVIDENCE/<specialist>.md` — one per dispatched specialist, with citations
- `HYPOTHESES.md` — competing hypotheses + prior probabilities
- `SYNTHESIS.md` — final cited handback to engineering, organized by sub-question
- `LOG.md` — turn-by-turn timeline of dispatches and gate transitions

## Adversarial gates (mandatory before "high confidence")

1. `research-skeptic` — attack reasoning chains in SYNTHESIS draft
2. `research-adversary` — attack the corpus (SEO blogs, citation laundering, astroturf)
3. `research-moderator` — 3-round debate on any flagged contradiction
4. `research-evaluator` — 5-dim rubric: Goal alignment, Communication, Output quality, Safety, Efficiency

## Hard rules

- Synthesist must cite ≥3 primary sources per claim.
- No SEO blogs. No "X is awesome" Medium posts.
- All PyTorch issue references include URL + status (open/closed) + last-update date.
- Use `mcp__plugin_context7_context7__query-docs` for any current `torch.mps` API claim.
- Use `gh api search/issues` for live PyTorch bug-tracker data.
- Tolerance recommendations must include the source bug or measurement that justifies them.

## Out of scope (not this question)

- Implementation choices (those are engineering-lead's call)
- Which kernel to ship in the swarm fuzzer (that's testing-lead's call)
- README/docs wording (docs-lead)
