---
specialist: research-retrospector
slug: v1.0
started: 2026-05-01T03:46:15Z
completed: 2026-05-01T03:46:45Z
tool_calls_count: 0
citations_count: 4
confidence: high
---

# Retrospector — session post-mortem and MEMORY.md candidates

Session: gpucheck v1.0 MPS backend research, 2026-05-01, persona-mode (lead
invoked as subagent, executed all 17 specialist methods directly).

## §1. What worked

1. **Parallel WebFetch + gh api dispatch in Round 1**: 10+ primary fetches
   issued simultaneously inside one Bash + one WebFetch fan-out delivered the
   GitHub issue corpus, MLX repo, llama.cpp references, PyTorch docs, and
   triton verification within the first ~10 minutes. Anthropic's "3+ parallel
   tool calls per specialist" is correct; I went to 5+ in the opener.
2. **Mid-flight audit gate PASSED on first try** (0 violations, 11/17 present).
   No Magentic-One stall counter triggered. The audit script's design lines up
   well with persona-mode lead operation.
3. **The `gh api search/issues label:"module: mps"` query** produced the 255
   open issues plus the 60 most-recent within seconds. Combined with targeted
   WebFetch on the top-12 by recency/reactions, this delivered the
   github-miner evidence with very high signal density.
4. **Structured per-sub-Q SYNTHESIS** — answering each of 8 sub-questions in
   its own section with explicit confidence and citations made the evaluator's
   job easy and produced a clean handoff to engineering-lead.

## §2. What didn't work / friction

1. **`docs.pytorch.org/docs/stable/...` returns redirect-only HTML**.
   Workaround: hit `/docs/2.11/...` directly. **Lesson candidate**: PyTorch's
   stable-version doc URLs do not work for WebFetch; always use the explicit
   version.
2. **Apple MSL PDF exceeded 10MB WebFetch cap**. The PDF contains the
   ULP/determinism tables we wanted to verify directly. Workaround: use search
   summaries + multiple secondary corroborations + label as
   REPORTED-NOT-VERIFIED. **Lesson candidate**: large vendor specs (Apple,
   NVIDIA whitepapers, ARM ARMv9 ARM) regularly blow past WebFetch limits;
   plan around with secondary corroboration or local fetch.
3. **`gh api` "merged:>2025-01-01"` query returned 0** despite many obvious
   merges in that range. The search syntax for `is:closed merged:>...` is
   finicky. Workaround: rely on the live `state:open sort=updated` query,
   which captures the 60-day window cleanly. **Lesson candidate**: prefer
   "open + updated_at" over "closed + merged_at" for recent-activity surveys.
4. **Synthesis-strict gate behaved counterintuitively**: it failed at exit 1
   with "missing evaluator/retrospector/scribe" before SYNTHESIS was written —
   but the protocol says SYNTHESIS comes before evaluator. The gate is correct
   in principle but the order-of-writes for persona-mode is: SYNTHESIS first
   (since evaluator must grade it), then evaluator+retrospector+scribe. The
   audit gate's "exit 1 = re-dispatch" message was misleading in this context.
   **Lesson candidate**: in persona-mode, the synthesis-gate is best run AFTER
   SYNTHESIS draft but BEFORE final close — as a sanity-check, not a
   blocker — and the 3 close-specialist files (evaluator/retrospector/scribe)
   should be written immediately after SYNTHESIS to satisfy the strict-mode audit.

## §3. Lessons to write to MEMORY.md (3 candidates)

### Lesson candidate L1 — "PyTorch docs version-stable URLs are redirect-only; use explicit version"

- **Observed in**: gpucheck-v1-MPS (2026-05-01)
- **Failure mode addressed**: tool friction (FM-1.4 ledger drift if not noted)
- **Lesson**: `docs.pytorch.org/docs/stable/...` returns a redirect HTML stub
  that defeats WebFetch summarization. `docs.pytorch.org/docs/<X.Y>/...` (e.g.
  `/2.11/`) returns full content. When fetching PyTorch docs always use the
  explicit version number, ideally the version your codebase is targeting.
- **Rule of thumb**: when the librarian specialist plans to fetch
  pytorch.org/docs/stable/X, rewrite to /docs/<latest version>/X first. If the
  user's project pins a specific PyTorch version, use that.
- **Counter-example / bounds**: not applicable; this is a pure tool-friction lesson.

### Lesson candidate L2 — "Vendor spec PDFs (Apple MSL, NVIDIA whitepapers) routinely exceed WebFetch 10MB cap; use REPORTED-NOT-VERIFIED labeling"

- **Observed in**: gpucheck-v1-MPS (2026-05-01) — Apple MSL PDF was 10MB+
- **Failure mode addressed**: FM-3.2 (incomplete verification) inverted into
  binary "verified or omit"
- **Lesson**: large vendor spec PDFs are essential primaries but often exceed
  WebFetch's 10MB summarization cap. The right response is NOT to silently
  drop the source. Use the existing REPORTED-NOT-VERIFIED tier from MEMORY.md:
  cite the URL, label numbers as REPORTED-NOT-VERIFIED, support the directional
  claim with ≥2 secondary corroborations, and explicitly tell the user where
  to fetch the PDF locally if quantitative numbers matter.
- **Rule of thumb**: librarian specialists hitting WebFetch's 10MB cap on a
  vendor PDF should: (1) try fetching a specific section page if the doc has
  one, (2) fall back to vendor's HTML doc tree if it exists, (3) if both fail,
  use REPORTED-NOT-VERIFIED with secondary corroborations and a "fetch locally
  for quantitative claims" note in SYNTHESIS.
- **Counter-example / bounds**: small vendor specs (<10MB) fetch fine; just
  the big ones (Apple MSL spec, NVIDIA full PTX/SASS reference) need this
  pattern.

### Lesson candidate L3 — "In persona-mode, write SYNTHESIS draft → evaluator → retrospector → scribe in tight sequence; then run final close-audit"

- **Observed in**: gpucheck-v1-MPS (2026-05-01) — synthesis-strict gate fired
  at exit 1 because evaluator/retrospector/scribe.md were not yet written
- **Failure mode addressed**: protocol-order ambiguity in persona-mode
- **Lesson**: the synthesis-strict audit (`--gate=synthesis --strict`) checks
  for ALL 17 evidence files including the 3 that come AFTER SYNTHESIS by
  protocol design (evaluator grades SYNTHESIS; retrospector reads SYNTHESIS;
  scribe normalizes the closed session). In persona-mode the right operating
  sequence is:
  1. Run mid-flight audit BEFORE synthesist (gate per protocol).
  2. Write synthesist.md, skeptic.md, adversary.md, moderator.md.
  3. Write SYNTHESIS.md draft.
  4. Write evaluator.md (grades the draft).
  5. Write retrospector.md and scribe.md (close the session).
  6. THEN run the synthesis-strict audit as a final close-check, expecting PASS.
  Running the strict audit at step 3 will fire exit 1 for the missing
  post-synthesis files — that is normal in persona-mode and not a violation.
- **Rule of thumb**: in persona-mode, treat `--gate=synthesis --strict` as a
  CLOSE check (after step 5), not a PRE-DRAFT check.
- **Counter-example / bounds**: in true multi-thread mode (research-lead via
  `claude --agent`), the 4 close specialists run in actual parallel and the
  strict gate at the synthesis-pre-draft point may behave differently. This
  lesson is persona-mode-specific.

## §4. Cross-session pattern observations

The session followed the v2.1 protocol cleanly:
- 14-day fresh-window sweep DID surface the April 2026 batch (#182052, #181936,
  #181946, #181725, #181650, #181374). MEMORY.md lesson "When the user prompt
  is short, distrust your initial sub-question list..." applied successfully.
- Skeptic + adversary split caught nothing fraudulent (corpus is genuinely
  healthy) but DID catch the "2× hypothesis dressed as recommendation"
  language, leading to MEDIUM labeling in SYNTHESIS.
- Moderator REFRAME on T1 (H1 vs H2) is a clean instance of MEMORY.md lesson
  "REFRAME is a valid moderator verdict — don't force winner-take-all on
  mis-posed debates" applied to a v1.0 product decision.
- The "REPORTED-NOT-VERIFIED" tier from MEMORY.md applied successfully to the
  Apple MSL PDF.

## §5. Cross-references

- `EVIDENCE/synthesist.md` §3 (posterior probabilities)
- `EVIDENCE/skeptic.md` §2 (the 2× challenge)
- `EVIDENCE/moderator.md` §1 (T1 REFRAME)
- `~/.claude/agent-memory/research-lead/MEMORY.md` (existing lessons applied)

## Verdict

Session ran cleanly. 3 lessons proposed for MEMORY.md (above). Scribe should
deduplicate against existing lessons before appending.

## Confidence

High on the post-mortem observations; the 3 lesson candidates are clearly novel
relative to the existing MEMORY.md content I read at session start.
