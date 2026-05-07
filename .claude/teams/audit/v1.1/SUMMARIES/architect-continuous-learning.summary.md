# architect-continuous-learning summary (W2)

## 4 hook attach points

1. **SessionStart**: rank top-K lessons from STARTER + SHARED + per-lead MEMORY.md → inject as `<continuous-learning-context>` block (≤500ms target).
2. **Agent dispatch (orchestrator-side)**: prepend ranked lesson set to every Agent payload. Application-layer is the source of truth (harness PreToolUse is unreliable per existing research-lead lesson L236-241).
3. **SessionEnd**: extend `~/.claude/hooks/session-capture.sh` to **automatically run scribe-merge** (today only writes to staging — never merges; this CLOSES THE LOOP). flock+atomic-rename for the 92-process concurrency case.
4. **Pattern-extraction**: out-of-session script (cron/launchd/`/loop`) — scans corpus, proposes skills to forge-lead.

## Triggers

- SessionStart event → 1a always
- PreToolUse on Task + orchestrator caching → 1b
- Stop event → 1c (branches team/adhoc/trivial)
- cron + on-demand `/loop` + 1c-set marker file → 1d

## Ranker design

Hybrid score: tag-overlap **60%** (Jaccard on tags) + recency-decay **20%** (90-day half-life) + helpful/harmful ratio **20%**.
Top-K = 15 per-lead + 10 shared + always-include STARTER (~9% of context).
Index pre-built by scribe-merge, sub-100ms at SessionStart.

**Rejected:** vector embeddings (latency), LLM judge (cost+circular), static first-200-lines (current — wrong).

## Most important failure mode + mitigation

**FM-2 Lesson rot.** A 6-month workaround for a fixed bug stays injected → system gets confidently worse than no-memory.
**Mitigation:** schema MUST include `harmful_count`; when `harmful_count > helpful_count` AND total ≥3, scribe **auto-archives** the lesson (no human review). Plus recency-decay in ranker as belt-and-suspenders. Without this reflex the corpus only ever grows monotonically wrong.

## Migration: 10 ordered steps, no data loss until step 4 (preserves originals in `_migrated/`), full rollback ≤5 min
