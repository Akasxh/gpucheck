# DIFF_LOG — claude-forge v0.4 hook 1b (Agent-dispatch wrapper)

Owner: engineering-executor
Source spec: `~/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/architect-continuous-learning.md` §1b

## Iteration 1 — Task 1b.1: cf_dispatch_wrapper.py main entrypoint
- **File**: `/Users/cero/.claude/scripts/cf_dispatch_wrapper.py`
- **Change**: Created the wrapper module with `wrap_agent_prompt(original_prompt, subagent_type, top_k=5)`, `lead_for_subagent()`, `_load_ranker()` (graceful fallback if `lesson_ranker.py` is absent), and a `_cli()` entrypoint with `--subagent-type / --top-k / --json-out`. Stdlib only. Defensive: never raises — every external call wrapped, errors logged to stderr.
- **Reason**: Architect §1b mandates application-layer enforcement because the harness `PreToolUse` on `Task` is unreliable (cited research-lead/MEMORY.md L236-241). The wrapper is the single place that injects ranked lesson context into a sub-agent's first prompt.
- **Acceptance criterion addressed**: charter pt 1 (function signature, lead auto-detection from `subagent_type`, graceful fallback when ranker missing, output format `--- relevant lessons (claude-forge v0.4 hook 1b) --- ... --- end lessons --- <original_prompt>`).

## Iteration 2 — Task 1b.2: cf-rank bash shim
- **File**: `/Users/cero/.claude/scripts/cf-rank`
- **Change**: 9-line bash wrapper that `exec`s python3 against `cf_dispatch_wrapper.py`, forwarding all argv. Made executable (chmod +x).
- **Reason**: Charter pt 2 — orchestrators use the shim as a Unix-pipe-friendly invocation: `echo "$prompt" | cf-rank --subagent-type X`. Decouples orchestrator scripting from the python interpreter path.
- **Acceptance criterion addressed**: charter pt 2 (CLI ergonomics for orchestrator adoption).

## Iteration 3 — Task 1b.3: SKILL.md
- **File**: `/Users/cero/.claude/skills/cf-dispatch-wrapper/SKILL.md`
- **Change**: 78-line skill (≤80 LOC budget) with frontmatter (`name: cf-dispatch-wrapper`, `description: When an orchestrator dispatches a subagent via the Agent tool, prepend relevant memory lessons to the prompt. Invoke before calling Agent. Returns the augmented prompt.`) and a 5-section body: When/How to invoke, Behaviour, Output format, Example, Source.
- **Reason**: Charter pt 3 — skills are the documented invocation path orchestrators see in their skill-list (registered: confirmed by post-write skills reload). Body has the literal `cf-rank --subagent-type` example so a fresh orchestrator can adopt without re-reading the architect doc.
- **Acceptance criterion addressed**: charter pt 3 (skill registration + ≤80 LOC + 1 worked example).

## Iteration 4 — Task 1b.4: test suite
- **File**: `/Users/cero/.claude/scripts/tests/test_cf_dispatch_wrapper.py`
- **Change**: 18 tests (parametrized) covering: lead-prefix mapping (9 cases incl. `general-purpose`, empty, unknown), three fall-back paths (missing ranker, raising ranker, empty-list ranker), happy path (format / order / top_k), `general-purpose` → `scope: shared-only` tag, never-raises invariant on adversarial prompts, missing-field lessons, and three subprocess-driven CLI smoke tests including `--json-out`.
- **Reason**: Same tmp_path/monkeypatch fixture pattern as task #1 (charter directive). Tests use stdlib + pytest only and never touch the real `~/.claude/agent-memory` tree.
- **Acceptance criterion addressed**: charter pt 4 (test parity with task #1).

---

# DIFF_LOG — claude-forge v0.4 hook 1a (SessionStart lesson ranker + injector)

Owner: engineering-executor
Source spec: `~/Code/gpucheck/.claude/teams/audit/v1.1/EVIDENCE/architect-continuous-learning.md` §1a + §4

## Iteration 5 — Task 1a.1: SessionStart lesson ranker (Python)
- **File**: `/Users/cero/.claude/scripts/lesson_ranker.py`
- **Change**: New stdlib-only ranker module. `rank_lessons_for_question(question, lead, top_k=15, shared_top_k=10)` consumes `~/.claude/agent-memory/INDEX.md` (parsed via row regex), filters status=stale and harmful_count>=2, scores `0.6 * tag_jaccard + 0.2 * recency_decay(90d_halflife) + 0.2 * helpful_ratio`, partitions per-lead-top-K + cross-lead-shared-top-K, always-includes any lesson under `~/.claude/agent-memory/_starter/` (tolerates missing). Body retrieval reads each chosen lesson verbatim from its lead's MEMORY.md. CLI: `--question --lead --top-k --shared-top-k --format markdown|json`. Defensive: never raises; on internal failure returns starter-only or [].
- **Reason**: Charter pt 1 — the ranker must be the single deterministic scoring authority for SessionStart so the hook is dumb-glue. Consuming INDEX.md (already shipped by `build_memory_index.py`) avoids re-walking every MEMORY.md per the architect's "offline-indexed" §4 directive.
- **Acceptance criterion addressed**: charter pt 1 (hybrid score, filters, top-K + shared-top-K + starter, defensive contract, stdlib only).

## Iteration 6 — Task 1a.2: SessionStart hook script (bash)
- **File**: `/Users/cero/.claude/hooks/session-start.sh`
- **Change**: New 70-LOC Stop-style hook. Reads PAYLOAD JSON from stdin, extracts `session_id` + `cwd` via python3 stdlib, invokes `python3 ~/.claude/scripts/lesson_ranker.py --question "" --top-k 15 --shared-top-k 10 --format markdown`, persists output to `/tmp/claude-forge-context-<session_id>.md` (idempotent: overwrite-safe, content deterministic), logs one line to `~/.claude/agent-memory/_session-start.log`, emits `{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "<markdown>"}}` on stdout. Always `exit 0` per architect's "do not block session start" rule. `chmod +x`.
- **Reason**: Charter pt 2 — Claude Code's documented SessionStart hook contract is JSON-on-stdout with `additionalContext`; this is what the harness prepends to the conversation. Smoke-tested with synthetic payload, hook returned a parseable envelope (5046-byte additionalContext) and the persisted /tmp file matched.
- **Acceptance criterion addressed**: charter pt 2 (hook contract, payload parsing, /tmp persistence, log line, return 0, JSON injection mechanism).

## Iteration 7 — Task 1a.3: settings.json wiring
- **File**: `/Users/cero/.claude/settings.json` (resolved via dotfiles symlink)
- **Change**: Appended one entry to `hooks.SessionStart` array (created since absent): `{"matcher": "", "hooks": [{"type": "command", "command": "$HOME/.claude/hooks/session-start.sh"}]}`. Existing `hooks.Stop` (1 entry, `session-capture.sh`) and `hooks.PostToolUse` (1 entry, `log-evidence-writes.sh`) preserved verbatim. Edit performed via `python3 -c "json.load() ... json.dump()"` for safe round-trip; backup written to `~/.claude/settings.json.bak.20260507`.
- **Reason**: Charter pt 3 — the harness only honors hooks declared in settings.json; appending (not overwriting) preserves the v0.3 Stop hook (scribe-merge entrypoint) and the PostToolUse evidence-write tracer. JSON round-trip with idempotency check (skip-if-already-wired) prevents double-registration on rerun.
- **Acceptance criterion addressed**: charter pt 3 (append to SessionStart array, preserve Stop + PostToolUse, JSON validity).

## Iteration 8 — Task 1a.4: ranker test suite
- **File**: `/Users/cero/.claude/scripts/tests/test_lesson_ranker.py`
- **Change**: New 19-case pytest module. Sandboxed (monkeypatch redirects `MEM_ROOT`/`INDEX_FILE`/`STARTER_DIR` to `tmp_path`): Jaccard correctness/empty/disjoint, recency-decay (today=1.0, 90d=0.5, 365d=`pow(0.5, 365/90)`, unparseable->0), feedback-score (zero->0.5, helpful>0.5, harmful<0.5), tokenize drops stopwords, top_k respected, harmful_count>=2 filter, status=stale filter, lead-partition vs cross-lead, Jaccard drives ranking, empty INDEX->[], garbage INDEX->[], `_starter/` always-included, `_starter/` missing tolerated. Live: 1 fixture test against real `research-lead/MEMORY.md` (skipped if INDEX.md absent). Plus `__init__.py` to make the dir a package.
- **Reason**: Charter pt 4 — test parity with the spec checklist (top_k, Jaccard, recency, harmful, stale, empty MEMORY, live fixture). Stdlib + pytest only; no real `~/.claude/agent-memory` mutations.
- **Acceptance criterion addressed**: charter pt 4 (all spec test cases covered).
