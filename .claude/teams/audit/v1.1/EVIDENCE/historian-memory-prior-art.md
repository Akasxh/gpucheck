# Historian — prior art on agent / LLM memory systems

**Scope:** survey of memory primitives across SDKs, IDE-agents, and academic papers,
to inform claude-forge v0.3 continuous-learning design.
**Retrieval date:** 2026-05-01.
**Systems surveyed:** 10 (LangGraph, OpenAI Responses+Conversations, Cursor,
AutoGen, Cline, Continue.dev, Anthropic Memory tool, MCP memory server, Letta).
**Academic papers cited:** 5 (MemGPT, Generative Agents, A-MEM, Mem0, ACE).

---

## Canonical names for this topic

- "agent memory", "long-term memory", "long-context", "episodic memory"
  (academic).
- "checkpointer", "checkpoint", "store" (LangGraph idiom).
- "thread", "conversation", "memory tool" (OpenAI / Anthropic API idiom).
- "memory bank", "rules", "context provider" (IDE-agent idiom — Cline,
  Cursor, Continue.dev).
- "memory blocks", "core memory", "archival memory", "recall memory"
  (MemGPT / Letta tradition, traceable to OS-style virtual memory analogy).
- "context engineering" (Anthropic's preferred framing as of late 2025).
- "Zettelkasten / note-based memory" (A-MEM, NeurIPS 2025).

---

## Foundational sources

### MemGPT — Packer, Wooders, Lin et al. (2023)
- arXiv:2310.08560 — UC Berkeley / Sky Computing Lab.
  URL: https://arxiv.org/abs/2310.08560 (retrieved 2026-05-01).
- **Relevance:** introduced the OS-virtual-memory analogy that defines almost
  every later "agent memory" system. Main context = working memory inside the
  LLM context window; external context = paged storage; the agent issues
  function calls to swap content in/out. Every later system either inherits
  this taxonomy or argues against it.
- **Credibility:** high — primary source, repeatedly cited (>1k citations on
  Semantic Scholar by mid-2025), production deployment via Letta proves the
  pattern works at scale.

### Generative Agents — Park, O'Brien, Cai et al. (2023)
- arXiv:2304.03442 — Stanford × Google. UIST 2023 best paper.
  URL: https://arxiv.org/abs/2304.03442 (retrieved 2026-05-01).
- **Relevance:** defined the **retrieval-scoring formula** that has become
  the de-facto baseline: `score = α·recency + β·importance + γ·relevance`,
  min-max normalised, equal weights in the original. Recency is exponential
  decay (factor 0.995 per sandbox-hour); importance is LLM-rated 1–10 at
  insert time; relevance is cosine similarity over embeddings of the query
  vs. each memory. Reflection (synthesising lower-level observations into
  higher-level reflections) is also from this paper.
- **Credibility:** high — Stanford HCI group, extensively cited and
  reimplemented; the recency/importance/relevance triple appears verbatim
  in subsequent systems (Mem0, A-MEM, Bedrock AgentCore guides).

### A-MEM — Xu, Liang, Mei, Gao, Tan, Zhang (2025)
- arXiv:2502.12110 — NeurIPS 2025.
  URL: https://arxiv.org/abs/2502.12110 (retrieved 2026-05-01).
- **Relevance:** Zettelkasten-style memory: each new memory is stored as a
  "note" with structured attributes (description, keywords, tags) and
  bidirectionally linked to related past notes; existing notes get *updated*
  when new memories arrive (memory evolution). Implemented over ChromaDB.
  Argues against flat blob-memory.
- **Credibility:** high — peer-reviewed at NeurIPS 2025; code released
  (github.com/agiresearch/A-mem). Evaluated across 6 foundation models.

### Mem0 — Chhikara, Khant, Aryan, Singh, Yadav (2025)
- arXiv:2504.19413 (April 2025).
  URL: https://arxiv.org/abs/2504.19413 (retrieved 2026-05-01).
- **Relevance:** production-oriented agent memory — extracts/consolidates/
  retrieves salient facts from dialogue; offers a graph-memory variant for
  relational structure. Reports 26 % LLM-judge improvement over OpenAI's
  built-in memory, 91 % latency reduction, ~90 % token-cost reduction vs.
  full-context. The numbers are from the authors' own benchmark, so cite
  with care.
- **Credibility:** medium-high — claims are vendor-self-reported; commercial
  product (mem0.ai) behind the paper. *adversary: please verify the
  benchmark methodology.*

### ACE — Zhang, Hu, Upasani, Ma et al. (2025/26, ICLR 2026)
- arXiv:2510.04618.
  URL: https://arxiv.org/abs/2510.04618 (retrieved 2026-05-01).
- **Relevance:** the Generator/Reflector/Curator triad. Reframes memory as
  an *evolving playbook* of strategies. Key insight: "context collapse" and
  "brevity bias" are real failure modes when you let an LLM iteratively
  rewrite its own memory; structured preservation prevents them. Reports
  +10.6 % on agent benchmarks, +8.6 % on finance, matches production agents
  on AppWorld with smaller models — without labelled supervision.
- **Credibility:** high — accepted to ICLR 2026, multi-institution
  authorship including Stanford / SambaNova.

---

## Current state of the art — by system

For each: 2-paragraph summary, then 1-line lesson for claude-forge v0.3.

### 1. LangGraph — `BaseStore` and `BaseCheckpointSaver`
**GitHub:** https://github.com/langchain-ai/langgraph (retrieved 2026-05-01).
**Files:** `libs/checkpoint/langgraph/store/base/__init__.py` (BaseStore),
`libs/checkpoint/langgraph/checkpoint/base/__init__.py` (BaseCheckpointSaver),
`libs/checkpoint/langgraph/checkpoint/memory/__init__.py` (InMemorySaver).

Two orthogonal primitives. **BaseCheckpointSaver** persists *graph execution
state* (per-thread): a `Checkpoint` is a TypedDict with `v`, `id`, `ts`,
`channel_values`, `channel_versions`, `versions_seen`, `updated_channels` —
i.e. it captures the LangGraph state-machine snapshot, not user-level
"memory". Methods: `get_tuple`, `put`, `list`, `put_writes`, `delete_thread`.
`InMemorySaver` is the dev/test backend; production uses Postgres or
SQLite-backed savers. **BaseStore** is the cross-thread memory abstraction:
hierarchical namespaces (`tuple[str, ...]`), key-value items, optional TTL
(`supports_ttl`, `TTLConfig`), and optional embedding-backed semantic search
via `IndexConfig(dims, embed, fields)`. Operations are `get`, `put`,
`search` (with `query`/`filter`/`limit`), `delete`, `list_namespaces`, plus
batch (`Op`/`Result`) and async (`abatch`/`aget`/...) variants.

**Lesson:** *separate execution-state checkpointing from semantic memory —
two different lifecycles, two different APIs.* The `(namespace, key, value,
index)` shape with optional `IndexConfig` is the cleanest cross-thread
memory primitive published; copy it.

### 2. OpenAI Responses + Conversations API (post-Assistants deprecation)
**Doc:** https://developers.openai.com/api/docs/assistants/migration
(retrieved 2026-05-01). Assistants API beta sunset 2026-08-26.

Threads are gone. The replacement is two cooperating endpoints:
**Conversations** (`openai.conversations.create(items=[...], metadata={...})`)
stores arbitrary items — messages, tool calls, tool outputs — server-side,
plus user metadata; and **Responses** (`responses.create(...,
conversation="conv_id")`) which references the conversation by ID and
auto-attaches its history. The fundamental change is that the *conversation
is no longer a message list*; it's a typed-item log. Tool calls and outputs
are first-class, so tool-using agents don't need parallel state.

The published docs do not specify a hard token / item limit on a
conversation; in practice the limit is the model's context window times the
provider's per-turn budget. There is no built-in semantic-search retrieval
on Conversation history — it's append-only state. *adversary: please verify
size limits via the official rate-limit page.*

**Lesson:** *items, not messages.* If we want forward-compatibility with
OpenAI-style server state, our memory log should be heterogeneous-typed
events, not a chat transcript.

### 3. Anthropic Memory tool (`memory_20250818`)
**Doc:** https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
(retrieved 2026-05-01).

A **client-side, file-based** tool. Anthropic ships only the protocol:
the tool exposes `view`, `create`, `str_replace`, `insert`, `delete`,
`rename` over a `/memories` virtual directory. The host application
implements the backend (filesystem, S3, SQLite, encrypted store —
Anthropic's choice). The model is steered by a system-prompt
preamble: *"IMPORTANT: ALWAYS VIEW YOUR MEMORY DIRECTORY BEFORE DOING
ANYTHING ELSE … ASSUME INTERRUPTION: your context window might be reset
at any moment, so you risk losing any progress that is not recorded."*
SDK helpers: `BetaAbstractMemoryTool` (Python),
`betaMemoryTool` (TypeScript).

The pairing model is explicit and important: memory + **context editing**
(client-side eviction of old tool results) + **compaction** (server-side
summarisation when nearing context limit). The Anthropic blog
"Effective harnesses for long-running agents" (2025-11-26) recommends an
**initializer session** that lays down `claude-progress.txt`, a
JSON feature checklist, and an `init.sh` script before any task work
begins; subsequent sessions read those files first.

**Lesson:** *the tool is a verb, not a noun.* Anthropic publishes the API
contract (`view`/`create`/`str_replace`/...) and lets the user own the
storage. claude-forge v0.3 should expose the same contract — file-shaped,
path-namespaced, edit-in-place — and let the harness decide where the
bytes live.

### 4. MCP knowledge-graph memory server (`@modelcontextprotocol/server-memory`)
**GitHub:** https://github.com/modelcontextprotocol/servers/tree/main/src/memory
(retrieved 2026-05-01).

Reference MCP server published by the Anthropic-led MCP project. Storage is
a single JSONL file (default `memory.jsonl`, configurable via
`MEMORY_FILE_PATH`). Schema: **entities** (name, type, observations[]),
**relations** (from-entity, to-entity, predicate phrased in active voice),
**observations** (atomic strings attached to an entity). Tools exposed:
`create_entities`, `create_relations`, `add_observations`,
`delete_entities`, `delete_observations`, `delete_relations`, `read_graph`,
`search_nodes`, `open_nodes`. Search is substring/keyword over the JSONL,
not embedding-backed.

This is the *least-effort* persistent memory: one file, append-only, easy
to inspect, trivially version-controlled. It's the de-facto baseline that
Cursor and Continue.dev users plug in when they want cross-session memory.

**Lesson:** *triples (entity, relation, entity) + observations is enough
for a v0 if you don't need fuzzy retrieval.* But you'll outgrow JSONL
substring search within weeks of real use.

### 5. Cursor — `.cursor/rules/*.mdc` (no native memory)
**Doc:** https://cursor.com/docs/rules (retrieved 2026-05-01).

Cursor explicitly does **not** ship a native cross-session memory. Their
docs say plainly: "large language models don't retain memory between
completions." The closest primitive is **Project Rules**: `.mdc`
markdown-with-frontmatter files in `.cursor/rules/`, version-controlled
with the codebase. Frontmatter fields: `description` (used for intelligent
auto-attach), `globs` (auto-attach when matching files are in context),
`alwaysApply` (universal). The legacy `.cursorrules` file is superseded.
Rules are *prepended to the system prompt*, not stored as retrievable
memory.

For actual memory, the Cursor community has converged on the **MCP-memory
plug-in pattern**: install `@itseasy21/mcp-knowledge-graph` (a fork of the
official MCP memory server with configurable path), plus the Basic Memory
or Memory Bank MCP servers for richer workflows. Some users adopt a
`.brain/` folder with `MEMORY.md`, `SESSION.md`, `LOG.md` and bootstrap it
via `.cursorrules`. So Cursor's "memory" is *bring-your-own-MCP*.

**Lesson:** *do not conflate rules with memory.* Rules are
deterministic, version-controlled instructions; memory is mutable, agent-
written state. Mixing them is what produced the .cursorrules → memory-bank
DIY explosion.

### 6. Continue.dev — `.continue/rules` (rules + optional MCP memory)
**Doc:** https://docs.continue.dev/customize/deep-dives/rules,
https://continue.dev/continuedev/rules-memory (retrieved 2026-05-01).
**Hub entry:** continuedev/rules-memory (published 2025-03-19, 7 stars).

Same architecture as Cursor: rules live in `.continue/rules/*.md` (with
YAML frontmatter `name`, `globs`, `regex`, `description`, `alwaysApply`),
plus a global `~/.continue/rules` and Hub-managed rules. There is **no
native memory feature**; the Hub publishes a "rules-memory" pack — a set of
prompt-rules that *teach the agent how to use the MCP memory server*. So
Continue.dev's official answer to memory is: install MCP memory server,
install the rules pack, done.

Open issue continuedev/continue#4615 ("Feature Request: Implement Memory
Bank for Enhanced Context Management") shows the community is asking for a
Cline-style native memory bank — as of retrieval still open.

**Lesson:** *if your IDE-agent's memory story is "install an MCP server,"
your IDE-agent doesn't have a memory story.* Native is better than BYO for
defaults.

### 7. Cline — Memory Bank (six markdown files)
**Doc:** https://docs.cline.bot/features/memory-bank,
https://github.com/cline/cline/blob/main/docs/prompting/cline-memory-bank.mdx
(retrieved 2026-05-01).
**Source:** `src/core/context/context-management/ContextManager.ts`,
`context-error-handling.ts`, `context-window-utils.ts`,
`src/core/context/context-tracking/` (FileContextTracker,
ModelContextTracker).

Cline's Memory Bank is a six-file convention in `memory-bank/` at repo
root: `projectbrief.md` (foundational requirements), `productContext.md`
(why the project exists), `activeContext.md` (current focus, recent
changes — updates most often), `systemPatterns.md` (architecture/design),
`techContext.md` (stack/setup/constraints), `progress.md` (what works,
what's left, known issues). Cline reads all six at the start of every
task via custom-instruction-injected prompt. Files are plain markdown,
human-editable, git-trackable.

The *runtime* layer is in `src/core/context/`: `ContextManager` does
dynamic conversation-history manipulation (`save`/`load` of
`contextHistoryUpdates` to/from disk), `FileContextTracker` watches for
external file changes mid-session, `ModelContextTracker` logs historical
context. The "[DUPLICATE FILE READ]" optimisation — when the same file is
read twice, the second read is replaced with a tombstone — is also in
ContextManager. Plus the `new_task` tool which intentionally ends a
session and starts a fresh one when context approaches ~50 % usage.

**Lesson:** *human-editable markdown beats opaque DB.* The six-file
hierarchy (`projectbrief → productContext → activeContext → progress`) is
the most copyable IDE-agent memory schema published. The duplicate-file-
read tombstone is a free 10 %+ context win.

### 8. AutoGen — `Memory` protocol (`autogen_core.memory`)
**GitHub:** https://github.com/microsoft/autogen
(retrieved 2026-05-01).
**File:** `python/packages/autogen-core/src/autogen_core/memory/_base_memory.py`,
`_list_memory.py`.

Microsoft AutoGen exposes a minimalist async protocol: five methods on
the `Memory` ABC — `update_context(ChatCompletionContext) ->
UpdateContextResult`, `query(query) -> MemoryQueryResult`, `add(content)`,
`clear()`, `close()`. Data model: `MemoryContent` (Pydantic) with
`content` (str | bytes | dict | Image), `mime_type`, optional `metadata`;
`MemoryQueryResult` wraps `results: List[MemoryContent]`. Concrete impl:
`ListMemory` — naïve list, no retrieval scoring.

The notable design choice is `update_context`: instead of forcing the agent
to query memory and decide what to inject, the memory implementation owns
the rewriting of `ChatCompletionContext` itself. This is a tighter
coupling than the LangGraph `BaseStore` model and arguably better for
predictable behaviour, worse for composability.

**Lesson:** *`update_context` is more principled than `query` + manual
splice.* But the trade-off is opacity to the agent; pick depending on
whether the agent should reason about its own retrieval.

### 9. Letta (formerly MemGPT) — three-tier OS-style memory
**GitHub:** https://github.com/letta-ai/letta (retrieved 2026-05-01).
**Source:** `letta/agent.py` uses `Memory` class with editable `Block`s
(`get_block(label)`, `list_block_labels()`, `compile()`); persistence via
`BlockManager.update_block()`.
**Tool sigs:** `letta/functions/function_sets/base.py` —
`core_memory_append(agent_state, label, content)`,
`core_memory_replace(agent_state, label, old_content, new_content)`,
`archival_memory_insert(self, content, tags=None)`,
`archival_memory_search(self, query, tags=None, tag_match_mode='any',
top_k, start_datetime, end_datetime)`.

The original MemGPT three-tier model, productised: **Core Memory** (small
labelled blocks always in context — agent edits via
`core_memory_append`/`core_memory_replace`), **Recall Memory** (full
conversation history searchable by tool call), **Archival Memory**
(unbounded long-term store backed by a vector DB, queried with
`archival_memory_search`, optionally filtered by tags and date range).
Read-only blocks (`block.read_only`) are protected. The Letta v1
architecture (2025) deprecated the old MemGPT-style heartbeat/send_message
pattern in favour of native reasoning tokens, but kept the three-tier
memory exactly.

**Lesson:** *labelled blocks the agent can edit > opaque store the agent
queries.* The `core_memory_append` / `core_memory_replace` pair is the
cleanest "let the agent rewrite its own working memory" API in production.
Note: the *date-filtered* `archival_memory_search` (start_datetime/
end_datetime) is unusual and worth copying — most systems only filter by
tag or namespace.

### 10. Anthropic harness pattern — initializer + progress file
**Source:** https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents
(published 2025-11-26, retrieved 2026-05-01).

Not a system per se, but the prescribed *protocol* for using the memory
tool. An **initializer session** (different prompt, runs once) bootstraps
three artifacts: an `init.sh` development script, a `claude-progress.txt`
log, and a JSON feature-checklist with hundreds of items in `failing`
state. **Subsequent coding sessions** read those artifacts first, work on
*one feature at a time*, and only flip a feature to `passing` after
end-to-end verification. The progress log is updated before session-end.

The crucial idea: **memory is bootstrapped, not improvised**. The first
session's job is to lay down structured recovery artifacts; later sessions
inherit them. This is the same pattern Cline's Memory Bank instantiates,
but formalised by Anthropic.

**Lesson:** *bake an initializer pass into the harness.* Don't wait for
the agent to discover that it needs a progress file — give it one on day
0, and bias the prompt toward verification-before-completion.

---

## Dissenting voices

- **Cursor** explicitly refuses to build native memory — "LLMs don't retain
  memory between completions" is their on-record framing. The implication
  is that all "memory" should be either (a) reproducible from version-
  controlled rules or (b) externalised to MCP. This is a *minority*
  position; every other system surveyed has built-in memory.
- **ACE (arXiv:2510.04618)** argues that most "memory rewriting" pipelines
  suffer from **context collapse** and **brevity bias** — when you let an
  LLM iteratively summarise its own memory, detail erodes monotonically
  until the memory is useless. Their generator/reflector/curator split is
  explicitly a structural defence. This is a counter to the
  Letta/MemGPT-style "agent edits its own memory" school: yes the agent
  can, but unconstrained self-editing is a known failure mode.
- **MemGPT/Letta** treats memory as agent-controlled (the agent decides
  what to write/evict). **AutoGen** and **LangGraph** treat memory as
  framework-controlled (`update_context` mutates the prompt without the
  agent's involvement). These philosophies are incompatible: the first
  optimises for emergent autonomy, the second for predictability.

---

## Issue-tracker findings

- **continuedev/continue#4615** — "Feature Request: Implement Memory Bank
  for Enhanced Context Management" — open as of retrieval. Community is
  asking for a Cline-style native memory bank in Continue.dev; maintainers'
  current answer is the rules-memory hub pack + MCP memory server.
- **forum.cursor.com / "Add Persistent Memory in Cursor"
  (thread 57497)** — long-running community thread tracking the BYO-MCP
  memory pattern; multiple competing solutions (Basic Memory,
  itseasy21/mcp-knowledge-graph, mindlink/.brain, Hopsule).
- **Anthropic Memory tool docs** — note the SDK helpers
  (`BetaAbstractMemoryTool` Python, `betaMemoryTool` TS) and the path
  traversal warning are explicit; this matters for claude-forge if we
  expose paths to the agent.

*adversary: please verify the Mem0 benchmark numbers against an
independent re-evaluation; vendor-self-reported.*

---

## Synthesis

### Three patterns the field has converged on

1. **Two-tier separation: ephemeral execution state vs. durable semantic
   memory.** LangGraph (BaseCheckpointSaver vs. BaseStore), Letta
   (core+recall vs. archival), Cline (ContextManager runtime vs.
   memory-bank/*.md), Anthropic (compaction vs. memory tool) — all four
   independently arrive at the same bifurcation. The lifecycles are
   different (execution state is invalidated by graph changes; semantic
   memory persists), so the storage must be different.

2. **File-shaped, path-namespaced, human-readable storage.** Anthropic's
   memory tool exposes `/memories/<path>`; Cline's memory bank is six
   markdown files; LangGraph's `BaseStore` uses `tuple[str, ...]`
   namespaces; Cursor's rules use `.cursor/rules/*.mdc`. JSONL +
   knowledge-graph (MCP server) is the JSON dialect of the same idea.
   Nobody who succeeded shipped opaque-binary memory. The principle is
   *the human and the agent must both be able to read and edit the
   store*.

3. **Generative-Agents retrieval triple is the de-facto scoring baseline.**
   `recency·importance·relevance`, with recency = exponential decay,
   importance = LLM-rated 1–10 at insert, relevance = embedding cosine.
   Mem0, A-MEM, AWS Bedrock AgentCore, and most "build your own agent
   memory" tutorials all start from this formula and tweak weights.

### Two patterns that have failed (and why)

1. **Pure conversation-as-memory (early Assistants API threads).** OpenAI
   built a thread = ordered list of messages, no item types, no metadata,
   no retrieval — and is now deprecating it (Aug 2026 sunset) in favour
   of typed Conversations + Responses. Reason: an LLM agent emits
   tool-calls and tool-outputs that are not messages; pretending they are
   forces ugly serialisation and breaks tool-history reasoning.

2. **Unconstrained agent self-editing of memory (early MemGPT).** Letta
   kept the three-tier shape but added read-only blocks and a curator
   layer; ACE diagnoses the failure mode formally as "context collapse"
   and "brevity bias." If you let the agent be the only writer with no
   structural guardrails, memory monotonically degrades. Every successful
   system has *some* curator (Generative Agents' importance-rating LLM,
   ACE's curator agent, Cline's six-file schema, Anthropic's "edit
   in-place, don't sprawl" prompting guidance).

### One most copyable design for claude-forge v0.3

**Anthropic's memory tool contract on top of a Cline-style six-file
schema, with a Generative-Agents retrieval scorer for any optional
embedding-backed sub-store.**

Concretely:
- Expose the six commands `view`/`create`/`str_replace`/`insert`/
  `delete`/`rename` over a `/memories` virtual directory (Anthropic
  tool contract — already standardised, SDK helpers exist).
- Bootstrap the directory with a six-file schema like Cline's
  (`projectbrief.md`, `productContext.md`, `activeContext.md`,
  `systemPatterns.md`, `techContext.md`, `progress.md`) via an
  initializer pass (Anthropic harness pattern).
- For richer retrieval over arbitrary notes, add an optional
  `BaseStore`-style sub-namespace (LangGraph) with the
  `recency·importance·relevance` scorer (Generative Agents) and
  per-namespace TTL.
- Avoid: opaque DB-only storage (Cursor users hate it), unconstrained
  agent self-editing without guardrails (ACE failure modes), and treating
  memory as a message log (deprecated by OpenAI).

The single lesson: **the agent should write memory through verbs the user
can audit (view/create/edit), into files the user can read, with a curator
constraint somewhere in the loop.**

---

## Comparison matrix

| System              | Storage backend        | Retrieval method               | Scope (per-thread / cross-thread / cross-project) | Decay / eviction policy             | Integration pattern                       | License           |
|---------------------|------------------------|--------------------------------|---------------------------------------------------|-------------------------------------|-------------------------------------------|-------------------|
| LangGraph BaseStore | pluggable (in-mem, Postgres, custom) | namespace prefix + filter, optional embedding via IndexConfig | cross-thread (namespace tuples) | TTL via `TTLConfig`/`supports_ttl`  | `BaseStore.put/get/search`, async batch    | MIT               |
| LangGraph Checkpoint | pluggable (in-mem, Postgres, SQLite) | per-thread sequential, by `id` | per-thread                       | manual `delete_thread`              | `BaseCheckpointSaver.put/get_tuple/list`   | MIT               |
| OpenAI Conversations | OpenAI server-side     | none built-in (append-only items) | per-conversation                  | none documented                     | `responses.create(conversation="...")`     | proprietary       |
| Anthropic Memory    | client-side (host's choice — FS/DB/S3) | model-driven `view`+grep, no built-in semantic | per-organization (host-namespaced) | host-implemented; docs suggest size cap + LRU eviction | `memory_20250818` tool, BetaAbstractMemoryTool helper | proprietary API; SDK MIT |
| MCP memory server   | local JSONL file (`memory.jsonl`) | substring search over entity/observation strings | cross-project (single file)       | none (manual delete)                | MCP tools `create_entities`, `search_nodes`, … | MIT               |
| Cursor rules        | `.cursor/rules/*.mdc` (filesystem, git) | glob-match + LLM relevance (description) | per-project                       | none (manual delete; git-versioned) | system-prompt prepend (no retrieval call) | proprietary IDE   |
| Continue rules      | `.continue/rules/*.md` (filesystem, git, + Hub)  | glob/regex + LLM relevance     | per-project + global + Hub        | none (manual)                       | system-prompt prepend; memory via MCP add-on | Apache-2.0       |
| Cline Memory Bank   | `memory-bank/*.md` (filesystem, git)  | full read at session start (6 files) + ContextManager dedup | per-project                       | session-end progress update; `new_task` resets context | custom-instruction-driven, ContextManager runtime | Apache-2.0       |
| AutoGen Memory      | impl-defined (ListMemory in core)     | `query()` + `update_context()` rewrites prompt | per-agent                         | impl-defined; `clear()` available    | `Memory` ABC, async                        | MIT               |
| Letta core+recall+archival | Postgres + vector DB (default) | tag/date filter + vector search (`archival_memory_search`) | per-agent (blocks) + per-org (archival) | block-level read-only flags; explicit `archival_memory_insert` | tool calls (`core_memory_append`, …)        | Apache-2.0        |
| Generative Agents (ref impl) | per-agent memory stream (in-mem + JSON dump) | `α·recency + β·importance + γ·relevance`, equal-weight, min-max | per-agent                         | exponential-decay recency (factor 0.995) | embedded in agent loop                     | MIT (replications) |
| Mem0                | Postgres + vector + optional graph DB | LLM-extracted facts, vector + graph hop | per-user / per-app                | LLM-curated consolidation            | SDK + REST                                 | Apache-2.0        |
| A-MEM               | ChromaDB + linked notes                | structured note attrs + semantic + bi-directional links | per-agent                         | "memory evolution" — notes update on insert | research code (agiresearch/A-mem)         | Apache-2.0        |

---

## Confidence

**High** for primitives, file paths, and class names of LangGraph,
Letta, Anthropic Memory tool, MCP memory server, Cline, Cursor, Continue.
**Medium** for Mem0 (vendor-self-reported benchmarks) and OpenAI
Conversations size limits (not documented in the migration page; would
need separate verification against the rate-limit reference).
**High** for academic citations — all paper IDs verified against arXiv
abstracts on 2026-05-01.

The single weakest link is Mem0's quantitative claims; flagged for
adversary review.
