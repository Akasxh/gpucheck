# historian-memory-prior-art summary (W2)

## 3 patterns the field has converged on

1. **Two-tier separation: ephemeral execution state vs durable semantic memory.** LangGraph (`BaseCheckpointSaver` vs `BaseStore`), Letta (core/recall vs archival), Cline (`ContextManager` vs `memory-bank/*.md`), Anthropic (compaction vs memory tool) — all bifurcate. Different lifecycles → different storage.
2. **File-shaped, path-namespaced, human-readable storage.** Anthropic's `/memories/<path>`, Cline's six markdown files, LangGraph's `tuple[str, ...]` namespaces, MCP server's JSONL — nobody who succeeded shipped opaque-binary memory.
3. **Generative-Agents retrieval triple is de-facto baseline.** `recency · importance · relevance` (exp-decay 0.995, LLM-rated 1-10, embedding cosine, min-max equal-weight). Reused by Mem0, A-MEM, Bedrock AgentCore.

## 2 patterns that have failed

1. **Conversation-as-memory** — OpenAI's original Threads model. Deprecated 2026-08-26 because tool-calls and tool-outputs aren't messages. Replacement uses typed `items`.
2. **Unconstrained agent self-editing** — early MemGPT. ACE (ICLR 2026, arXiv 2510.04618) formalises the failure as **"context collapse"** and **"brevity bias"**: let an LLM iteratively rewrite its own memory and detail erodes monotonically. Every successful system has a curator constraint.

## Most copyable design for claude-forge v0.3

**Anthropic's memory-tool contract** (`view`/`create`/`str_replace`/`insert`/`delete`/`rename` over `/memories`) **layered over a Cline-style six-file schema** (`projectbrief / productContext / activeContext / systemPatterns / techContext / progress`), bootstrapped by an Anthropic-style initializer pass. Add an optional `BaseStore`-style namespaced sub-store with the Generative-Agents scorer for fuzzy notes. **Avoid**: opaque DBs, unconstrained self-edit, message-log memory.

## Sources surveyed

10 systems (LangGraph, Letta, Cline, Anthropic memory-tool, Cursor, AutoGen, Continue.dev, Bedrock AgentCore, Mem0, MemGPT) + 5 academic papers (ACE, Generative Agents, MemGPT, A-MEM, Letta paper). Mem0 benchmarks flagged for adversary review.
