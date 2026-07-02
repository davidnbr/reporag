# reporag — Local RAG MCP Server

**Fully local, zero-cost RAG knowledge layer for AI coding tools.**
No SaaS. No API keys. No pricing tiers. All computation on your machine.

Works with **agy (antigravity)**, **Claude Code**, **Cursor**, **Codex CLI**, and any MCP-compatible client.

---

## Why reporag

Your AI assistant doesn't know your codebase. It hallucinates APIs, reinvents functions that already exist, and guesses at code it has never seen. reporag fixes this by giving the model real, retrieved context from *your* repo — locally.

**The headline result:** on private or large codebases the model has never seen, reporag delivers a **+71–81% answer-quality improvement** — turning unusable baseline answers (1.1–1.3 / 5) into competent ones (3.8–4.2 / 5).

| Codebase           | Files | Baseline |  RAG | Composite Δ |
| ------------------ | ----: | -------: | ---: | ----------: |
| private Go project |    74 |     2.24 | 3.84 |  **+71.3%** |
| Django             | 2,955 |     2.35 | 4.25 |  **+81.1%** |

_Claude-as-judge, baseline vs. RAG-injected context, scored on correctness, completeness, and hallucination avoidance (1–5). The less the model knows from training, the larger the gain._

![LLM response quality: baseline vs RAG](https://github.com/user-attachments/assets/f338658b-c62c-4314-aad9-361736c98cf2)

→ Full methodology, retrieval-recall tables, and anti-duplication ("discovery") results in **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)**.

---

## Quick start

**Prerequisite:** [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — single binary, no Python needed upfront.

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Configure your client(s) — writes MCP config + proactive-use rules
uvx --from "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git" \
    reporag setup --client all
```

Restart your client. That's it — the server launches automatically via `uvx`, indexes on first use, and retrieves context before the model answers.

**Set up one tool at a time** instead of `--client all`:

```bash
reporag setup --client claude   # Claude Code  (installs hooks; register the server with `claude mcp add` — see docs/CLIENTS.md)
reporag setup --client cursor   # Cursor       (~/.cursor/mcp.json + rules)
reporag setup --client codex    # Codex CLI    (~/.codex/config.toml)
```

> agy (antigravity) uses the same MCP config format as Claude Code — see [docs/CLIENTS.md](docs/CLIENTS.md).
> Use `reporag[ml]` instead of `[ml-cpu]` for CUDA-accelerated embedding on an NVIDIA GPU.

→ Per-client manual config, hooks, and Cursor/Codex details in **[docs/CLIENTS.md](docs/CLIENTS.md)**.

---

## Tools

| Tool             | What it does                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| `index_codebase` | Parse, embed, and graph-index a project. Run once, then incrementally on changes. |
| `query_code`     | Hybrid RAG retrieval (dense + BM25 + RRF + PPR + optional rerank).                 |
| `find_existing`  | Pre-implementation discovery — surface code that already does the task.            |
| `get_symbol`     | Exact symbol lookup by name.                                                       |
| `remember` / `recall` | Persistent knowledge store across sessions.                                   |

<details>
<summary>Example calls</summary>

```jsonc
// index_codebase
{ "path": "/path/to/project", "incremental": true, "languages": ["go", "python"] }

// query_code — use `project` to scope when multiple repos are indexed
{ "query": "how does authentication work", "k": 10, "project": "/path/to/project" }

// find_existing — call BEFORE writing new code to avoid reimplementing
{ "task": "validate API error codes and map them to user messages", "project": "/path/to/project" }

// get_symbol
{ "name": "UserController", "language": "python" }

// remember / recall
{ "content": "Use JWT with 15min expiry for auth", "tags": ["auth"], "category": "decision" }
{ "query": "auth token decisions", "tags": ["auth"] }
```

`find_existing` runs the same hybrid pipeline as `query_code` (no reranker), deduplicated by file (max 2/file), with a `reuse_hint` per result. It prevents the "model reimplements logic that already exists" failure mode — see [discovery benchmarks](docs/BENCHMARKS.md#3-discovery-mode-anti-duplication).

</details>

---

## How it works

```
AI tool (agy / Claude Code / Cursor / Codex)
       │
       ▼ MCP stdio
reporag bridge          → thin stdio↔HTTP proxy, one per client (loads no ML)
       │
       ▼ streamable HTTP (127.0.0.1:7800, loopback only)
reporag daemon          → ONE per machine; auto-spawned, deduplicated, idle-shutdown
       ├── tree-sitter   → AST chunks + semantic text (per-language)
       ├── SCIP CLIs     → compiler-grade dependency graph (+ heuristic fallback)
       ├── LanceDB       → dense vector search (nomic-embed-text-v1, 768-dim, local)
       ├── BM25          → sparse retrieval (k1=1.2, b=0.75)
       ├── RRF k=60      → hybrid fusion
       ├── Reverse PPR   → architectural hub ranking
       ├── Cross-encoder → reranking (bge-reranker-base, local)
       └── SQLite FTS5   → persistent memory store
```

Every client still launches `reporag` the same way (via `uvx`), but each launch is now
a lightweight **bridge** that connects to a single shared **daemon**. The first bridge
starts the daemon; the rest reuse it — so N editor windows share **one** embedding model
and **serialized** indexing instead of N copies competing for CPU/GPU. The daemon binds
loopback only, and shuts itself down after 15 min idle (configurable). Set
`REPORAG_NO_DAEMON=1` to revert to the classic self-contained per-client server.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++.

---

## Documentation

| Doc                                          | Contents                                                          |
| -------------------------------------------- | ---------------------------------------------------------------- |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | Full benchmarks: retrieval recall, LLM quality, discovery mode.  |
| **[docs/CLIENTS.md](docs/CLIENTS.md)**       | Per-client config, hooks, Cursor rules, Codex managed block.     |
| **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)** | Config file, env vars, install variants, SCIP, development. |
| [docs/codebase_rag_research.md](docs/codebase_rag_research.md) | Architectural & mathematical foundations.        |

## License

See [LICENSE](LICENSE).
