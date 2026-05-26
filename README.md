# codebrain — Local RAG MCP Server

Fully local, zero-cost RAG knowledge layer for AI coding tools.
No SaaS. No pricing tiers. All computation on your machine.

Works with **agy (antigravity)**, **Claude Code**, **Cursor**, and any MCP-compatible client.

## Architecture

```
AI tool (agy / Claude Code / Cursor)
       │
       ▼ MCP stdio
codebrain MCP server
       ├── tree-sitter  → AST chunks + semantic text (per-language)
       ├── SCIP CLIs    → compiler-grade dependency graph (+ heuristic fallback)
       ├── LanceDB      → dense vector search (nomic-embed-text-v1, 768-dim, local)
       ├── BM25         → sparse retrieval (k1=1.2, b=0.75)
       ├── RRF k=60     → hybrid fusion
       ├── Reverse PPR  → architectural hub ranking
       ├── Cross-encoder → reranking (ms-marco-MiniLM-L-6-v2, local)
       └── SQLite FTS5  → persistent memory store
```

## Benchmarks

Synthetic golden set: 40 named functions/classes from the live index.
Query template: `"implementation of {name}"` / `"how does {name} work"`.
Metric: fraction of queries where the exact chunk appears in top-k (Recall@k), plus MRR@k.

| Stage      | Recall@5 | Recall@10 | MRR@10 | ms/query |
|------------|----------|-----------|--------|----------|
| dense      | 0.875    | 0.900     | 0.791  | 6        |
| bm25       | 0.875    | 0.875     | 0.765  | <1       |
| rrf        | 0.875    | 0.925     | 0.778  | 6        |
| rrf + ppr  | 0.875    | 0.925     | 0.768  | 8        |
| **full**   | **0.900**| **0.900** |**0.814**| 430     |

`full` = RRF fusion → Reverse PPR hub re-ranking → cross-encoder rerank.
Reranker cost (~430 ms) trades latency for top-1 precision; disable with `rerank=false` in `query_code`.

Run your own ablation:

```bash
devenv shell -- python scripts/benchmark.py --samples 100 --k 5 10
# filter to a single project:
devenv shell -- python scripts/benchmark.py --project /path/to/project --stages dense rrf full
# functions only, quiet output:
devenv shell -- python scripts/benchmark.py --filter-chunk-types function --quiet
```

## Install (any machine)

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — single binary, no Python required upfront.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

MCP clients then launch the server automatically via `uvx`. No manual install step.

## Configure

### Claude Code (`~/.claude/.mcp.json`)

```json
{
  "mcpServers": {
    "rag-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "rag-mcp[ml,treesitter] @ git+https://github.com/davidnbr/codebrain.git",
        "rag-mcp"
      ],
      "env": {
        "RAG_MCP_DATA_DIR": "~/.local/share/rag-mcp"
      }
    }
  }
}
```

### agy / antigravity (`~/.gemini/antigravity/mcp_config.json`)

Same format as above.

### Any other MCP client

```json
{
  "command": "uvx",
  "args": ["--from", "rag-mcp[ml,treesitter] @ git+https://github.com/davidnbr/codebrain.git", "rag-mcp"],
  "env": { "RAG_MCP_DATA_DIR": "~/.local/share/rag-mcp" }
}
```

## Tools

### `index_codebase`
Parse, embed, and graph-index a project. Run once, then incrementally on changes.

```json
{ "path": "/path/to/project", "incremental": true, "languages": ["go", "python"] }
```

### `query_code`
Hybrid RAG retrieval: dense + BM25 + RRF + PPR + cross-encoder rerank.

```json
{ "query": "how does authentication work", "k": 10, "project": "/path/to/project" }
```

Use `project` to restrict results to a single codebase when multiple are indexed.

### `get_symbol`
Exact symbol lookup by name.

```json
{ "name": "UserController", "language": "python" }
```

### `remember` / `recall`
Persistent knowledge store across sessions.

```json
{ "content": "Use JWT with 15min expiry for auth", "tags": ["auth"], "category": "decision" }
{ "query": "auth token decisions", "tags": ["auth"] }
```

## Configuration

Config file: `~/.config/rag-mcp/config.json` (optional — all fields have defaults)

```json
{
  "embed_model": "nomic-ai/nomic-embed-text-v1",
  "embed_backend": "sentence-transformers",
  "data_dir": "~/.local/share/rag-mcp",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "reranker_k": 50,
  "bm25_k1": 1.2,
  "bm25_b": 0.75,
  "rrf_k": 60,
  "ppr_alpha": 0.85,
  "ppr_seed_k": 20,
  "dense_candidates": 50,
  "sparse_candidates": 50,
  "snippet_chars": 600
}
```

### Environment variable overrides

| Variable | Field |
|----------|-------|
| `RAG_MCP_DATA_DIR` | `data_dir` |
| `RAG_MCP_EMBED_MODEL` | `embed_model` |
| `RAG_MCP_EMBED_BACKEND` | `embed_backend` |
| `RAG_MCP_RERANKER_K` | `reranker_k` |
| `RAG_MCP_BM25_K1` | `bm25_k1` |
| `RAG_MCP_BM25_B` | `bm25_b` |
| `RAG_MCP_RRF_K` | `rrf_k` |
| `RAG_MCP_PPR_ALPHA` | `ppr_alpha` |
| `RAG_MCP_DENSE_CANDIDATES` | `dense_candidates` |
| `RAG_MCP_SPARSE_CANDIDATES` | `sparse_candidates` |
| `RAG_MCP_SNIPPET_CHARS` | `snippet_chars` |

## Optional: SCIP CLIs (compiler-grade dependency graph)

Without these, codebrain uses tree-sitter import heuristics (~70% graph accuracy).

```bash
pip install scip-python                              # Python
npm install -g @sourcegraph/scip-typescript          # JS/TS
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest  # Go
```

## Development

```bash
git clone https://github.com/davidnbr/codebrain
cd codebrain
devenv shell   # Nix-based reproducible env (requires devenv)
# or:
uv sync --extra dev --extra treesitter
pytest tests/ -v
```

## Supported languages

Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
