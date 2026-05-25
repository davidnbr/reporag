# codebrain — Local RAG MCP Server

Fully local, zero-cost RAG knowledge layer for AI coding tools.
No SaaS. No pricing tiers. All computation on your machine.

Works with **agy**, **Claude Code**, **Cursor**, and any MCP client.

## Architecture

```
AI tool (agy / Claude Code / Cursor)
       │
       ▼ MCP stdio
codebrain MCP server
       ├── tree-sitter → AST chunks + semantic text
       ├── SCIP CLIs  → compiler-grade dependency graph (+ heuristic fallback)
       ├── LanceDB    → dense vector search (nomic-embed-text-v1, local)
       ├── BM25       → sparse retrieval (k1=1.2, b=0.75)
       ├── RRF k=60   → hybrid fusion
       ├── Reverse PPR → architectural hub ranking
       ├── Cross-encoder → reranking
       └── SQLite FTS5 → persistent memory store
```

## Install

```bash
cd ~/Projects/codebrain
pip install -e ".[dev]"
```

### Optional: SCIP CLIs (compiler-grade graph, higher accuracy)

```bash
# Python
pip install scip-python

# JS/TS
npm install -g @sourcegraph/scip-typescript

# Go
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest

# Rust: use rust-analyzer (already ships with rustup)
```

Without SCIP CLIs installed, codebrain falls back to tree-sitter import heuristics (~70% graph accuracy). All other features are unaffected.

## Configure with agy / Claude Code

Add to your MCP config (`~/.config/agy/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "codebrain": {
      "command": "rag-mcp",
      "args": []
    }
  }
}
```

## Tools

### `index_codebase`
Parse and index your project.

```json
{ "path": "/path/to/your/project", "incremental": true }
```

### `query_code`
Retrieve relevant code context via hybrid RAG.

```json
{ "query": "how does authentication work", "k": 10 }
```

### `get_symbol`
Exact symbol lookup.

```json
{ "name": "UserController", "language": "python" }
```

### `remember`
Store a decision or discovery persistently.

```json
{ "content": "Use JWT with 15min expiry for auth tokens", "tags": ["auth", "jwt"], "category": "decision" }
```

### `recall`
Search stored knowledge.

```json
{ "query": "auth token decisions", "tags": ["auth"] }
```

## Configuration

Default config: `~/.config/rag-mcp/config.json`

```json
{
  "embed_model": "nomic-ai/nomic-embed-text-v1",
  "embed_backend": "sentence-transformers",
  "data_dir": ".rag-mcp",
  "reranker_k": 15,
  "bm25_k1": 1.2,
  "bm25_b": 0.75,
  "rrf_k": 60,
  "ppr_alpha": 0.85
}
```

Override via env vars: `RAG_MCP_EMBED_MODEL`, `RAG_MCP_DATA_DIR`, etc.

## Run tests

```bash
pytest tests/ -v
```
