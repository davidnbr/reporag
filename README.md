# reporag — Local RAG MCP Server

Fully local, zero-cost RAG knowledge layer for AI coding tools.
No SaaS. No pricing tiers. All computation on your machine.

Works with **agy (antigravity)**, **Claude Code**, **Cursor**, and any MCP-compatible client.

## Architecture

```
AI tool (agy / Claude Code / Cursor)
       │
       ▼ MCP stdio
reporag MCP server
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

Synthetic golden set: named functions/classes sampled from the live index.
Query template: `"implementation of {name}"` / `"how does {name} work"`.
Metric: fraction of queries where the exact chunk appears in top-k (Recall@k), plus MRR@k.

### reporag (small, 246 chunks, AST strategy)

| Stage     | Recall@5  | Recall@10 | MRR@10    | ms/query |
| --------- | --------- | --------- | --------- | -------- |
| dense     | 0.875     | 0.900     | 0.791     | 6        |
| bm25      | 0.875     | 0.875     | 0.765     | <1       |
| rrf       | 0.875     | 0.925     | 0.778     | 6        |
| rrf + ppr | 0.875     | 0.925     | 0.768     | 8        |
| **full**  | **0.900** | **0.900** | **0.814** | 430      |

`full` = RRF fusion → Reverse PPR hub re-ranking → cross-encoder rerank.
Reranker cost (~430 ms) trades latency for top-1 precision; disable with `rerank=false` in `query_code`.

### Django (large, 869 files / ~11k chunks, 100 samples)

Validates chunking strategy at scale. PPR requires a dense dependency graph to help — sparse heuristic graphs (857 edges / 11k chunks) show no benefit or slight regression.

| Stage     | Strategy | Recall@5 | Recall@10 | MRR@10 | ms/query |
| --------- | -------- | -------- | --------- | ------ | -------- |
| dense     | ast      | 0.770    | 0.790     | 0.625  | 35       |
| dense     | hybrid   | 0.690    | 0.740     | 0.574  | 37       |
| bm25      | ast      | 0.610    | 0.660     | 0.497  | 1        |
| bm25      | hybrid   | 0.600    | 0.640     | 0.495  | 1        |
| rrf       | ast      | 0.750    | **0.850** | 0.634  | 37       |
| rrf       | hybrid   | 0.670    | 0.780     | 0.566  | 38       |
| rrf + ppr | ast      | 0.750    | 0.840     | 0.617  | 41       |
| rrf + ppr | hybrid   | 0.670    | 0.780     | 0.566  | 43       |

AST consistently outperforms hybrid by 5–7 pp on R@10 for **named-symbol queries** (e.g. "implementation of QuerySet"). Hybrid window chunks dilute the dense vector space when exact AST chunks compete against overlapping windows.

**Benchmark caveat:** both tables measure exact named-chunk recall — structurally biased toward AST. The cited paper (arXiv:2605.04763) where sliding window wins tests _code completion at cursor positions_, not NL→code architectural queries. For queries like "how does the ORM query execution flow work?" or "where is middleware processing defined?", hybrid may outperform AST by providing import context and cross-function windows that AST drops — this is untested.

Choose your strategy:

- `"ast"` (default) — best for symbol lookup, exact function/class retrieval, smaller index, lower latency
- `"hybrid"` — may improve architectural/contextual queries at the cost of 5–7 pp recall regression on symbol lookup and ~2× indexing latency
- `"sliding"` — pure windows, no symbol-level precision

Run your own ablation:

```bash
devenv shell -- python scripts/benchmark.py --samples 100 --k 5 10
# filter to a single project:
devenv shell -- python scripts/benchmark.py --project /path/to/project --stages dense rrf full
# functions only, quiet output:
devenv shell -- python scripts/benchmark.py --filter-chunk-types function --quiet
```

### LLM response quality (downstream impact)

Retrieval recall measures _whether_ the right chunk is returned. This benchmark measures _whether Claude gives a better answer_ when that chunk is injected as context.

**Method:** For each sampled function/class, generate a question (`"How does {name} work?"`), call Claude twice — once with no context (baseline), once with retrieved chunks injected (RAG) — then use Claude as judge to score both responses on correctness, completeness, and hallucination avoidance (1–5 each). Uses `claude -p` (Claude Code CLI, no API key required).

#### Summary

| Codebase         | Files | Chunks | Baseline | RAG  | Composite Δ |
| ---------------- | ----- | ------ | -------- | ---- | ----------- |
| reporag          | 39    | 266    | 3.98     | 4.16 | **+4.4%**   |
| "other_codebase" | 74    | 488    | 2.24     | 3.84 | **+71.3%**  |
| Django           | 2,955 | 45k    | 2.35     | 4.25 | **+81.1%**  |

The less Claude knows about a codebase from training, the larger the RAG gain. reporag uses textbook patterns Claude has seen extensively — baseline is already strong. Django has large implementation-specific surfaces Claude cannot reconstruct from training alone.

#### reporag (39 files, 266 chunks, 17/30 scored)

| Metric        | Baseline | RAG      | Δ (%)     |
| ------------- | -------- | -------- | --------- |
| correctness   | 4.18     | 4.29     | +2.8%     |
| completeness  | 3.94     | 4.24     | +7.5%     |
| hallucination | 3.82     | 3.94     | +3.1%     |
| **composite** | **3.98** | **4.16** | **+4.4%** |

Small improvement: well-known patterns (MCP, Python, RAG). RAG helps most on completeness — actual source code makes answers more thorough.

#### "other_codebase" (74 files, 488 chunks, 30/30 scored)

| Metric        | Baseline | RAG      | Δ (%)      |
| ------------- | -------- | -------- | ---------- |
| correctness   | 1.27     | 3.57     | +181.6%    |
| completeness  | 1.13     | 3.53     | +211.8%    |
| hallucination | 4.33     | 4.43     | +2.3%      |
| **composite** | **2.24** | **3.84** | **+71.3%** |

Baseline near 1 — Claude cannot answer without context. Hallucination high at baseline (Claude hedges rather than inventing). RAG raises quality by ~2.5× on correctness and completeness.

#### Django (2,955 files, 45k chunks, 24/30 scored)

| Metric        | Baseline | RAG      | Δ (%)      |
| ------------- | -------- | -------- | ---------- |
| correctness   | 1.33     | 4.25     | +218.8%    |
| completeness  | 1.17     | 4.21     | +260.7%    |
| hallucination | 4.54     | 4.29     | -5.5%      |
| **composite** | **2.35** | **4.25** | **+81.1%** |

Largest correctness/completeness gains. Hallucination slightly regresses with RAG: injecting 10 chunks causes Claude to synthesize across chunks in ways that diverge from the single ground-truth function. Baseline Claude hedges cleanly; RAG Claude answers but over-extends retrieved context.

**Takeaway:** on private or large codebases where Claude has no training data, reporag delivers **+71–81% composite quality improvement** — turning unusable baselines (1.1–1.3) into competent answers (3.8–4.2).

Run on your own codebase:

```bash
# index first (via MCP tool in Claude Code), then:
devenv shell -- python scripts/llm_eval.py --project /path/to/project --samples 30 --output results.json
```

## Install (any machine)

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — single binary, no Python required upfront.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

MCP clients then launch the server automatically via `uvx`. No manual install step.

Two install variants:

| Extra      | Torch    | First-run download | Use when                         |
| ---------- | -------- | ------------------ | -------------------------------- |
| `[ml]`     | CUDA     | ~2 GB              | NVIDIA GPU available             |
| `[ml-cpu]` | CPU-only | ~250 MB            | No GPU / most developer machines |

`[ml-cpu]` uses `[tool.uv.sources]` to redirect torch to the PyTorch CPU index — **requires uv** (ignored by pip). reporag runs purely on CPU regardless; the embed models fit in RAM and latency is acceptable.

## Configure

### Claude Code (`~/.claude/.mcp.json`)

```json
{
  "mcpServers": {
    "reporag": {
      "command": "uvx",
      "args": [
        "--from",
        "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
        "reporag"
      ],
      "env": {
        "REPORAG_DATA_DIR": "~/.local/share/reporag"
      }
    }
  }
}
```

Use `reporag[ml]` instead if you have an NVIDIA GPU and want CUDA-accelerated embedding.

### agy / antigravity (`~/.gemini/antigravity/mcp_config.json`)

Same format as above.

### Any other MCP client

```json
{
  "command": "uvx",
  "args": [
    "--from",
    "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
    "reporag"
  ],
  "env": { "REPORAG_DATA_DIR": "~/.local/share/reporag" }
}
```

## Tools

### `index_codebase`

Parse, embed, and graph-index a project. Run once, then incrementally on changes.

```json
{
  "path": "/path/to/project",
  "incremental": true,
  "languages": ["go", "python"]
}
```

### `query_code`

Hybrid RAG retrieval: dense + BM25 + RRF + PPR + cross-encoder rerank.

```json
{
  "query": "how does authentication work",
  "k": 10,
  "project": "/path/to/project"
}
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

Config file: `~/.config/reporag/config.json` (optional — all fields have defaults)

```json
{
  "embed_model": "nomic-ai/nomic-embed-text-v1",
  "embed_backend": "sentence-transformers",
  "data_dir": "~/.local/share/reporag",
  "reranker_model": "BAAI/bge-reranker-base",
  "reranker_k": 50,
  "rerank_by_default": false,
  "chunk_strategy": "ast",
  "chunk_window_lines": 64,
  "chunk_overlap_lines": 16,
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

**`chunk_strategy`**: `"ast"` (default) = function/class-level symbols only. `"hybrid"` = AST named symbols + sliding windows over uncovered lines. `"sliding"` = pure 64-line overlapping windows. Hybrid fills import and module-level context gaps that pure AST drops.

**`rerank_by_default`**: off by default. MS MARCO-trained rerankers degrade code retrieval quality (CodeRAG-Bench, arXiv:2406.14497). Enable per-query with `"rerank": true` in `query_code`. Default reranker is `BAAI/bge-reranker-base` (nDCG@10 0.699 vs MiniLM-L6's 0.662, requires HF token on first download).

### Environment variable overrides

| Variable                      | Field                 |
| ----------------------------- | --------------------- |
| `REPORAG_DATA_DIR`            | `data_dir`            |
| `REPORAG_EMBED_MODEL`         | `embed_model`         |
| `REPORAG_EMBED_BACKEND`       | `embed_backend`       |
| `REPORAG_RERANKER_K`          | `reranker_k`          |
| `REPORAG_RERANK_BY_DEFAULT`   | `rerank_by_default`   |
| `REPORAG_CHUNK_STRATEGY`      | `chunk_strategy`      |
| `REPORAG_CHUNK_WINDOW_LINES`  | `chunk_window_lines`  |
| `REPORAG_CHUNK_OVERLAP_LINES` | `chunk_overlap_lines` |
| `REPORAG_BM25_K1`             | `bm25_k1`             |
| `REPORAG_BM25_B`              | `bm25_b`              |
| `REPORAG_RRF_K`               | `rrf_k`               |
| `REPORAG_PPR_ALPHA`           | `ppr_alpha`           |
| `REPORAG_DENSE_CANDIDATES`    | `dense_candidates`    |
| `REPORAG_SPARSE_CANDIDATES`   | `sparse_candidates`   |
| `REPORAG_SNIPPET_CHARS`       | `snippet_chars`       |

## Optional: SCIP CLIs (compiler-grade dependency graph)

Without these, reporag uses tree-sitter import heuristics (~70% graph accuracy).

```bash
pip install scip-python                              # Python
npm install -g @sourcegraph/scip-typescript          # JS/TS
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest  # Go
```

## Development

```bash
git clone https://github.com/davidnbr/reporag
cd reporag
devenv shell   # Nix-based reproducible env (requires devenv)
# or:
uv sync --extra dev
pytest tests/ -v
```

### Regenerating `scip_pb2.py`

`reporag/indexer/scip_pb2.py` is a generated protobuf file built from the
[Sourcegraph SCIP proto](https://github.com/sourcegraph/scip/blob/main/scip.proto).
It ships pre-generated so users don't need build tools. Regenerate when the SCIP
protocol version changes:

```bash
pip install grpcio-tools   # one-time — bundles protoc, no system install needed
python scripts/generate_scip_pb2.py
```

The script fetches the latest `scip.proto` from Sourcegraph's repo and writes
`reporag/indexer/scip_pb2.py`. Commit the result alongside any `scip_indexer.py` changes.

## Supported languages

Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
