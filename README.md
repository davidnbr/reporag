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

Three questions, three benchmarks:

| #   | Question                                        | Method                                        | Section                                              |
| --- | ----------------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| 1   | Does the right chunk appear in top-k?           | Recall@k / MRR@k on named-symbol queries      | [Retrieval quality](#1-retrieval-quality)            |
| 2   | Does retrieved context improve Claude's answer? | Claude-as-judge, baseline vs. RAG (1–5 scale) | [LLM response quality](#2-llm-response-quality)      |
| 3   | Can Claude find code it doesn't know exists?    | Description-only queries (no symbol names)    | [Discovery mode](#3-discovery-mode-anti-duplication) |

All commands to reproduce are in [Reproducing these benchmarks](#reproducing-these-benchmarks).

### 1. Retrieval quality

Synthetic golden set: named functions/classes sampled from the live index.
Query template: `"implementation of {name}"` / `"how does {name} work"`.
Metric: fraction of queries where the exact chunk appears in top-k (Recall@k), plus MRR@k.

#### reporag (246 chunks, AST strategy)

| Stage     |  Recall@5 | Recall@10 |    MRR@10 | ms/query |
| --------- | --------: | --------: | --------: | -------: |
| dense     |     0.875 |     0.900 |     0.791 |        6 |
| bm25      |     0.875 |     0.875 |     0.765 |       <1 |
| rrf       |     0.875 |     0.925 |     0.778 |        6 |
| rrf + ppr |     0.875 |     0.925 |     0.768 |        8 |
| **full**  | **0.900** | **0.900** | **0.814** |      430 |

`full` = RRF fusion → Reverse PPR hub re-ranking → cross-encoder rerank. Reranker cost (~430 ms) trades latency for top-1 precision; disable with `rerank=false` in `query_code`.

#### Django (869 files / ~11k chunks, 100 samples)

Validates chunking strategy at scale. PPR requires a dense dependency graph to help — sparse heuristic graphs (857 edges / 11k chunks) show no benefit or slight regression.

| Stage     | Strategy | Recall@5 | Recall@10 | MRR@10 | ms/query |
| --------- | -------- | -------: | --------: | -----: | -------: |
| dense     | ast      |    0.770 |     0.790 |  0.625 |       35 |
| dense     | hybrid   |    0.690 |     0.740 |  0.574 |       37 |
| bm25      | ast      |    0.610 |     0.660 |  0.497 |        1 |
| bm25      | hybrid   |    0.600 |     0.640 |  0.495 |        1 |
| rrf       | ast      |    0.750 | **0.850** |  0.634 |       37 |
| rrf       | hybrid   |    0.670 |     0.780 |  0.566 |       38 |
| rrf + ppr | ast      |    0.750 |     0.840 |  0.617 |       41 |
| rrf + ppr | hybrid   |    0.670 |     0.780 |  0.566 |       43 |

AST consistently outperforms hybrid by 5–7 pp on Recall@10 for **named-symbol queries** (e.g. "implementation of QuerySet"). Hybrid window chunks dilute the dense vector space when exact AST chunks compete against overlapping windows.

> **Caveat:** both tables measure exact named-chunk recall — structurally biased toward AST. The cited paper (arXiv:2605.04763) where sliding window wins tests _code completion at cursor positions_, not NL→code architectural queries. For queries like "how does the ORM query execution flow work?" or "where is middleware processing defined?", hybrid may outperform AST by providing import context and cross-function windows that AST drops — this is untested.

**Strategy guidance:**

| Strategy          | Best for                                      | Trade-off                                             |
| ----------------- | --------------------------------------------- | ----------------------------------------------------- |
| `"ast"` (default) | Symbol lookup, exact function/class retrieval | Smallest index, lowest latency                        |
| `"hybrid"`        | Architectural/contextual queries (untested)   | −5–7 pp recall on symbol lookup, ~2× indexing latency |
| `"sliding"`       | Pure window coverage                          | No symbol-level precision                             |

### 2. LLM response quality

Retrieval recall measures _whether_ the right chunk is returned. This benchmark measures _whether Claude gives a better answer_ when that chunk is injected as context.

**Method:** for each sampled function/class, generate a question (`"How does {name} work?"`), call Claude twice — once with no context (baseline), once with retrieved chunks injected (RAG) — then use Claude as judge to score both responses on correctness, completeness, and hallucination avoidance (1–5 each). Uses `claude -p` (Claude Code CLI, no API key required).

| Codebase           | Files | Chunks | Scored | Baseline |  RAG | Composite Δ |
| ------------------ | ----: | -----: | -----: | -------: | ---: | ----------: |
| reporag            |    39 |    266 |  17/30 |     3.98 | 4.16 |   **+4.4%** |
| private Go project |    74 |    488 |  30/30 |     2.24 | 3.84 |  **+71.3%** |
| Django             | 2,955 |   ~45k |  24/30 |     2.35 | 4.25 |  **+81.1%** |

![LLM response quality: baseline vs RAG](https://github.com/user-attachments/assets/f338658b-c62c-4314-aad9-361736c98cf2)

_"Discovery" = description-only queries (anti-duplication use case), see [§3](#3-discovery-mode-anti-duplication)._

<details>
<summary>Per-metric breakdown</summary>

**reporag** — well-known patterns (MCP, Python, RAG); baseline already strong. RAG helps most on completeness — actual source makes answers more thorough.

| Metric        | Baseline |      RAG |     Δ (%) |
| ------------- | -------: | -------: | --------: |
| correctness   |     4.18 |     4.29 |     +2.8% |
| completeness  |     3.94 |     4.24 |     +7.5% |
| hallucination |     3.82 |     3.94 |     +3.1% |
| **composite** | **3.98** | **4.16** | **+4.4%** |

**Private Go project** — baseline near 1: Claude cannot answer without context. Hallucination score is high at baseline because Claude hedges rather than inventing. RAG raises correctness/completeness ~2.5×.

| Metric        | Baseline |      RAG |      Δ (%) |
| ------------- | -------: | -------: | ---------: |
| correctness   |     1.27 |     3.57 |    +181.6% |
| completeness  |     1.13 |     3.53 |    +211.8% |
| hallucination |     4.33 |     4.43 |      +2.3% |
| **composite** | **2.24** | **3.84** | **+71.3%** |

**Django** — largest correctness/completeness gains. Hallucination slightly regresses with RAG: injecting 10 chunks causes Claude to synthesize across chunks in ways that diverge from the single ground-truth function. Baseline hedges cleanly; RAG over-extends retrieved context.

| Metric        | Baseline |      RAG |      Δ (%) |
| ------------- | -------: | -------: | ---------: |
| correctness   |     1.33 |     4.25 |    +218.8% |
| completeness  |     1.17 |     4.21 |    +260.7% |
| hallucination |     4.54 |     4.29 |      −5.5% |
| **composite** | **2.35** | **4.25** | **+81.1%** |

</details>

**Takeaway:** the less Claude knows about a codebase from training, the larger the RAG gain. On private or large codebases, reporag delivers **+71–81% composite quality improvement** — turning unusable baselines (1.1–1.3) into competent answers (3.8–4.2).

### 3. Discovery mode (anti-duplication)

The benchmarks above test **named-symbol lookup** — Claude already knows what it's looking for ("how does `selfCorrectLoop` work"). The real failure mode in multi-file features is the opposite: Claude doesn't know a function/pattern already exists, so it reimplements it.

**Method:** strip the function name from `semantic_text`, leaving only the description (parameters/docstring). Use that as the query — simulates "I need to implement X" without knowing any names. Grep can't help here (no name to search for); semantic retrieval is the only option.

**Retrieval recall** (private Go project, 488 chunks, 50 samples):

| Stage          |  Recall@5 | Recall@10 | MRR@10 | ms/query |
| -------------- | --------: | --------: | -----: | -------: |
| grep-discovery |     0.660 |     0.760 |  0.390 |   1598.7 |
| dense          |     1.000 |     1.000 |  0.911 |     88.3 |
| bm25           |     1.000 |     1.000 |  0.910 |     13.7 |
| **full**       | **1.000** | **1.000** |  0.907 |    120.8 |

![Discovery mode Recall@10 by stage](https://github.com/user-attachments/assets/70ea1c4c-1e27-40d8-8cff-3370a1de9e07)

RAG vs grep-discovery Recall@10: **+0.240**. Grep word-overlap on raw content gets close (0.760) but is 16× slower and misses 1 in 4. RAG hits every case.

**LLM response quality** (same project, 5 samples → 15 `claude -p` calls):

| Metric        | Baseline |      RAG |       Δ (%) |
| ------------- | -------: | -------: | ----------: |
| correctness   |     1.00 |     4.20 |     +320.0% |
| completeness  |     1.20 |     4.00 |     +233.3% |
| hallucination |     3.20 |     5.00 |      +56.2% |
| **composite** | **1.80** | **4.40** | **+144.4%** |

4/5 samples: baseline either refused, hallucinated the wrong language, or invented a different implementation. RAG returned the **exact existing function verbatim with file:line location**, ready to import instead of reimplement. The 1 miss was a query with no real signal (`"Parameters: result types.ExecutionResult."` — no docstring, no useful description).

> **Caveat:** queries are derived from `semantic_text` (leave-one-out), so this is optimistic vs. true unknown-unknowns — it validates the embedding space is dense enough for description→code retrieval, not a blind real-world test. n=5 for the LLM eval is small.

**Takeaway:** when Claude is told "before writing new code, call `find_existing(task=...)`," semantic retrieval reliably surfaces existing implementations from a one-sentence description — the core anti-duplication use case.

### Reproducing these benchmarks

```bash
# Retrieval quality (named-symbol lookup)
devenv shell -- python scripts/benchmark.py --samples 100 --k 5 10
devenv shell -- python scripts/benchmark.py --project /path/to/project --stages dense rrf full
devenv shell -- python scripts/benchmark.py --filter-chunk-types function --quiet

# LLM response quality (named-symbol lookup) — index project first via index_codebase
devenv shell -- python scripts/llm_eval.py --project /path/to/project --samples 30 --output results.json

# Discovery mode (anti-duplication)
devenv shell -- python scripts/benchmark.py --project /path/to/project --mode discovery --samples 50
devenv shell -- python scripts/llm_eval.py --project /path/to/project --mode discovery --samples 5 --output results.json
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

### Cursor (`~/.cursor/mcp.json`)

Run `reporag setup --client cursor` to write this automatically, or add manually:

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
      "env": { "REPORAG_DATA_DIR": "~/.local/share/reporag" }
    }
  }
}
```

### Codex CLI (`~/.codex/config.toml`)

Run `reporag setup --client codex` to write this automatically (preserves the rest of your hand-edited `config.toml` — comments, other servers, settings — via a marker-delimited managed block), or add manually:

```toml
[mcp_servers.reporag]
command = "uvx"
args = ["--from", "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git", "reporag"]
env = { REPORAG_DATA_DIR = "~/.local/share/reporag" }
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 120
```

Honors `CODEX_HOME` (defaults to `~/.codex`). Restart Codex to activate.

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

Create an empty `.reporag-ignore` file in a project's root to opt it out of indexing
entirely — `index_codebase` returns `{"status": "skipped", ...}` without starting a
background task, and the `reporag-hint` hook stays silent for that directory.

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

### `find_existing`

Pre-implementation discovery — call before writing new code to surface existing functions/classes that already handle the task. Same hybrid pipeline as `query_code` (no reranker), deduplicated by file (max 2/file), with `reuse_hint` explaining why each result is relevant.

```json
{
  "task": "validate API error codes and map them to user-facing messages",
  "project": "/path/to/project",
  "k": 10
}
```

Prevents the "Claude reimplements logic that already exists in another module" failure mode in multi-file features. See [Discovery mode](#3-discovery-mode-anti-duplication) for benchmark results.

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

## Client Setup

```bash
# Configure Claude Code, Cursor, and Codex CLI at once:
uvx --from "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git" \
    reporag setup --client all

# Or configure individually:
reporag setup --client claude   # Claude Code only
reporag setup --client cursor   # Cursor only
reporag setup --client codex    # Codex CLI only
```

Then restart the client.

### Claude Code

Two `UserPromptSubmit` hooks ship inside the package and are **installed automatically** the first time the MCP server connects — no manual step needed.

The hooks install into `~/.claude/hooks/` and register in `~/.claude/settings.json`:

**`reporag-autoindex`** — fires on every prompt. If the current directory is not indexed:

```
[reporag] /path/to/project has not been indexed yet.
Call index_codebase with path="/path/to/project" to enable code search.
```

Claude automatically calls `index_codebase` before answering.

**`reporag-hint`** — fires on code-related prompts. If the project is indexed:

```
[reporag] /path/to/project is indexed (285 chunks).
Use query_code to retrieve relevant context before answering.
```

Claude proactively calls `query_code` (dense + BM25 + RRF + PPR pipeline) before answering.

Both hooks read `~/.local/share/reporag/projects.json` — no ML imports, < 5 ms overhead.

### Cursor

Writes `~/.cursor/mcp.json` with the reporag server config and creates `~/.cursor/rules/reporag.mdc` (Cursor ≥0.50 global rules, `alwaysApply: true`) instructing Cursor to use `query_code` and `index_codebase` proactively.

For older Cursor versions, add to your project's `.cursorrules`:

```
Use the reporag MCP tools for all code questions:
query_code before answering, index_codebase if not indexed, get_symbol for lookups.
```

### Codex CLI

Writes `~/.codex/config.toml` with the reporag MCP server registration plus `UserPromptSubmit` (`reporag-hint`), `PreToolUse` (`reporag-dupcheck`, matcher `apply_patch`), and `SessionStart` (`reporag-autoindex`) command hooks — using a marker-delimited managed block (`# >>> reporag managed (do not edit) >>>` … `# <<< reporag managed <<<`) so hand-edited comments, other `[mcp_servers.*]` tables, and top-level settings survive untouched. The assembled file is validated with `tomllib` before writing; an invalid or conflicting existing file is left untouched and reported, never corrupted.

Codex hooks emit `{"hookSpecificOutput": {...}}` JSON (set via `REPORAG_HOOK_FORMAT=codex` in the hook command) since Codex doesn't read plain stdout as context the way Claude Code does. Re-running setup is idempotent — byte-identical output, no rewrite.

### Additional CLI

```bash
reporag status --project /path/to/project     # check if a project is indexed
reporag setup-hooks [--claude-dir ~/.claude]   # reinstall Claude Code hooks only
reporag setup --client codex [--codex-dir ~/.codex]  # reinstall Codex config only
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
