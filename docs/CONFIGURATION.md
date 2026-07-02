# Configuration

Config file: `~/.config/reporag/config.json` (optional — all fields have defaults).

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

**`chunk_strategy`**: `"ast"` (default) = function/class-level symbols only. `"hybrid"` = AST named symbols + sliding windows over uncovered lines. `"sliding"` = pure 64-line overlapping windows. Hybrid fills import and module-level context gaps that pure AST drops. See [Benchmarks §1](BENCHMARKS.md#1-retrieval-quality) for strategy trade-offs.

**`rerank_by_default`**: off by default. MS MARCO-trained rerankers degrade code retrieval quality (CodeRAG-Bench, arXiv:2406.14497). Enable per-query with `"rerank": true` in `query_code`. Default reranker is `BAAI/bge-reranker-base` (nDCG@10 0.699 vs MiniLM-L6's 0.662, requires HF token on first download).

## Environment variable overrides

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

## Shared daemon

Each client launch of `reporag` is a thin **bridge** (stdio↔HTTP) that connects to a
single machine-wide **daemon** over loopback HTTP. The first bridge auto-spawns the
daemon; every other client reuses it — so all editor windows share **one** embedding
model and **serialized** indexing (one `index_codebase` at a time) instead of one heavy
process per client. Spawning is deduplicated via a `flock` + fixed-port bind, so
simultaneous launches can't race into two daemons. The daemon runs `reporag serve` under
the hood and is detached, so it outlives the bridge that started it.

The daemon auto-shuts-down after `REPORAG_IDLE_TIMEOUT` seconds with no active MCP
sessions, then re-spawns on the next request — no orphaned process between coding sessions.

| Variable               | Default       | Meaning                                                              |
| ---------------------- | ------------- | -------------------------------------------------------------------- |
| `REPORAG_HTTP_HOST`    | `127.0.0.1`   | Daemon bind address. Loopback only — never expose on the network.    |
| `REPORAG_HTTP_PORT`    | `7800`        | Daemon TCP port.                                                     |
| `REPORAG_IDLE_TIMEOUT` | `900` (15 min)| Idle seconds before auto-shutdown. `0` (or negative) = never shut down. |
| `REPORAG_NO_DAEMON`    | _(unset)_     | `1` = skip the daemon; run a classic self-contained per-client server. |

`reporag serve [--host H] [--port P]` starts the daemon in the foreground (host/port fall
back to the env vars above) — useful for debugging. Normal usage never needs it; the
bridge spawns the daemon on demand.

## Install variants

| Extra      | Torch    | First-run download | Use when                         |
| ---------- | -------- | ------------------ | -------------------------------- |
| `[ml]`     | CUDA     | ~2 GB              | NVIDIA GPU available             |
| `[ml-cpu]` | CPU-only | ~250 MB            | No GPU / most developer machines |

`[ml-cpu]` uses `[tool.uv.sources]` to redirect torch to the PyTorch CPU index — **requires uv** (ignored by pip). reporag runs purely on CPU regardless; the embed models fit in RAM and latency is acceptable.

## Optional: SCIP CLIs (compiler-grade dependency graph)

Without these, reporag uses tree-sitter import heuristics (~70% graph accuracy).

```bash
pip install scip-python                                        # Python
npm install -g @sourcegraph/scip-typescript                   # JS/TS
go install github.com/sourcegraph/scip-go/cmd/scip-go@latest  # Go
```

## Per-project opt-out

Create an empty `.reporag-ignore` file in a project's root to opt it out of indexing entirely — `index_codebase` returns `{"status": "skipped", ...}` without starting a background task, and the `reporag-hint` hook stays silent for that directory.

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
