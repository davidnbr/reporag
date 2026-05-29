from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel

CONFIG_PATH = Path.home() / ".config" / "reporag" / "config.json"

_ENV_MAP = {
    "REPORAG_EMBED_MODEL": "embed_model",
    "REPORAG_EMBED_BACKEND": "embed_backend",
    "REPORAG_OLLAMA_URL": "ollama_url",
    "REPORAG_DATA_DIR": "data_dir",
    "REPORAG_RERANKER_MODEL": "reranker_model",
    "REPORAG_RERANKER_K": "reranker_k",
    "REPORAG_RERANK_BY_DEFAULT": "rerank_by_default",
    "REPORAG_BM25_K1": "bm25_k1",
    "REPORAG_BM25_B": "bm25_b",
    "REPORAG_RRF_K": "rrf_k",
    "REPORAG_PPR_ALPHA": "ppr_alpha",
    "REPORAG_PPR_SEED_K": "ppr_seed_k",
    "REPORAG_DENSE_CANDIDATES": "dense_candidates",
    "REPORAG_SPARSE_CANDIDATES": "sparse_candidates",
    "REPORAG_SUBGRAPH_HOPS": "subgraph_hops",
    "REPORAG_SNIPPET_CHARS": "snippet_chars",
    "REPORAG_RRF_DENSE_WEIGHT": "rrf_dense_weight",
    "REPORAG_RRF_SPARSE_WEIGHT": "rrf_sparse_weight",
    "REPORAG_MIN_GRAPH_EDGES_PPR": "min_graph_edges_for_ppr",
    "REPORAG_CHUNK_STRATEGY": "chunk_strategy",
    "REPORAG_CHUNK_WINDOW_LINES": "chunk_window_lines",
    "REPORAG_CHUNK_OVERLAP_LINES": "chunk_overlap_lines",
    "REPORAG_INDEX_BATCH_SIZE": "index_batch_size",
    "REPORAG_WATCH_DEBOUNCE_S": "watch_debounce_s",
}


class Config(BaseModel):
    embed_model: str = "nomic-ai/nomic-embed-text-v1"
    embed_backend: str = "sentence-transformers"  # "sentence-transformers" | "ollama"
    ollama_url: str = "http://localhost:11434"
    data_dir: str = "~/.local/share/reporag"
    # Reranker: bge-reranker-base outperforms ms-marco-MiniLM-L6 (nDCG@10 0.699 vs 0.662)
    # rerank_by_default=False: CodeRAG-Bench shows MS-MARCO rerankers degrade code retrieval
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_k: int = 50  # rerank when final candidates <= this
    rerank_by_default: bool = False  # off by default — rerankers trained on doc-IR hurt code tasks
    bm25_k1: float = 1.2  # BM25 term frequency saturation (research §4)
    bm25_b: float = 0.75  # BM25 length normalization (research §4)
    rrf_k: int = 60  # RRF smoothing constant (research §4, standardized)
    ppr_alpha: float = 0.85  # PageRank damping factor (research §3)
    ppr_seed_k: int = 20  # top RRF hits used as PPR teleport seeds
    dense_candidates: int = 50  # dense retrieval pool before RRF
    sparse_candidates: int = 50  # BM25 retrieval pool before RRF
    subgraph_hops: int = 1  # k-hop expansion for subgraph results
    snippet_chars: int = 600  # max chars of semantic_text returned per result
    use_ollama_docstrings: bool = False  # LLM docstring gen via Ollama (optional)
    rrf_dense_weight: float = 1.0  # RRF weight for dense retriever
    rrf_sparse_weight: float = 0.5  # RRF weight for BM25 (down-weighted; dense is stronger)
    min_graph_edges_for_ppr: int = (
        50  # minimum graph edges to enable PPR (avoids noise on sparse graphs)
    )
    # Chunking strategy: "ast" (default), "sliding" (window-only), "hybrid" (ast + gap windows)
    # arXiv:2605.04763 tests code completion at cursor — not NL→code retrieval; don't cite for RAG
    # "hybrid" fills import/module-level gaps; "ast" wins on named-symbol recall (benchmark data)
    chunk_strategy: str = "ast"
    chunk_window_lines: int = 64  # sliding window size in lines
    chunk_overlap_lines: int = 16  # overlap between adjacent windows
    # Progressive indexing
    index_batch_size: int = 20  # files per embedding batch; smaller = faster first results
    watch_debounce_s: float = 2.0  # seconds to debounce watchdog file-change events
    auto_index_paths: list[str] = []  # absolute paths to index on server startup


def load_config() -> Config:
    data: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    fields = Config.model_fields
    for env_key, field_name in _ENV_MAP.items():
        if val := os.environ.get(env_key):
            annotation = fields[field_name].annotation if field_name in fields else str
            try:
                if annotation is int:
                    data[field_name] = int(val)
                elif annotation is float:
                    data[field_name] = float(val)
                elif annotation == list[str]:
                    # Accept JSON array or colon-separated paths
                    if val.startswith("["):
                        data[field_name] = json.loads(val)
                    else:
                        data[field_name] = [p for p in val.split(":") if p]
                else:
                    data[field_name] = val
            except (ValueError, TypeError):
                data[field_name] = val
    return Config(**data)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
