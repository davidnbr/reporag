from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel, Field

CONFIG_PATH = Path.home() / ".config" / "rag-mcp" / "config.json"

_ENV_MAP = {
    "RAG_MCP_EMBED_MODEL": "embed_model",
    "RAG_MCP_EMBED_BACKEND": "embed_backend",
    "RAG_MCP_OLLAMA_URL": "ollama_url",
    "RAG_MCP_DATA_DIR": "data_dir",
    "RAG_MCP_RERANKER_MODEL": "reranker_model",
}


class Config(BaseModel):
    embed_model: str = "nomic-ai/nomic-embed-text-v1"
    embed_backend: str = "sentence-transformers"  # "sentence-transformers" | "ollama"
    ollama_url: str = "http://localhost:11434"
    data_dir: str = ".rag-mcp"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_k: int = 15          # only rerank when candidates <= this
    bm25_k1: float = 1.2          # BM25 term frequency saturation (research §4)
    bm25_b: float = 0.75          # BM25 length normalization (research §4)
    rrf_k: int = 60               # RRF smoothing constant (research §4, standardized)
    ppr_alpha: float = 0.85       # PageRank damping factor (research §3)
    ppr_seed_k: int = 20          # top RRF hits used as PPR teleport seeds
    dense_candidates: int = 50    # dense retrieval pool before RRF
    sparse_candidates: int = 50   # BM25 retrieval pool before RRF
    subgraph_hops: int = 1        # k-hop expansion for subgraph results
    use_ollama_docstrings: bool = False  # LLM docstring gen via Ollama (optional)


def load_config() -> Config:
    data: dict = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            data = json.load(f)
    for env_key, field in _ENV_MAP.items():
        if val := os.environ.get(env_key):
            data[field] = val
    return Config(**data)


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = load_config()
    return _config
