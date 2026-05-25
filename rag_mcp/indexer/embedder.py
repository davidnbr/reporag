"""
Local embedding generation — zero API cost.

Primary:  nomic-ai/nomic-embed-text-v1 via sentence-transformers (768-dim, Apache 2.0)
Fallback: all-MiniLM-L6-v2 (384-dim, MIT, smaller/faster)
Optional: Ollama REST API (nomic-embed-text or any local model)

nomic-embed-text requires prefix tokens:
  "search_document: " for corpus texts at index time
  "search_query: "    for query vectors at retrieval time
"""
from __future__ import annotations

import asyncio
from typing import Literal

import numpy as np

EmbedBackend = Literal["sentence-transformers", "ollama"]

_NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1"
_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_NOMIC_PREFIXES = {"index": "search_document: ", "query": "search_query: "}


class Embedder:
    """Unified local embedding interface."""

    def __init__(
        self,
        model: str = _NOMIC_MODEL,
        backend: EmbedBackend = "sentence-transformers",
        ollama_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.backend = backend
        self.ollama_url = ollama_url
        self._st_model: object | None = None
        self._use_nomic_prefix = "nomic" in model.lower()

    def _load_st(self) -> None:
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model, trust_remote_code=True)
        except Exception:
            # Fall back to MiniLM if nomic fails (e.g., first-time download issue)
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(_MINILM_MODEL)
            self._use_nomic_prefix = False

    def _apply_prefix(self, texts: list[str], mode: Literal["index", "query"]) -> list[str]:
        if not self._use_nomic_prefix:
            return texts
        prefix = _NOMIC_PREFIXES[mode]
        return [prefix + t for t in texts]

    def encode_corpus(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of corpus texts for indexing. Returns (N, dim) float32 array."""
        prefixed = self._apply_prefix(texts, "index")
        if self.backend == "sentence-transformers":
            self._load_st()
            return self._st_model.encode(  # type: ignore[union-attr]
                prefixed,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 100,
                normalize_embeddings=True,
            ).astype(np.float32)
        return self._encode_ollama_batch(prefixed)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string. Returns (dim,) float32 vector."""
        prefixed = self._apply_prefix([query], "query")
        if self.backend == "sentence-transformers":
            self._load_st()
            vec = self._st_model.encode(  # type: ignore[union-attr]
                prefixed,
                normalize_embeddings=True,
            )
            return np.array(vec[0], dtype=np.float32)
        return self._encode_ollama_batch(prefixed)[0]

    def _encode_ollama_batch(self, texts: list[str]) -> np.ndarray:
        import httpx

        vecs: list[np.ndarray] = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                resp = client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                resp.raise_for_status()
                vecs.append(np.array(resp.json()["embedding"], dtype=np.float32))
        return np.stack(vecs)

    async def encode_corpus_async(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Async wrapper — offloads blocking encode to thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode_corpus, texts, batch_size)

    @property
    def dim(self) -> int:
        """Return embedding dimension."""
        return 768 if "nomic" in self.model.lower() else 384
