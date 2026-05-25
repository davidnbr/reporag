"""
Cross-encoder reranker for precision boost on final top-K — research §4.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (MIT, ~84MB, local-only)
Only activated when len(candidates) <= reranker_k (default 15) — precision over speed.
Lazy-loaded on first use to avoid startup delay.
"""
from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    """Reranks (query, chunk) pairs using a local cross-encoder model."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        text_key: str = "semantic_text",
    ) -> list[dict[str, Any]]:
        """
        Rerank chunks by cross-encoder relevance to query.

        Args:
            query: The user's natural language query.
            chunks: List of chunk dicts, each must have text_key field.
            text_key: Dict key for the text to score against.

        Returns:
            Chunks sorted by cross-encoder score descending, score added as 'rerank_score'.
        """
        if not chunks:
            return chunks
        self._load()
        pairs = [(query, c.get(text_key, c.get("raw_content", ""))) for c in chunks]
        scores: list[float] = self._model.predict(pairs).tolist()
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = score
        return sorted(chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
