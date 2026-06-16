"""
BM25 sparse retrieval — research §4.

BM25 params (pinned per research §4 formulation):
  k1 = 1.2  (term frequency saturation)
  b  = 0.75 (length normalization)

Uses bm25s for fast vectorized BM25. Index persisted as numpy arrays.
Code-aware tokenization splits snake_case and camelCase so identifiers
like `rrf_fuse` and `DenseIndex` match their constituent tokens.
"""

from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def _code_tokenize(text: str) -> str:
    """Split code identifiers for BM25 matching.

    rrf_fuse      -> "rrf fuse rrf_fuse"
    DenseIndex    -> "Dense Index DenseIndex"
    encode_query  -> "encode query encode_query"
    """
    # preserve original alongside splits for exact-match recall
    tokens = [text]
    # snake_case split
    snake_split = re.sub(r"_+", " ", text)
    tokens.append(snake_split)
    # camelCase / PascalCase split
    camel_split = re.sub(r"([a-z\d])([A-Z])", r"\1 \2", snake_split)
    tokens.append(camel_split)
    # letter-digit boundary
    tokens.append(re.sub(r"([a-zA-Z])(\d)", r"\1 \2", camel_split))
    return " ".join(tokens).lower()


class BM25Index:
    """Persistent BM25 index over chunk semantic texts."""

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._retriever: object | None = None
        self._doc_ids: list[str] = []
        self._doc_files: list[str] = []
        self._corpus: list[str] = []

    def build(
        self, doc_ids: list[str], texts: list[str], doc_files: list[str] | None = None
    ) -> None:
        """Build BM25 index from doc IDs, their semantic texts, and source file paths.

        `doc_files` (parallel to `doc_ids`) enables project-scoped search via
        path-prefix filtering. If omitted, project filtering is disabled.
        """
        import bm25s

        self._doc_ids = doc_ids
        self._doc_files = doc_files if doc_files is not None else []
        self._corpus = texts
        tokenized = [_code_tokenize(t) for t in texts]
        corpus_tokens = bm25s.tokenize(tokenized, stopwords="en", show_progress=False)
        self._retriever = bm25s.BM25(k1=self.k1, b=self.b)
        self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, k: int = 50, project: str | None = None) -> list[str]:
        """Return top-k doc IDs ranked by BM25 score.

        If `project` is given and doc_files were provided at build time, retrieves
        a deeper pool internally (k*4) and filters to chunks under that project
        root before truncating to k.
        """
        if self._retriever is None or not self._doc_ids:
            return []
        import bm25s

        use_project_filter = bool(project) and bool(self._doc_files)
        retrieve_k = k * 4 if use_project_filter else k
        retrieve_k = min(retrieve_k, len(self._doc_ids))

        query_tokens = bm25s.tokenize([_code_tokenize(query)], stopwords="en", show_progress=False)
        results, _ = self._retriever.retrieve(query_tokens, k=retrieve_k, show_progress=False)
        # results shape: (n_queries, k) — first query only
        indices = results[0].tolist() if hasattr(results[0], "tolist") else list(results[0])
        ids = [self._doc_ids[int(i)] for i in indices if int(i) < len(self._doc_ids)]

        if use_project_filter:
            safe_root = project.rstrip("/") + "/"  # type: ignore[union-attr]
            files = [self._doc_files[int(i)] for i in indices if int(i) < len(self._doc_files)]
            ids = [doc_id for doc_id, f in zip(ids, files) if f.startswith(safe_root)]

        return ids[:k]

    def save(self, path: Path) -> None:
        """Persist index and metadata to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self._doc_ids, f)
        with open(path / "doc_files.json", "w") as f:
            json.dump(self._doc_files, f)
        with open(path / "corpus.json", "w") as f:
            json.dump(self._corpus, f)
        with open(path / "retriever.pkl", "wb") as f:
            pickle.dump(self._retriever, f)

    @classmethod
    def load(cls, path: Path, k1: float = 1.2, b: float = 0.75) -> BM25Index:
        """Load a previously saved index."""
        idx = cls(k1=k1, b=b)
        with open(path / "doc_ids.json") as f:
            idx._doc_ids = json.load(f)
        doc_files_path = path / "doc_files.json"
        if doc_files_path.exists():
            with open(doc_files_path) as f:
                idx._doc_files = json.load(f)
        else:
            idx._doc_files = []
            logger.warning(
                "%s missing — project-scoped BM25 search disabled until next full reindex",
                doc_files_path,
            )
        with open(path / "corpus.json") as f:
            idx._corpus = json.load(f)
        with open(path / "retriever.pkl", "rb") as f:
            idx._retriever = pickle.load(f)  # noqa: S301 — local trusted data only
        return idx

    @property
    def is_ready(self) -> bool:
        return self._retriever is not None and bool(self._doc_ids)
