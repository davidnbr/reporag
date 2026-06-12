"""End-to-end tests for the query_code MCP tool pipeline (Task 12), fully mocked runtime.

Covers:
- project filter is forwarded to dense/BM25 search.
- final results are truncated to `k`.
- subgraph neighbor expansion populates `neighbors` from the dependency graph.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import networkx as nx

from reporag.tools.query import run

_CFG = SimpleNamespace(
    dense_candidates=50,
    sparse_candidates=50,
    rrf_k=60,
    rrf_dense_weight=1.0,
    rrf_sparse_weight=0.5,
    ppr_seed_k=20,
    min_graph_edges_for_ppr=50,
    ppr_alpha=0.85,
    reranker_k=50,
    rerank_by_default=False,
    subgraph_hops=1,
    snippet_chars=600,
)


class _FakeEmbedder:
    def encode_query(self, text: str) -> Any:
        return [0.0]


class _FakeBM25:
    is_ready = False

    def search(self, query: str, k: int, project: str | None = None) -> list[str]:
        raise AssertionError("bm25.search should not be called when is_ready is False")


class _FakeDense:
    def __init__(self, ids: list[str], chunks: dict[str, dict[str, Any]]) -> None:
        self._ids = ids
        self._chunks = chunks
        self.search_calls: list[dict[str, Any]] = []

    def search(
        self,
        q_vec: Any,
        k: int,
        project: str | None = None,
        languages: list[str] | None = None,
    ) -> list[str]:
        self.search_calls.append({"k": k, "project": project, "languages": languages})
        return self._ids[:k]

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self._chunks[i] for i in ids if i in self._chunks]


def _chunk(chunk_id: str, file_path: str, name: str = "f") -> dict[str, Any]:
    return {
        "id": chunk_id,
        "name": name,
        "chunk_type": "function",
        "language": "python",
        "file_path": file_path,
        "start_line": 1,
        "end_line": 5,
        "semantic_text": f"Function {name}.",
    }


def _runtime(
    ids: list[str], chunks: dict[str, dict[str, Any]], graph: nx.DiGraph | None = None
) -> Any:
    return SimpleNamespace(
        config=_CFG,
        embedder=_FakeEmbedder(),
        dense=_FakeDense(ids, chunks),
        bm25=_FakeBM25(),
        graph=graph,
        reranker=None,
    )


def test_project_filter_forwarded_to_dense_search() -> None:
    chunks = {"a": _chunk("a", "/repo/proj/a.py")}
    runtime = _runtime(["a"], chunks)

    asyncio.run(run({"query": "find a", "project": "/repo/proj", "rerank": False}, runtime))

    assert runtime.dense.search_calls[0]["project"] == "/repo/proj"


def test_results_truncated_to_k() -> None:
    ids = [f"c{i}" for i in range(8)]
    chunks = {cid: _chunk(cid, f"/repo/{cid}.py") for cid in ids}
    runtime = _runtime(ids, chunks)

    result = asyncio.run(run({"query": "find stuff", "k": 3, "rerank": False}, runtime))

    assert len(result["results"]) == 3
    assert result["pipeline"]["dense_hits"] == 8


def test_subgraph_expansion_populates_neighbors() -> None:
    chunks = {"a": _chunk("a", "/repo/a.py")}
    graph = nx.DiGraph()
    graph.add_edge("/repo/a.py", "/repo/b.py")

    runtime = _runtime(["a"], chunks, graph=graph)

    result = asyncio.run(run({"query": "find a", "k": 1, "rerank": False}, runtime))

    assert result["results"][0]["neighbors"] == ["/repo/b.py"]
    assert result["pipeline"]["ppr_applied"] is False
