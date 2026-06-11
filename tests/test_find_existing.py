"""Tests for the find_existing tool's output-quality fields (Task 8).

Covers:
1. reuse_hint contains a real description (not a tautological name echo).
2. signature field is the first line of raw_content.
3. Low-relevance candidates (score < 0.35 * top_score) are dropped.
4. When no candidates survive chunk-type filtering, return the
   "safe to implement from scratch" message.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from reporag.tools.find_existing import run

_CFG = SimpleNamespace(
    dense_candidates=50,
    sparse_candidates=50,
    rrf_k=60,
    rrf_dense_weight=1.0,
    rrf_sparse_weight=0.5,
)


class _FakeEmbedder:
    def encode_query(self, text: str) -> Any:
        return [0.0]


class _FakeDense:
    def __init__(self, dense_ids: list[str], chunks: dict[str, dict[str, Any]]) -> None:
        self._dense_ids = dense_ids
        self._chunks = chunks

    def search(self, q_vec: Any, k: int, project: str | None = None, languages: Any = None) -> list[str]:
        return self._dense_ids[:k]

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self._chunks[i] for i in ids if i in self._chunks]


class _FakeBM25:
    is_ready = True

    def __init__(self, sparse_ids: list[str]) -> None:
        self._sparse_ids = sparse_ids

    def search(self, query: str, k: int, project: str | None = None) -> list[str]:
        return self._sparse_ids[:k]


def _runtime(dense_ids: list[str], sparse_ids: list[str], chunks: dict[str, dict[str, Any]]) -> Any:
    return SimpleNamespace(
        config=_CFG,
        embedder=_FakeEmbedder(),
        dense=_FakeDense(dense_ids, chunks),
        bm25=_FakeBM25(sparse_ids),
    )


def _func_chunk(
    chunk_id: str, name: str, doc: str, raw_content: str, file_path: str = "module.py"
) -> dict[str, Any]:
    name_readable = name.replace("_", " ")
    return {
        "id": chunk_id,
        "name": name,
        "chunk_type": "function",
        "language": "python",
        "file_path": file_path,
        "start_line": 1,
        "end_line": 5,
        "semantic_text": f"Function {name_readable}. Parameters: x. Returns: bool. {doc}",
        "raw_content": raw_content,
    }


def test_reuse_hint_has_description_not_tautology() -> None:
    chunk = _func_chunk(
        "c1",
        "run_task",
        "Executes the scheduled task and returns success.",
        "def run_task(x: int) -> bool:\n    return x > 0\n",
    )
    runtime = _runtime(dense_ids=["c1"], sparse_ids=[], chunks={"c1": chunk})

    result = asyncio.run(run({"task": "run a task"}, runtime))

    assert result["total_found"] == 1
    hint = result["existing_code"][0]["reuse_hint"]
    assert "may already handle 'run task'" not in hint
    assert "Executes the scheduled task and returns success" in hint


def test_signature_is_first_line_of_raw_content() -> None:
    chunk = _func_chunk(
        "c1",
        "run_task",
        "Executes the scheduled task.",
        "def run_task(x: int) -> bool:\n    return x > 0\n",
    )
    runtime = _runtime(dense_ids=["c1"], sparse_ids=[], chunks={"c1": chunk})

    result = asyncio.run(run({"task": "run a task"}, runtime))

    assert result["existing_code"][0]["signature"] == "def run_task(x: int) -> bool:"


def test_low_score_tail_dropped() -> None:
    chunk_a = _func_chunk("a", "func_a", "Does A.", "def func_a():\n    pass\n")
    chunk_b = _func_chunk("b", "func_b", "Does B.", "def func_b():\n    pass\n")

    # 'a' is the top dense hit; 'b' only shows up at the tail of the sparse
    # ranking, giving it a much lower fused RRF score (well below 0.35x top).
    dummies = [f"dummy{i}" for i in range(49)]
    sparse_ids = dummies + ["b"]
    dummy_chunks = {d: {"id": d, "name": d, "chunk_type": "module"} for d in dummies}

    chunks = {"a": chunk_a, "b": chunk_b, **dummy_chunks}
    runtime = _runtime(dense_ids=["a"], sparse_ids=sparse_ids, chunks=chunks)

    result = asyncio.run(run({"task": "do something"}, runtime))

    names = [r["name"] for r in result["existing_code"]]
    assert "func_a" in names
    assert "func_b" not in names


def test_no_actionable_chunks_returns_safe_message() -> None:
    module_chunk = {
        "id": "m1",
        "name": "module",
        "chunk_type": "module",
        "language": "python",
        "file_path": "module.py",
        "start_line": 1,
        "end_line": 50,
        "semantic_text": "Module module in file module.py.",
        "raw_content": "import os\n",
    }
    runtime = _runtime(dense_ids=["m1"], sparse_ids=[], chunks={"m1": module_chunk})

    result = asyncio.run(run({"task": "do something"}, runtime))

    assert result["total_found"] == 0
    assert result["existing_code"] == []
    assert "Safe to implement from scratch" in result["message"]
