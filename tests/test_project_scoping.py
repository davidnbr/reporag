"""Tests for project-scoped retrieval (Task 2).

Verifies that DenseIndex.search and BM25Index.search prefilter by project
root before truncation, so a large foreign-project index never crowds out
the target project's results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ── BM25 project filter ─────────────────────────────────────────────────────


def test_bm25_search_filters_by_project():
    pytest.importorskip("bm25s", reason="bm25s not installed")
    from reporag.retrieval.sparse import BM25Index

    doc_ids = ["a1", "a2", "b1", "b2", "b3"]
    doc_files = [
        "/rootA/foo.py",
        "/rootA/bar.py",
        "/rootB/foo.py",
        "/rootB/bar.py",
        "/rootB/baz.py",
    ]
    texts = [
        "function authenticate user login",
        "function logout user session",
        "function authenticate admin login",
        "function logout admin session",
        "function authenticate guest login",
    ]

    idx = BM25Index()
    idx.build(doc_ids, texts, doc_files)

    results = idx.search("authenticate login", k=10, project="/rootA")
    assert results
    assert set(results).issubset({"a1", "a2"})

    results_b = idx.search("authenticate login", k=10, project="/rootB")
    assert results_b
    assert set(results_b).issubset({"b1", "b2", "b3"})


def test_bm25_search_no_project_returns_all():
    pytest.importorskip("bm25s", reason="bm25s not installed")
    from reporag.retrieval.sparse import BM25Index

    doc_ids = ["a1", "b1"]
    doc_files = ["/rootA/foo.py", "/rootB/foo.py"]
    texts = ["authenticate login function", "authenticate login function"]

    idx = BM25Index()
    idx.build(doc_ids, texts, doc_files)

    results = idx.search("authenticate login", k=10)
    assert set(results) == {"a1", "b1"}


def test_bm25_load_tolerates_missing_doc_files(tmp_path: Path):
    pytest.importorskip("bm25s", reason="bm25s not installed")
    from reporag.retrieval.sparse import BM25Index

    idx = BM25Index()
    idx.build(["a1", "b1"], ["foo bar", "baz qux"])  # no doc_files
    save_path = tmp_path / "bm25"
    idx.save(save_path)
    (save_path / "doc_files.json").unlink()

    loaded = BM25Index.load(save_path)
    assert loaded._doc_files == []
    # project filter silently disabled — falls back to unfiltered search,
    # ranked by BM25 score (the matching doc still ranks first)
    results = loaded.search("foo", k=10, project="/rootA")
    assert results[0] == "a1"


# ── Dense (LanceDB) project filter ──────────────────────────────────────────


def test_dense_search_filters_by_project(tmp_path: Path):
    pytest.importorskip("lancedb", reason="lancedb not installed")
    from reporag.retrieval.dense import DenseIndex

    dim = 4
    dense = DenseIndex(tmp_path / "data", dim=dim)

    rng = np.random.default_rng(42)
    records = []
    for i in range(5):
        vec = rng.random(dim).astype(np.float32)
        records.append(
            {
                "id": f"a{i}",
                "file_path": f"/rootA/file{i}.py",
                "language": "python",
                "chunk_type": "function",
                "name": f"func_a{i}",
                "semantic_text": f"text a{i}",
                "raw_content": "",
                "start_line": 1,
                "end_line": 2,
                "parent_name": "",
                "vector": vec,
            }
        )
    for i in range(5):
        vec = rng.random(dim).astype(np.float32)
        records.append(
            {
                "id": f"b{i}",
                "file_path": f"/rootB/file{i}.py",
                "language": "python",
                "chunk_type": "function",
                "name": f"func_b{i}",
                "semantic_text": f"text b{i}",
                "raw_content": "",
                "start_line": 1,
                "end_line": 2,
                "parent_name": "",
                "vector": vec,
            }
        )
    dense.upsert(records)

    q_vec = rng.random(dim).astype(np.float32)

    results_a = dense.search(q_vec, k=10, project="/rootA")
    assert results_a
    assert all(r.startswith("a") for r in results_a)

    results_b = dense.search(q_vec, k=10, project="/rootB")
    assert results_b
    assert all(r.startswith("b") for r in results_b)

    # Without a project filter, both pools are eligible
    results_all = dense.search(q_vec, k=10)
    assert {r[0] for r in results_all} == {"a", "b"}


def test_dense_search_filters_by_language(tmp_path: Path):
    pytest.importorskip("lancedb", reason="lancedb not installed")
    from reporag.retrieval.dense import DenseIndex

    dim = 4
    dense = DenseIndex(tmp_path / "data", dim=dim)

    rng = np.random.default_rng(7)
    records = [
        {
            "id": "py1",
            "file_path": "/root/a.py",
            "language": "python",
            "chunk_type": "function",
            "name": "py_func",
            "semantic_text": "python function",
            "raw_content": "",
            "start_line": 1,
            "end_line": 2,
            "parent_name": "",
            "vector": rng.random(dim).astype(np.float32),
        },
        {
            "id": "go1",
            "file_path": "/root/a.go",
            "language": "go",
            "chunk_type": "function",
            "name": "go_func",
            "semantic_text": "go function",
            "raw_content": "",
            "start_line": 1,
            "end_line": 2,
            "parent_name": "",
            "vector": rng.random(dim).astype(np.float32),
        },
    ]
    dense.upsert(records)

    q_vec = rng.random(dim).astype(np.float32)
    results = dense.search(q_vec, k=10, languages=["python"])
    assert results == ["py1"]
