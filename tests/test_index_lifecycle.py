"""Tests for the background indexing lifecycle (Task 3).

Covers two regressions:
1. An all-unchanged incremental run must skip graph rebuild, graph reload,
   reranker invalidation, and registry update entirely (no per-prompt rebuilds).
2. The project registry must be updated with project-wide totals (via
   DenseIndex.count_by_project / ChunkIndexer.count_files), not per-run deltas
   — regression for the live "reporag chunks: 0" bug.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from reporag.server import IndexTask
from reporag.tools.index import _run_index_bg


def test_no_change_skips_graph_and_registry(monkeypatch):
    build_graph_mock = Mock()
    monkeypatch.setattr(
        "reporag.indexer.graph_builder.build_graph_for_project", build_graph_mock
    )
    update_mock = Mock()
    monkeypatch.setattr("reporag.projects.update", update_mock)

    runtime = SimpleNamespace(
        chunker=SimpleNamespace(
            index_files_batched_async=AsyncMock(
                return_value={"files": 0, "chunks": 0, "skipped": 3}
            )
        ),
        dense=Mock(),
        graph_db=Mock(),
        reranker=Mock(),
        reload_graph=Mock(),
        index_sem=asyncio.Semaphore(1),
        config=SimpleNamespace(index_batch_size=20),
        index_tasks={},
    )

    root = Path("/tmp/some-project")
    task = IndexTask(
        task_id="t1",
        project=str(root),
        started_at=time.monotonic(),
        total_files=3,
    )
    runtime.index_tasks["t1"] = task

    asyncio.run(_run_index_bg("t1", root, [], incremental=True, runtime=runtime))

    assert task.status == "done"
    assert task.skipped_files == 3
    build_graph_mock.assert_not_called()
    runtime.reload_graph.assert_not_called()
    runtime.reranker.invalidate_cache.assert_not_called()
    update_mock.assert_not_called()


@pytest.fixture()
def registry_path(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORAG_DATA_DIR", str(tmp_path / "registry"))
    monkeypatch.setitem(__import__("sys").modules, "reporag.config", None)
    import importlib

    import reporag.projects as proj_mod

    importlib.reload(proj_mod)
    return proj_mod


def test_registry_totals_match_project_chunk_count(tmp_path: Path, registry_path):
    pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    from reporag.indexer.chunker import ChunkIndexer
    from reporag.indexer.graph_builder import GraphDB
    from reporag.retrieval.dense import DenseIndex
    from reporag.retrieval.sparse import BM25Index

    proj = registry_path

    dim = 4

    class _FakeEmbedder:
        def encode_corpus(self, texts: list[str], batch_size: int = 64):
            import numpy as np

            return np.zeros((len(texts), dim), dtype=np.float32)

    root = tmp_path / "project"
    root.mkdir()
    py_file = root / "module_a.py"
    py_file.write_text(
        textwrap.dedent(
            '''
            def my_func(x: int) -> int:
                """Add one."""
                return x + 1
            '''
        )
    )

    data = tmp_path / "data"
    graph_db = GraphDB(data / "dependency_graph.db")
    dense = DenseIndex(data, dim=dim)
    bm25 = BM25Index()
    chunker = ChunkIndexer(
        data_dir=data,
        embedder=_FakeEmbedder(),
        dense_index=dense,
        bm25_index=bm25,
        graph_db=graph_db,
    )

    runtime = SimpleNamespace(
        chunker=chunker,
        dense=dense,
        graph_db=graph_db,
        reranker=None,
        reload_graph=Mock(),
        index_sem=asyncio.Semaphore(1),
        config=SimpleNamespace(index_batch_size=20),
        index_tasks={},
    )

    def _new_task(task_id: str) -> IndexTask:
        task = IndexTask(
            task_id=task_id,
            project=str(root),
            started_at=time.monotonic(),
            total_files=1,
        )
        runtime.index_tasks[task_id] = task
        return task

    # First run: file is new — full pipeline runs, registry gets totals.
    _new_task("t1")
    asyncio.run(_run_index_bg("t1", root, [py_file], incremental=True, runtime=runtime))
    assert runtime.index_tasks["t1"].status == "done"

    expected_chunks = dense.count_by_project(str(root))
    assert expected_chunks > 0

    entry = proj.get(str(root))
    assert entry is not None
    assert entry["chunks"] == expected_chunks
    assert entry["files"] == chunker.count_files(str(root))

    # Second run: nothing changed — early-return path must not zero the registry.
    _new_task("t2")
    asyncio.run(_run_index_bg("t2", root, [py_file], incremental=True, runtime=runtime))
    assert runtime.index_tasks["t2"].status == "done"
    assert runtime.index_tasks["t2"].skipped_files == 1

    entry_after = proj.get(str(root))
    assert entry_after is not None
    assert entry_after["chunks"] == expected_chunks
    assert entry_after["chunks"] == dense.count_by_project(str(root))
