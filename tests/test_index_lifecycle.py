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
from reporag.tools.index import _run_index_bg, run


def test_no_change_skips_graph_and_registry(monkeypatch):
    build_graph_mock = Mock()
    monkeypatch.setattr(
        "reporag.indexer.graph_builder.build_graph_for_project", build_graph_mock
    )
    update_mock = Mock()
    monkeypatch.setattr("reporag.projects.update", update_mock)

    runtime = SimpleNamespace(
        chunker=SimpleNamespace(
            files_under=Mock(return_value=[]),
            forget_file=Mock(),
            index_files_batched_async=AsyncMock(
                return_value={"files": 0, "chunks": 0, "skipped": 3}
            ),
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


def test_orphaned_file_chunks_purged_on_reindex(tmp_path: Path):
    pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    from reporag.indexer.chunker import ChunkIndexer
    from reporag.indexer.graph_builder import GraphDB
    from reporag.retrieval.dense import DenseIndex
    from reporag.retrieval.sparse import BM25Index

    dim = 4

    class _FakeEmbedder:
        def encode_corpus(self, texts: list[str], batch_size: int = 64):
            import numpy as np

            return np.zeros((len(texts), dim), dtype=np.float32)

    root = tmp_path / "project"
    root.mkdir()
    a_file = root / "a.py"
    b_file = root / "b.py"
    a_file.write_text("def func_a():\n    return 1\n")
    b_file.write_text("def func_b():\n    return 2\n")

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

    def _new_task(task_id: str, total_files: int) -> None:
        runtime.index_tasks[task_id] = IndexTask(
            task_id=task_id,
            project=str(root),
            started_at=time.monotonic(),
            total_files=total_files,
        )

    _new_task("t1", 2)
    asyncio.run(_run_index_bg("t1", root, [a_file, b_file], incremental=True, runtime=runtime))
    assert runtime.index_tasks["t1"].indexed_files == 2

    b_path = str(b_file)
    assert b_path in chunker.files_under(str(root))
    total_before = dense.count_by_project(str(root))
    assert total_before > 0

    b_file.unlink()

    _new_task("t2", 1)
    asyncio.run(_run_index_bg("t2", root, [a_file], incremental=True, runtime=runtime))

    assert b_path not in chunker.files_under(str(root))
    assert dense.count_by_project(str(root)) < total_before
    assert dense._table.search().where(f"file_path = '{b_path}'").to_list() == []


def _build_real_runtime(tmp_path: Path) -> SimpleNamespace:
    from reporag.indexer.chunker import ChunkIndexer
    from reporag.indexer.graph_builder import GraphDB
    from reporag.retrieval.dense import DenseIndex
    from reporag.retrieval.sparse import BM25Index

    dim = 4

    class _FakeEmbedder:
        def encode_corpus(self, texts: list[str], batch_size: int = 64):
            import numpy as np

            return np.zeros((len(texts), dim), dtype=np.float32)

    data = tmp_path / "data"
    graph_db = GraphDB(data / "dependency_graph.db")
    dense = DenseIndex(data, dim=dim)
    chunker = ChunkIndexer(
        data_dir=data,
        embedder=_FakeEmbedder(),
        dense_index=dense,
        bm25_index=BM25Index(),
        graph_db=graph_db,
    )
    return SimpleNamespace(
        chunker=chunker,
        dense=dense,
        graph_db=graph_db,
        reranker=None,
        reload_graph=Mock(),
        index_sem=asyncio.Semaphore(1),
        config=SimpleNamespace(index_batch_size=20),
        index_tasks={},
    )


def _start_task(runtime: SimpleNamespace, task_id: str, root: Path, total_files: int) -> None:
    runtime.index_tasks[task_id] = IndexTask(
        task_id=task_id,
        project=str(root),
        started_at=time.monotonic(),
        total_files=total_files,
    )


def test_orphan_only_run_refreshes_registry(tmp_path: Path, registry_path):
    """Regression: 0 changed files + purged orphans must still update registry totals."""
    pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    proj = registry_path
    root = tmp_path / "project"
    root.mkdir()
    a_file = root / "a.py"
    b_file = root / "b.py"
    a_file.write_text("def func_a():\n    return 1\n")
    b_file.write_text("def func_b():\n    return 2\n")

    runtime = _build_real_runtime(tmp_path)

    _start_task(runtime, "t1", root, 2)
    asyncio.run(_run_index_bg("t1", root, [a_file, b_file], incremental=True, runtime=runtime))
    chunks_before = proj.get(str(root))["chunks"]
    assert chunks_before > 0

    # Delete b.py; a.py unchanged — orphan-purge-only run.
    b_file.unlink()
    _start_task(runtime, "t2", root, 1)
    asyncio.run(_run_index_bg("t2", root, [a_file], incremental=True, runtime=runtime))

    assert runtime.index_tasks["t2"].indexed_files == 0
    entry = proj.get(str(root))
    assert entry["chunks"] == runtime.dense.count_by_project(str(root))
    assert entry["chunks"] < chunks_before


def test_desync_meta_without_chunks_forces_full_reindex(tmp_path: Path, registry_path):
    """Regression: file_meta.db rows + empty dense store must not skip all files."""
    pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    root = tmp_path / "project"
    root.mkdir()
    py_file = root / "a.py"
    py_file.write_text("def func_a():\n    return 1\n")

    runtime = _build_real_runtime(tmp_path)

    _start_task(runtime, "t1", root, 1)
    asyncio.run(_run_index_bg("t1", root, [py_file], incremental=True, runtime=runtime))
    assert runtime.dense.count_by_project(str(root)) > 0

    # Simulate dense store wipe while file_meta.db survives.
    safe_root = str(root).replace("'", "''")
    runtime.dense._table.delete(f"file_path LIKE '{safe_root}/%'")
    assert runtime.dense.count_by_project(str(root)) == 0
    assert runtime.chunker.files_under(str(root)) != []

    _start_task(runtime, "t2", root, 1)
    asyncio.run(_run_index_bg("t2", root, [py_file], incremental=True, runtime=runtime))

    assert runtime.index_tasks["t2"].status == "done"
    assert runtime.index_tasks["t2"].indexed_files == 1
    assert runtime.dense.count_by_project(str(root)) > 0


def test_interrupted_index_resumes_from_last_batch(tmp_path: Path):
    """Mid-index crash (or killed Claude session) must not lose completed batches.

    Per-batch write order is dense upsert -> file_meta record, so every file
    with a meta row already has its chunks persisted. A later run with
    incremental=True skips those files and indexes only the remainder.
    """
    pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
    pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    import numpy as np

    class _FlakyEmbedder:
        """Succeeds on the first batch, then raises — simulates a killed session."""

        def __init__(self) -> None:
            self.calls = 0
            self.fail_from_call = 2

        def encode_corpus(self, texts: list[str], batch_size: int = 64):
            self.calls += 1
            if self.calls >= self.fail_from_call:
                raise RuntimeError("session killed mid-index")
            return np.zeros((len(texts), 4), dtype=np.float32)

    root = tmp_path / "project"
    root.mkdir()
    files = []
    for name in ("a", "b", "c"):
        f = root / f"{name}.py"
        f.write_text(f"def func_{name}():\n    return 1\n")
        files.append(f)

    runtime = _build_real_runtime(tmp_path)
    embedder = _FlakyEmbedder()
    runtime.chunker._embedder = embedder

    with pytest.raises(RuntimeError, match="session killed"):
        asyncio.run(
            runtime.chunker.index_files_batched_async(
                files, incremental=True, file_batch_size=1
            )
        )

    # First batch persisted: chunks in dense, meta row recorded.
    assert runtime.dense.count_by_project(str(root)) > 0
    done_files = runtime.chunker.files_under(str(root))
    assert len(done_files) == 1

    # "New session": embedder healthy again. Only the remaining files index.
    embedder.fail_from_call = 999
    stats = asyncio.run(
        runtime.chunker.index_files_batched_async(files, incremental=True, file_batch_size=1)
    )

    assert stats["skipped"] == 1
    assert stats["files"] == 2
    assert sorted(runtime.chunker.files_under(str(root))) == sorted(str(f) for f in files)
    assert runtime.dense.count_by_project(str(root)) >= 3


def test_run_skips_project_with_reporag_ignore(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / ".reporag-ignore").touch()
    (root / "main.py").write_text("x = 1\n")

    runtime = SimpleNamespace(index_tasks={})
    result = asyncio.run(run({"path": str(root)}, runtime))

    assert result == {
        "status": "skipped",
        "reason": ".reporag-ignore present",
        "project": str(root),
    }
    assert runtime.index_tasks == {}
