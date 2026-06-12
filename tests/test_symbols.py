"""Tests for symbol-table population during indexing (Task 1)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest

tree_sitter = pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")
pytest.importorskip("lancedb", reason="lancedb not installed")
pytest.importorskip("bm25s", reason="bm25s not installed")

from reporag.indexer.chunker import ChunkIndexer  # noqa: E402
from reporag.indexer.graph_builder import GraphDB  # noqa: E402
from reporag.retrieval.dense import DenseIndex  # noqa: E402
from reporag.retrieval.sparse import BM25Index  # noqa: E402

_DIM = 4


class _FakeEmbedder:
    """Returns deterministic zero vectors — no model download needed."""

    def encode_corpus(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return np.zeros((len(texts), _DIM), dtype=np.float32)


@pytest.fixture()
def py_file(tmp_path: Path) -> Path:
    code = textwrap.dedent('''
        """Module docstring."""

        class MyClass:
            """A class."""

            def my_method(self) -> int:
                return 1


        def my_func(x: int) -> int:
            """Add one."""
            return x + 1
    ''')
    p = tmp_path / "module_a.py"
    p.write_text(code)
    return p


@pytest.fixture()
def indexer(tmp_path: Path) -> tuple[ChunkIndexer, GraphDB]:
    data = tmp_path / "data"
    graph_db = GraphDB(data / "dependency_graph.db")
    dense = DenseIndex(data, dim=_DIM)
    bm25 = BM25Index()
    chunker = ChunkIndexer(
        data_dir=data,
        embedder=_FakeEmbedder(),
        dense_index=dense,
        bm25_index=bm25,
        graph_db=graph_db,
    )
    return chunker, graph_db


def test_get_symbol_finds_function(indexer: tuple[ChunkIndexer, GraphDB], py_file: Path):
    chunker, graph_db = indexer
    chunker.index_files([py_file], incremental=False)

    rows = graph_db.get_symbol("my_func")
    assert len(rows) == 1
    assert rows[0]["file_path"] == str(py_file)
    assert rows[0]["symbol_type"] == "function"
    assert rows[0]["start_line"] >= 1


def test_get_symbol_finds_class_and_method(indexer: tuple[ChunkIndexer, GraphDB], py_file: Path):
    chunker, graph_db = indexer
    chunker.index_files([py_file], incremental=False)

    class_rows = graph_db.get_symbol("MyClass")
    assert len(class_rows) == 1
    assert class_rows[0]["symbol_type"] == "class"

    method_rows = graph_db.get_symbol("my_method")
    assert len(method_rows) == 1
    assert method_rows[0]["symbol_type"] == "function"


def test_get_symbol_project_filter_excludes_other_roots(tmp_path: Path):
    """Same-named symbols in another project must not leak into a scoped lookup."""
    graph_db = GraphDB(tmp_path / "graph.db")
    for root in ("/projA", "/projB"):
        graph_db.upsert_symbol(
            {
                "id": f"{root}-my_func",
                "name": "my_func",
                "file_path": f"{root}/mod.py",
                "language": "python",
                "symbol_type": "function",
                "start_line": 1,
                "end_line": 2,
            }
        )
    graph_db.commit()

    rows = graph_db.get_symbol("my_func", project="/projA")
    assert [r["file_path"] for r in rows] == ["/projA/mod.py"]

    rows_all = graph_db.get_symbol("my_func")
    assert len(rows_all) == 2


def test_reindex_modified_file_no_duplicate_symbols(
    indexer: tuple[ChunkIndexer, GraphDB], py_file: Path
):
    chunker, graph_db = indexer
    chunker.index_files([py_file], incremental=False)
    assert len(graph_db.get_symbol("my_func")) == 1

    # Modify the file (add a comment) and re-index incrementally
    py_file.write_text(py_file.read_text() + "\n# trailing comment\n")
    chunker.index_files([py_file], incremental=True)

    assert len(graph_db.get_symbol("my_func")) == 1
    assert len(graph_db.get_symbol("MyClass")) == 1


def test_unchanged_reindex_keeps_symbols(indexer: tuple[ChunkIndexer, GraphDB], py_file: Path):
    chunker, graph_db = indexer
    chunker.index_files([py_file], incremental=False)
    assert len(graph_db.get_symbol("my_func")) == 1

    # Re-index with no changes — incremental skip must not wipe symbols
    chunker.index_files([py_file], incremental=True)
    assert len(graph_db.get_symbol("my_func")) == 1
