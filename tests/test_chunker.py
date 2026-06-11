"""Unit tests for AST parser and semantic text extraction."""

import textwrap
from pathlib import Path

import pytest

tree_sitter = pytest.importorskip("tree_sitter", reason="tree-sitter not installed")
pytest.importorskip("tree_sitter_python", reason="tree-sitter-python not installed")

from reporag.indexer.ast_parser import Chunk, detect_language, parse_file  # noqa: E402
from reporag.indexer.semantic_text import chunk_to_semantic_text  # noqa: E402


@pytest.fixture()
def py_file(tmp_path: Path) -> Path:
    code = textwrap.dedent('''
        """Module docstring."""

        class UserService:
            """Manages user operations."""

            def create_user(self, name: str, email: str) -> dict:
                """Create a new user record."""
                return {"name": name, "email": email}

            async def delete_user(self, user_id: int) -> bool:
                return True

        def standalone_function(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y
    ''')
    p = tmp_path / "user_service.py"
    p.write_text(code)
    return p


@pytest.fixture()
def ts_file(tmp_path: Path) -> Path:
    code = textwrap.dedent("""
        interface User {
            id: number;
            name: string;
        }

        class AuthService {
            login(email: string, password: string): Promise<User> {
                return Promise.resolve({ id: 1, name: "test" });
            }
        }

        function hashPassword(pw: string): string {
            return pw;
        }
    """)
    p = tmp_path / "auth.ts"
    p.write_text(code)
    return p


def test_detect_language_python():
    assert detect_language(Path("foo.py")) == "python"


def test_detect_language_typescript():
    assert detect_language(Path("bar.ts")) == "typescript"


def test_detect_language_unsupported():
    assert detect_language(Path("readme.md")) is None


def test_parse_python_extracts_class(py_file: Path):
    chunks = parse_file(py_file)
    types = [c.chunk_type for c in chunks]
    assert "class" in types


def test_parse_python_extracts_functions(py_file: Path):
    chunks = parse_file(py_file)
    names = [c.name for c in chunks]
    assert "create_user" in names or "standalone_function" in names


def test_parse_python_module_chunk(py_file: Path):
    chunks = parse_file(py_file)
    assert chunks[0].chunk_type == "module"


def test_parse_python_line_ranges(py_file: Path):
    chunks = parse_file(py_file)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_parse_python_chunk_ids_unique(py_file: Path):
    chunks = parse_file(py_file)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_parse_typescript(ts_file: Path):
    chunks = parse_file(ts_file)
    assert len(chunks) > 0
    names = [c.name for c in chunks]
    assert any(n in names for n in ("AuthService", "hashPassword", "login"))


def test_semantic_text_function():
    chunk = Chunk(
        id="x",
        file_path="auth.py",
        language="python",
        chunk_type="function",
        name="create_user",
        raw_content="def create_user(name: str, email: str) -> dict:\n    pass",
        start_line=1,
        end_line=2,
        existing_docstring="Create a new user record.",
    )
    text = chunk_to_semantic_text(chunk)
    assert "create user" in text.lower() or "function" in text.lower()
    assert "Create a new user record" in text


def test_semantic_text_class():
    chunk = Chunk(
        id="y",
        file_path="service.py",
        language="python",
        chunk_type="class",
        name="UserService",
        raw_content="class UserService(BaseService):\n    pass",
        start_line=1,
        end_line=2,
        existing_docstring="Manages user operations.",
    )
    text = chunk_to_semantic_text(chunk)
    assert "UserService" in text or "user service" in text.lower()


def test_semantic_text_no_empty(py_file: Path):
    chunks = parse_file(py_file)
    for c in chunks:
        text = chunk_to_semantic_text(c)
        assert text.strip(), f"Empty semantic text for chunk {c.name}"


def test_rebuild_bm25_on_unopened_table(tmp_path: Path):
    """_rebuild_bm25 must not crash when DenseIndex._table is still None.

    Regression for: 'NoneType' object has no attribute 'search' — happened
    when no batch ever upserted records (e.g. all files skipped via
    incremental check), leaving DenseIndex._table unset.
    """
    pytest.importorskip("lancedb", reason="lancedb not installed")
    pytest.importorskip("bm25s", reason="bm25s not installed")

    from reporag.indexer.chunker import ChunkIndexer
    from reporag.retrieval.dense import DenseIndex
    from reporag.retrieval.sparse import BM25Index

    dense = DenseIndex(tmp_path / "data", dim=4)
    assert dense._table is None  # never opened

    indexer = ChunkIndexer(
        data_dir=tmp_path / "data",
        embedder=None,
        dense_index=dense,
        bm25_index=BM25Index(),
    )

    indexer._rebuild_bm25()  # must not raise

    assert dense._table is not None
    assert indexer._bm25.is_ready is False
