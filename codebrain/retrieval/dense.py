"""
Dense vector retrieval via LanceDB (embedded, Apache 2.0) — research §4.

LanceDB stores chunk embeddings in Lance columnar format.
Cosine similarity search: score(q, d) = (v_q · v_d) / (||v_q|| ||v_d||)
Embeddings pre-normalized at index time so dot product == cosine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TABLE_NAME = "code_chunks"


def _schema(dim: int) -> Any:
    import pyarrow as pa
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("file_path", pa.string()),
        pa.field("language", pa.string()),
        pa.field("chunk_type", pa.string()),
        pa.field("name", pa.string()),
        pa.field("semantic_text", pa.string()),
        pa.field("raw_content", pa.string()),
        pa.field("start_line", pa.int32()),
        pa.field("end_line", pa.int32()),
        pa.field("parent_name", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])


class DenseIndex:
    """LanceDB-backed dense retrieval index."""

    def __init__(self, data_dir: Path, dim: int = 768) -> None:
        self._data_dir = data_dir
        self._dim = dim
        self._db: Any = None
        self._table: Any = None

    def _connect(self) -> None:
        if self._db is not None:
            return
        import lancedb
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._data_dir / "vectors"))

    def _open_or_create_table(self) -> None:
        self._connect()
        try:
            self._table = self._db.open_table(TABLE_NAME)
        except Exception:
            self._table = self._db.create_table(TABLE_NAME, schema=_schema(self._dim))

    def upsert(self, records: list[dict[str, Any]]) -> None:
        """Insert or overwrite records. Each record must include 'id' and 'vector'."""
        if not records:
            return
        self._open_or_create_table()
        import pyarrow as pa
        # Normalize vectors to list[float] before building arrow table
        for r in records:
            v = r.get("vector")
            if isinstance(v, np.ndarray):
                r["vector"] = v.tolist()
            elif not isinstance(v, list):
                r["vector"] = list(v)
        schema = _schema(self._dim)
        table = pa.Table.from_pylist(records, schema=schema)
        self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(table)

    def delete_by_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a file (for incremental re-index)."""
        self._open_or_create_table()
        self._table.delete(f"file_path = '{file_path}'")

    def delete_by_project(self, project_root: str) -> int:
        """Remove all chunks whose file_path starts with project_root. Returns deleted count."""
        self._open_or_create_table()
        before = self._table.count_rows()
        # LanceDB delete supports SQL-like WHERE; use starts_with via LIKE
        safe = project_root.replace("'", "''")
        self._table.delete(f"file_path LIKE '{safe}%'")
        return before - self._table.count_rows()

    def search(self, query_vec: np.ndarray, k: int = 50) -> list[str]:
        """Return top-k chunk IDs by cosine similarity."""
        self._open_or_create_table()
        results = (
            self._table.search(query_vec.tolist())
            .metric("cosine")
            .limit(k)
            .select(["id", "_distance"])
            .to_list()
        )
        return [r["id"] for r in results]

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        """Fetch full chunk records by IDs."""
        self._open_or_create_table()
        if not ids:
            return []
        id_list = ", ".join(f"'{i}'" for i in ids)
        return self._table.search().where(f"id IN ({id_list})").to_list()

    def count(self) -> int:
        """Return total number of indexed chunks."""
        self._open_or_create_table()
        return self._table.count_rows()
