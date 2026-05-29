"""
Persistent memory store for decisions, discoveries, and project knowledge.

Storage: SQLite with FTS5 virtual table for keyword search.
Retrieval: FTS5 MATCH (keyword) + optional numpy cosine for embedding similarity.
Zero external dependencies beyond stdlib sqlite3.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

CREATE_MEMORIES = """
CREATE TABLE IF NOT EXISTS memories (
    id       TEXT PRIMARY KEY,
    content  TEXT NOT NULL,
    tags     TEXT NOT NULL DEFAULT '[]',
    category TEXT NOT NULL DEFAULT 'general',
    created_at REAL NOT NULL,
    embedding BLOB
);
"""

CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
USING fts5(content, tags, content=memories, content_rowid=rowid);
"""

CREATE_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.rowid, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.rowid, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.rowid, new.content, new.tags);
END;
"""


class MemoryStore:
    """SQLite-backed persistent memory with FTS5 keyword search."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(CREATE_MEMORIES + CREATE_FTS + CREATE_FTS_TRIGGERS)
        self._conn.commit()

    def remember(
        self,
        content: str,
        tags: list[str] | None = None,
        category: str = "general",
        embedding: bytes | None = None,
    ) -> str:
        """Store a memory entry. Returns the generated ID."""
        mem_id = hashlib.sha256(f"{content}{time.time()}".encode()).hexdigest()[:16]
        tags_json = json.dumps(tags or [])
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO memories (id, content, tags, category, created_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (mem_id, content, tags_json, category, time.time(), embedding),
        )
        self._conn.commit()
        return mem_id

    def recall(
        self,
        query: str,
        tags: list[str] | None = None,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        Search memories by keyword (FTS5) with optional tag + category filter.

        Returns list of memory dicts sorted by FTS5 relevance.
        """
        cur = self._conn.cursor()
        fts_query = _build_fts_query(query, tags)
        base_sql = """
            SELECT m.id, m.content, m.tags, m.category, m.created_at,
                   bm25(memories_fts) AS score
            FROM memories_fts
            JOIN memories m ON memories_fts.rowid = m.rowid
            WHERE memories_fts MATCH ?
        """
        params: list[Any] = [fts_query]
        if category:
            base_sql += " AND m.category = ?"
            params.append(category)
        base_sql += " ORDER BY score LIMIT ?"
        params.append(limit)

        rows = cur.execute(base_sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_by_id(self, mem_id: str) -> dict[str, Any] | None:
        """Retrieve a single memory by ID."""
        cur = self._conn.cursor()
        row = cur.execute("SELECT * FROM memories WHERE id = ?", (mem_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def delete(self, mem_id: str) -> bool:
        """Delete a memory by ID. Returns True if found and deleted."""
        cur = self._conn.cursor()
        cur.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._conn.close()


def _build_fts_query(query: str, tags: list[str] | None) -> str:
    """
    Build FTS5 MATCH expression from query + optional tags.

    Splits query into individual tokens (OR match) so "RRF fusion" matches
    documents containing either "RRF" or "fusion", not the exact phrase.
    """
    terms: list[str] = [t for t in query.split() if t.strip()]
    if tags:
        terms.extend(t.strip() for t in tags if t.strip())
    if not terms:
        return '""'
    # Sanitize: remove FTS5 special chars that would cause syntax errors
    safe = [t.replace('"', "").replace("(", "").replace(")", "") for t in terms]
    safe = [t for t in safe if t]
    return " OR ".join(safe) if safe else '""'


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["tags"] = json.loads(d.get("tags") or "[]")
    d.pop("embedding", None)
    return d
