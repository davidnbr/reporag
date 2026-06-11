"""
Dependency graph builder — orchestrates SCIP (primary) + heuristic fallback.

Research §6: "To build a mathematically correct dependency graph, we bypass
regex-based parsing and use compiler-level linkers."

Stores all edges in SQLite dependency_graph.db.
Loads NetworkX DiGraph for PageRank computation.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx

from reporag.indexer import scip_indexer
from reporag.indexer.heuristic_graph import extract_imports

logger = logging.getLogger(__name__)

CREATE_SYMBOLS = """
CREATE TABLE IF NOT EXISTS symbols (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    language    TEXT NOT NULL,
    symbol_type TEXT NOT NULL,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
"""

CREATE_EDGES = """
CREATE TABLE IF NOT EXISTS edges (
    src       TEXT NOT NULL,
    dst       TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    source    TEXT NOT NULL DEFAULT 'scip',
    PRIMARY KEY (src, dst, edge_type)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""

CREATE_INDEX_LOG = """
CREATE TABLE IF NOT EXISTS index_log (
    file_path TEXT PRIMARY KEY,
    language  TEXT,
    graph_source TEXT,  -- 'scip' or 'heuristic'
    indexed_at REAL
);
"""


class GraphDB:
    """SQLite graph store for symbols and dependency edges."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(CREATE_SYMBOLS + CREATE_EDGES + CREATE_INDEX_LOG)
        self._conn.commit()

    def upsert_symbol(self, sym: dict[str, Any]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO symbols (id, name, file_path, language, symbol_type, start_line, end_line) "
            "VALUES (:id, :name, :file_path, :language, :symbol_type, :start_line, :end_line)",
            sym,
        )

    def upsert_edge(self, src: str, dst: str, edge_type: str, source: str = "scip") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO edges (src, dst, edge_type, source) VALUES (?, ?, ?, ?)",
            (src, dst, edge_type, source),
        )

    def delete_file(self, file_path: str) -> None:
        self._conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
        self._conn.execute("DELETE FROM edges WHERE src = ?", (file_path,))
        self._conn.execute("DELETE FROM index_log WHERE file_path = ?", (file_path,))

    def delete_symbols_for_file(self, file_path: str) -> None:
        self._conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))

    def log_file(self, file_path: str, language: str, graph_source: str) -> None:
        import time

        self._conn.execute(
            "INSERT OR REPLACE INTO index_log (file_path, language, graph_source, indexed_at) VALUES (?, ?, ?, ?)",
            (file_path, language, graph_source, time.time()),
        )

    def commit(self) -> None:
        self._conn.commit()

    def get_symbol(self, name: str, language: str | None = None) -> list[dict[str, Any]]:
        if language:
            rows = self._conn.execute(
                "SELECT * FROM symbols WHERE name = ? AND language = ?", (name, language)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM symbols WHERE name = ?", (name,)).fetchall()
        return [dict(r) for r in rows]

    def load_networkx_graph(self) -> nx.DiGraph:
        """Load all edges into a NetworkX DiGraph for PageRank computation."""
        G: nx.DiGraph = nx.DiGraph()
        rows = self._conn.execute("SELECT src, dst FROM edges").fetchall()
        for row in rows:
            G.add_edge(row["src"], row["dst"])
        return G

    def coverage_report(self) -> dict[str, int]:
        """Return count of files indexed by SCIP vs heuristic."""
        rows = self._conn.execute(
            "SELECT graph_source, COUNT(*) as cnt FROM index_log GROUP BY graph_source"
        ).fetchall()
        return {r["graph_source"]: r["cnt"] for r in rows}

    def close(self) -> None:
        self._conn.close()


def build_graph_for_project(
    root: Path,
    files: list[Path],
    db: GraphDB,
    scip_languages: list[str] | None = None,
) -> dict[str, int]:
    """
    Build dependency graph for a list of files.

    For each language: try SCIP first, fall back to heuristic.
    Returns coverage stats: {'scip': N, 'heuristic': M}.
    """
    # Detect languages present in this file list
    from reporag.indexer.ast_parser import detect_language

    if scip_languages is None:
        scip_languages = scip_indexer.available_languages()
        if scip_languages:
            logger.info("SCIP available for: %s", scip_languages)
        else:
            logger.info("No SCIP CLIs found — using heuristic graph for all languages")

    # Group files by language
    lang_files: dict[str, list[Path]] = {}
    for f in files:
        lang = detect_language(f)
        if lang:
            lang_files.setdefault(lang, []).append(f)

    stats = {"scip": 0, "heuristic": 0}

    for language, lang_file_list in lang_files.items():
        if language in scip_languages:
            _build_scip(root, language, lang_file_list, db, stats)
        else:
            _build_heuristic(root, language, lang_file_list, db, stats)

    db.commit()
    return stats


def _build_scip(
    root: Path,
    language: str,
    files: list[Path],
    db: GraphDB,
    stats: dict[str, int],
) -> None:
    logger.info("Running SCIP for %s (%d files)", language, len(files))
    edges = scip_indexer.run_scip(root, language)

    if not edges:
        logger.warning("SCIP returned no edges for %s — falling back to heuristic", language)
        _build_heuristic(root, language, files, db, stats)
        return

    for edge in edges:
        db.upsert_edge(edge.src_file, edge.dst_file, edge.edge_type, source="scip")

    for f in files:
        db.log_file(str(f), language, "scip")
        stats["scip"] += 1


def _build_heuristic(
    root: Path,
    language: str,
    files: list[Path],
    db: GraphDB,
    stats: dict[str, int],
) -> None:
    for f in files:
        try:
            edges = extract_imports(f, root)
            for edge in edges:
                db.upsert_edge(edge.src_file, edge.dst_file, edge.edge_type, source="heuristic")
            db.log_file(str(f), language, "heuristic")
            stats["heuristic"] += 1
        except Exception as exc:
            logger.warning("Heuristic graph failed for %s: %s", f, exc)
