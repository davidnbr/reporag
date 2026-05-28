"""
Orchestrates AST parsing → semantic text extraction → embedding → LanceDB upsert.

This is the main indexing pipeline entry point.
Supports incremental re-index: skips files unchanged since last index (mtime + hash).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from codebrain.indexer.ast_parser import Chunk, parse_file
from codebrain.indexer.semantic_text import chunk_to_semantic_text
from codebrain.indexer.sliding_window import hybrid_chunks, sliding_window_chunks

logger = logging.getLogger(__name__)

CREATE_FILE_INDEX = """
CREATE TABLE IF NOT EXISTS file_index (
    file_path  TEXT PRIMARY KEY,
    mtime      REAL NOT NULL,
    file_hash  TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    indexed_at REAL NOT NULL
);
"""


class ChunkIndexer:
    """Full pipeline: parse → embed → store chunks."""

    def __init__(
        self,
        data_dir: Path,
        embedder: Any,
        dense_index: Any,
        bm25_index: Any,
        chunk_strategy: str = "hybrid",
        chunk_window_lines: int = 64,
        chunk_overlap_lines: int = 16,
    ) -> None:
        self._data_dir = data_dir
        self._embedder = embedder
        self._dense = dense_index
        self._bm25 = bm25_index
        self._chunk_strategy = chunk_strategy
        self._chunk_window_lines = chunk_window_lines
        self._chunk_overlap_lines = chunk_overlap_lines
        self._meta_conn = self._init_meta_db()

    def _init_meta_db(self) -> sqlite3.Connection:
        self._data_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._data_dir / "file_meta.db"), check_same_thread=False)
        conn.executescript(CREATE_FILE_INDEX)
        conn.commit()
        return conn

    def _file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _is_unchanged(self, path: Path) -> bool:
        row = self._meta_conn.execute(
            "SELECT mtime, file_hash FROM file_index WHERE file_path = ?", (str(path),)
        ).fetchone()
        if not row:
            return False
        current_mtime = path.stat().st_mtime
        if abs(current_mtime - row[0]) > 0.01:
            return False
        return self._file_hash(path) == row[1]

    def _record_file(self, path: Path, chunk_count: int) -> None:
        self._meta_conn.execute(
            "INSERT OR REPLACE INTO file_index (file_path, mtime, file_hash, chunk_count, indexed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(path), path.stat().st_mtime, self._file_hash(path), chunk_count, time.time()),
        )
        self._meta_conn.commit()

    def index_files(
        self,
        files: list[Path],
        incremental: bool = True,
        batch_size: int = 64,
    ) -> dict[str, int]:
        """
        Index a list of files. Returns stats dict.

        Args:
            files: Source files to index.
            incremental: Skip files unchanged since last index.
            batch_size: Embedding batch size.
        """
        to_index = [f for f in files if not (incremental and self._is_unchanged(f))]
        logger.info("Indexing %d/%d files (incremental=%s)", len(to_index), len(files), incremental)

        all_chunks: list[Chunk] = []
        for path in to_index:
            try:
                if self._chunk_strategy == "sliding":
                    chunks = sliding_window_chunks(path, self._chunk_window_lines, self._chunk_overlap_lines)
                elif self._chunk_strategy == "hybrid":
                    chunks = hybrid_chunks(path, self._chunk_window_lines, self._chunk_overlap_lines)
                else:
                    chunks = parse_file(path)
                all_chunks.extend(chunks)
            except Exception as exc:
                logger.warning("Parse failed for %s: %s", path, exc)

        if not all_chunks:
            return {"files": 0, "chunks": 0, "skipped": len(files) - len(to_index)}

        # Generate semantic text for all chunks
        semantic_texts = [chunk_to_semantic_text(c) for c in all_chunks]

        # Embed in batches
        logger.info("Embedding %d chunks...", len(all_chunks))
        embeddings: np.ndarray = self._embedder.encode_corpus(semantic_texts, batch_size=batch_size)

        # Build records for LanceDB
        records: list[dict[str, Any]] = []
        for chunk, text, vec in zip(all_chunks, semantic_texts, embeddings):
            records.append({
                "id": chunk.id,
                "file_path": chunk.file_path,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "semantic_text": text,
                "raw_content": chunk.raw_content[:4000],  # cap at 4KB
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "parent_name": chunk.parent_name or "",
                "vector": vec,
            })

        # Remove stale chunks for re-indexed files before upsert
        for path in to_index:
            self._dense.delete_by_file(str(path))

        # Upsert into LanceDB
        self._dense.upsert(records)

        # Rebuild BM25 index (full rebuild — fast, <10s for 200k LOC)
        self._rebuild_bm25()

        # Record file metadata
        file_chunk_counts: dict[str, int] = {}
        for chunk in all_chunks:
            file_chunk_counts[chunk.file_path] = file_chunk_counts.get(chunk.file_path, 0) + 1
        for path in to_index:
            count = file_chunk_counts.get(str(path), 0)
            self._record_file(path, count)

        return {
            "files": len(to_index),
            "chunks": len(all_chunks),
            "skipped": len(files) - len(to_index),
        }

    async def index_files_batched_async(
        self,
        files: list[Path],
        incremental: bool = True,
        file_batch_size: int = 20,
        on_batch: Callable[[int, int, int], None] | None = None,
    ) -> dict[str, int]:
        """
        Index files in batches, upserting to LanceDB after each batch.
        Results become queryable after the first batch (~seconds vs minutes).

        on_batch(files_indexed, chunks_indexed, files_skipped) called after each batch.
        """
        to_index = [f for f in files if not (incremental and self._is_unchanged(f))]
        skipped = len(files) - len(to_index)
        total_chunks = 0
        files_done = 0

        loop = asyncio.get_event_loop()

        for batch_start in range(0, len(to_index), file_batch_size):
            batch = to_index[batch_start : batch_start + file_batch_size]
            batch_chunks: list[Any] = []

            for path in batch:
                try:
                    if self._chunk_strategy == "sliding":
                        chunks = sliding_window_chunks(path, self._chunk_window_lines, self._chunk_overlap_lines)
                    elif self._chunk_strategy == "hybrid":
                        chunks = hybrid_chunks(path, self._chunk_window_lines, self._chunk_overlap_lines)
                    else:
                        chunks = parse_file(path)
                    batch_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning("Parse failed for %s: %s", path, exc)

            if batch_chunks:
                texts = [chunk_to_semantic_text(c) for c in batch_chunks]
                # Offload CPU-bound embedding to thread pool — keeps event loop responsive
                embeddings = await loop.run_in_executor(
                    None, self._embedder.encode_corpus, texts, 64
                )
                records = [
                    {
                        "id": chunk.id,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "chunk_type": chunk.chunk_type,
                        "name": chunk.name,
                        "semantic_text": text,
                        "raw_content": chunk.raw_content[:4000],
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "parent_name": chunk.parent_name or "",
                        "vector": vec,
                    }
                    for chunk, text, vec in zip(batch_chunks, texts, embeddings)
                ]
                for path in batch:
                    self._dense.delete_by_file(str(path))
                self._dense.upsert(records)

                file_chunk_counts: dict[str, int] = {}
                for chunk in batch_chunks:
                    file_chunk_counts[chunk.file_path] = file_chunk_counts.get(chunk.file_path, 0) + 1
                for path in batch:
                    self._record_file(path, file_chunk_counts.get(str(path), 0))

                total_chunks += len(batch_chunks)

            files_done += len(batch)
            if on_batch:
                on_batch(files_done, total_chunks, skipped)

            await asyncio.sleep(0)  # yield to event loop between batches

        # BM25 rebuild once at end (full scan of LanceDB — fast, <10s for 200k LOC)
        await loop.run_in_executor(None, self._rebuild_bm25)

        return {"files": files_done, "chunks": total_chunks, "skipped": skipped}

    def _rebuild_bm25(self) -> None:
        """Rebuild BM25 index from all chunks currently in LanceDB."""
        logger.info("Rebuilding BM25 index...")
        all_records = self._dense._table.search().select(["id", "semantic_text"]).to_list()
        ids = [r["id"] for r in all_records]
        texts = [r["semantic_text"] for r in all_records]
        if ids:
            self._bm25.build(ids, texts)
            bm25_path = self._data_dir / "bm25"
            self._bm25.save(bm25_path)
