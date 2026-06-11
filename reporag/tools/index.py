"""MCP tool: index_codebase — parse, embed, and graph-index a project directory.

Returns immediately with a task_id. Indexing runs in background; first batch
of results is queryable within seconds. Use index_status to track progress.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EXCLUDE_DEFAULTS = {
    "node_modules",
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".mypy_cache",
    ".ruff_cache",
    "target",
    ".codebrain",
    ".reporag",
    ".devenv",
    ".direnv",
    "vendor",
    "pkg",
    ".cache",
}

# Files indexed first — LLMs ask about entry points immediately
_ENTRY_POINTS = {
    "main.py",
    "app.py",
    "server.py",
    "__init__.py",
    "cli.py",
    "wsgi.py",
    "asgi.py",
    "main.ts",
    "index.ts",
    "app.ts",
    "server.ts",
    "index.js",
    "main.js",
    "main.go",
    "main.rs",
    "lib.rs",
    "mod.rs",
    "Main.java",
    "Application.java",
    "main.c",
    "main.cpp",
}


def _priority_sort(files: list[Path]) -> list[Path]:
    """Entry points first, then most-recently-modified first."""

    def score(f: Path) -> tuple:
        return (0 if f.name in _ENTRY_POINTS else 1, -f.stat().st_mtime)

    return sorted(files, key=score)


async def _run_index_bg(
    task_id: str,
    root: Path,
    files: list[Path],
    incremental: bool,
    runtime: Any,
) -> None:
    """Background coroutine: runs full index pipeline, updates task progress."""
    task = runtime.index_tasks[task_id]
    try:
        async with runtime.index_sem:
            known_files = runtime.chunker.files_under(str(root))
            current_files = {str(f) for f in files}
            orphans = [f for f in known_files if f not in current_files]
            for orphan in orphans:
                runtime.dense.delete_by_file(orphan)
                runtime.chunker.forget_file(orphan)
                runtime.graph_db.delete_file(orphan)
            if orphans:
                runtime.graph_db.commit()
                logger.info("Index task %s: purged %d orphaned files", task_id, len(orphans))

            def on_batch(files_done: int, chunks_done: int, skipped: int) -> None:
                task.indexed_files = files_done
                task.indexed_chunks = chunks_done
                task.skipped_files = skipped

            chunk_stats = await runtime.chunker.index_files_batched_async(
                files,
                incremental=incremental,
                file_batch_size=runtime.config.index_batch_size,
                on_batch=on_batch,
                force_bm25_rebuild=bool(orphans),
            )

            task.indexed_files = chunk_stats["files"]
            task.indexed_chunks = chunk_stats["chunks"]
            task.skipped_files = chunk_stats["skipped"]

            if chunk_stats["files"] == 0:
                # Nothing changed — skip graph rebuild, reload, reranker
                # invalidation, and registry update entirely.
                task.status = "done"
                task.finished_at = time.monotonic()
                logger.info(
                    "Index task %s done: no changes (%d files skipped, %.1fs)",
                    task_id,
                    task.skipped_files,
                    task.finished_at - task.started_at,
                )
                return

            from reporag.indexer.graph_builder import build_graph_for_project

            loop = asyncio.get_running_loop()
            graph_stats = await loop.run_in_executor(
                None, build_graph_for_project, root, files, runtime.graph_db
            )
            runtime.reload_graph()
            if runtime.reranker is not None:
                runtime.reranker.invalidate_cache()

            task.graph_edges_scip = graph_stats.get("scip", 0)
            task.graph_edges_heuristic = graph_stats.get("heuristic", 0)
            task.status = "done"
            task.finished_at = time.monotonic()
            try:
                from reporag.projects import update as _reg_update

                total_chunks = runtime.dense.count_by_project(str(root))
                total_files = runtime.chunker.count_files(str(root))
                _reg_update(str(root), total_chunks, total_files)
            except Exception:
                logger.exception("Index task %s: registry update failed", task_id)
            logger.info(
                "Index task %s done: %d files, %d chunks (%.1fs)",
                task_id,
                task.indexed_files,
                task.indexed_chunks,
                task.finished_at - task.started_at,
            )
    except Exception as exc:
        task.status = "error"
        task.error = str(exc)
        task.finished_at = time.monotonic()
        logger.exception("Index task %s failed", task_id)


async def run(
    arguments: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    raw_path = arguments.get("path", ".")
    root = Path(raw_path).expanduser().resolve()

    if not root.exists():
        return {"error": f"Path does not exist: {root}"}
    if not root.is_dir():
        return {"error": f"Path is not a directory: {root}"}

    incremental: bool = arguments.get("incremental", True)
    languages_filter: list[str] | None = arguments.get("languages")
    extra_excludes: list[str] = arguments.get("exclude_patterns", [])
    exclude = _EXCLUDE_DEFAULTS | set(extra_excludes)

    from reporag.indexer.ast_parser import detect_language

    files: list[Path] = []
    for f in root.rglob("*"):
        if f.is_file() and not any(part in exclude for part in f.parts):
            lang = detect_language(f)
            if lang and (not languages_filter or lang in languages_filter):
                files.append(f)

    if not files:
        return {
            "indexed_files": 0,
            "chunks": 0,
            "graph_edges": 0,
            "message": "No supported source files found.",
        }

    files = _priority_sort(files)

    task_id = uuid.uuid4().hex[:12]
    from reporag.server import IndexTask

    task = IndexTask(
        task_id=task_id,
        project=str(root),
        started_at=time.monotonic(),
        total_files=len(files),
        incremental=incremental,
    )
    runtime.index_tasks[task_id] = task

    # Track project for file watcher
    runtime.watched_projects.add(str(root))
    runtime._start_watcher(str(root))

    asyncio.create_task(_run_index_bg(task_id, root, files, incremental, runtime))

    return {
        "status": "indexing_started",
        "task_id": task_id,
        "total_files": len(files),
        "incremental": incremental,
        "project": str(root),
        "message": (
            f"Indexing {len(files)} files in background. "
            f"First results available after batch 1 (~{runtime.config.index_batch_size} files). "
            f"Use index_status to track progress."
        ),
    }
