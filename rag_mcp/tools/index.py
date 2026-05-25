"""MCP tool: index_codebase — parse, embed, and graph-index a project directory."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_EXCLUDE_DEFAULTS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".mypy_cache", ".ruff_cache", "target", ".rag-mcp",
    ".devenv", ".direnv", "vendor", "pkg", ".cache",
}


async def run(
    arguments: dict[str, Any],
    runtime: "Runtime",  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    """
    Execute index_codebase tool.

    Args:
        arguments: {
            path: str (required) — absolute or relative project root,
            incremental: bool (default True),
            languages: list[str] | None — restrict to these languages,
            exclude_patterns: list[str] | None — additional dir names to skip,
        }
        runtime: shared server runtime state.
    """
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

    from rag_mcp.indexer.ast_parser import LANGUAGE_EXT, detect_language

    # Collect all source files
    files: list[Path] = []
    for f in root.rglob("*"):
        if f.is_file() and not any(part in exclude for part in f.parts):
            lang = detect_language(f)
            if lang and (not languages_filter or lang in languages_filter):
                files.append(f)

    if not files:
        return {"indexed_files": 0, "chunks": 0, "graph_edges": 0, "message": "No supported source files found."}

    logger.info("Found %d source files in %s", len(files), root)

    # Chunk + embed
    import time
    t0 = time.monotonic()
    if not incremental:
        deleted = runtime.dense.delete_by_project(str(root))
        logger.info("Full re-index: removed %d stale chunks for %s", deleted, root)
    chunk_stats = runtime.chunker.index_files(files, incremental=incremental)

    # Build dependency graph
    from rag_mcp.indexer.graph_builder import build_graph_for_project
    graph_stats = build_graph_for_project(root, files, runtime.graph_db)

    # Reload NetworkX graph into runtime
    runtime.reload_graph()

    # Invalidate reranker cache — indexed content changed
    if runtime.reranker is not None:
        runtime.reranker.invalidate_cache()

    elapsed = time.monotonic() - t0
    coverage = runtime.graph_db.coverage_report()

    return {
        "indexed_files": chunk_stats["files"],
        "skipped_files": chunk_stats["skipped"],
        "chunks": chunk_stats["chunks"],
        "graph_edges_scip": graph_stats.get("scip", 0),
        "graph_edges_heuristic": graph_stats.get("heuristic", 0),
        "graph_coverage": coverage,
        "duration_s": round(elapsed, 2),
        "root": str(root),
    }
