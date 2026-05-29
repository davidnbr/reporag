"""MCP tool: get_symbol — exact symbol lookup via SQLite graph DB."""
from __future__ import annotations

from typing import Any


async def run(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    """
    Execute get_symbol tool.

    Args:
        arguments: {
            name: str (required),
            language: str | None,
            fuzzy: bool (default False),
        }
    """
    name: str = arguments.get("name", "").strip()
    if not name:
        return {"error": "name is required"}

    language: str | None = arguments.get("language")
    fuzzy: bool = arguments.get("fuzzy", False)

    if fuzzy:
        # LanceDB semantic search for approximate name match
        q_vec = runtime.embedder.encode_query(name)
        chunk_ids = runtime.dense.search(q_vec, k=5)
        chunks = runtime.dense.get_chunks(chunk_ids)
        if language:
            chunks = [c for c in chunks if c.get("language") == language]
        return {"results": _format_chunks(chunks[:5]), "mode": "fuzzy"}

    # Exact SQLite lookup — <50ms
    rows = runtime.graph_db.get_symbol(name, language)
    if not rows:
        return {"results": [], "mode": "exact", "message": f"Symbol '{name}' not found"}

    return {
        "results": [
            {
                "name": r["name"],
                "type": r["symbol_type"],
                "file": r["file_path"],
                "language": r["language"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
            }
            for r in rows
        ],
        "mode": "exact",
    }


def _format_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": c.get("name", ""),
            "type": c.get("chunk_type", ""),
            "file": c.get("file_path", ""),
            "language": c.get("language", ""),
            "start_line": c.get("start_line", 0),
            "end_line": c.get("end_line", 0),
            "snippet": c.get("semantic_text", "")[:200],
        }
        for c in chunks
    ]
