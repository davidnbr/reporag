"""MCP tools: remember + recall — persistent cross-session knowledge store."""

from __future__ import annotations

from typing import Any

VALID_CATEGORIES = {"decision", "discovery", "pattern", "architecture", "note", "general"}


async def run_remember(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    """
    Execute remember tool.

    Args:
        arguments: {
            content: str (required),
            tags: list[str] (default []),
            category: str (default 'general'),
        }
    """
    content: str = arguments.get("content", "").strip()
    if not content:
        return {"error": "content is required"}

    tags: list[str] = arguments.get("tags", [])
    category: str = arguments.get("category", "general")

    if category not in VALID_CATEGORIES:
        category = "general"

    mem_id = runtime.memory.remember(content, tags=tags, category=category)
    return {"id": mem_id, "stored": True, "category": category, "tags": tags}


async def run_recall(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    """
    Execute recall tool.

    Args:
        arguments: {
            query: str (required),
            tags: list[str] | None,
            category: str | None,
            limit: int (default 10),
        }
    """
    query: str = arguments.get("query", "").strip()
    if not query:
        return {"error": "query is required"}

    tags: list[str] | None = arguments.get("tags")
    category: str | None = arguments.get("category")
    limit: int = int(arguments.get("limit", 10))

    results = runtime.memory.recall(query, tags=tags, category=category, limit=limit)
    return {"results": results, "count": len(results)}
