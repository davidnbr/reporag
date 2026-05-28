"""MCP tool: ask_project — natural language routing to project intelligence tools."""
from __future__ import annotations

from typing import Any

_ROUTES: list[tuple[list[str], str]] = [
    (
        ["what does", "what is this", "purpose", "overview", "summary", "about",
         "describe", "tell me about", "what kind", "what type"],
        "summarize",
    ),
    (
        ["architecture", "structure", "organized", "how is it", "layers", "modules",
         "design", "dependency", "dependencies", "components", "layout"],
        "architecture",
    ),
    (
        ["todo", "status", "what's left", "what is left", "implemented", "progress",
         "where are we", "stubs", "health", "fixme", "remaining", "done", "complete"],
        "status",
    ),
]


async def run(
    arguments: dict[str, Any],
    runtime: "Runtime",  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    raw_query: str = arguments.get("query", "").strip()
    project: str = arguments.get("project", "")

    if not raw_query:
        return {"error": "query is required"}

    query_lower = raw_query.lower()

    best_tool: str | None = None
    for patterns, tool in _ROUTES:
        if any(p in query_lower for p in patterns):
            best_tool = tool
            break

    if best_tool == "summarize":
        from codebrain.tools import summarize
        result = await summarize.run({"project": project}, runtime)
        result["source_tool"] = "summarize_project"
        return result

    if best_tool == "architecture":
        from codebrain.tools import architecture
        result = await architecture.run({"project": project}, runtime)
        result["source_tool"] = "get_architecture"
        return result

    if best_tool == "status":
        from codebrain.tools import status
        result = await status.run({"project": project}, runtime)
        result["source_tool"] = "project_status"
        return result

    # Fallback: semantic code search
    from codebrain.tools import query as query_tool
    result = await query_tool.run({"query": raw_query, "project": project, "k": 10}, runtime)
    result["source_tool"] = "query_code"
    return result
