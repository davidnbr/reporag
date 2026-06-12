"""MCP tool: summarize_project — structured project overview."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _extract_description(root: Path) -> str:
    """First substantive paragraph from README, or empty string."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        readme = root / name
        if not readme.exists():
            continue
        text = readme.read_text(errors="replace")[:4000]
        current: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith(("#", "!", "[", "```", "---", "===")):
                if current:
                    break
                continue
            if stripped:
                current.append(stripped)
            elif current:
                break
        if current:
            return " ".join(current)[:500]
    return ""


_ENTRY_NAMES = {"main.py", "__main__.py", "server.py", "cli.py", "app.py", "manage.py", "run.py"}
_SKIP_DIRS = {
    ".devenv",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".git",
    "__pycache__",
    ".terraform",
    "_build",
    "deps",
}


async def run(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    project = arguments.get("project", "").strip()
    if not project:
        return {"error": "project is required"}

    root = Path(project).expanduser().resolve()
    if not root.exists():
        return {"error": f"Path does not exist: {root}"}

    root_str = str(root)
    # LanceDB WHERE pre-filters for performance; Python startswith is authoritative
    # (LIKE treats _ and % as wildcards — paths containing either would silently mismatch)
    _sql_safe = root_str.replace("'", "''")

    # ── chunks from LanceDB ──────────────────────────────────────────────────
    runtime.dense._open_or_create_table()
    rows = [
        r
        for r in runtime.dense._table.search()
        .where(f"file_path LIKE '{_sql_safe}%'")
        .limit(50000)
        .to_list()
        if (r.get("file_path") or "").startswith(root_str)
    ]

    if not rows:
        return {"error": "No chunks indexed for this project. Run index_codebase first."}

    # ── tech stack ───────────────────────────────────────────────────────────
    lang_counts: dict[str, int] = {}
    for r in rows:
        lang = r.get("language") or "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    # ── entry points ─────────────────────────────────────────────────────────
    entry_points: list[str] = []
    try:
        for f in root.rglob("*"):
            if f.name in _ENTRY_NAMES and not any(p in _SKIP_DIRS for p in f.parts):
                entry_points.append(str(f.relative_to(root)))
    except Exception:
        pass

    # ── components: top imported files in project subgraph ───────────────────
    components: list[dict[str, Any]] = []
    if runtime.graph is not None:
        project_nodes = [n for n in runtime.graph.nodes() if n.startswith(root_str)]
        ranked = sorted(project_nodes, key=lambda n: runtime.graph.in_degree(n), reverse=True)[:10]
        for node in ranked:
            rel = node[len(root_str) :].lstrip("/")
            out_deg = runtime.graph.out_degree(node)
            in_deg = runtime.graph.in_degree(node)
            role = "hub" if out_deg > 5 else ("utility" if in_deg > 3 else "module")
            components.append(
                {"file": rel, "role": role, "imported_by": in_deg, "imports": out_deg}
            )

    # ── public symbols (exported functions/classes) ───────────────────────────
    public_api = [
        r["name"]
        for r in rows
        if r.get("chunk_type") in ("function", "class")
        and r.get("name")
        and not r.get("name", "").startswith("_")
    ]
    public_api = sorted(set(public_api))[:20]

    return {
        "project": root_str,
        "description": _extract_description(root),
        "tech_stack": dict(sorted(lang_counts.items(), key=lambda x: x[1], reverse=True)),
        "entry_points": entry_points[:5],
        "components": components,
        "public_api": public_api,
        "chunk_count": len(rows),
        "indexed_files": len({r.get("file_path") for r in rows}),
    }
