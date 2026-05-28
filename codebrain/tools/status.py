"""MCP tool: project_status — TODOs, stubs, test coverage, git activity."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

_TODO_RE = re.compile(r'\b(TODO|FIXME|HACK|XXX)\b[:\s]*(.*)', re.IGNORECASE)
_STUB_RE = re.compile(r'^\s*(pass|raise\s+NotImplementedError[^\n]*)\s*$', re.MULTILINE)


def _git_activity(project: str) -> dict[str, Any]:
    try:
        changed = subprocess.run(
            ["git", "-C", project, "log", "--since=30.days", "--name-only", "--format=", "--no-merges"],
            capture_output=True, text=True, timeout=5,
        )
        files_30d = len({f for f in changed.stdout.strip().split("\n") if f.strip()})

        last = subprocess.run(
            ["git", "-C", project, "log", "-1", "--format=%cr"],
            capture_output=True, text=True, timeout=5,
        )
        return {"files_changed_30d": files_30d, "last_commit": last.stdout.strip()}
    except Exception:
        return {"available": False}


async def run(
    arguments: dict[str, Any],
    runtime: "Runtime",  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    project = arguments.get("project", "").strip()
    if not project:
        return {"error": "project is required"}

    root = Path(project).expanduser().resolve()
    root_str = str(root)
    safe = root_str.replace("'", "''")

    runtime.dense._open_or_create_table()
    rows = runtime.dense._table.search().where(f"file_path LIKE '{safe}%'").limit(50000).to_list()

    if not rows:
        return {"error": "No chunks indexed for this project. Run index_codebase first."}

    # ── TODOs ─────────────────────────────────────────────────────────────────
    todos: list[dict[str, str]] = []
    for r in rows:
        text = (r.get("raw_content") or "") + " " + (r.get("semantic_text") or "")
        for m in _TODO_RE.finditer(text):
            todos.append({
                "file": r.get("file_path", "")[len(root_str):].lstrip("/"),
                "type": m.group(1).upper(),
                "text": m.group(2).strip()[:120],
            })
        if len(todos) >= 100:
            break

    # ── stubs: named functions whose body is only pass / raise NotImplementedError ──
    stubs: list[dict[str, str]] = []
    for r in rows:
        if r.get("chunk_type") not in ("function", "method"):
            continue
        content = r.get("raw_content", "")
        lines = content.strip().split("\n")
        if len(lines) > 5:
            continue
        body = "\n".join(lines[1:]).strip()
        if body and _STUB_RE.match(body) and len(body) < 80:
            stubs.append({
                "file": r.get("file_path", "")[len(root_str):].lstrip("/"),
                "name": r.get("name", ""),
                "type": r.get("chunk_type", ""),
            })

    # ── test coverage ─────────────────────────────────────────────────────────
    source_files = {r.get("file_path", "") for r in rows}
    test_files = [
        f for f in source_files
        if "/test" in f or f.endswith("_test.py") or "/tests/" in f
    ]

    # ── git ───────────────────────────────────────────────────────────────────
    git_info = _git_activity(root_str)

    # ── health ────────────────────────────────────────────────────────────────
    health = "good"
    if len(todos) > 20 or len(stubs) > 5:
        health = "needs-attention"
    if not git_info.get("available", True) is False and git_info.get("files_changed_30d", 1) == 0:
        health = "stale"

    return {
        "project": root_str,
        "health": health,
        "todos": todos[:20],
        "todo_count": len(todos),
        "stubs": stubs[:10],
        "stub_count": len(stubs),
        "test_coverage": {
            "test_files": len(test_files),
            "source_files": len(source_files),
            "ratio": round(len(test_files) / max(len(source_files), 1), 2),
        },
        "git": git_info,
        "chunk_count": len(rows),
    }
