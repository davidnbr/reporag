"""MCP tool: index_status — query background indexing task progress."""

from __future__ import annotations

import time
from typing import Any


async def run(arguments: dict[str, Any], runtime: Any) -> dict[str, Any]:
    task_id: str | None = arguments.get("task_id")

    if task_id:
        task = runtime.index_tasks.get(task_id)
        if not task:
            return {"error": f"No index task with id {task_id!r}"}
        return _task_dict(task)

    tasks = list(runtime.index_tasks.values())
    if not tasks:
        return {"tasks": [], "message": "No index tasks have run yet."}

    # Most recent first
    tasks.sort(key=lambda t: t.started_at, reverse=True)
    return {"tasks": [_task_dict(t) for t in tasks]}


def _task_dict(task: Any) -> dict[str, Any]:
    now = time.monotonic()
    elapsed = (task.finished_at if task.finished_at else now) - task.started_at
    eta: float | None = None
    if task.status == "running" and task.indexed_files > 0:
        rate = task.indexed_files / elapsed
        remaining = task.total_files - task.indexed_files
        eta = round(remaining / rate, 0) if rate > 0 else None

    return {
        "task_id": task.task_id,
        "project": task.project,
        "status": task.status,
        "incremental": task.incremental,
        "progress_pct": (
            round(100 * task.indexed_files / task.total_files, 1) if task.total_files > 0 else 100.0
        ),
        "indexed_files": task.indexed_files,
        "total_files": task.total_files,
        "skipped_files": task.skipped_files,
        "indexed_chunks": task.indexed_chunks,
        "elapsed_s": round(elapsed, 1),
        "eta_s": eta,
        "error": task.error,
    }
