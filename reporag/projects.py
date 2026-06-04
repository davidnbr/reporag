"""Lightweight registry of indexed projects.

Stored at $REPORAG_DATA_DIR/projects.json — read by Claude Code hooks without
importing any ML dependencies (fast, < 5 ms).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

_registry_lock = threading.Lock()


def _registry_path() -> Path:
    try:
        from reporag.config import get_config  # noqa: PLC0415
        return Path(get_config().data_dir).expanduser() / "projects.json"
    except ImportError:
        data_dir = os.environ.get("REPORAG_DATA_DIR", "~/.local/share/reporag")
        return Path(data_dir).expanduser() / "projects.json"


def _load() -> dict:
    path = _registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(registry: dict) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    tmp.replace(path)


def _is_subpath(child: str, parent: str) -> bool:
    try:
        Path(child).relative_to(parent)
        return True
    except ValueError:
        return False


def update(project: str, chunks: int, files: int) -> None:
    with _registry_lock:
        registry = _load()
        registry[project] = {
            "chunks": chunks,
            "files": files,
            "indexed_at": datetime.now(UTC).isoformat(),
        }
        _save(registry)


def get(project: str) -> dict | None:
    registry = _load()
    if project in registry:
        return registry[project]
    # path-aware match — subdirectory of an indexed project
    for key, val in registry.items():
        if _is_subpath(project, key):
            return val
    return None


def all_projects() -> dict:
    return _load()
