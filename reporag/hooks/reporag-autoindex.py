#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook — tells Claude to index unindexed projects.

If the current working directory has not been indexed by reporag, outputs a
system reminder instructing Claude to call index_codebase. Claude will then
call the MCP tool automatically on the next response.

No ML imports — reads only the lightweight projects.json registry.
"""
import json
import os
import sys
from pathlib import Path

try:
    data = json.load(sys.stdin)
    cwd = data.get("cwd", "").rstrip("/")
    if not cwd:
        sys.exit(0)

    data_dir = Path(os.environ.get("REPORAG_DATA_DIR", "~/.local/share/reporag")).expanduser()
    registry_path = data_dir / "projects.json"

    registry: dict = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            pass

    from pathlib import Path as _Path
    is_indexed = any(
        _Path(cwd) == _Path(proj) or _Path(cwd).is_relative_to(_Path(proj))
        for proj in registry
    )

    if not is_indexed:
        print(
            f"[reporag] {cwd} has not been indexed yet. "
            f'Call index_codebase with path="{cwd}" to enable code search and retrieval.'
        )
except Exception:
    pass  # never break Claude Code
