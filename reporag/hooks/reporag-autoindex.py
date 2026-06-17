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


def _emit(text: str, event: str) -> None:
    if os.environ.get("REPORAG_HOOK_FORMAT") == "codex":
        print(json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}}))
    else:
        print(text)


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

    _cwd_path = Path(cwd)

    # Respect opt-out: .reporag-ignore in project root silences this hook
    if (_cwd_path / ".reporag-ignore").exists():
        sys.exit(0)

    is_indexed = any(
        _cwd_path == Path(proj) or _cwd_path.is_relative_to(Path(proj)) for proj in registry
    )

    if not is_indexed:
        _emit(
            f'[reporag] ACTION: {cwd} is unindexed — call index_codebase(path="{cwd}") '
            f"now, before doing anything else. It runs in the background (first results "
            f"in seconds, check via index_status), so starting it costs nothing. Once "
            f"indexed, ground answers in real code with query_code / get_symbol / "
            f"get_architecture / ask_project instead of guessing — that's the difference "
            f"between a correct answer and a plausible-sounding one. "
            f"(Create .reporag-ignore in project root to silence this.)",
            "SessionStart",
        )
except Exception:
    pass  # never break Claude Code
