#!/usr/bin/env python3
"""
Claude Code PreToolUse hook — write-time duplicate-symbol detection.

Before a Write/Edit tool call lands, scans the new content for top-level
`def`/`class`/`function` definitions and checks the indexed symbols table
for exact-name collisions in OTHER files. Emits a non-blocking warning so
Claude can verify it isn't reimplementing something that already exists.

Exact-name match only (v1) — no embeddings/ML, must stay fast (<50ms).
Reads dependency_graph.db read-only; never blocks the tool call.
"""

import json
import os
import re
import sqlite3
import sys
from pathlib import Path


def _emit(text: str, event: str) -> None:
    if os.environ.get("REPORAG_HOOK_FORMAT") == "codex":
        print(
            json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": text}})
        )
    else:
        print(text)


_DEF_RE = re.compile(r"^\s*(?:def|class|function)\s+(\w+)", re.MULTILINE)

_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Add|Update) File: (.+)$", re.MULTILINE)

_MAX_WARNINGS = 5


def extract_names(content: str) -> set[str]:
    """Return the set of top-level def/class/function names in `content`."""
    return set(_DEF_RE.findall(content))


def parse_apply_patch(command: str) -> tuple[str, str]:
    """Extract (first target file, added-line content) from a Codex apply_patch body.

    Codex exposes file writes via the apply_patch tool as a single patch string in
    tool_input.command. Added lines start with '+' (excluding the '*** Add File:'
    headers); we strip the leading '+' so the def/class regex matches recovered code.
    """
    if "*** Begin Patch" not in command and "*** Add File:" not in command:
        return "", ""
    files = _PATCH_FILE_RE.findall(command)
    added = [
        line[1:]
        for line in command.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return (files[0].strip() if files else ""), "\n".join(added)


def find_collisions(
    db_path: Path, names: set[str], current_file: str
) -> list[tuple[str, str, int]]:
    """Return (name, file_path, start_line) for symbols matching `names` in other files."""
    if not names or not db_path.exists():
        return []
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        placeholders = ",".join("?" * len(names))
        rows = conn.execute(
            f"SELECT name, file_path, start_line FROM symbols WHERE name IN ({placeholders})",
            tuple(names),
        ).fetchall()
    finally:
        conn.close()
    return [(name, fp, line) for name, fp, line in rows if fp != current_file]


def main() -> None:
    try:
        data = json.load(sys.stdin)
        tool_input = data.get("tool_input", {})
        file_path = tool_input.get("file_path", "")
        content = tool_input.get("content") or tool_input.get("new_string") or ""
        if not content:
            # Codex apply_patch: file writes arrive as a patch string in tool_input.command
            file_path, content = parse_apply_patch(tool_input.get("command", "") or "")
        if not content:
            return

        names = extract_names(content)
        if not names:
            return

        data_dir = Path(os.environ.get("REPORAG_DATA_DIR", "~/.local/share/reporag")).expanduser()
        collisions = find_collisions(data_dir / "dependency_graph.db", names, file_path)

        warnings = [
            f"[reporag] WARNING: '{name}' already exists at {fp}:{line} — "
            f"verify you're not duplicating before writing."
            for name, fp, line in collisions[:_MAX_WARNINGS]
        ]
        if warnings:
            _emit("\n".join(warnings), "PreToolUse")
    except Exception:
        pass  # never break Claude Code


if __name__ == "__main__":
    main()
