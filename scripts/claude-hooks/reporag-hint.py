#!/usr/bin/env python3
"""
Claude Code UserPromptSubmit hook — injects a query_code reminder for code questions.

When the current project is indexed and the user asks a code-related question,
outputs a one-line system reminder so Claude proactively uses query_code instead
of answering from training data alone.

No ML imports — reads only the lightweight projects.json registry.
"""
import json
import os
import re
import sys
from pathlib import Path

# Prompts that are likely asking about the codebase
_CODE_RE = re.compile(
    r"\b(how|what|where|explain|implement|show|find|why|does|work|trace|debug|"
    r"bug|error|fix|test|refactor|add|create|update|call|return|class|function|"
    r"method|module|import|define|declare|use|change|remove|rename)\b",
    re.IGNORECASE,
)

try:
    data = json.load(sys.stdin)
    prompt: str = data.get("prompt", "")
    cwd: str = data.get("cwd", "").rstrip("/")

    if not cwd or len(prompt) < 8 or not _CODE_RE.search(prompt):
        sys.exit(0)

    data_dir = Path(os.environ.get("REPORAG_DATA_DIR", "~/.local/share/reporag")).expanduser()
    registry_path = data_dir / "projects.json"

    registry: dict = {}
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception:
            pass

    # Find the longest matching indexed project for cwd
    best_proj: str | None = None
    best_info: dict | None = None
    for proj, info in registry.items():
        proj = proj.rstrip("/")
        if cwd.startswith(proj) and (best_proj is None or len(proj) > len(best_proj)):
            best_proj = proj
            best_info = info

    if best_proj and best_info:
        chunks = best_info.get("chunks", 0)
        print(
            f"[reporag] {best_proj} is indexed ({chunks:,} chunks). "
            f"Use query_code to retrieve relevant context before answering."
        )
except Exception:
    pass  # never break Claude Code
