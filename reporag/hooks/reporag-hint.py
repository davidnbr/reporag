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
import sys
from pathlib import Path

try:
    data = json.load(sys.stdin)
    cwd: str = data.get("cwd", "").rstrip("/")

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

    # Find the longest matching indexed project for cwd (path-aware, not string prefix)
    _cwd = Path(cwd)
    best_proj: str | None = None
    best_info: dict | None = None
    for proj, info in registry.items():
        _proj = Path(proj)
        if (_cwd == _proj or _cwd.is_relative_to(_proj)) and (
            best_proj is None or len(proj) > len(best_proj)
        ):
            best_proj = proj
            best_info = info

    if best_proj and best_info:
        chunks = best_info.get("chunks", 0)

        # Detect implementation intent from user prompt
        prompt: str = data.get("prompt", "").lower()
        impl_keywords = (
            "implement",
            "add ",
            "create ",
            "build ",
            "write a",
            "write the",
            "new endpoint",
            "new view",
            "new service",
            "new function",
            "new class",
            "new feature",
            "new handler",
            "new middleware",
            "new model",
        )
        is_impl = any(kw in prompt for kw in impl_keywords)

        if is_impl:
            print(
                f"[reporag] {best_proj} is indexed ({chunks:,} chunks) — "
                f"BEFORE WRITING NEW CODE: call find_existing(task=<your task description>, "
                f"project={best_proj!r}) to surface existing functions and patterns you should "
                f"reuse. Prevents duplication. "
                f"Also available: query_code (semantic search), get_symbol (exact lookup), "
                f"get_architecture (module topology)."
            )
        else:
            print(
                f"[reporag] {best_proj} is indexed ({chunks:,} chunks) — USE IT before answering, "
                f"don't guess from training data: query_code (semantic search for relevant snippets), "
                f"get_symbol (jump to a function/class def + refs), get_architecture (module/dependency "
                f"overview), ask_project (natural-language Q&A grounded in the indexed code), "
                f"summarize_project (high-level summary). Pick the one matching the question; "
                f"grounding in real code beats a plausible-sounding guess."
            )
except Exception:
    pass  # never break Claude Code
