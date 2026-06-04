"""Unit tests for _setup_hooks_impl and _setup_cursor_impl in reporag.server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from reporag.server import _setup_cursor_impl, _setup_hooks_impl


# ── helpers ──────────────────────────────────────────────────────────────────

def _fake_hooks_dir(tmp_path: Path) -> Path:
    """Create tmp_path/reporag/hooks with one hook script.

    Returns the package dir so callers can set __file__ = pkg / "server.py",
    making Path(__file__).parent / "hooks" resolve to the real hooks dir.
    """
    pkg = tmp_path / "reporag"
    pkg.mkdir()
    hooks = pkg / "hooks"
    hooks.mkdir()
    hook = hooks / "reporag-test.py"
    hook.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    hook.chmod(0o755)
    return pkg


# ── _setup_hooks_impl ─────────────────────────────────────────────────────────

def test_hooks_impl_installs_to_claude_dir(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    import reporag.server as srv
    monkeypatch.setattr(srv, "__file__", str(pkg / "server.py"))

    _setup_hooks_impl(claude_dir, verbose=False)

    assert (claude_dir / "hooks" / "reporag-test.py").exists()
    settings = json.loads((claude_dir / "settings.json").read_text())
    hooks_cfg = settings["hooks"]["UserPromptSubmit"]
    assert any("reporag-test.py" in str(h) for h in hooks_cfg)


def test_hooks_impl_idempotent(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    import reporag.server as srv
    monkeypatch.setattr(srv, "__file__", str(pkg / "server.py"))

    _setup_hooks_impl(claude_dir)
    mtime1 = (claude_dir / "settings.json").stat().st_mtime

    _setup_hooks_impl(claude_dir)
    mtime2 = (claude_dir / "settings.json").stat().st_mtime

    assert mtime1 == mtime2  # second call must not rewrite


def test_hooks_impl_skips_corrupt_json(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    (claude_dir / "settings.json").write_text("{invalid json{{")

    import reporag.server as srv
    monkeypatch.setattr(srv, "__file__", str(pkg / "server.py"))

    result = _setup_hooks_impl(claude_dir, verbose=False)
    assert result is False
    assert "{invalid json{{" in (claude_dir / "settings.json").read_text()


def test_hooks_impl_preserves_existing_settings(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    existing = {"theme": "dark", "keybindings": {"save": "ctrl+s"}, "hooks": {"OtherEvent": []}}
    (claude_dir / "settings.json").write_text(json.dumps(existing))

    import reporag.server as srv
    monkeypatch.setattr(srv, "__file__", str(pkg / "server.py"))
    _setup_hooks_impl(claude_dir)

    result = json.loads((claude_dir / "settings.json").read_text())
    assert result["theme"] == "dark"
    assert result["keybindings"]["save"] == "ctrl+s"
    assert result["hooks"]["OtherEvent"] == []


def test_hooks_impl_handles_empty_hooks_list_entry(tmp_path, monkeypatch):
    """Regression: hook entry with hooks:[] must not raise IndexError."""
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    settings = {"hooks": {"UserPromptSubmit": [{"matcher": ".*", "hooks": []}]}}
    (claude_dir / "settings.json").write_text(json.dumps(settings))

    import reporag.server as srv
    monkeypatch.setattr(srv, "__file__", str(pkg / "server.py"))

    result = _setup_hooks_impl(claude_dir, verbose=False)
    assert isinstance(result, bool)


# ── _setup_cursor_impl ────────────────────────────────────────────────────────

def test_cursor_impl_writes_mcp_json(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    _setup_cursor_impl(cursor_dir)
    mcp = json.loads((cursor_dir / "mcp.json").read_text())
    assert "reporag" in mcp["mcpServers"]


def test_cursor_impl_idempotent_no_rewrite(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    _setup_cursor_impl(cursor_dir)
    mtime1 = (cursor_dir / "mcp.json").stat().st_mtime

    _setup_cursor_impl(cursor_dir)
    mtime2 = (cursor_dir / "mcp.json").stat().st_mtime

    assert mtime1 == mtime2


def test_cursor_impl_preserves_other_mcp_servers(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    existing = {"mcpServers": {"other-tool": {"command": "npx", "args": ["other"]}}}
    (cursor_dir / "mcp.json").write_text(json.dumps(existing))

    _setup_cursor_impl(cursor_dir)

    result = json.loads((cursor_dir / "mcp.json").read_text())
    assert "other-tool" in result["mcpServers"]
    assert "reporag" in result["mcpServers"]


def test_cursor_impl_skips_corrupt_json(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    (cursor_dir / "mcp.json").write_text("{bad json{{")

    result = _setup_cursor_impl(cursor_dir, verbose=False)
    assert result is False
    assert "{bad json{{" in (cursor_dir / "mcp.json").read_text()


def test_cursor_impl_writes_rules_mdc(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    _setup_cursor_impl(cursor_dir)
    rules = (cursor_dir / "rules" / "reporag.mdc").read_text()
    assert "alwaysApply: true" in rules
    assert "query_code" in rules


def test_cursor_impl_does_not_overwrite_existing_rules(tmp_path):
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    rules_dir = cursor_dir / "rules"
    rules_dir.mkdir()
    rules_path = rules_dir / "reporag.mdc"
    rules_path.write_text("my custom rules")

    _setup_cursor_impl(cursor_dir)
    assert rules_path.read_text() == "my custom rules"


# ── hook scripts (subprocess) ─────────────────────────────────────────────────

def _run_hook(script: str, stdin_data: dict, env: dict | None = None) -> str:
    pkg_hooks = Path(__file__).parent.parent / "reporag" / "hooks"
    result = subprocess.run(
        [sys.executable, str(pkg_hooks / script)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
    )
    return result.stdout.strip()


def test_autoindex_hook_silent_when_no_cwd(tmp_path):
    out = _run_hook("reporag-autoindex.py", {"prompt": "hello"}, {"REPORAG_DATA_DIR": str(tmp_path)})
    assert out == ""


def test_autoindex_hook_fires_when_unindexed(tmp_path):
    out = _run_hook(
        "reporag-autoindex.py",
        {"cwd": "/home/user/myproject", "prompt": "fix the bug"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert "index_codebase" in out


def test_autoindex_hook_silent_when_indexed(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 100, "files": 10}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-autoindex.py",
        {"cwd": "/home/user/myproject/src", "prompt": "explain this"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out == ""


def test_autoindex_hook_respects_ignore_file(tmp_path):
    ignore_dir = tmp_path / "proj"
    ignore_dir.mkdir()
    (ignore_dir / ".reporag-ignore").touch()
    out = _run_hook(
        "reporag-autoindex.py",
        {"cwd": str(ignore_dir), "prompt": "explain this"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out == ""


def test_hint_hook_fires_on_code_prompt_indexed(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "explain the function and debug the error"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert "query_code" in out
    assert "285" in out


def test_hint_hook_silent_for_short_prompt(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "fix it"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out == ""


def test_hint_hook_silent_for_non_code_prompt(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "please add a comma after the word hello there"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out == ""


def test_hint_hook_no_sibling_collision(tmp_path):
    """Regression: hint must not fire for /myproject-fork when /myproject is indexed."""
    registry = {"/home/user/myproject": {"chunks": 100, "files": 10}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject-fork", "prompt": "explain the function and debug the error"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out == ""
