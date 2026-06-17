"""Unit tests for _setup_hooks_impl and _setup_cursor_impl in reporag.server."""

import json
import os
import subprocess
import sys
from pathlib import Path

from reporag.setup import _setup_codex_impl, _setup_cursor_impl, _setup_hooks_impl

# ── helpers ──────────────────────────────────────────────────────────────────


def _fake_hooks_dir(tmp_path: Path) -> Path:
    """Create tmp_path/reporag/hooks with reporag-hint.py and reporag-autoindex.py.

    Returns the package dir so callers can set __file__ = pkg / "server.py",
    making Path(__file__).parent / "hooks" resolve to the real hooks dir.
    """
    pkg = tmp_path / "reporag"
    pkg.mkdir()
    hooks = pkg / "hooks"
    hooks.mkdir()
    for name in ("reporag-hint.py", "reporag-autoindex.py"):
        hook = hooks / name
        hook.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
        hook.chmod(0o755)
    return pkg


def _fake_hooks_dir_full(tmp_path: Path) -> Path:
    """Like _fake_hooks_dir but also includes reporag-dupcheck.py, for codex setup tests."""
    pkg = tmp_path / "reporag"
    pkg.mkdir()
    hooks = pkg / "hooks"
    hooks.mkdir()
    for name in ("reporag-hint.py", "reporag-dupcheck.py", "reporag-autoindex.py"):
        hook = hooks / name
        hook.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
        hook.chmod(0o755)
    return pkg


# ── _setup_hooks_impl ─────────────────────────────────────────────────────────


def test_hooks_impl_installs_to_claude_dir(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

    _setup_hooks_impl(claude_dir, verbose=False)

    assert (claude_dir / "hooks" / "reporag-hint.py").exists()
    assert (claude_dir / "hooks" / "reporag-autoindex.py").exists()
    settings = json.loads((claude_dir / "settings.json").read_text())
    hooks_cfg = settings["hooks"]["UserPromptSubmit"]
    # mcp_tool entry for index_codebase (replaces old autoindex command hook)
    assert any(
        e.get("type") == "mcp_tool" and e.get("tool") == "index_codebase"
        for h in hooks_cfg
        for e in (h.get("hooks") or [])
    )
    # command entry for reporag-hint.py
    assert any("reporag-hint.py" in str(h) for h in hooks_cfg)
    # reporag-autoindex.py must NOT be wired as a command hook
    assert not any("reporag-autoindex.py" in str(h) for h in hooks_cfg)


def test_hooks_impl_idempotent(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

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

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

    result = _setup_hooks_impl(claude_dir, verbose=False)
    assert result is False
    assert "{invalid json{{" in (claude_dir / "settings.json").read_text()


def test_hooks_impl_preserves_existing_settings(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir(tmp_path)
    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    existing = {"theme": "dark", "keybindings": {"save": "ctrl+s"}, "hooks": {"OtherEvent": []}}
    (claude_dir / "settings.json").write_text(json.dumps(existing))

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
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

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

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
    out = _run_hook(
        "reporag-autoindex.py", {"prompt": "hello"}, {"REPORAG_DATA_DIR": str(tmp_path)}
    )
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


def test_hint_hook_fires_for_short_prompt(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "fix it"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert "query_code" in out


def test_hint_hook_fires_for_non_code_prompt(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "please add a comma after the word hello there"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert "query_code" in out


# ── _setup_codex_impl ─────────────────────────────────────────────────────────


def test_codex_impl_fresh_create(tmp_path, monkeypatch):
    import tomllib

    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

    result = _setup_codex_impl(codex_dir, verbose=False)
    assert result is True

    config_path = codex_dir / "config.toml"
    text = config_path.read_text()
    assert srv._CODEX_BEGIN_MARKER in text
    assert srv._CODEX_END_MARKER in text

    parsed = tomllib.loads(text)
    assert "reporag" in parsed["mcp_servers"]


def test_codex_impl_idempotent(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

    _setup_codex_impl(codex_dir, verbose=False)
    text1 = (codex_dir / "config.toml").read_text()

    result = _setup_codex_impl(codex_dir, verbose=False)
    text2 = (codex_dir / "config.toml").read_text()

    assert result is False
    assert text1 == text2


def test_codex_impl_preserves_existing_content(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    existing = '# my hand-written config\nmodel = "o3"\n\n[mcp_servers.other]\ncommand = "npx"\n'
    (codex_dir / "config.toml").write_text(existing)

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
    _setup_codex_impl(codex_dir, verbose=False)

    text = (codex_dir / "config.toml").read_text()
    assert "# my hand-written config" in text
    assert 'model = "o3"' in text
    assert "[mcp_servers.other]" in text
    assert "[mcp_servers.reporag]" in text


def test_codex_impl_marker_replace_only_touches_managed_region(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)

    _setup_codex_impl(codex_dir, verbose=False)
    config_path = codex_dir / "config.toml"
    text = config_path.read_text()

    # user appends content after the managed block
    text_with_addition = text + '\n[mcp_servers.user_added]\ncommand = "foo"\n'
    config_path.write_text(text_with_addition)

    _setup_codex_impl(codex_dir, verbose=False)
    text2 = config_path.read_text()
    assert "[mcp_servers.user_added]" in text2
    assert 'command = "foo"' in text2


def test_codex_impl_invalid_existing_toml_left_untouched(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    bad = "this is not [valid toml\n"
    (codex_dir / "config.toml").write_text(bad)

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
    result = _setup_codex_impl(codex_dir, verbose=False)

    assert result is False
    assert (codex_dir / "config.toml").read_text() == bad
    assert not (codex_dir / "config.tmp").exists()


def test_codex_impl_duplicate_table_outside_markers_safe_skip(tmp_path, monkeypatch):
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    conflicting = '[mcp_servers.reporag]\ncommand = "something-else"\n'
    (codex_dir / "config.toml").write_text(conflicting)

    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
    result = _setup_codex_impl(codex_dir, verbose=False)

    assert result is False
    assert (codex_dir / "config.toml").read_text() == conflicting
    assert not (codex_dir / "config.tmp").exists()


def test_codex_impl_all_hooks_shell_wrapped_with_format(tmp_path, monkeypatch):
    """F1+F3: every hook command is sh -c wrapped and sets REPORAG_HOOK_FORMAT=codex."""
    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
    _setup_codex_impl(codex_dir, verbose=False)
    text = (codex_dir / "config.toml").read_text()

    for script in ("reporag-hint.py", "reporag-dupcheck.py", "reporag-autoindex.py"):
        # find the command line referencing this script and assert wrapping + env
        line = next(ln for ln in text.splitlines() if script in ln and "command" in ln)
        assert "sh -c" in line
        assert "REPORAG_HOOK_FORMAT=codex" in line


def test_codex_impl_userpromptsubmit_omits_matcher(tmp_path, monkeypatch):
    """Codex ignores matcher for UserPromptSubmit — it must not be emitted there."""
    import tomllib

    pkg = _fake_hooks_dir_full(tmp_path)
    codex_dir = tmp_path / "codex"
    import reporag.setup as srv

    monkeypatch.setattr(srv, "PACKAGE_DIR", pkg)
    _setup_codex_impl(codex_dir, verbose=False)
    parsed = tomllib.loads((codex_dir / "config.toml").read_text())

    ups = parsed["hooks"]["UserPromptSubmit"]
    assert all("matcher" not in group for group in ups)
    # events that DO support matcher still carry it
    assert parsed["hooks"]["PreToolUse"][0]["matcher"] == "apply_patch"
    assert parsed["hooks"]["SessionStart"][0]["matcher"] == ".*"


def test_codex_impl_honors_codex_home_default(tmp_path, monkeypatch):
    import reporag.setup as srv

    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom_codex"))
    assert srv._default_codex_dir() == tmp_path / "custom_codex"


# ── hook script _emit format ──────────────────────────────────────────────────


def test_hint_hook_codex_format(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "explain the function"},
        {"REPORAG_DATA_DIR": str(tmp_path), "REPORAG_HOOK_FORMAT": "codex"},
    )
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "query_code" in parsed["hookSpecificOutput"]["additionalContext"]


def test_hint_hook_plain_format_unset(tmp_path):
    registry = {"/home/user/myproject": {"chunks": 285, "files": 39}}
    (tmp_path / "projects.json").write_text(json.dumps(registry))
    out = _run_hook(
        "reporag-hint.py",
        {"cwd": "/home/user/myproject", "prompt": "explain the function"},
        {"REPORAG_DATA_DIR": str(tmp_path)},
    )
    assert out.startswith("[reporag]")
    assert "hookSpecificOutput" not in out


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


def test_autoindex_hook_codex_format(tmp_path):
    """F1: autoindex must emit Codex JSON (SessionStart) for an unindexed project."""
    (tmp_path / "projects.json").write_text(json.dumps({}))
    out = _run_hook(
        "reporag-autoindex.py",
        {"cwd": str(tmp_path / "unindexed_proj")},
        {"REPORAG_DATA_DIR": str(tmp_path), "REPORAG_HOOK_FORMAT": "codex"},
    )
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "index_codebase" in parsed["hookSpecificOutput"]["additionalContext"]


def _seed_symbols_db(tmp_path: Path, rows: list[tuple[str, str, int]]) -> None:
    import sqlite3

    conn = sqlite3.connect(tmp_path / "dependency_graph.db")
    conn.execute("CREATE TABLE symbols (name TEXT, file_path TEXT, start_line INTEGER)")
    conn.executemany("INSERT INTO symbols VALUES (?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_dupcheck_hook_codex_apply_patch(tmp_path):
    """F2: dupcheck must parse Codex apply_patch payloads and warn on collisions."""
    _seed_symbols_db(tmp_path, [("existing_func", "other/mod.py", 12)])
    patch = (
        "*** Begin Patch\n"
        "*** Add File: newmod.py\n"
        "+def existing_func():\n"
        "+    return 1\n"
        "*** End Patch\n"
    )
    out = _run_hook(
        "reporag-dupcheck.py",
        {"tool_name": "apply_patch", "tool_input": {"command": patch}},
        {"REPORAG_DATA_DIR": str(tmp_path), "REPORAG_HOOK_FORMAT": "codex"},
    )
    parsed = json.loads(out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
    ctx = parsed["hookSpecificOutput"]["additionalContext"]
    assert "existing_func" in ctx
    assert "other/mod.py:12" in ctx


def test_dupcheck_hook_apply_patch_no_collision_silent(tmp_path):
    """apply_patch defining a name not in the index produces no output."""
    _seed_symbols_db(tmp_path, [("unrelated", "other/mod.py", 12)])
    patch = (
        "*** Begin Patch\n"
        "*** Add File: newmod.py\n"
        "+def brand_new_func():\n"
        "+    return 1\n"
        "*** End Patch\n"
    )
    out = _run_hook(
        "reporag-dupcheck.py",
        {"tool_name": "apply_patch", "tool_input": {"command": patch}},
        {"REPORAG_DATA_DIR": str(tmp_path), "REPORAG_HOOK_FORMAT": "codex"},
    )
    assert out == ""
