"""Install reporag into MCP-capable AI clients.

Each supported client stores its configuration differently, so each gets a
dedicated writer. All three share the same guarantees:

  * **Idempotent** — re-running makes no change once configured.
  * **Atomic** — writes go to a temp file and are renamed into place.
  * **Non-destructive** — a config file that cannot be parsed is left untouched
    rather than overwritten, so hand-edited files are never corrupted.

Supported clients:

  ===========  =========================  ==========================================
  Client       Config file                Mechanism
  ===========  =========================  ==========================================
  Claude Code  ~/.claude/settings.json    JSON merge of hook entries
  Cursor       ~/.cursor/mcp.json + rules JSON merge + a rules/reporag.mdc file
  Codex CLI    ~/.codex/config.toml       marker-delimited managed TOML block
  ===========  =========================  ==========================================

The hook scripts under ``reporag/hooks`` are client-agnostic: they read the same
stdin fields everywhere and switch output format via ``REPORAG_HOOK_FORMAT``
(plain text for Claude/Cursor, JSON for Codex).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# Directory of the installed ``reporag`` package; hook scripts live alongside it.
# Exposed as a module global so tests can point it at a fixture tree.
PACKAGE_DIR: Path = Path(__file__).resolve().parent

# Canonical command an MCP client uses to launch the server. Targets uvx-from-git
# installs; users on `pip install reporag` or a local checkout should edit their
# client config to point at their own `reporag` binary instead.
MCP_LAUNCH: Final[dict] = {
    "command": "uvx",
    "args": [
        "--from",
        "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
        "reporag",
    ],
    "env": {"REPORAG_DATA_DIR": "~/.local/share/reporag"},
}

# Hook scripts set this env var to select their stdout format. Codex consumes JSON
# (`hookSpecificOutput.additionalContext`); Claude/Cursor consume plain text.
HOOK_FORMAT_ENV: Final[str] = "REPORAG_HOOK_FORMAT"

_CODEX_BEGIN_MARKER: Final[str] = "# >>> reporag managed (do not edit) >>>"
_CODEX_END_MARKER: Final[str] = "# <<< reporag managed <<<"


### shared IO helpers #####################


def _say(verbose: bool, msg: str) -> None:
    """Print progress, but only in verbose (interactive `reporag setup`) mode."""
    if verbose:
        print(msg)


def _warn(verbose: bool, msg: str) -> None:
    """Surface a warning: to stdout when interactive, to the log otherwise."""
    if verbose:
        print(msg)
    else:
        logger.warning(msg)


def _atomic_write(path: Path, text: str) -> None:
    """Write `text` to `path` atomically (temp file + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _load_json(path: Path, verbose: bool) -> dict | None:
    """Load a JSON object from `path`.

    Returns ``{}`` if the file is absent, the parsed dict if valid, or ``None`` if
    the file exists but is malformed — the signal for callers to skip writing.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        _warn(
            verbose,
            f"  warning: {path} contains invalid JSON — skipping to avoid data loss. "
            "Fix manually and re-run.",
        )
        return None


def _packaged_hooks_dir() -> Path:
    return PACKAGE_DIR / "hooks"


def _copy_hook_scripts(dest_dir: Path, verbose: bool = False) -> bool:
    """Copy packaged ``reporag-*.py`` hook scripts into `dest_dir`.

    Copies only when the source is newer (mtime), keeps the executable bit, and
    returns ``True`` if any file was written.
    """
    pkg_hooks = _packaged_hooks_dir()
    if not pkg_hooks.exists():
        _say(verbose, f"Hook scripts not found at {pkg_hooks}")
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    changed = False
    for hook_file in sorted(pkg_hooks.glob("reporag-*.py")):
        dest = dest_dir / hook_file.name
        if not dest.exists() or hook_file.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(hook_file, dest)
            dest.chmod(0o755)
            changed = True
            _say(verbose, f"  copied: {dest}")
        else:
            _say(verbose, f"  up-to-date: {dest}")
    return changed


### default config directories (env-overridable) #####################


def _default_claude_dir() -> Path:
    """Resolve Claude Code config dir, honoring CLAUDE_CONFIG_DIR override."""
    return Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()


def _default_cursor_dir() -> Path:
    """Resolve Cursor config dir."""
    return Path("~/.cursor").expanduser()


def _default_codex_dir() -> Path:
    """Resolve Codex CLI config dir, honoring CODEX_HOME override."""
    return Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()


### Claude Code #####################


def _setup_hooks_impl(claude_dir: Path, verbose: bool = False) -> bool:
    """Install reporag Claude Code hooks. Returns True if settings changed.

    Wires three behaviors into ``settings.json``:
      * an ``index_codebase`` mcp_tool hook (auto-index on prompt),
      * a ``reporag-hint.py`` command hook (inject retrieval guidance), and
      * a ``reporag-dupcheck.py`` PreToolUse hook (duplicate-symbol warnings).

    Hook commands use the portable ``${CLAUDE_CONFIG_DIR:-$HOME/.claude}`` form so
    the settings file is valid across machines with different home directories.
    """
    if not _packaged_hooks_dir().exists():
        _say(verbose, f"Hook scripts not found at {_packaged_hooks_dir()}")
        return False

    hooks_dir = claude_dir / "hooks"
    _copy_hook_scripts(hooks_dir, verbose=verbose)

    settings_path = claude_dir / "settings.json"
    settings = _load_json(settings_path, verbose)
    if settings is None:
        return False

    hooks_cfg = settings.setdefault("hooks", {})
    up_hooks: list = hooks_cfg.setdefault("UserPromptSubmit", [])
    pre_hooks: list = hooks_cfg.setdefault("PreToolUse", [])

    def _commands(entry: dict) -> list[str]:
        return [e.get("command", "") for e in (entry.get("hooks") or [])]

    def _mcp_tools(entry: dict) -> list[str]:
        return [
            e.get("tool", "")
            for e in (entry.get("hooks") or [])
            if e.get("type") == "mcp_tool"
        ]

    def _drop(entries: list, predicate: Callable[[dict], bool], note: str) -> None:
        nonlocal changed
        before = len(entries)
        entries[:] = [e for e in entries if not predicate(e)]
        if len(entries) < before:
            changed = True
            _say(verbose, note)

    changed = False

    # Drop the legacy autoindex command hook — superseded by the mcp_tool entry below.
    autoindex_script = str(hooks_dir / "reporag-autoindex.py")
    _drop(
        up_hooks,
        lambda e: autoindex_script in _commands(e),
        "  removed stale reporag-autoindex.py command hook",
    )

    if not any("index_codebase" in _mcp_tools(e) for e in up_hooks):
        up_hooks.append(
            {
                "matcher": ".*",
                "hooks": [
                    {
                        "type": "mcp_tool",
                        "server": "reporag",
                        "tool": "index_codebase",
                        "input": {"path": "${cwd}"},
                    }
                ],
            }
        )
        changed = True
        _say(verbose, "  added mcp_tool hook: index_codebase")

    hint_cmd = "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/hooks/reporag-hint.py"
    _drop(
        up_hooks,
        lambda e: any(
            c.endswith("/reporag-hint.py") and c != hint_cmd for c in _commands(e)
        ),
        "  removed stale reporag-hint.py command hook",
    )
    if not any(hint_cmd in _commands(e) for e in up_hooks):
        up_hooks.append(
            {"matcher": ".*", "hooks": [{"type": "command", "command": hint_cmd}]}
        )
        changed = True
        _say(verbose, f"  added command hook: {hint_cmd}")

    dupcheck_cmd = "${CLAUDE_CONFIG_DIR:-$HOME/.claude}/hooks/reporag-dupcheck.py"
    if not any(dupcheck_cmd in _commands(e) for e in pre_hooks):
        pre_hooks.append(
            {
                "matcher": "Write|Edit",
                "hooks": [{"type": "command", "command": dupcheck_cmd}],
            }
        )
        changed = True
        _say(verbose, f"  added command hook: {dupcheck_cmd}")

    if changed:
        _atomic_write(settings_path, json.dumps(settings, indent=2))
        _say(verbose, f"  updated: {settings_path}")
    return changed


def _auto_setup_hooks() -> None:
    """Install Claude Code hooks on MCP server start; never raises into the server."""
    try:
        if _setup_hooks_impl(_default_claude_dir(), verbose=False):
            logger.info(
                "reporag: Claude Code hooks installed automatically "
                "(restart Claude Code to activate)"
            )
    except Exception:  # noqa: BLE001 — must never break the MCP server
        logger.debug("reporag: auto hook setup skipped", exc_info=True)


### Cursor #####################

_CURSOR_RULES_BODY: Final[str] = """\
# reporag — local RAG code search
# When answering questions about code in this project:
# 1. Call query_code to retrieve relevant context before answering
# 2. Call index_codebase if the project has not been indexed yet
# 3. Call get_symbol for exact function/class lookups
Use the reporag MCP tools proactively for all code questions.
"""


def _setup_cursor_impl(cursor_dir: Path, verbose: bool = False) -> bool:
    """Write ~/.cursor/mcp.json and install Cursor rules. Returns True if changed."""
    mcp_path = cursor_dir / "mcp.json"
    mcp = _load_json(mcp_path, verbose)
    if mcp is None:
        return False

    servers = mcp.setdefault("mcpServers", {})
    changed = servers.get("reporag") != MCP_LAUNCH
    servers["reporag"] = MCP_LAUNCH
    if changed:
        _atomic_write(mcp_path, json.dumps(mcp, indent=2))
        _say(verbose, f"  written: {mcp_path}")
    else:
        _say(verbose, f"  unchanged: {mcp_path}")

    # Cursor ≥0.50 reads global rules from ~/.cursor/rules/*.mdc.
    rules_path = cursor_dir / "rules" / "reporag.mdc"
    if not rules_path.exists():
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(
            "---\ndescription: Use reporag MCP for code questions\nalwaysApply: true\n---\n"
            + _CURSOR_RULES_BODY
        )
        _say(verbose, f"  written: {rules_path}")
        changed = True
    else:
        _say(verbose, f"  exists: {rules_path} (not overwritten)")

    _say(
        verbose,
        "\nFor older Cursor versions, add to your project's .cursorrules:\n"
        + _CURSOR_RULES_BODY,
    )
    return changed


### Codex CLI #####################


def _toml_str(s: str) -> str:
    """Quote `s` as a TOML basic string."""
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _codex_hook_command(script: Path) -> str:
    """Build the shell command that runs a reporag hook under Codex.

    Wrapped in ``sh -c`` so the ``REPORAG_HOOK_FORMAT`` env prefix is honored
    whether Codex runs the command through a shell or splits it into argv. All
    reporag hooks emit Codex JSON, so every command sets the env. ``shlex.quote``
    guards paths containing spaces or quotes.
    """
    inner = (
        f"{HOOK_FORMAT_ENV}=codex exec /usr/bin/env python3 {shlex.quote(str(script))}"
    )
    return f"sh -c {shlex.quote(inner)}"


def _codex_hook_table(event: str, matcher: str | None, script: Path) -> list[str]:
    """Render one ``[[hooks.<event>]]`` command-hook table as TOML lines.

    `matcher` is omitted when None — Codex ignores it for events like
    UserPromptSubmit, so emitting one there would just be misleading noise.
    """
    lines = [f"[[hooks.{event}]]"]
    if matcher is not None:
        lines.append(f"matcher = {_toml_str(matcher)}")
    lines += [
        f"[[hooks.{event}.hooks]]",
        'type = "command"',
        f"command = {_toml_str(_codex_hook_command(script))}",
        "timeout = 30",
        "",
    ]
    return lines


def _codex_managed_block(codex_dir: Path) -> str:
    """Build the reporag-managed TOML block (MCP server + hooks) for config.toml."""
    hooks_dir = codex_dir / "hooks"
    args_toml = ", ".join(_toml_str(a) for a in MCP_LAUNCH["args"])
    env_toml = ", ".join(f"{k} = {_toml_str(v)}" for k, v in MCP_LAUNCH["env"].items())

    lines = [
        _CODEX_BEGIN_MARKER,
        "[mcp_servers.reporag]",
        f"command = {_toml_str(MCP_LAUNCH['command'])}",
        f"args = [{args_toml}]",
        f"env = {{ {env_toml} }}",
        "enabled = true",
        "startup_timeout_sec = 30",
        "tool_timeout_sec = 120",
        "",
        *_codex_hook_table("UserPromptSubmit", None, hooks_dir / "reporag-hint.py"),
        *_codex_hook_table(
            "PreToolUse", "apply_patch", hooks_dir / "reporag-dupcheck.py"
        ),
        *_codex_hook_table("SessionStart", ".*", hooks_dir / "reporag-autoindex.py"),
    ]
    # The last hook table leaves a trailing "" — replace it with the end marker.
    lines[-1] = _CODEX_END_MARKER
    return "\n".join(lines)


def _splice_managed_block(existing_text: str, block: str) -> str:
    """Replace the marker-delimited region in `existing_text`, or append `block`."""
    if _CODEX_BEGIN_MARKER in existing_text and _CODEX_END_MARKER in existing_text:
        pattern = re.compile(
            re.escape(_CODEX_BEGIN_MARKER) + r".*?" + re.escape(_CODEX_END_MARKER),
            re.DOTALL,
        )
        return pattern.sub(lambda _: block, existing_text, count=1)
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    sep = "\n" if existing_text else ""
    return existing_text + sep + block + "\n"


def _setup_codex_impl(codex_dir: Path, verbose: bool = False) -> bool:
    """Write ~/.codex/config.toml managed block + install hook scripts.

    The managed block is spliced between markers so the rest of a hand-edited
    config.toml (comments, other servers, user settings) is preserved verbatim.
    The assembled file is validated as TOML before writing; a conflicting
    ``[mcp_servers.reporag]`` outside the markers makes validation fail, and the
    original file is left untouched. Returns True if the file changed.
    """
    _copy_hook_scripts(codex_dir / "hooks", verbose=verbose)

    config_path = codex_dir / "config.toml"
    existing_text = config_path.read_text() if config_path.exists() else ""
    final_text = _splice_managed_block(existing_text, _codex_managed_block(codex_dir))

    try:
        tomllib.loads(final_text)
    except tomllib.TOMLDecodeError as exc:
        _warn(
            verbose,
            f"  warning: assembled {config_path} would be invalid TOML ({exc}) — "
            "skipping to avoid data loss. Check for a conflicting [mcp_servers.reporag] "
            "table outside the reporag-managed block.",
        )
        return False

    if final_text == existing_text:
        _say(verbose, f"  unchanged: {config_path}")
        return False

    _atomic_write(config_path, final_text)
    _say(verbose, f"  written: {config_path}")
    return True


### client registry + CLI #####################


@dataclass(frozen=True)
class _Client:
    """One configurable target: how to find its config dir and apply our config."""

    key: str
    default_dir: Callable[[], Path]
    configure: Callable[[Path, bool], bool]
    restart_hint: str


_CLIENTS: Final[dict[str, _Client]] = {
    "claude": _Client(
        "claude",
        _default_claude_dir,
        _setup_hooks_impl,
        "Restart Claude Code to activate hooks.",
    ),
    "cursor": _Client(
        "cursor", _default_cursor_dir, _setup_cursor_impl, "Restart Cursor to activate."
    ),
    "codex": _Client(
        "codex", _default_codex_dir, _setup_codex_impl, "Restart Codex to activate."
    ),
}


def run_setup(
    clients: list[str], overrides: dict[str, Path], verbose: bool = True
) -> None:
    """Configure each named client, using `overrides[key]` as its dir when present."""
    for key in clients:
        client = _CLIENTS[key]
        target = overrides.get(key) or client.default_dir()
        _say(verbose, f"\n[{key}]")
        client.configure(target, verbose)
        _say(verbose, f"  {client.restart_hint}")


def cli() -> None:
    """Entry point for ``reporag setup``."""
    parser = argparse.ArgumentParser(
        prog="reporag setup",
        description="Configure reporag for Claude Code, Cursor, and/or Codex CLI.",
    )
    parser.add_argument(
        "--client",
        choices=[*_CLIENTS, "all"],
        default="all",
        help="Which AI client to configure (default: all)",
    )
    parser.add_argument(
        "--claude-dir",
        help="Override Claude config dir ($CLAUDE_CONFIG_DIR or ~/.claude)",
    )
    parser.add_argument("--cursor-dir", help="Override Cursor config dir (~/.cursor)")
    parser.add_argument(
        "--codex-dir", help="Override Codex config dir ($CODEX_HOME or ~/.codex)"
    )
    args = parser.parse_args(sys.argv[2:])

    selected = list(_CLIENTS) if args.client == "all" else [args.client]
    overrides = {
        key: Path(value).expanduser()
        for key, value in (
            ("claude", args.claude_dir),
            ("cursor", args.cursor_dir),
            ("codex", args.codex_dir),
        )
        if value
    }
    run_setup(selected, overrides, verbose=True)


def cli_hooks() -> None:
    """Backward-compatible entry point for ``reporag setup-hooks`` (Claude only)."""
    parser = argparse.ArgumentParser(prog="reporag setup-hooks")
    parser.add_argument(
        "--claude-dir",
        help="Claude config directory (default: $CLAUDE_CONFIG_DIR or ~/.claude)",
    )
    args = parser.parse_args(sys.argv[2:])
    claude_dir = (
        Path(args.claude_dir).expanduser() if args.claude_dir else _default_claude_dir()
    )
    _setup_hooks_impl(claude_dir, verbose=True)
    print("\nDone. Restart Claude Code to activate hooks.")
