"""Tests for the reporag-dupcheck.py PreToolUse hook (Task 11).

The hook script's filename contains a hyphen so it's loaded via
importlib from its file path rather than a normal import.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

_HOOK_PATH = Path(__file__).parent.parent / "reporag" / "hooks" / "reporag-dupcheck.py"


def _load_hook() -> Any:
    spec = importlib.util.spec_from_file_location("reporag_dupcheck", _HOOK_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def hook() -> Any:
    return _load_hook()


def _make_symbols_db(path: Path, rows: list[tuple[str, str, str, str, str, int, int]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE symbols (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            file_path   TEXT NOT NULL,
            language    TEXT NOT NULL,
            symbol_type TEXT NOT NULL,
            start_line  INTEGER NOT NULL,
            end_line    INTEGER NOT NULL
        )
        """
    )
    conn.executemany(
        "INSERT INTO symbols VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_extract_names_finds_def_and_class(hook: Any) -> None:
    content = "def existing_fn(x):\n    return x\n\n\nclass Helper:\n    pass\n"
    assert hook.extract_names(content) == {"existing_fn", "Helper"}


def test_extract_names_empty_for_no_definitions(hook: Any) -> None:
    assert hook.extract_names("x = 1\nprint(x)\n") == set()


def test_find_collisions_excludes_same_file(hook: Any, tmp_path: Path) -> None:
    db_path = tmp_path / "dependency_graph.db"
    _make_symbols_db(
        db_path,
        [
            ("a", "existing_fn", "/repo/lib/utils.py", "python", "function", 10, 20),
            ("b", "existing_fn", "/repo/new/file.py", "python", "function", 1, 5),
        ],
    )

    collisions = hook.find_collisions(db_path, {"existing_fn"}, current_file="/repo/new/file.py")

    assert collisions == [("existing_fn", "/repo/lib/utils.py", 10)]


def test_main_emits_warning_for_existing_symbol(
    hook: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_symbols_db(
        data_dir / "dependency_graph.db",
        [("a", "existing_fn", "/repo/lib/utils.py", "python", "function", 10, 20)],
    )
    monkeypatch.setenv("REPORAG_DATA_DIR", str(data_dir))

    payload = {
        "tool_input": {
            "file_path": "/repo/new/file.py",
            "content": "def existing_fn(x):\n    return x + 1\n",
        }
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    hook.main()

    out = capsys.readouterr().out
    assert "existing_fn" in out
    assert "/repo/lib/utils.py:10" in out


def test_main_silent_for_new_name(
    hook: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_symbols_db(
        data_dir / "dependency_graph.db",
        [("a", "existing_fn", "/repo/lib/utils.py", "python", "function", 10, 20)],
    )
    monkeypatch.setenv("REPORAG_DATA_DIR", str(data_dir))

    payload = {
        "tool_input": {
            "file_path": "/repo/new/file.py",
            "content": "def brand_new_fn(x):\n    return x + 1\n",
        }
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    hook.main()

    assert capsys.readouterr().out == ""
