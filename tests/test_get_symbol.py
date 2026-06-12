"""End-to-end tests for the get_symbol MCP tool (Task 12) — exact + fuzzy modes, mocked runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from reporag.tools.symbol import run


class _FakeGraphDB:
    def __init__(self, rows: dict[str, list[dict[str, Any]]]) -> None:
        self._rows = rows
        self.last_project: str | None = None

    def get_symbol(
        self, name: str, language: str | None = None, project: str | None = None
    ) -> list[dict[str, Any]]:
        self.last_project = project
        rows = self._rows.get(name, [])
        if language:
            rows = [r for r in rows if r["language"] == language]
        if project:
            rows = [r for r in rows if r["file_path"].startswith(project.rstrip("/") + "/")]
        return rows


class _FakeEmbedder:
    def encode_query(self, text: str) -> Any:
        return [0.0]


class _FakeDense:
    def __init__(self, ids: list[str], chunks: dict[str, dict[str, Any]]) -> None:
        self._ids = ids
        self._chunks = chunks

    def search(self, q_vec: Any, k: int, project: str | None = None) -> list[str]:
        self.last_project = project
        return self._ids[:k]

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        return [self._chunks[i] for i in ids if i in self._chunks]


def test_exact_match_returns_results() -> None:
    runtime = SimpleNamespace(
        graph_db=_FakeGraphDB(
            {
                "my_func": [
                    {
                        "name": "my_func",
                        "symbol_type": "function",
                        "file_path": "/repo/module_a.py",
                        "language": "python",
                        "start_line": 10,
                        "end_line": 12,
                    }
                ]
            }
        ),
        embedder=_FakeEmbedder(),
        dense=_FakeDense([], {}),
    )

    result = asyncio.run(run({"name": "my_func", "project": "/repo"}, runtime))

    assert result["mode"] == "exact"
    assert result["results"] == [
        {
            "name": "my_func",
            "type": "function",
            "file": "/repo/module_a.py",
            "language": "python",
            "start_line": 10,
            "end_line": 12,
        }
    ]


def test_exact_match_not_found_returns_message() -> None:
    runtime = SimpleNamespace(
        graph_db=_FakeGraphDB({}),
        embedder=_FakeEmbedder(),
        dense=_FakeDense([], {}),
    )

    result = asyncio.run(run({"name": "missing_fn", "project": "/repo"}, runtime))

    assert result == {
        "results": [],
        "mode": "exact",
        "message": "Symbol 'missing_fn' not found",
    }


def test_fuzzy_match_uses_dense_search_and_filters_language() -> None:
    chunks = {
        "c1": {
            "id": "c1",
            "name": "my_funcion",
            "chunk_type": "function",
            "file_path": "/repo/a.py",
            "language": "python",
            "start_line": 1,
            "end_line": 3,
            "semantic_text": "Function my funcion. Does a thing.",
        },
        "c2": {
            "id": "c2",
            "name": "myFunction",
            "chunk_type": "function",
            "file_path": "/repo/b.ts",
            "language": "typescript",
            "start_line": 5,
            "end_line": 8,
            "semantic_text": "Function myFunction. Does a thing in TS.",
        },
    }
    runtime = SimpleNamespace(
        graph_db=_FakeGraphDB({}),
        embedder=_FakeEmbedder(),
        dense=_FakeDense(["c1", "c2"], chunks),
    )

    result = asyncio.run(run({"name": "my_function", "fuzzy": True, "language": "python"}, runtime))

    assert result["mode"] == "fuzzy"
    assert [r["name"] for r in result["results"]] == ["my_funcion"]
    assert result["results"][0]["file"] == "/repo/a.py"


def test_exact_match_defaults_to_current_project(monkeypatch) -> None:
    """Omitting `project` must scope to the server's own project — symbols
    from other indexed repos must never leak into the results."""
    import reporag.projects as projects

    monkeypatch.setattr(projects, "default_root", lambda: "/repo")

    graph_db = _FakeGraphDB(
        {
            "my_func": [
                {
                    "name": "my_func",
                    "symbol_type": "function",
                    "file_path": "/repo/module_a.py",
                    "language": "python",
                    "start_line": 10,
                    "end_line": 12,
                },
                {
                    "name": "my_func",
                    "symbol_type": "function",
                    "file_path": "/other_project/module_b.py",
                    "language": "python",
                    "start_line": 1,
                    "end_line": 3,
                },
            ]
        }
    )
    runtime = SimpleNamespace(graph_db=graph_db, embedder=_FakeEmbedder(), dense=_FakeDense([], {}))

    result = asyncio.run(run({"name": "my_func"}, runtime))

    assert graph_db.last_project == "/repo"
    assert [r["file"] for r in result["results"]] == ["/repo/module_a.py"]


def test_fuzzy_match_defaults_to_current_project(monkeypatch) -> None:
    import reporag.projects as projects

    monkeypatch.setattr(projects, "default_root", lambda: "/repo")

    dense = _FakeDense([], {})
    runtime = SimpleNamespace(graph_db=_FakeGraphDB({}), embedder=_FakeEmbedder(), dense=dense)

    asyncio.run(run({"name": "my_func", "fuzzy": True}, runtime))

    assert dense.last_project == "/repo"


def test_missing_name_returns_error() -> None:
    runtime = SimpleNamespace(
        graph_db=_FakeGraphDB({}), embedder=_FakeEmbedder(), dense=_FakeDense([], {})
    )

    result = asyncio.run(run({}, runtime))

    assert result == {"error": "name is required"}
