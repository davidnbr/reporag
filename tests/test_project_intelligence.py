"""
Integration tests for project intelligence tools:
  summarize_project, get_architecture, project_status, ask_project

Requires ML extras and at least one indexed project.
Skipped automatically if no data exists.

Run:
    devenv shell -- pytest tests/test_project_intelligence.py -m integration -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.integration

_PROJECT = str(Path(__file__).parent.parent.resolve())


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def runtime() -> Any:
    try:
        from codebrain.config import get_config
        from codebrain.server import Runtime
    except ImportError as e:
        pytest.skip(f"codebrain not importable: {e}")

    try:
        import lancedb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("ML extras not installed")

    rt = Runtime(config=get_config())
    try:
        rt.initialize()
    except Exception as e:
        pytest.skip(f"Runtime init failed: {e}")

    if rt.dense.count() == 0:
        pytest.skip("No chunks indexed. Run index_codebase first.")

    return rt


# ── summarize_project ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_summarize_returns_required_keys(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": _PROJECT}, runtime)

    assert "error" not in result, f"summarize failed: {result['error']}"
    for key in ("description", "tech_stack", "entry_points", "components", "public_api",
                "chunk_count", "indexed_files"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_summarize_detects_python(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert "python" in result["tech_stack"], "Expected python in tech_stack"
    assert result["tech_stack"]["python"] > 0


@pytest.mark.asyncio
async def test_summarize_finds_server_entry_point(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    entry_points = result["entry_points"]
    assert any("server.py" in ep for ep in entry_points), (
        f"Expected server.py in entry_points, got: {entry_points}"
    )


@pytest.mark.asyncio
async def test_summarize_has_public_api(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert len(result["public_api"]) > 0, "Expected at least one public symbol"
    for sym in result["public_api"]:
        assert not sym.startswith("_"), f"Private symbol leaked into public_api: {sym}"


@pytest.mark.asyncio
async def test_summarize_missing_project_returns_error(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": ""}, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_summarize_nonexistent_project_returns_error(runtime: Any) -> None:
    from codebrain.tools import summarize
    result = await summarize.run({"project": "/tmp/nonexistent_project_xyzzy"}, runtime)
    assert "error" in result


# ── get_architecture ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_architecture_returns_required_keys(runtime: Any) -> None:
    from codebrain.tools import architecture
    result = await architecture.run({"project": _PROJECT}, runtime)

    assert "error" not in result, f"architecture failed: {result.get('error')}"
    for key in ("node_count", "edge_count", "role_summary", "nodes", "edges", "layers"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_architecture_nodes_have_required_fields(runtime: Any) -> None:
    from codebrain.tools import architecture
    result = await architecture.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert result["node_count"] > 0, "Expected at least one node"
    for node in result["nodes"][:5]:
        for field in ("file", "role", "out_degree", "in_degree"):
            assert field in node, f"Node missing field '{field}': {node}"
        assert node["role"] in ("hub", "utility", "bridge", "leaf"), (
            f"Unknown role: {node['role']}"
        )


@pytest.mark.asyncio
async def test_architecture_layers_contain_codebrain(runtime: Any) -> None:
    from codebrain.tools import architecture
    result = await architecture.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert "codebrain" in result["layers"], (
        f"Expected 'codebrain' layer, got: {list(result['layers'].keys())}"
    )


@pytest.mark.asyncio
async def test_architecture_missing_project_returns_error(runtime: Any) -> None:
    from codebrain.tools import architecture
    result = await architecture.run({"project": ""}, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_architecture_counts_match_lists(runtime: Any) -> None:
    from codebrain.tools import architecture
    result = await architecture.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert result["node_count"] == len(result["nodes"])
    assert result["edge_count"] == len(result["edges"])


# ── project_status ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_status_returns_required_keys(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": _PROJECT}, runtime)

    assert "error" not in result, f"status failed: {result.get('error')}"
    for key in ("health", "todos", "todo_count", "stubs", "stub_count",
                "test_coverage", "git", "chunk_count"):
        assert key in result, f"missing key: {key}"


@pytest.mark.asyncio
async def test_status_health_is_valid(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert result["health"] in ("good", "needs-attention", "stale"), (
        f"Unknown health value: {result['health']}"
    )


@pytest.mark.asyncio
async def test_status_todo_counts_match(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    assert result["todo_count"] >= len(result["todos"]), (
        "todo_count must be >= len(todos) (todos is capped at 20)"
    )
    assert result["stub_count"] >= len(result["stubs"]), (
        "stub_count must be >= len(stubs) (stubs is capped at 10)"
    )


@pytest.mark.asyncio
async def test_status_test_coverage_ratio(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    tc = result["test_coverage"]
    assert tc["test_files"] >= 1, "Expected at least one test file"
    assert tc["source_files"] >= tc["test_files"]
    assert 0.0 <= tc["ratio"] <= 1.0


@pytest.mark.asyncio
async def test_status_git_info_present(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": _PROJECT}, runtime)

    assert "error" not in result
    git = result["git"]
    if git.get("available") is False:
        pytest.skip("git not available in this environment")
    assert "files_changed_30d" in git
    assert "last_commit" in git
    assert isinstance(git["files_changed_30d"], int)


@pytest.mark.asyncio
async def test_status_missing_project_returns_error(runtime: Any) -> None:
    from codebrain.tools import status
    result = await status.run({"project": ""}, runtime)
    assert "error" in result


# ── ask_project ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ask_routes_to_summarize(runtime: Any) -> None:
    from codebrain.tools import ask
    for query in ("what does this project do", "give me an overview", "describe this codebase"):
        result = await ask.run({"query": query, "project": _PROJECT}, runtime)
        assert "error" not in result, f"ask failed for '{query}': {result.get('error')}"
        assert result.get("source_tool") == "summarize_project", (
            f"Expected summarize_project, got {result.get('source_tool')} for '{query}'"
        )


@pytest.mark.asyncio
async def test_ask_routes_to_architecture(runtime: Any) -> None:
    from codebrain.tools import ask
    for query in ("show the architecture", "how is the code structured", "what are the layers"):
        result = await ask.run({"query": query, "project": _PROJECT}, runtime)
        assert "error" not in result
        assert result.get("source_tool") == "get_architecture", (
            f"Expected get_architecture, got {result.get('source_tool')} for '{query}'"
        )


@pytest.mark.asyncio
async def test_ask_routes_to_status(runtime: Any) -> None:
    from codebrain.tools import ask
    for query in ("what's the project status", "any todos left", "show stubs and fixmes"):
        result = await ask.run({"query": query, "project": _PROJECT}, runtime)
        assert "error" not in result
        assert result.get("source_tool") == "project_status", (
            f"Expected project_status, got {result.get('source_tool')} for '{query}'"
        )


@pytest.mark.asyncio
async def test_ask_falls_back_to_query(runtime: Any) -> None:
    from codebrain.tools import ask
    result = await ask.run({"query": "how does RRF fusion work", "project": _PROJECT}, runtime)
    assert "error" not in result
    assert result.get("source_tool") == "query_code", (
        f"Expected query_code fallback, got {result.get('source_tool')}"
    )
    assert "results" in result
    assert len(result["results"]) > 0


@pytest.mark.asyncio
async def test_ask_missing_query_returns_error(runtime: Any) -> None:
    from codebrain.tools import ask
    result = await ask.run({"query": "", "project": _PROJECT}, runtime)
    assert "error" in result


@pytest.mark.asyncio
async def test_ask_result_includes_source_tool(runtime: Any) -> None:
    from codebrain.tools import ask
    result = await ask.run({"query": "what is this", "project": _PROJECT}, runtime)
    assert "source_tool" in result, "ask_project must always set source_tool"
