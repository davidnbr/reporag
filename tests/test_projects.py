"""Unit tests for reporag.projects — registry CRUD and path matching."""

import json
import threading
from pathlib import Path

import pytest


@pytest.fixture()
def registry_path(tmp_path, monkeypatch):
    """Point the registry at a temp dir with no ML config involved."""
    monkeypatch.setenv("REPORAG_DATA_DIR", str(tmp_path))
    # Prevent get_config() from succeeding so we exercise the ImportError fallback path
    monkeypatch.setitem(__import__("sys").modules, "reporag.config", None)
    # Re-import so _registry_path() picks up the patched env
    import importlib
    import reporag.projects as proj_mod
    importlib.reload(proj_mod)
    return tmp_path / "projects.json", proj_mod


def test_update_and_get_exact(registry_path):
    path, proj = registry_path
    proj.update("/home/user/myproject", chunks=100, files=10)
    result = proj.get("/home/user/myproject")
    assert result is not None
    assert result["chunks"] == 100
    assert result["files"] == 10
    assert "indexed_at" in result


def test_get_subpath_match(registry_path):
    _, proj = registry_path
    proj.update("/home/user/myproject", chunks=50, files=5)
    result = proj.get("/home/user/myproject/src/foo")
    assert result is not None
    assert result["chunks"] == 50


def test_get_sibling_no_collision(registry_path):
    """Regression: /home/user/myproject-fork must NOT match /home/user/myproject."""
    _, proj = registry_path
    proj.update("/home/user/myproject", chunks=50, files=5)
    result = proj.get("/home/user/myproject-fork")
    assert result is None


def test_get_sibling_prefix_no_collision(registry_path):
    """Regression: /home/user/myprojectextra must NOT match /home/user/myproject."""
    _, proj = registry_path
    proj.update("/home/user/myproject", chunks=50, files=5)
    result = proj.get("/home/user/myprojectextra")
    assert result is None


def test_get_returns_copy_not_reference(registry_path):
    """Mutating the returned dict must not affect the registry on next load."""
    _, proj = registry_path
    proj.update("/home/user/proj", chunks=10, files=2)
    result = proj.get("/home/user/proj")
    result["chunks"] = 9999
    fresh = proj.get("/home/user/proj")
    assert fresh["chunks"] == 10


def test_get_missing_project(registry_path):
    _, proj = registry_path
    assert proj.get("/nonexistent/path") is None


def test_all_projects_empty(registry_path):
    _, proj = registry_path
    assert proj.all_projects() == {}


def test_all_projects_returns_all(registry_path):
    _, proj = registry_path
    proj.update("/a", chunks=1, files=1)
    proj.update("/b", chunks=2, files=2)
    all_ = proj.all_projects()
    assert "/a" in all_ and "/b" in all_


def test_atomic_write_no_partial_on_concurrent_updates(registry_path, tmp_path):
    """Two concurrent update() calls must not corrupt the registry."""
    path, proj = registry_path
    errors = []

    def _write(name, n):
        try:
            proj.update(f"/project/{name}", chunks=n, files=n)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_write, args=(f"p{i}", i)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    all_ = proj.all_projects()
    assert len(all_) == 10


def test_registry_file_is_valid_json_after_update(registry_path):
    path, proj = registry_path
    proj.update("/x", chunks=5, files=1)
    content = json.loads(path.read_text())
    assert "/x" in content


def test_corrupt_registry_returns_empty(registry_path):
    path, proj = registry_path
    path.write_text("not json{{{")
    assert proj._load() == {}
    assert proj.get("/anything") is None
