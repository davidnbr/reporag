"""Tests for heuristic Python import extraction (Task 5).

Covers the `_PY_IMPORT` multiline-eating regression, multi-import
splitting (`import a, b`), and relative-import resolution.
"""

from __future__ import annotations

from pathlib import Path

from reporag.indexer.heuristic_graph import extract_imports


def test_multiline_imports_no_newline_in_dst(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text("import inspect\nimport os\n\ndef foo():\n    pass\n")

    edges = extract_imports(f, tmp_path)
    names = {e.import_name for e in edges}
    assert names == {"inspect", "os"}
    for edge in edges:
        assert "\n" not in edge.dst_file
        assert "\n" not in edge.import_name


def test_import_a_b_yields_two_edges(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text("import a, b\n")

    edges = extract_imports(f, tmp_path)
    assert len(edges) == 2
    assert {e.import_name for e in edges} == {"a", "b"}
    for edge in edges:
        assert "\n" not in edge.dst_file


def test_relative_import_resolves_to_sibling(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "config.py").write_text("VALUE = 1\n")
    mod = pkg / "mod.py"
    mod.write_text("from .config import VALUE\n")

    edges = extract_imports(mod, tmp_path)
    assert len(edges) == 1
    assert edges[0].dst_file == str(pkg / "config.py")


def test_relative_import_dot_only_resolves_to_package_init(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    mod = pkg / "mod.py"
    mod.write_text("from . import config\n")

    edges = extract_imports(mod, tmp_path)
    assert len(edges) == 1
    assert edges[0].dst_file == str(pkg / "__init__.py")


def test_parent_relative_import_resolves(tmp_path: Path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "shared.py").write_text("X = 1\n")
    sub = pkg / "sub"
    sub.mkdir()
    (sub / "__init__.py").write_text("")
    mod = sub / "mod.py"
    mod.write_text("from ..shared import X\n")

    edges = extract_imports(mod, tmp_path)
    assert len(edges) == 1
    assert edges[0].dst_file == str(pkg / "shared.py")


def test_unresolved_module_kept_as_name(tmp_path: Path):
    f = tmp_path / "main.py"
    f.write_text("import requests\n")

    edges = extract_imports(f, tmp_path)
    assert len(edges) == 1
    assert edges[0].dst_file == "requests"


def test_ruby_require_relative_resolves_to_sibling(tmp_path: Path):
    helper = tmp_path / "helper.rb"
    helper.write_text("def help_me; end\n")
    main = tmp_path / "main.rb"
    main.write_text("require 'json'\nrequire_relative 'helper'\n")

    edges = extract_imports(main, tmp_path)
    by_name = {e.import_name: e for e in edges}

    assert by_name["helper"].dst_file == str(helper.resolve())
    # external gem stays unresolved as its require name
    assert by_name["json"].dst_file == "json"


def test_ruby_require_resolves_from_lib(tmp_path: Path):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "billing.rb").write_text("module Billing; end\n")
    main = tmp_path / "main.rb"
    main.write_text("require 'billing'\n")

    edges = extract_imports(main, tmp_path)
    assert len(edges) == 1
    assert edges[0].dst_file == str((lib / "billing.rb").resolve())
