"""
Fallback dependency graph builder using tree-sitter import heuristics.

Used when SCIP CLI is not installed for a language. Accuracy ~70% vs SCIP ~99%.
Resolves import statements to file paths using project-relative heuristics.
All edges logged as 'heuristic' source so callers can report coverage.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class HeuristicEdge:
    __slots__ = ("src_file", "dst_file", "import_name", "edge_type", "source")

    def __init__(
        self,
        src_file: str,
        dst_file: str,
        import_name: str,
        edge_type: str = "import",
        source: str = "heuristic",
    ) -> None:
        self.src_file = src_file
        self.dst_file = dst_file
        self.import_name = import_name
        self.edge_type = edge_type
        self.source = source


def extract_imports(file_path: Path, root: Path) -> list[HeuristicEdge]:
    """
    Extract import edges from a source file using regex + path heuristics.

    Args:
        file_path: Absolute path to source file.
        root: Project root directory for resolving relative imports.

    Returns:
        List of HeuristicEdge objects (src=file_path, dst=resolved file or module).
    """
    suffix = file_path.suffix.lower()
    src = file_path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".py":
        return _python_imports(src, file_path, root)
    if suffix in (".js", ".jsx", ".ts", ".tsx"):
        return _js_ts_imports(src, file_path, root)
    if suffix == ".go":
        return _go_imports(src, file_path, root)
    if suffix == ".rs":
        return _rust_imports(src, file_path, root)
    if suffix == ".java":
        return _java_imports(src, file_path, root)
    if suffix == ".rb":
        return _ruby_imports(src, file_path, root)
    return []


# ── Python ───────────────────────────────────────────────────────────────────

_PY_IMPORT = re.compile(
    r"^(?:from\s+([.\w]+)\s+import|import[ \t]+([\w.]+(?:[ \t]*,[ \t]*[\w.]+)*))",
    re.MULTILINE,
)


def _python_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    for m in _PY_IMPORT.finditer(src):
        modules = [m.group(1)] if m.group(1) else [s.strip() for s in m.group(2).split(",")]
        for module in modules:
            resolved = _resolve_python_module(module, file_path, root)
            edges.append(HeuristicEdge(str(file_path), resolved, module))
    return edges


def _resolve_python_module(module: str, src_file: Path, root: Path) -> str:
    """Try to resolve dotted module name to a file path within the project."""
    if module.startswith("."):
        # Relative import: leading dots count parent levels above src_file's package.
        stripped = module.lstrip(".")
        for _ in range(len(module) - len(stripped) - 1):
            src_file = src_file.parent
        base = src_file.parent
        parts = stripped.split(".") if stripped else []
        candidates = (
            [(base / Path(*parts)).with_suffix(".py"), base / Path(*parts) / "__init__.py"]
            if parts
            else [base / "__init__.py"]
        )
        for c in candidates:
            if c.exists():
                return str(c)
        return module

    parts = module.split(".")
    candidates = [
        root / Path(*parts).with_suffix(".py"),
        root / Path(*parts) / "__init__.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return module  # unresolved — keep as module name


# ── JS / TS ──────────────────────────────────────────────────────────────────

_JS_IMPORT = re.compile(
    r"""(?:import\s+.*?\s+from\s+|require\s*\(\s*)['"]([^'"]+)['"]""",
    re.DOTALL,
)


def _js_ts_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    for m in _JS_IMPORT.finditer(src):
        spec = m.group(1)
        resolved = _resolve_js_module(spec, file_path, root)
        edges.append(HeuristicEdge(str(file_path), resolved, spec))
    return edges


def _resolve_js_module(spec: str, src_file: Path, root: Path) -> str:
    if not spec.startswith(("..", ".")):
        return spec  # external package — unresolvable without node_modules
    base = src_file.parent / spec
    for ext in ("", ".ts", ".tsx", ".js", ".jsx", "/index.ts", "/index.js"):
        candidate = Path(str(base) + ext)
        if candidate.exists():
            return str(candidate.resolve())
    return spec


# ── Go ───────────────────────────────────────────────────────────────────────

_GO_IMPORT = re.compile(r'"([^"]+)"')
_GO_IMPORT_BLOCK = re.compile(r"import\s*\(([^)]+)\)", re.DOTALL)
_GO_IMPORT_SINGLE = re.compile(r'^import\s+"([^"]+)"', re.MULTILINE)


def _go_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    specs: list[str] = []
    for block in _GO_IMPORT_BLOCK.finditer(src):
        specs.extend(_GO_IMPORT.findall(block.group(1)))
    for m in _GO_IMPORT_SINGLE.finditer(src):
        specs.append(m.group(1))
    for spec in specs:
        # Resolve project-local packages only (contain root module path)
        parts = spec.split("/")
        candidate = root / Path(*parts)
        dst = str(candidate) if candidate.exists() else spec
        edges.append(HeuristicEdge(str(file_path), dst, spec))
    return edges


# ── Rust ─────────────────────────────────────────────────────────────────────

_RUST_USE = re.compile(r"^use\s+([\w::{}, \n]+);", re.MULTILINE)
_RUST_MOD = re.compile(r"^mod\s+(\w+);", re.MULTILINE)


def _rust_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    for m in _RUST_MOD.finditer(src):
        mod_name = m.group(1)
        sibling = file_path.parent / f"{mod_name}.rs"
        sub_mod = file_path.parent / mod_name / "mod.rs"
        dst = str(sibling) if sibling.exists() else (str(sub_mod) if sub_mod.exists() else mod_name)
        edges.append(HeuristicEdge(str(file_path), dst, mod_name))
    return edges


# ── Java ─────────────────────────────────────────────────────────────────────

_JAVA_IMPORT = re.compile(r"^import\s+([\w.]+);", re.MULTILINE)


def _java_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    for m in _JAVA_IMPORT.finditer(src):
        fqn = m.group(1)
        parts = fqn.split(".")
        candidate = root / Path("src", "main", "java", *parts).with_suffix(".java")
        dst = str(candidate) if candidate.exists() else fqn
        edges.append(HeuristicEdge(str(file_path), dst, fqn))
    return edges


# ── Ruby ─────────────────────────────────────────────────────────────────────

_RUBY_REQUIRE = re.compile(
    r"""^\s*(require_relative|require)\s+['"]([^'"]+)['"]""",
    re.MULTILINE,
)


def _ruby_imports(src: str, file_path: Path, root: Path) -> list[HeuristicEdge]:
    edges: list[HeuristicEdge] = []
    for m in _RUBY_REQUIRE.finditer(src):
        kind, spec = m.group(1), m.group(2)
        base = file_path.parent if kind == "require_relative" else root
        candidates = [
            (base / spec).with_suffix(".rb"),
            base / "lib" / f"{spec}.rb",
        ]
        dst = next((str(c.resolve()) for c in candidates if c.exists()), spec)
        edges.append(HeuristicEdge(str(file_path), dst, spec))
    return edges
