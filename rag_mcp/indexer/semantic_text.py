"""
NL-like text extraction from AST chunks for embedding — research §4.

Goal: project code latent space → NL query space (manifold alignment).
Greptile proof: embedding docstrings >> embedding raw code for MIPS precision.
No LLM required. Optional Ollama path for richer text.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_mcp.indexer.ast_parser import Chunk


def _clean(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _extract_params_python(raw: str) -> str:
    """Extract parameter names from raw Python function source."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", raw)
    if not match:
        return ""
    params = match.group(1)
    # Strip type annotations for readability
    cleaned = re.sub(r":\s*[^,=)]+", "", params)
    return _clean(cleaned)


def _extract_params_generic(raw: str) -> str:
    """Heuristic param extraction for JS/TS/Go/Rust/Java."""
    match = re.search(r"\(([^)]{0,200})\)", raw)
    if not match:
        return ""
    return _clean(match.group(1))


def _extract_return_type(raw: str) -> str:
    """Heuristic return type extraction."""
    # Python: -> Type
    m = re.search(r"->\s*([^\n:]{1,60})", raw)
    if m:
        return _clean(m.group(1))
    # Go: func name(...) RetType
    m = re.search(r"\)\s+([A-Z][a-zA-Z0-9*\[\]]+)\s*\{", raw)
    if m:
        return _clean(m.group(1))
    return ""


def chunk_to_semantic_text(chunk: "Chunk") -> str:
    """
    Convert a Chunk to embeddable natural-language-like text.

    This is the core manifold alignment step: strips syntactic entropy,
    keeps conceptual entropy in NL form that shares feature distribution
    with user queries.
    """
    raw = chunk.raw_content
    doc = _clean(chunk.existing_docstring) if chunk.existing_docstring else ""
    name_readable = chunk.name.replace("_", " ").replace("-", " ")

    if chunk.chunk_type == "module":
        imports = _extract_imports(raw)
        return (
            f"Module {chunk.name} in file {chunk.file_path}. "
            f"Language: {chunk.language}. "
            f"{('Imports: ' + imports + '.') if imports else ''} "
            f"{doc}"
        ).strip()

    if chunk.chunk_type == "function":
        params = _extract_params_python(raw) if chunk.language == "python" else _extract_params_generic(raw)
        ret = _extract_return_type(raw)
        parent = f" in {chunk.parent_name}" if chunk.parent_name else ""
        return (
            f"Function {name_readable}{parent}. "
            f"{('Parameters: ' + params + '.') if params else ''} "
            f"{('Returns: ' + ret + '.') if ret else ''} "
            f"{doc}"
        ).strip()

    if chunk.chunk_type == "method":
        params = _extract_params_python(raw) if chunk.language == "python" else _extract_params_generic(raw)
        ret = _extract_return_type(raw)
        cls = chunk.parent_name or "unknown class"
        return (
            f"Method {name_readable} of class {cls}. "
            f"{('Parameters: ' + params + '.') if params else ''} "
            f"{('Returns: ' + ret + '.') if ret else ''} "
            f"{doc}"
        ).strip()

    if chunk.chunk_type == "class":
        bases = _extract_bases(raw, chunk.language)
        return (
            f"Class {name_readable}. "
            f"{('Extends: ' + bases + '.') if bases else ''} "
            f"{doc}"
        ).strip()

    if chunk.chunk_type == "interface":
        return (
            f"Interface {name_readable}. "
            f"{doc}"
        ).strip()

    if chunk.chunk_type == "window":
        # Sliding window chunk — raw code content for BM25/dense code matching
        # No NL transformation: the code itself is the best representation
        return (
            f"Code from {chunk.file_path} lines {chunk.start_line}-{chunk.end_line}. "
            f"Language: {chunk.language}. "
            f"{chunk.raw_content[:800]}"
        ).strip()

    return f"{chunk.chunk_type} {name_readable}. {doc}".strip()


def _extract_imports(raw: str) -> str:
    """Extract import names from module source (first 50 lines)."""
    lines = raw.splitlines()[:50]
    imports: list[str] = []
    for line in lines:
        line = line.strip()
        if line.startswith(("import ", "from ", "require(", "use ", "using ")):
            imports.append(line[:80])
        if len(imports) >= 10:
            break
    return ", ".join(imports)


def _extract_bases(raw: str, language: str) -> str:
    """Extract base class / interface names."""
    if language == "python":
        m = re.search(r"class\s+\w+\s*\(([^)]+)\)", raw)
    elif language in ("typescript", "javascript", "java"):
        m = re.search(r"(?:extends|implements)\s+([A-Za-z0-9_, <>\[\]]+)", raw)
    else:
        return ""
    return _clean(m.group(1)) if m else ""
