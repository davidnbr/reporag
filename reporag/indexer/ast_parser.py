"""AST-aware code chunk extraction via tree-sitter — research §1, §2."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import tree_sitter

LANGUAGE_EXT: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".ex": "elixir",
    ".exs": "elixir",
    ".tf": "hcl",
    ".tfvars": "hcl",
}

# Node types to extract per language (type → chunk_type label)
_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {
        "function_definition": "function",
        "async_function_definition": "function",
        "class_definition": "class",
    },
    "javascript": {
        "function_declaration": "function",
        "function_expression": "function",
        "arrow_function": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "typescript": {
        "function_declaration": "function",
        "function_expression": "function",
        "arrow_function": "function",
        "class_declaration": "class",
        "method_definition": "method",
        "interface_declaration": "interface",
    },
    "tsx": {
        "function_declaration": "function",
        "arrow_function": "function",
        "class_declaration": "class",
        "method_definition": "method",
    },
    "go": {
        "function_declaration": "function",
        "method_declaration": "method",
        "type_declaration": "class",
    },
    "rust": {
        "function_item": "function",
        "struct_item": "class",
        "impl_item": "class",
        "trait_item": "interface",
    },
    "java": {
        "method_declaration": "method",
        "class_declaration": "class",
        "interface_declaration": "interface",
        "constructor_declaration": "function",
    },
    "c": {
        "function_definition": "function",
        "struct_specifier": "class",
    },
    "cpp": {
        "function_definition": "function",
        "class_specifier": "class",
        "struct_specifier": "class",
    },
    "ruby": {
        "method": "method",
        "singleton_method": "method",
        "class": "class",
        "module": "class",
    },
    # Elixir and HCL have no dedicated def/class node types — `defmodule Foo`,
    # `def bar`, `resource "x" "y" {}` etc. all parse as generic `call`/`block`
    # nodes. Dispatched via _CUSTOM_CHUNK_INFO instead of this table.
    "elixir": {},
    "hcl": {},
}

# Name field node types per language
_NAME_FIELDS: dict[str, list[str]] = {
    "python": ["name"],
    "javascript": ["name", "key"],
    "typescript": ["name", "key"],
    "tsx": ["name", "key"],
    "go": ["name"],
    "rust": ["name"],
    "java": ["name"],
    "c": ["declarator"],
    "cpp": ["declarator"],
    "ruby": ["name"],
}


@dataclass
class Chunk:
    """Single semantic code unit extracted from an AST node."""

    id: str
    file_path: str
    language: str
    chunk_type: str
    name: str
    raw_content: str
    start_line: int
    end_line: int
    parent_name: str | None = None
    existing_docstring: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def make_id(cls, file_path: str, name: str, start_line: int) -> str:
        """Stable content-addressed ID for a chunk."""
        key = f"{file_path}::{name}::{start_line}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]


def detect_language(path: Path) -> str | None:
    """Return tree-sitter language name for file, or None if unsupported."""
    return LANGUAGE_EXT.get(path.suffix.lower())


def _get_node_name(node: tree_sitter.Node, language: str, src: bytes) -> str:
    """Extract the name identifier from a named AST node."""
    fields = _NAME_FIELDS.get(language, ["name"])
    for field_name in fields:
        child = node.child_by_field_name(field_name)
        if child:
            # For C/C++ declarators, recurse one level
            inner = child.child_by_field_name("declarator") or child
            return src[inner.start_byte : inner.end_byte].decode("utf-8", errors="replace").strip()
    # Fallback: first named child that looks like an identifier
    for child in node.children:
        if child.type in (
            "identifier",
            "type_identifier",
            "field_identifier",
            "property_identifier",
        ):
            return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace").strip()
    return "<anonymous>"


def _extract_docstring(node: tree_sitter.Node, language: str, src: bytes) -> str | None:
    """Extract leading docstring/comment for a node if present."""
    body = node.child_by_field_name("body")
    if not body:
        return None
    for child in body.children:
        if child.type in ("expression_statement", "block"):
            for sub in child.children:
                if sub.type in ("string", "raw_string_literal"):
                    text = src[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace")
                    return text.strip("\"' \n\t").strip()
        if child.type == "comment":
            return (
                src[child.start_byte : child.end_byte]
                .decode("utf-8", errors="replace")
                .lstrip("/# ")
                .strip()
            )
    return None


_ELIXIR_DEF_KINDS: dict[str, str] = {
    "defmodule": "class",
    "defprotocol": "interface",
    "defimpl": "class",
    "def": "function",
    "defp": "function",
    "defmacro": "function",
    "defmacrop": "function",
    "defguard": "function",
    "defguardp": "function",
    "defdelegate": "function",
}


def _elixir_chunk_info(node: tree_sitter.Node, src: bytes) -> tuple[str, str] | None:
    """Elixir has no def/class node types — `def foo`, `defmodule Foo`, etc.
    are all `call` nodes whose first child identifier names the construct."""
    if node.type != "call" or not node.children:
        return None
    head = node.children[0]
    if head.type != "identifier":
        return None
    keyword = src[head.start_byte : head.end_byte].decode("utf-8", errors="replace")
    chunk_type = _ELIXIR_DEF_KINDS.get(keyword)
    if chunk_type is None:
        return None

    for child in node.children[1:]:
        if child.type != "arguments":
            continue
        for arg in child.children:
            if arg.type == "alias":  # defmodule Foo.Bar
                return chunk_type, src[arg.start_byte : arg.end_byte].decode(
                    "utf-8", errors="replace"
                )
            if arg.type in ("call", "identifier"):  # def foo(...), do: .. / def foo, do: ..
                target = arg.children[0] if arg.type == "call" and arg.children else arg
                if target.type == "identifier":
                    return chunk_type, src[target.start_byte : target.end_byte].decode(
                        "utf-8", errors="replace"
                    )
    return chunk_type, "<anonymous>"


_HCL_BLOCK_KINDS: dict[str, str] = {
    "resource": "class",
    "data": "class",
    "module": "class",
    "provider": "class",
    "variable": "function",
    "output": "function",
    "locals": "function",
}


def _hcl_chunk_info(node: tree_sitter.Node, src: bytes) -> tuple[str, str] | None:
    """HCL top-level constructs are all `block` nodes — the block kind and its
    labels are its first identifier + string_lit children."""
    if node.type != "block" or not node.children:
        return None
    head = node.children[0]
    if head.type != "identifier":
        return None
    keyword = src[head.start_byte : head.end_byte].decode("utf-8", errors="replace")
    chunk_type = _HCL_BLOCK_KINDS.get(keyword)
    if chunk_type is None:
        return None

    labels: list[str] = []
    for child in node.children[1:]:
        if child.type != "string_lit":
            continue
        for sub in child.children:
            if sub.type == "template_literal":
                labels.append(src[sub.start_byte : sub.end_byte].decode("utf-8", errors="replace"))

    if not labels:
        return chunk_type, "<anonymous>"
    if keyword in ("resource", "data"):
        return chunk_type, ".".join(labels)
    return chunk_type, labels[-1]


# Languages where def/class constructs aren't distinguishable by node.type
# alone — see _elixir_chunk_info / _hcl_chunk_info for why.
_CUSTOM_CHUNK_INFO: dict[str, Callable[[tree_sitter.Node, bytes], tuple[str, str] | None]] = {
    "elixir": _elixir_chunk_info,
    "hcl": _hcl_chunk_info,
}


def _walk_extract(
    node: tree_sitter.Node,
    src: bytes,
    file_path: str,
    language: str,
    target_types: dict[str, str],
    parent_name: str | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    # is_named excludes keyword tokens — in Ruby the `class`/`module` keywords
    # themselves have node.type "class"/"module" and would match target_types.
    chunk_type = target_types.get(node.type) if node.is_named else None
    name: str | None = None

    if chunk_type is None and node.is_named:
        custom = _CUSTOM_CHUNK_INFO.get(language)
        if custom:
            info = custom(node, src)
            if info:
                chunk_type, name = info

    if chunk_type:
        if name is None:
            name = _get_node_name(node, language, src)
        raw = src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        docstring = _extract_docstring(node, language, src)
        chunk_id = Chunk.make_id(file_path, name, node.start_point[0])
        chunks.append(
            Chunk(
                id=chunk_id,
                file_path=file_path,
                language=language,
                chunk_type=chunk_type,
                name=name,
                raw_content=raw,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_name=parent_name,
                existing_docstring=docstring,
            )
        )
        # For class/interface nodes, recurse with this as parent
        new_parent = name if chunk_type in ("class", "interface") else parent_name
    else:
        new_parent = parent_name

    for child in node.children:
        chunks.extend(_walk_extract(child, src, file_path, language, target_types, new_parent))

    return chunks


def _make_parser(language: str):  # type: ignore[return]
    """Build a tree-sitter Parser for the given language using per-language packages."""
    from tree_sitter import Language, Parser

    try:
        if language == "python":
            import tree_sitter_python as mod

            lang_obj = Language(mod.language())
        elif language == "javascript":
            import tree_sitter_javascript as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "typescript":
            import tree_sitter_typescript as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language_typescript())
        elif language == "tsx":
            import tree_sitter_typescript as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language_tsx())
        elif language == "go":
            import tree_sitter_go as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "rust":
            import tree_sitter_rust as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "java":
            import tree_sitter_java as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "c":
            import tree_sitter_c as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "cpp":
            import tree_sitter_cpp as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "ruby":
            import tree_sitter_ruby as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "elixir":
            import tree_sitter_elixir as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        elif language == "hcl":
            import tree_sitter_hcl as mod  # type: ignore[no-redef]

            lang_obj = Language(mod.language())
        else:
            return None
        return Parser(lang_obj)
    except ImportError:
        return None


def parse_file(path: Path) -> list[Chunk]:
    """
    Parse a source file and return all semantic chunks.

    Returns empty list for unsupported languages or parse errors.
    """
    language = detect_language(path)
    if not language:
        return []

    parser = _make_parser(language)
    if parser is None:
        return []

    src = path.read_bytes()
    tree = parser.parse(src)
    target_types = _NODE_TYPES.get(language, {})

    chunks = _walk_extract(tree.root_node, src, str(path), language, target_types)

    # Add a module-level chunk for the whole file (imports, exports summary)
    if chunks:
        file_chunk = Chunk(
            id=Chunk.make_id(str(path), "__module__", 0),
            file_path=str(path),
            language=language,
            chunk_type="module",
            name=path.stem,
            raw_content=src.decode("utf-8", errors="replace")[:2000],
            start_line=1,
            end_line=tree.root_node.end_point[0] + 1,
        )
        chunks.insert(0, file_chunk)

    return chunks
