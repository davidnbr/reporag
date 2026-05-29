"""
SCIP compiler-grade dependency graph builder — research §6.

SCIP (Source Code Indexing Protocol, Sourcegraph) produces exact symbol definitions
and references with file+line coordinates, resolved at compiler/LSP level.
This gives mathematically correct edges vs tree-sitter heuristics (~70% accuracy).

SCIP CLIs (install once per language):
  Python:     pip install scip-python
  JS/TS:      npm install -g @sourcegraph/scip-typescript
  Go:         go install github.com/sourcegraph/scip-go/cmd/scip-go@latest
  Rust:       rust-analyzer generates SCIP via --scip flag
  Java:       scip-java (Sourcegraph)

Output: .scip protobuf file → parsed to exact (src_file, dst_file, edge_type) edges.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# CLI commands per language — {language: [cmd, ...args_with_{root}_placeholder]}
SCIP_COMMANDS: dict[str, list[str]] = {
    "python": ["scip-python", "index", "--project-root", "{root}", "--output", "{output}"],
    "typescript": ["scip-typescript", "--projectRoot", "{root}", "--output", "{output}"],
    "javascript": ["scip-typescript", "--projectRoot", "{root}", "--output", "{output}"],
    "go": ["scip-go", "--root", "{root}", "--output", "{output}"],
    "java": ["scip-java", "index", "--project-root", "{root}", "--output", "{output}"],
}


class SCIPEdge(NamedTuple):
    src_symbol: str
    src_file: str
    src_line: int
    dst_symbol: str
    dst_file: str
    dst_line: int
    edge_type: str  # "reference" | "definition"


def is_available(language: str) -> bool:
    """Return True if the SCIP CLI for this language is installed."""
    cmd = SCIP_COMMANDS.get(language, [])
    if not cmd:
        return False
    return shutil.which(cmd[0]) is not None


def available_languages() -> list[str]:
    """Return languages with SCIP CLI installed."""
    return [lang for lang in SCIP_COMMANDS if is_available(lang)]


def run_scip(root: Path, language: str) -> list[SCIPEdge]:
    """
    Run SCIP CLI for the given language rooted at root.

    Returns list of SCIPEdge objects. Returns [] on failure (caller falls back to heuristic).
    """
    if not is_available(language):
        logger.debug("SCIP CLI not found for %s, skipping", language)
        return []

    with tempfile.NamedTemporaryFile(suffix=".scip", delete=False) as tmp:
        output_path = Path(tmp.name)

    cmd = [
        part.replace("{root}", str(root)).replace("{output}", str(output_path))
        for part in SCIP_COMMANDS[language]
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning("SCIP %s failed: %s", language, result.stderr[:500])
            return []
        return _parse_scip_output(output_path, root)
    except subprocess.TimeoutExpired:
        logger.warning("SCIP %s timed out after 300s", language)
        return []
    except Exception as exc:
        logger.warning("SCIP %s error: %s", language, exc)
        return []
    finally:
        output_path.unlink(missing_ok=True)


def _parse_scip_output(scip_path: Path, root: Path) -> list[SCIPEdge]:
    """Parse .scip protobuf → list of SCIPEdge."""
    if not scip_path.exists() or scip_path.stat().st_size == 0:
        return []

    # Try bundled scip_pb2 first (generated from scip.proto, ships with reporag)
    Index = None
    try:
        from reporag.indexer.scip_pb2 import Index  # type: ignore[import]  # noqa: PLC0415
    except (ImportError, AttributeError):
        try:
            import scip_pb2  # type: ignore[import]  # noqa: PLC0415
            Index = scip_pb2.Index
        except ImportError:
            try:
                import importlib

                from google.protobuf import descriptor_pb2 as _  # noqa: F401, PLC0415
                scip_mod = importlib.import_module("scip.scip_pb2")
                Index = scip_mod.Index
            except (ImportError, AttributeError):
                pass

    if Index is None:
        logger.warning("'scip' Python package not installed or incompatible. Run: pip install reporag")
        return []

    try:
        data = scip_path.read_bytes()
        index = Index()
        index.ParseFromString(data)
    except Exception as exc:
        logger.warning("Failed to parse SCIP output: %s", exc)
        return []

    edges: list[SCIPEdge] = []
    # Build symbol → definition location map
    symbol_defs: dict[str, tuple[str, int]] = {}

    for doc in index.documents:
        rel_path = doc.relative_path
        abs_path = str(root / rel_path)
        for occurrence in doc.occurrences:
            if occurrence.symbol_roles & 1:  # DEFINITION role bit
                symbol_defs[occurrence.symbol] = (abs_path, occurrence.range[0] + 1)

    # Build reference edges
    for doc in index.documents:
        rel_path = doc.relative_path
        abs_path = str(root / rel_path)
        for occurrence in doc.occurrences:
            if not (occurrence.symbol_roles & 1):  # REFERENCE (not definition)
                dst_loc = symbol_defs.get(occurrence.symbol)
                if dst_loc and dst_loc[0] != abs_path:  # cross-file reference
                    edges.append(SCIPEdge(
                        src_symbol=f"{rel_path}::{occurrence.range[0] + 1}",
                        src_file=abs_path,
                        src_line=occurrence.range[0] + 1,
                        dst_symbol=occurrence.symbol,
                        dst_file=dst_loc[0],
                        dst_line=dst_loc[1],
                        edge_type="reference",
                    ))

    return edges
