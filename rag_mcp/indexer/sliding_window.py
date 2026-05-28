"""
Sliding window chunker — line-based overlap chunking.

arXiv:2605.04763 (2025): sliding window beats function-level AST chunking
by 3.5-5.6 points exact match on RepoEval. Window=64 lines, overlap=16 lines
is the empirically best configuration at standard token budgets.

Used in "sliding" and "hybrid" chunk strategies.
"""
from __future__ import annotations

from pathlib import Path

from rag_mcp.indexer.ast_parser import Chunk, detect_language


def sliding_window_chunks(
    path: Path,
    window_lines: int = 64,
    overlap_lines: int = 16,
) -> list[Chunk]:
    """
    Chunk a file into overlapping line windows.

    Each window is stride=(window_lines - overlap_lines) lines apart.
    Produces chunk_type="window" — no symbol name, but preserves local context.
    """
    language = detect_language(path)
    if not language:
        return []

    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = src.splitlines(keepends=True)
    if not lines:
        return []

    stride = max(1, window_lines - overlap_lines)
    chunks: list[Chunk] = []
    i = 0
    while i < len(lines):
        end = min(i + window_lines, len(lines))
        content = "".join(lines[i:end])
        start_line = i + 1
        end_line = end
        chunks.append(Chunk(
            id=Chunk.make_id(str(path), f"__window_{i}__", i),
            file_path=str(path),
            language=language,
            chunk_type="window",
            name=f"{path.stem}:{start_line}-{end_line}",
            raw_content=content,
            start_line=start_line,
            end_line=end_line,
        ))
        if end >= len(lines):
            break
        i += stride

    return chunks


def hybrid_chunks(
    path: Path,
    window_lines: int = 64,
    overlap_lines: int = 16,
) -> list[Chunk]:
    """
    AST named chunks + sliding window on lines not covered by any AST chunk.

    Preserves symbol lookup (named AST chunks) while filling module-level code,
    imports, and inter-function regions that function-level chunking drops.
    """
    from rag_mcp.indexer.ast_parser import parse_file

    ast_chunks = parse_file(path)
    if not ast_chunks:
        return sliding_window_chunks(path, window_lines, overlap_lines)

    # Determine lines covered by named AST chunks (exclude the module chunk)
    named = [c for c in ast_chunks if c.chunk_type != "module"]
    covered: set[int] = set()
    for c in named:
        covered.update(range(c.start_line, c.end_line + 1))

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return ast_chunks

    language = detect_language(path) or "unknown"
    gap_chunks: list[Chunk] = []

    # Emit sliding windows over contiguous uncovered blocks (min 5 lines)
    block_start: int | None = None
    for ln in range(1, len(lines) + 1):
        in_gap = ln not in covered
        if in_gap and block_start is None:
            block_start = ln
        elif not in_gap and block_start is not None:
            gap_chunks.extend(
                _window_block(lines, block_start, ln - 1, str(path), language, window_lines, overlap_lines)
            )
            block_start = None
    if block_start is not None:
        gap_chunks.extend(
            _window_block(lines, block_start, len(lines), str(path), language, window_lines, overlap_lines)
        )

    return ast_chunks + gap_chunks


def _window_block(
    all_lines: list[str],
    start_line: int,
    end_line: int,
    file_path: str,
    language: str,
    window_lines: int,
    overlap_lines: int,
) -> list[Chunk]:
    """Emit overlapping windows over [start_line, end_line] (1-based, inclusive)."""
    block = all_lines[start_line - 1:end_line]
    if len(block) < 5:
        return []

    stride = max(1, window_lines - overlap_lines)
    chunks: list[Chunk] = []
    i = 0
    while i < len(block):
        end = min(i + window_lines, len(block))
        content = "".join(block[i:end])
        abs_start = start_line + i
        abs_end = start_line + end - 1
        stem = Path(file_path).stem
        chunks.append(Chunk(
            id=Chunk.make_id(file_path, f"__gap_{abs_start}__", abs_start),
            file_path=file_path,
            language=language,
            chunk_type="window",
            name=f"{stem}:{abs_start}-{abs_end}",
            raw_content=content,
            start_line=abs_start,
            end_line=abs_end,
        ))
        if end >= len(block):
            break
        i += stride

    return chunks
