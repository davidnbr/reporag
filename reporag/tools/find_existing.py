"""
MCP tool: find_existing — pre-implementation discovery.

Before writing new code for a task, surface existing functions, classes, and
patterns in the indexed codebase that already handle the described functionality.
Prevents duplication and pattern drift in complex multi-file features.

Pipeline: same hybrid retrieval as query_code (dense + BM25 + RRF), then:
  - Deduplicate by file (max 2 results per file)
  - Add reuse_hint field explaining why each result is relevant
  - Return ranked list of existing code to consider reusing
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PER_FILE = 2


async def run(
    arguments: dict[str, Any],
    runtime: Any,
) -> dict[str, Any]:
    """
    Execute find_existing tool.

    Args:
        arguments: {
            task: str (required) — description of what you're about to implement,
            project: str | None — restrict to this project root path,
            k: int (default 10) — number of results to return,
        }
    """
    task: str = arguments.get("task", "").strip()
    if not task:
        return {"error": "task is required"}

    cfg = runtime.config
    k: int = int(arguments.get("k", 10))
    project_filter: str | None = arguments.get("project")

    # ── Hybrid retrieval (same pipeline as query_code, no reranker) ──────────
    q_vec = runtime.embedder.encode_query(task)
    dense_ids = runtime.dense.search(q_vec, k=cfg.dense_candidates, project=project_filter)
    sparse_ids = (
        runtime.bm25.search(task, k=cfg.sparse_candidates, project=project_filter)
        if runtime.bm25.is_ready
        else []
    )

    from reporag.retrieval.rrf import rrf_fuse

    fused = rrf_fuse(
        [dense_ids, sparse_ids],
        k=cfg.rrf_k,
        weights=[cfg.rrf_dense_weight, cfg.rrf_sparse_weight],
    )

    # k*6 (not k*4) — project prefilter already applied, so widen the pool to
    # compensate for the chunk_type filter dropping module/window chunks.
    candidate_ids = [doc_id for doc_id, _ in list(fused.items())[: k * 6]]
    candidates = runtime.dense.get_chunks(candidate_ids)

    # Restore merged score order
    id_rank = {doc_id: i for i, doc_id in enumerate(candidate_ids)}
    candidates.sort(key=lambda c: id_rank.get(c.get("id", ""), len(candidate_ids)))

    # Skip module-level chunks — functions and classes are actionable reuse targets
    candidates = [c for c in candidates if c.get("chunk_type") in {"function", "method", "class"}]

    # ── Deduplicate by file (max _MAX_PER_FILE per file) ─────────────────────
    file_counts: dict[str, int] = {}
    deduped = []
    for chunk in candidates:
        fp = chunk.get("file_path", "")
        if file_counts.get(fp, 0) < _MAX_PER_FILE:
            file_counts[fp] = file_counts.get(fp, 0) + 1
            deduped.append(chunk)
        if len(deduped) >= k:
            break

    # ── Format with reuse hints ───────────────────────────────────────────────
    results = []
    for chunk in deduped:
        name = chunk.get("name", "")
        chunk_type = chunk.get("chunk_type", "")
        file_path = chunk.get("file_path", "")
        start_line = chunk.get("start_line", 0)
        name_readable = name.replace("_", " ")
        reuse_hint = (
            f"Existing {chunk_type} '{name}' in {file_path}:{start_line} "
            f"may already handle '{name_readable}'. "
            f"Consider importing/extending instead of reimplementing."
        )
        results.append(
            {
                "file": file_path,
                "name": name,
                "chunk_type": chunk_type,
                "language": chunk.get("language", ""),
                "start_line": start_line,
                "end_line": chunk.get("end_line", 0),
                "snippet": chunk.get("semantic_text", "")[:400],
                "reuse_hint": reuse_hint,
            }
        )

    return {
        "task": task,
        "existing_code": results,
        "total_found": len(results),
        "message": (
            f"Found {len(results)} existing code items that may be relevant. "
            "Review before implementing to avoid duplication."
            if results
            else "No closely related existing code found. Safe to implement from scratch."
        ),
    }
