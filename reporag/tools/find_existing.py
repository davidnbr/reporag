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
import re
from typing import Any

logger = logging.getLogger(__name__)

_MAX_PER_FILE = 2
_RELEVANCE_RATIO = 0.35

_SEMANTIC_PREFIX_RE = re.compile(r"^(?:Function|Method|Class|Interface)\s+\S[^.]*\.\s*")
_PARAM_RETURN_RE = re.compile(r"^(?:(?:Parameters|Returns):[^.]*\.\s*)+")


def _description_from_semantic(semantic_text: str, name: str) -> str:
    """Strip the type/name/signature prefix from semantic_text, leaving the doc-derived description."""
    text = _SEMANTIC_PREFIX_RE.sub("", semantic_text, count=1).strip()
    text = _PARAM_RETURN_RE.sub("", text).strip()
    if not text:
        return f"implements {name.replace('_', ' ').replace('-', ' ')}"
    first_sentence = text.split(". ", 1)[0].rstrip(".")
    return first_sentence[:160]


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

    # ── Relevance threshold (relative to top RRF score) ──────────────────────
    top_score = max((fused.get(c.get("id", ""), 0.0) for c in candidates), default=0.0)
    if top_score > 0:
        candidates = [
            c for c in candidates if fused.get(c.get("id", ""), 0.0) >= _RELEVANCE_RATIO * top_score
        ]

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
        description = _description_from_semantic(chunk.get("semantic_text", ""), name)
        reuse_hint = (
            f"Existing {chunk_type} '{name}' at {file_path}:{start_line}. "
            f"{description}. Import/extend instead of reimplementing."
        )
        signature = chunk.get("raw_content", "").splitlines()[0][:200] if chunk.get("raw_content") else ""
        score = fused.get(chunk.get("id", ""), 0.0) / top_score if top_score > 0 else 0.0
        results.append(
            {
                "file": file_path,
                "name": name,
                "chunk_type": chunk_type,
                "language": chunk.get("language", ""),
                "start_line": start_line,
                "end_line": chunk.get("end_line", 0),
                "signature": signature,
                "snippet": chunk.get("semantic_text", "")[:400],
                "reuse_hint": reuse_hint,
                "score": round(score, 4),
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
