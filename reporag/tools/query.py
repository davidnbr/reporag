"""
MCP tool: query_code — full retrieval pipeline in a single round-trip.

Pipeline (research §3, §4):
  1. Dense search  → LanceDB cosine (top 50)
  2. Sparse search → BM25 k1=1.2, b=0.75 (top 50)
  3. RRF fusion    → k=60 (research §4)
  4. Reverse PPR   → hub-adjusted scores (research §3)
  5. Score merge   → 0.6·RRF + 0.4·PPR
  6. Rerank        → cross-encoder if ≤ reranker_k candidates
  7. Expand        → k-hop subgraph neighbors
  8. Return        → flat ranked list in ONE MCP response
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    """
    Execute query_code tool.

    Args:
        arguments: {
            query: str (required),
            k: int (default 10) — number of final results,
            rerank: bool (default True),
            languages: list[str] | None,
        }
    """
    query: str = arguments.get("query", "").strip()
    if not query:
        return {"error": "query is required"}

    cfg = runtime.config
    k: int = int(arguments.get("k", 10))
    do_rerank: bool = arguments.get("rerank", cfg.rerank_by_default)
    lang_filter: list[str] | None = arguments.get("languages")
    # Project scoping is mandatory: results from one repo must never leak into
    # another. When the caller omits `project`, scope to the server's project
    # (cwd-derived) instead of searching the shared index unscoped.
    from reporag.projects import default_root

    project_filter: str = arguments.get("project") or default_root()

    # ── 1. Dense retrieval ──────────────────────────────────────────────────
    q_vec = runtime.embedder.encode_query(query)
    dense_ids = runtime.dense.search(
        q_vec, k=cfg.dense_candidates, project=project_filter, languages=lang_filter
    )

    # ── 2. Sparse retrieval (BM25) ──────────────────────────────────────────
    sparse_ids = (
        runtime.bm25.search(query, k=cfg.sparse_candidates, project=project_filter)
        if runtime.bm25.is_ready
        else []
    )

    # ── 3. RRF fusion (weighted: dense=1.0, sparse=0.5) ─────────────────────
    from reporag.retrieval.rrf import rrf_fuse
    from reporag.retrieval.rrf import top_k as rrf_top_k

    fused = rrf_fuse(
        [dense_ids, sparse_ids],
        k=cfg.rrf_k,
        weights=[cfg.rrf_dense_weight, cfg.rrf_sparse_weight],
    )

    # ── 4. Reverse Personalized PageRank (gated on graph quality) ───────────
    # PPR operates on the dependency graph, whose nodes are file paths — not
    # chunk ids. Map RRF-top chunk ids -> their file paths to seed PPR, and
    # get back file_path -> PPR score (merged onto chunks in step 6).
    seed_ids = [doc_id for doc_id, _ in rrf_top_k(fused, cfg.ppr_seed_k)]
    file_ppr: dict[str, float] = {}
    graph_edge_count = runtime.graph.number_of_edges() if runtime.graph is not None else 0
    ppr_enabled = (
        runtime.graph is not None
        and runtime.graph.number_of_nodes() > 0
        and graph_edge_count >= cfg.min_graph_edges_for_ppr
    )
    # Use reduced PPR weight when graph is heuristic-only (no compiler-grade SCIP edges)
    ppr_weight = 0.4 if graph_edge_count >= cfg.min_graph_edges_for_ppr * 5 else 0.2
    if ppr_enabled:
        from reporag.retrieval.pagerank import reverse_personalized_pagerank

        seed_chunks = runtime.dense.get_chunks(seed_ids)
        seed_files = [c["file_path"] for c in seed_chunks if c.get("file_path")]
        file_ppr = reverse_personalized_pagerank(
            runtime.graph, seed_files, alpha=cfg.ppr_alpha, top_k=k * 3
        )

    # ── 5. Fetch candidate chunks (RRF pool only — PPR re-ranks within it) ──
    candidate_ids = [doc_id for doc_id, _ in rrf_top_k(fused, k * 3)]
    candidates = runtime.dense.get_chunks(candidate_ids)

    # Restore RRF order before re-scoring (get_chunks returns DB scan order)
    id_rank = {doc_id: i for i, doc_id in enumerate(candidate_ids)}
    candidates.sort(key=lambda c: id_rank.get(c.get("id", ""), len(candidate_ids)))

    # ── 6. Score merge (per-chunk RRF + its file's PPR score) ───────────────
    from reporag.retrieval.pagerank import merge_rrf_ppr

    chunk_files = {c["id"]: c.get("file_path", "") for c in candidates}
    rrf_pool = {doc_id: score for doc_id, score in fused.items() if doc_id in chunk_files}
    merged = merge_rrf_ppr(rrf_pool, chunk_files, file_ppr, ppr_weight=ppr_weight)
    candidates.sort(key=lambda c: merged.get(c.get("id", ""), 0.0), reverse=True)

    # ── 7. Cross-encoder rerank ─────────────────────────────────────────────
    if do_rerank and candidates and len(candidates) <= cfg.reranker_k:
        candidates = runtime.reranker.rerank(query, candidates)

    # ── 8. Subgraph expansion ───────────────────────────────────────────────
    final = candidates[:k]
    final = _expand_subgraph(final, runtime, cfg.subgraph_hops)

    return {
        "query": query,
        "results": _format_results(final, merged, cfg.snippet_chars),
        "total_candidates": len(candidates),
        "pipeline": {
            "dense_hits": len(dense_ids),
            "sparse_hits": len(sparse_ids),
            "rrf_merged": len(fused),
            "ppr_applied": len(file_ppr) > 0,
            "reranked": do_rerank and len(candidates) <= cfg.reranker_k,
        },
    }


def _expand_subgraph(
    chunks: list[dict[str, Any]],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
    hops: int,
) -> list[dict[str, Any]]:
    """Add k-hop neighbor file paths to each result's metadata."""
    if runtime.graph is None or hops == 0:
        return chunks

    for chunk in chunks:
        file_path = chunk.get("file_path", "")
        neighbors: set[str] = set()
        frontier = {file_path}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in runtime.graph:
                    next_frontier.update(runtime.graph.successors(node))
                    next_frontier.update(runtime.graph.predecessors(node))
            frontier = next_frontier - neighbors - {file_path}
            neighbors.update(next_frontier)
        chunk["neighbors"] = list(neighbors)[:10]

    return chunks


def _format_results(
    chunks: list[dict[str, Any]],
    scores: dict[str, float],
    snippet_chars: int = 600,
) -> list[dict[str, Any]]:
    out = []
    for chunk in chunks:
        out.append(
            {
                "file": chunk.get("file_path", ""),
                "name": chunk.get("name", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "language": chunk.get("language", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
                "score": round(scores.get(chunk.get("id", ""), 0.0), 4),
                "rerank_score": round(chunk.get("rerank_score", 0.0), 4),
                "snippet": chunk.get("semantic_text", "")[:snippet_chars],
                "neighbors": chunk.get("neighbors", []),
            }
        )
    return out
