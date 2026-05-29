"""
Reverse Personalized PageRank for code dependency graphs — research §3.

Standard PageRank on an import graph pools mass at leaf utilities (logger, utils).
Reverse PR transposes the graph so mass flows toward architectural hubs (App, Router,
Controller) — the nodes that are most *depended upon*.

Personalization vector is biased toward RRF hit nodes so the ranking is query-focused,
not just globally structural.

PR formulation (research §3):
  PR(u) = (1-d)/|V| + d * Σ_{v ∈ B_u} PR(v) / L(v)
  where B_u = nodes that reference u, d = 0.85

This is mathematically equivalent to running standard NetworkX pagerank on the
*reversed* graph (edges point dependency → dependent).
"""
from __future__ import annotations

import networkx as nx


def reverse_personalized_pagerank(
    graph: nx.DiGraph,
    seed_nodes: list[str],
    alpha: float = 0.85,
    top_k: int = 20,
) -> dict[str, float]:
    """
    Run Reverse Personalized PageRank.

    Args:
        graph: Directed dependency graph (edges: importer → imported).
        seed_nodes: Top RRF hit node IDs — used as teleportation targets.
        alpha: Damping factor (0.85 per research §3).
        top_k: Return only top-k nodes by PR score.

    Returns:
        Dict mapping node_id → PR score, top_k entries only.
    """
    if not graph.nodes:
        return {}

    g_rev = graph.reverse(copy=False)

    valid_seeds = [n for n in seed_nodes if n in g_rev]
    if not valid_seeds:
        personalization = None
    else:
        weight = 1.0 / len(valid_seeds)
        personalization = {n: 0.0 for n in g_rev.nodes()}
        for node in valid_seeds:
            personalization[node] = weight

    pr = nx.pagerank(g_rev, alpha=alpha, personalization=personalization, max_iter=100)
    return dict(sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k])


def merge_rrf_ppr(
    rrf_scores: dict[str, float],
    ppr_scores: dict[str, float],
    rrf_weight: float = 0.6,
    ppr_weight: float = 0.4,
) -> dict[str, float]:
    """
    Combine normalized RRF and PPR scores into a single ranking.
    Both inputs normalized to [0, 1] before merge.
    """
    def _normalize(d: dict[str, float]) -> dict[str, float]:
        if not d:
            return d
        max_v = max(d.values())
        if max_v == 0:
            return d
        return {k: v / max_v for k, v in d.items()}

    rrf_n = _normalize(rrf_scores)
    ppr_n = _normalize(ppr_scores)

    all_ids = set(rrf_n) | set(ppr_n)
    merged = {
        doc_id: rrf_weight * rrf_n.get(doc_id, 0.0) + ppr_weight * ppr_n.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    return dict(sorted(merged.items(), key=lambda x: x[1], reverse=True))
