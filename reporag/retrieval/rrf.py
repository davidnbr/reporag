"""
Reciprocal Rank Fusion (RRF) — research §4.

RRF(d) = Σ_{m ∈ M} 1 / (k + r_m(d))

k=60 is the standardized constant that smooths the impact of low-ranked outliers.
Documents sorted descending by RRF score form the final context ordering.
"""

from __future__ import annotations

from collections import defaultdict


def rrf_fuse(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[str, float]:
    """
    Fuse multiple ranked lists into a single RRF score dict.

    Args:
        rankings: Each inner list is a ranked result list (best first) of doc IDs.
        k: Smoothing constant (standardized at 60).
        weights: Per-list multipliers (default: equal 1.0 each).
                 Use [1.0, 0.5] to down-weight a weaker retriever (e.g. BM25).

    Returns:
        Dict mapping doc_id → RRF score, unsorted. Call sorted(..., reverse=True).
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for ranked_list, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += weight / (k + rank + 1)
    return dict(scores)


def top_k(fused: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Return top-k (doc_id, score) pairs sorted descending."""
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
