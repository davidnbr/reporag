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
) -> dict[str, float]:
    """
    Fuse multiple ranked lists into a single RRF score dict.

    Args:
        rankings: Each inner list is a ranked result list (best first) of doc IDs.
        k: Smoothing constant (standardized at 60).

    Returns:
        Dict mapping doc_id → RRF score, unsorted. Call sorted(..., reverse=True).
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return dict(scores)


def top_k(fused: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Return top-k (doc_id, score) pairs sorted descending."""
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
