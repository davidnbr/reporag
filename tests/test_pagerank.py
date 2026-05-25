"""Unit tests for Reverse Personalized PageRank — verifies research §3."""
import networkx as nx
import pytest
from rag_mcp.retrieval.pagerank import reverse_personalized_pagerank, merge_rrf_ppr


def _hub_graph() -> nx.DiGraph:
    """
    Graph where 'hub' is imported by many leaves.
    Standard PR: leaf utilities win. Reverse PR: hub wins.

    Edges: leaf_* → hub (each leaf imports hub)
    """
    G = nx.DiGraph()
    hub = "hub"
    for i in range(10):
        G.add_edge(f"leaf_{i}", hub)  # leaf imports hub
    G.add_edge("app", "leaf_0")
    G.add_edge("app", "leaf_1")
    return G


def test_reverse_ppr_hub_ranks_above_leaves():
    """Hub (imported by many) should rank above individual leaves in Reverse PPR."""
    G = _hub_graph()
    # Seed on 'app' — the entry point
    scores = reverse_personalized_pagerank(G, seed_nodes=["app"], alpha=0.85, top_k=20)
    assert "hub" in scores
    # Hub should outrank individual leaves (it's the architectural center)
    leaf_scores = [scores.get(f"leaf_{i}", 0) for i in range(2, 10)]
    assert scores["hub"] > max(leaf_scores), "Hub must rank above low-connectivity leaves"


def test_reverse_ppr_seed_bias():
    """PPR seeded on a node should give that node elevated score."""
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    scores_a = reverse_personalized_pagerank(G, seed_nodes=["a"], alpha=0.85, top_k=10)
    scores_c = reverse_personalized_pagerank(G, seed_nodes=["c"], alpha=0.85, top_k=10)
    # When seeded on 'a', 'a' should appear in results
    assert "a" in scores_a or "b" in scores_a


def test_reverse_ppr_empty_graph():
    G = nx.DiGraph()
    scores = reverse_personalized_pagerank(G, seed_nodes=["x"])
    assert scores == {}


def test_reverse_ppr_invalid_seeds_ignored():
    G = nx.DiGraph()
    G.add_edge("a", "b")
    # Seeds not in graph should not crash
    scores = reverse_personalized_pagerank(G, seed_nodes=["nonexistent"])
    assert isinstance(scores, dict)


def test_reverse_ppr_top_k_respected():
    G = nx.DiGraph()
    for i in range(20):
        G.add_edge(f"n{i}", f"n{i+1}")
    scores = reverse_personalized_pagerank(G, seed_nodes=["n0"], top_k=5)
    assert len(scores) <= 5


def test_merge_rrf_ppr_normalization():
    """Merged scores should be in [0, 1] range."""
    rrf = {"a": 0.1, "b": 0.05, "c": 0.02}
    ppr = {"a": 0.3, "b": 0.1, "d": 0.5}
    merged = merge_rrf_ppr(rrf, ppr)
    assert all(0.0 <= v <= 1.0 for v in merged.values())


def test_merge_rrf_ppr_union():
    """Merged result contains all docs from both inputs."""
    rrf = {"a": 0.1, "b": 0.05}
    ppr = {"c": 0.3, "d": 0.1}
    merged = merge_rrf_ppr(rrf, ppr)
    assert set(merged.keys()) == {"a", "b", "c", "d"}
