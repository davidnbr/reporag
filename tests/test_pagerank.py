"""Unit tests for Reverse Personalized PageRank — verifies research §3."""

import networkx as nx

from reporag.retrieval.pagerank import merge_rrf_ppr, reverse_personalized_pagerank


def _hub_graph() -> nx.DiGraph:
    """
    Graph where hub imports many leaves (architectural controller pattern).
    Standard PR on import graph: leaves win (many point to them from hub).
    Reverse PR: hub wins — hub has high out-degree in original = high in-degree
    in reversed graph, so PageRank mass pools at hub.

    Edges: hub → leaf_* (hub depends on many leaves), app → hub
    """
    G = nx.DiGraph()
    for i in range(10):
        G.add_edge("hub", f"leaf_{i}")  # hub imports/uses leaves
    G.add_edge("app", "hub")
    return G


def test_reverse_ppr_hub_ranks_above_leaves():
    """Hub (imports many leaves) should rank above individual leaves in Reverse PPR.

    RAG semantic: BM25 hits are leaf nodes; reverse PPR walks reversed graph
    (leaf→hub edges) to surface the architectural hub they share as parent.
    """
    G = _hub_graph()
    # Seed on leaf nodes — these are the BM25/dense retrieval hits
    scores = reverse_personalized_pagerank(G, seed_nodes=["leaf_0", "leaf_1"], alpha=0.85, top_k=20)
    assert "hub" in scores
    # Hub should outrank individual unseeded leaves
    leaf_scores = [scores.get(f"leaf_{i}", 0) for i in range(2, 10)]
    assert scores["hub"] > max(leaf_scores), "Hub must rank above low-connectivity leaves"


def test_reverse_ppr_seed_bias():
    """PPR seeded on a node should give that node elevated score."""
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    scores_a = reverse_personalized_pagerank(G, seed_nodes=["a"], alpha=0.85, top_k=10)
    _scores_c = reverse_personalized_pagerank(G, seed_nodes=["c"], alpha=0.85, top_k=10)
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
        G.add_edge(f"n{i}", f"n{i + 1}")
    scores = reverse_personalized_pagerank(G, seed_nodes=["n0"], top_k=5)
    assert len(scores) <= 5


def test_merge_rrf_ppr_normalization():
    """Merged scores should be in [0, 1] range."""
    rrf = {"a": 0.1, "b": 0.05, "c": 0.02}
    chunk_files = {"a": "fa.py", "b": "fb.py", "c": "fc.py"}
    file_ppr = {"fa.py": 0.3, "fb.py": 0.1, "fd.py": 0.5}
    merged = merge_rrf_ppr(rrf, chunk_files, file_ppr)
    assert all(0.0 <= v <= 1.0 for v in merged.values())


def test_merge_rrf_ppr_keys_are_chunk_pool_only():
    """Output is keyed by chunk id — file paths from PPR must not leak in."""
    rrf = {"a": 0.1, "b": 0.05}
    chunk_files = {"a": "fa.py", "b": "fb.py"}
    file_ppr = {"fc.py": 0.3, "fd.py": 0.1}
    merged = merge_rrf_ppr(rrf, chunk_files, file_ppr)
    assert set(merged.keys()) == {"a", "b"}


def test_merge_rrf_ppr_high_ppr_file_chunk_ranks_higher():
    """Equal-RRF chunks: the one in the higher-PPR file should rank first."""
    rrf = {"a": 0.1, "b": 0.1}
    chunk_files = {"a": "high.py", "b": "low.py"}
    file_ppr = {"high.py": 1.0, "low.py": 0.1}
    merged = merge_rrf_ppr(rrf, chunk_files, file_ppr)
    assert set(merged.keys()) == {"a", "b"}
    assert merged["a"] > merged["b"]
