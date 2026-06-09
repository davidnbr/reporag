"""
Integration tests for end-to-end retrieval quality.

Requires ML extras (sentence-transformers, lancedb) and at least one indexed project.
Skipped automatically if no data exists.

Run:
    devenv shell -- pytest tests/test_retrieval_quality.py -m integration -v

Synthetic golden set: named chunks (functions/classes) extracted from the live index.
Query: "implementation of {name}" — expects chunk's own ID in top-k.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pytestmark = pytest.mark.integration

_CHUNK_TYPES = {"function", "class", "method"}
_GOLDEN_SIZE = 40
_DISCOVERY_SIZE = 40
_MIN_DESCRIPTION_CHARS = 30
_SEED = 42


def _strip_name_from_semantic(semantic_text: str, name: str) -> str:
    """Remove leading 'Function/Method/Class {readable_name}.' prefix from semantic_text."""
    readable = name.replace("_", " ").replace("-", " ")
    for prefix in (f"Function {readable}.", f"Method {readable}.", f"Class {readable}."):
        if semantic_text.startswith(prefix):
            return semantic_text[len(prefix):].strip()
    return semantic_text


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def runtime() -> Any:
    try:
        from reporag.config import get_config
        from reporag.server import Runtime
    except ImportError as e:
        pytest.skip(f"reporag not importable: {e}")

    try:
        import lancedb  # noqa: F401
        import sentence_transformers  # noqa: F401
    except ImportError:
        pytest.skip("ML extras not installed (pip install reporag[ml])")

    rt = Runtime(config=get_config())
    try:
        rt.initialize()
    except Exception as e:
        pytest.skip(f"Runtime init failed: {e}")

    if rt.dense.count() == 0:
        pytest.skip("No chunks indexed. Run index_codebase first.")

    return rt


@pytest.fixture(scope="module")
def golden(runtime: Any) -> list[tuple[str, str]]:
    """(gold_id, query) pairs built from named chunks in the live index."""
    runtime.dense._open_or_create_table()
    rows = runtime.dense._table.search().limit(_GOLDEN_SIZE * 10).to_list()
    named = [r for r in rows if r.get("name") and r.get("chunk_type") in _CHUNK_TYPES]

    if len(named) < 5:
        pytest.skip(f"Too few named chunks ({len(named)}). Index more code first.")

    rng = random.Random(_SEED)
    rng.shuffle(named)
    named = named[:_GOLDEN_SIZE]

    templates = ["implementation of {name}", "how does {name} work"]
    return [(c["id"], templates[i % 2].format(name=c["name"])) for i, c in enumerate(named)]


# ── Retrieval helpers ─────────────────────────────────────────────────────────


def _top_ids_dense(rt: Any, q_vec: Any, k: int) -> list[str]:
    return rt.dense.search(q_vec, k=k)


def _top_ids_rrf(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
    from reporag.retrieval.rrf import rrf_fuse, top_k

    dense_ids = rt.dense.search(q_vec, k=50)
    sparse_ids = rt.bm25.search(query, k=50) if rt.bm25.is_ready else []
    fused = rrf_fuse([dense_ids, sparse_ids], k=60)
    return [doc_id for doc_id, _ in top_k(fused, k)]


def _top_ids_full(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
    from reporag.retrieval.pagerank import merge_rrf_ppr, reverse_personalized_pagerank
    from reporag.retrieval.rrf import rrf_fuse, top_k

    dense_ids = rt.dense.search(q_vec, k=50)
    sparse_ids = rt.bm25.search(query, k=50) if rt.bm25.is_ready else []
    fused = rrf_fuse([dense_ids, sparse_ids], k=60)

    ppr_scores: dict[str, float] = {}
    if rt.graph is not None and rt.graph.number_of_nodes() > 0:
        seeds = [doc_id for doc_id, _ in top_k(fused, 20)]
        ppr_scores = reverse_personalized_pagerank(rt.graph, seeds, alpha=0.85, top_k=k * 3)

    merged = merge_rrf_ppr(fused, ppr_scores)
    return list(merged.keys())[:k]


def _recall(pairs: list[tuple[str, list[str]]]) -> float:
    hits = sum(1 for gold, retrieved in pairs if gold in retrieved)
    return hits / len(pairs) if pairs else 0.0


def _mrr(pairs: list[tuple[str, list[str]]]) -> float:
    rr = [1.0 / (r.index(g) + 1) if g in r else 0.0 for g, r in pairs]
    return sum(rr) / len(rr) if rr else 0.0


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_full_pipeline_recall_at_10(runtime: Any, golden: list) -> None:
    """Full pipeline Recall@10 >= 0.5 on synthetic golden set."""
    pairs = []
    for gold_id, query in golden:
        q_vec = runtime.embedder.encode_query(query)
        retrieved = _top_ids_full(runtime, q_vec, query, k=10)
        pairs.append((gold_id, retrieved))

    recall = _recall(pairs)
    mrr = _mrr(pairs)
    print(f"\n  Full pipeline — Recall@10={recall:.3f}  MRR@10={mrr:.3f}")
    assert recall >= 0.75, f"Recall@10={recall:.3f} < 0.75 threshold"


def test_full_pipeline_recall_at_5(runtime: Any, golden: list) -> None:
    """Full pipeline Recall@5 >= 0.35 on synthetic golden set."""
    pairs = []
    for gold_id, query in golden:
        q_vec = runtime.embedder.encode_query(query)
        retrieved = _top_ids_full(runtime, q_vec, query, k=5)
        pairs.append((gold_id, retrieved))

    recall = _recall(pairs)
    assert recall >= 0.75, f"Recall@5={recall:.3f} < 0.75 threshold"


def test_rrf_not_worse_than_dense(runtime: Any, golden: list) -> None:
    """RRF Recall@10 must not regress more than 10pp vs dense-only."""
    dense_pairs, rrf_pairs = [], []
    for gold_id, query in golden:
        q_vec = runtime.embedder.encode_query(query)
        dense_pairs.append((gold_id, _top_ids_dense(runtime, q_vec, k=10)))
        rrf_pairs.append((gold_id, _top_ids_rrf(runtime, q_vec, query, k=10)))

    r_dense = _recall(dense_pairs)
    r_rrf = _recall(rrf_pairs)
    print(f"\n  dense Recall@10={r_dense:.3f}  rrf Recall@10={r_rrf:.3f}")
    assert r_rrf >= r_dense - 0.10, (
        f"RRF Recall@10={r_rrf:.3f} regressed >10pp vs dense={r_dense:.3f}"
    )


def test_full_not_worse_than_rrf(runtime: Any, golden: list) -> None:
    """Full pipeline Recall@10 must not regress more than 10pp vs RRF."""
    rrf_pairs, full_pairs = [], []
    for gold_id, query in golden:
        q_vec = runtime.embedder.encode_query(query)
        rrf_pairs.append((gold_id, _top_ids_rrf(runtime, q_vec, query, k=10)))
        full_pairs.append((gold_id, _top_ids_full(runtime, q_vec, query, k=10)))

    r_rrf = _recall(rrf_pairs)
    r_full = _recall(full_pairs)
    print(f"\n  rrf Recall@10={r_rrf:.3f}  full Recall@10={r_full:.3f}")
    assert r_full >= r_rrf - 0.10, (
        f"Full pipeline Recall@10={r_full:.3f} regressed >10pp vs rrf={r_rrf:.3f}"
    )


def test_mrr_at_10(runtime: Any, golden: list) -> None:
    """Full pipeline MRR@10 >= 0.3 — found chunks rank near the top."""
    pairs = []
    for gold_id, query in golden:
        q_vec = runtime.embedder.encode_query(query)
        retrieved = _top_ids_full(runtime, q_vec, query, k=10)
        pairs.append((gold_id, retrieved))

    mrr = _mrr(pairs)
    assert mrr >= 0.65, f"MRR@10={mrr:.3f} < 0.65 threshold"


@pytest.fixture(scope="module")
def golden_discovery(runtime: Any) -> list[tuple[str, str]]:
    """(gold_id, description_query) pairs for discovery mode.

    Simulates "I want to write X" without knowing the function name.
    Query = semantic_text minus the leading 'Function {name}.' prefix.
    Circularity note: queries are derived from semantic_text, so recall is
    optimistic vs. real-world unknown-unknowns; it validates the embedding
    space is usable for discovery, not a real-world IR benchmark.
    """
    runtime.dense._open_or_create_table()
    rows = runtime.dense._table.search().limit(_DISCOVERY_SIZE * 10).to_list()
    candidates = [
        r for r in rows
        if r.get("name")
        and r.get("chunk_type") in _CHUNK_TYPES
        and r.get("semantic_text")
    ]

    # Only chunks with meaningful description beyond just the name prefix
    meaningful = []
    for r in candidates:
        desc = _strip_name_from_semantic(r["semantic_text"], r["name"])
        if len(desc) >= _MIN_DESCRIPTION_CHARS:
            meaningful.append((r, desc))

    if len(meaningful) < 5:
        pytest.skip(f"Too few chunks with meaningful semantic_text ({len(meaningful)}). Index more code first.")

    rng = random.Random(_SEED)
    rng.shuffle(meaningful)
    meaningful = meaningful[:_DISCOVERY_SIZE]

    return [(r["id"], desc) for r, desc in meaningful]


def test_discovery_recall_at_10(runtime: Any, golden_discovery: list) -> None:
    """Discovery Recall@10 >= 0.50 using description-only queries (no function name).

    Validates that semantic retrieval can surface existing code when Claude
    describes what it wants to implement — the core anti-duplication use case.
    """
    pairs = []
    for gold_id, description in golden_discovery:
        q_vec = runtime.embedder.encode_query(description)
        retrieved = _top_ids_full(runtime, q_vec, description, k=10)
        pairs.append((gold_id, retrieved))

    recall = _recall(pairs)
    mrr = _mrr(pairs)
    print(f"\n  Discovery — Recall@10={recall:.3f}  MRR@10={mrr:.3f}  (n={len(pairs)})")
    assert recall >= 0.50, f"Discovery Recall@10={recall:.3f} < 0.50 threshold"
