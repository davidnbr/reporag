#!/usr/bin/env python3
"""
Retrieval quality benchmark — ablation across pipeline stages.

Synthetic golden set: named chunks (functions/classes) from the live index.
Query template: "implementation of {name}" / "how does {name} work"
Expected: chunk's own ID appears in top-k results.

Stages:
  dense    — LanceDB cosine only
  bm25     — BM25 sparse only
  rrf      — dense + BM25 fused (RRF k=60)
  rrf+ppr  — RRF + Reverse Personalized PageRank
  full     — RRF + PPR + cross-encoder rerank

Metrics:
  Recall@k — fraction of queries where gold ID in top-k
  MRR@k    — mean reciprocal rank (0 if not found in top-k)

Usage:
    devenv shell -- python scripts/benchmark.py
    devenv shell -- python scripts/benchmark.py --k 5 10 --samples 50
    devenv shell -- python scripts/benchmark.py --project /path/to/project
    devenv shell -- python scripts/benchmark.py --stages dense rrf full
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_QUERY_TEMPLATES = [
    "implementation of {name}",
    "how does {name} work",
]

_CHUNK_TYPES = {"function", "class", "method"}


# ── Runtime ──────────────────────────────────────────────────────────────────


def _build_runtime(data_dir: str | None) -> Any:
    from reporag.config import Config, get_config
    from reporag.server import Runtime

    cfg = get_config()
    if data_dir:
        cfg = Config(**{**cfg.model_dump(), "data_dir": data_dir})
    rt = Runtime(config=cfg)
    rt.initialize()
    return rt


def _get_named_chunks(
    runtime: Any,
    project: str | None,
    max_samples: int,
    seed: int,
    chunk_types: set[str] | None = None,
) -> list[dict]:
    runtime.dense._open_or_create_table()
    rows = runtime.dense._table.search().limit(max_samples * 20).to_list()
    types = chunk_types if chunk_types is not None else _CHUNK_TYPES
    named = [
        r
        for r in rows
        if r.get("name")
        and r.get("chunk_type") in types
        and (not project or r.get("file_path", "").startswith(project))
    ]
    if len(named) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(named)
        named = named[:max_samples]
    return named


# ── Pipeline stages ───────────────────────────────────────────────────────────


def _extract_symbol_name(query: str) -> str:
    """Extract bare symbol name from benchmark query templates."""
    for prefix in ("implementation of ", "how does "):
        if query.startswith(prefix):
            query = query[len(prefix):]
            break
    return query.removesuffix(" work").strip()


def _stage_grep(rt: Any, _q_vec: Any, query: str, k: int) -> list[str]:
    """Simulate grep: exact match on chunk name field — what Claude+tools would do."""
    name = _extract_symbol_name(query)
    rt.dense._open_or_create_table()
    escaped = name.replace("'", "''")
    rows = rt.dense._table.search().where(f"name = '{escaped}'").limit(k * 5).to_list()
    return [r["id"] for r in rows[:k]]


def _stage_dense(rt: Any, q_vec: Any, _query: str, k: int) -> list[str]:
    return rt.dense.search(q_vec, k=k)


def _stage_bm25(rt: Any, _q_vec: Any, query: str, k: int) -> list[str]:
    if not rt.bm25.is_ready:
        return []
    return rt.bm25.search(query, k=k)


def _stage_rrf(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
    from reporag.retrieval.rrf import rrf_fuse, top_k

    dense_ids = rt.dense.search(q_vec, k=50)
    sparse_ids = rt.bm25.search(query, k=50) if rt.bm25.is_ready else []
    fused = rrf_fuse([dense_ids, sparse_ids], k=60)
    return [doc_id for doc_id, _ in top_k(fused, k)]


def _stage_rrf_ppr(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
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


def _stage_full(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
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


def _stage_full_rerank(rt: Any, q_vec: Any, query: str, k: int) -> list[str]:
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
    candidate_ids = list(merged.keys())[: k * 3]
    candidates = rt.dense.get_chunks(candidate_ids)

    if candidates and len(candidates) <= 50:
        candidates = rt.reranker.rerank(query, candidates)

    return [c["id"] for c in candidates[:k]]


_STAGES: dict[str, Any] = {
    "grep": _stage_grep,
    "dense": _stage_dense,
    "bm25": _stage_bm25,
    "rrf": _stage_rrf,
    "rrf+ppr": _stage_rrf_ppr,
    "full": _stage_full,
    "full+rerank": _stage_full_rerank,
}


# ── Metrics ───────────────────────────────────────────────────────────────────


def _recall_mrr(pairs: list[tuple[str, list[str]]], k: int) -> tuple[float, float]:
    """(Recall@k, MRR@k) from (gold_id, retrieved_ids) pairs."""
    hits = 0
    rr_sum = 0.0
    for gold_id, retrieved in pairs:
        if gold_id in retrieved:
            hits += 1
            rr_sum += 1.0 / (retrieved.index(gold_id) + 1)
    n = len(pairs)
    return (hits / n if n else 0.0), (rr_sum / n if n else 0.0)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebrain retrieval quality benchmark")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10], metavar="K")
    parser.add_argument("--samples", type=int, default=100, help="Max golden queries")
    parser.add_argument("--project", type=str, help="Restrict to project root path")
    parser.add_argument("--data-dir", type=str, help="Override REPORAG_DATA_DIR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", choices=list(_STAGES), default=list(_STAGES))
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument(
        "--filter-chunk-types",
        nargs="+",
        metavar="TYPE",
        default=list(_CHUNK_TYPES),
        help="Chunk types to include in golden set (default: function class method)",
    )
    args = parser.parse_args()

    if not args.quiet:
        print("Loading runtime...", flush=True)
    t0 = time.monotonic()
    rt = _build_runtime(args.data_dir)
    if not args.quiet:
        print(f"Runtime loaded in {time.monotonic() - t0:.1f}s")

    chunk_types = set(args.filter_chunk_types)
    chunks = _get_named_chunks(rt, args.project, args.samples, args.seed, chunk_types)
    if not chunks:
        print("No named chunks found. Run: index_codebase path=<project>")
        sys.exit(1)

    if not args.quiet:
        print(f"\nGolden set : {len(chunks)} named chunks ({', '.join(sorted(chunk_types))})")
        if args.project:
            print(f"Project    : {args.project}")
        print(f"Stages     : {', '.join(args.stages)}")
        print(f"k values   : {args.k}\n")

    # Build (gold_id, query) pairs
    golden: list[tuple[str, str, str]] = []
    for i, chunk in enumerate(chunks):
        tmpl = _QUERY_TEMPLATES[i % len(_QUERY_TEMPLATES)]
        golden.append((chunk["id"], tmpl.format(name=chunk["name"]), chunk["name"]))

    max_k = max(args.k)

    # Accumulate per-stage per-k results
    results: dict[str, dict[int, list[tuple[str, list[str]]]]] = {
        stage: {k: [] for k in args.k} for stage in args.stages
    }
    latencies: dict[str, list[float]] = {stage: [] for stage in args.stages}

    if not args.quiet:
        print(f"Running {len(golden)} queries × {len(args.stages)} stages...", flush=True)
    for idx, (gold_id, query, _name) in enumerate(golden):
        if not args.quiet and idx % 10 == 0:
            print(f"  {idx:4d}/{len(golden)}", end="\r", flush=True)

        q_vec = rt.embedder.encode_query(query)

        for stage in args.stages:
            t_start = time.monotonic()
            retrieved = _STAGES[stage](rt, q_vec, query, max_k)
            latencies[stage].append(time.monotonic() - t_start)

            for k in args.k:
                results[stage][k].append((gold_id, retrieved[:k]))

    if not args.quiet:
        print(f"  {len(golden):4d}/{len(golden)} done    ")

    # Print results table
    k_headers = " │ ".join(f"Recall@{k:<2d}  MRR@{k:<2d}" for k in args.k)
    sep = "─" * (11 + 17 * len(args.k) + 14)
    print(f"\n{'Stage':<10} │ {k_headers} │ ms/query")
    print(sep)

    for stage in args.stages:
        parts = []
        for k in args.k:
            recall, mrr = _recall_mrr(results[stage][k], k)
            parts.append(f"  {recall:.3f}     {mrr:.3f}")
        avg_ms = sum(latencies[stage]) / len(latencies[stage]) * 1000
        print(f"{stage:<10} │ {'  │  '.join(parts)} │ {avg_ms:7.1f}")

    print()

    # Summary: improvement of full vs dense at largest k
    k = max(args.k)
    if "full" in args.stages and "dense" in args.stages:
        r_full, _ = _recall_mrr(results["full"][k], k)
        r_dense, _ = _recall_mrr(results["dense"][k], k)
        delta = r_full - r_dense
        print(f"Full pipeline vs dense-only Recall@{k}: {delta:+.3f}")
    if "full+rerank" in args.stages and "full" in args.stages:
        r_rerank, _ = _recall_mrr(results["full+rerank"][k], k)
        r_full, _ = _recall_mrr(results["full"][k], k)
        delta = r_rerank - r_full
        print(f"Reranker impact on full pipeline Recall@{k}: {delta:+.3f}")


if __name__ == "__main__":
    main()
