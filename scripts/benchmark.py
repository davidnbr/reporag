#!/usr/bin/env python3
"""
Retrieval quality benchmark — ablation across pipeline stages.

Two modes:
  named     — query by symbol name ("implementation of {name}"). Tests named-symbol lookup.
              Grep baseline included. Claude+tools baseline.
  discovery — query by description (semantic_text minus name). Tests unknown-unknown discovery:
              "I want to write something that does X, does existing code already do this?"
              Grep drops to near-zero; RAG must carry the load.

Stages:
  grep       — exact name match on chunk name field (named mode only)
  dense      — LanceDB cosine only
  bm25       — BM25 sparse only
  rrf        — dense + BM25 fused (RRF k=60)
  rrf+ppr    — RRF + Reverse Personalized PageRank
  full       — RRF + PPR (production default, no reranker)
  full+rerank — full + cross-encoder rerank

Metrics:
  Recall@k — fraction of queries where gold ID in top-k
  MRR@k    — mean reciprocal rank (0 if not found in top-k)

Usage:
    devenv shell -- python scripts/benchmark.py
    devenv shell -- python scripts/benchmark.py --mode discovery --samples 50
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

_DISCOVERY_TEMPLATES = [
    "{description}",
    "implement {description}",
]

_CHUNK_TYPES = {"function", "class", "method"}
# Min chars of semantic_text beyond the name prefix to be useful as a discovery query
_MIN_DESCRIPTION_CHARS = 30


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


def _strip_name_from_semantic(semantic_text: str, name: str) -> str:
    """Remove the leading 'Function/Method/Class {name}.' prefix from semantic_text."""
    readable = name.replace("_", " ").replace("-", " ")
    for prefix in (
        f"Function {readable}.",
        f"Method {readable}.",
        f"Class {readable}.",
        f"Interface {readable}.",
    ):
        if semantic_text.startswith(prefix):
            return semantic_text[len(prefix) :].strip()
    # Fallback: strip up to first period if it contains the name
    first_sentence, _, rest = semantic_text.partition(".")
    if name in first_sentence or readable in first_sentence:
        return rest.strip()
    return semantic_text


def _get_discovery_chunks(
    runtime: Any,
    project: str | None,
    max_samples: int,
    seed: int,
    chunk_types: set[str] | None = None,
) -> list[dict]:
    """Chunks with enough description in semantic_text to form a discovery query."""
    runtime.dense._open_or_create_table()
    rows = runtime.dense._table.search().limit(max_samples * 20).to_list()
    types = chunk_types if chunk_types is not None else _CHUNK_TYPES
    candidates = []
    for r in rows:
        if not r.get("name") or r.get("chunk_type") not in types:
            continue
        if project and not r.get("file_path", "").startswith(project):
            continue
        semantic = r.get("semantic_text", "")
        description = _strip_name_from_semantic(semantic, r["name"])
        if len(description) >= _MIN_DESCRIPTION_CHARS:
            r["_description"] = description
            candidates.append(r)
    if len(candidates) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(candidates)
        candidates = candidates[:max_samples]
    return candidates


# ── Pipeline stages ───────────────────────────────────────────────────────────


def _extract_symbol_name(query: str) -> str:
    """Extract bare symbol name from benchmark query templates."""
    for prefix in ("implementation of ", "how does "):
        if query.startswith(prefix):
            query = query[len(prefix) :]
            break
    return query.removesuffix(" work").strip()


def _stage_grep(rt: Any, _q_vec: Any, query: str, k: int) -> list[str]:
    """Simulate grep: exact match on chunk name field — what Claude+tools would do."""
    name = _extract_symbol_name(query)
    rt.dense._open_or_create_table()
    escaped = name.replace("'", "''")
    rows = rt.dense._table.search().where(f"name = '{escaped}'").limit(k * 5).to_list()
    return [r["id"] for r in rows[:k]]


_GREP_STOPWORDS = frozenset(
    "a an the and or in of to for with from by on at is are was were be been "
    "this that it its we they do does did can could would should have has had "
    "function method class implement returns parameters returns".split()
)


def _stage_grep_discovery(rt: Any, _q_vec: Any, query: str, k: int) -> list[str]:
    """
    Discovery-mode grep simulation: word-overlap on raw_content (actual source code).

    Represents `grep -r <keywords> .` when you know what you want to build but not
    what it's called. Uses raw_content (not semantic_text) to avoid the circularity
    of searching in the text that was used to generate discovery queries.

    Note: discovery queries are derived from semantic_text, which has the same
    vocabulary as semantic_text — so scores here are an upper bound on real-world
    grep effectiveness, not a lower bound.
    """
    words = {
        w.lower()
        for w in query.replace(".", " ").replace(",", " ").split()
        if len(w) > 3 and w.lower() not in _GREP_STOPWORDS
    }
    if not words:
        return []
    rt.dense._open_or_create_table()
    rows = rt.dense._table.search().limit(5000).to_list()
    scored: list[tuple[int, str]] = []
    for r in rows:
        # grep operates on raw source code, not the NL description
        text = (r.get("raw_content") or "").lower()
        overlap = sum(1 for w in words if w in text)
        if overlap > 0:
            scored.append((overlap, r["id"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc_id for _, doc_id in scored[:k]]


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
    "grep-discovery": _stage_grep_discovery,
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


_NAMED_DEFAULT_STAGES = ["grep", "dense", "bm25", "rrf", "rrf+ppr", "full", "full+rerank"]
_DISCOVERY_DEFAULT_STAGES = ["grep-discovery", "dense", "bm25", "rrf", "rrf+ppr", "full"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Codebrain retrieval quality benchmark")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10], metavar="K")
    parser.add_argument("--samples", type=int, default=100, help="Max golden queries")
    parser.add_argument("--project", type=str, help="Restrict to project root path")
    parser.add_argument("--data-dir", type=str, help="Override REPORAG_DATA_DIR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["named", "discovery"],
        default="named",
        help=(
            "named: query by symbol name (grep-friendly). "
            "discovery: query by description — tests unknown-unknown retrieval."
        ),
    )
    parser.add_argument("--stages", nargs="+", choices=list(_STAGES), default=None)
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
    discovery_mode = args.mode == "discovery"

    if discovery_mode:
        chunks = _get_discovery_chunks(rt, args.project, args.samples, args.seed, chunk_types)
        default_stages = _DISCOVERY_DEFAULT_STAGES
    else:
        chunks = _get_named_chunks(rt, args.project, args.samples, args.seed, chunk_types)
        default_stages = _NAMED_DEFAULT_STAGES

    stages = args.stages if args.stages is not None else default_stages

    if not chunks:
        label = "discovery chunks with descriptions" if discovery_mode else "named chunks"
        print(f"No {label} found. Run: index_codebase path=<project>")
        sys.exit(1)

    if not args.quiet:
        label = "discovery chunks" if discovery_mode else "named chunks"
        print(f"\nMode       : {args.mode}")
        print(f"Golden set : {len(chunks)} {label} ({', '.join(sorted(chunk_types))})")
        if args.project:
            print(f"Project    : {args.project}")
        print(f"Stages     : {', '.join(stages)}")
        print(f"k values   : {args.k}\n")

    # Build (gold_id, query) pairs
    golden: list[tuple[str, str, str]] = []
    if discovery_mode:
        templates = _DISCOVERY_TEMPLATES
        for i, chunk in enumerate(chunks):
            tmpl = templates[i % len(templates)]
            description = chunk.get("_description", chunk.get("semantic_text", chunk["name"]))
            golden.append((chunk["id"], tmpl.format(description=description), chunk["name"]))
    else:
        for i, chunk in enumerate(chunks):
            tmpl = _QUERY_TEMPLATES[i % len(_QUERY_TEMPLATES)]
            golden.append((chunk["id"], tmpl.format(name=chunk["name"]), chunk["name"]))

    max_k = max(args.k)

    # Accumulate per-stage per-k results
    results: dict[str, dict[int, list[tuple[str, list[str]]]]] = {
        stage: {k: [] for k in args.k} for stage in stages
    }
    latencies: dict[str, list[float]] = {stage: [] for stage in stages}

    if not args.quiet:
        print(f"Running {len(golden)} queries × {len(stages)} stages...", flush=True)
    for idx, (gold_id, query, _name) in enumerate(golden):
        if not args.quiet and idx % 10 == 0:
            print(f"  {idx:4d}/{len(golden)}", end="\r", flush=True)

        q_vec = rt.embedder.encode_query(query)

        for stage in stages:
            t_start = time.monotonic()
            retrieved = _STAGES[stage](rt, q_vec, query, max_k)
            latencies[stage].append(time.monotonic() - t_start)

            for k in args.k:
                results[stage][k].append((gold_id, retrieved[:k]))

    if not args.quiet:
        print(f"  {len(golden):4d}/{len(golden)} done    ")

    # Print results table
    stage_col = max(len(s) for s in stages) + 1
    k_headers = " │ ".join(f"Recall@{k:<2d}  MRR@{k:<2d}" for k in args.k)
    sep = "─" * (stage_col + 3 + 17 * len(args.k) + 14)
    print(f"\n{'':<{stage_col}} │ {k_headers} │ ms/query")
    print(sep)

    for stage in stages:
        parts = []
        for k in args.k:
            recall, mrr = _recall_mrr(results[stage][k], k)
            parts.append(f"  {recall:.3f}     {mrr:.3f}")
        avg_ms = sum(latencies[stage]) / len(latencies[stage]) * 1000
        print(f"{stage:<{stage_col}} │ {'  │  '.join(parts)} │ {avg_ms:7.1f}")

    print()

    # Summary
    k = max(args.k)
    if "full" in stages and "dense" in stages:
        r_full, _ = _recall_mrr(results["full"][k], k)
        r_dense, _ = _recall_mrr(results["dense"][k], k)
        print(f"Full pipeline vs dense-only Recall@{k}: {r_full - r_dense:+.3f}")
    if "full+rerank" in stages and "full" in stages:
        r_rerank, _ = _recall_mrr(results["full+rerank"][k], k)
        r_full, _ = _recall_mrr(results["full"][k], k)
        print(f"Reranker impact Recall@{k}: {r_rerank - r_full:+.3f}")
    if discovery_mode and "grep-discovery" in stages and "full" in stages:
        r_grep, _ = _recall_mrr(results["grep-discovery"][k], k)
        r_full, _ = _recall_mrr(results["full"][k], k)
        print(f"RAG vs grep-discovery Recall@{k}: {r_full - r_grep:+.3f} (RAG advantage)")


if __name__ == "__main__":
    main()
