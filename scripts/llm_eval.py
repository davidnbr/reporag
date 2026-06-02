#!/usr/bin/env python3
"""
LLM response quality benchmark — measures how much reporag context improves Claude answers.

Samples named chunks from the live index, generates a question per chunk, then asks
Claude WITH and WITHOUT retrieved codebase context. Uses Claude as judge to score
correctness, completeness, and hallucination rate, then reports a comparison table.

This complements scripts/benchmark.py (retrieval quality) by measuring the downstream
impact on actual LLM response quality — the metric that matters to end users.

Usage:
    python scripts/llm_eval.py --project /path/to/project --samples 30
    python scripts/llm_eval.py --project /path/to/project /other --samples 50
    python scripts/llm_eval.py --project /path/to/project --samples 20 --k 8 --output results.json
    ANTHROPIC_API_KEY=sk-... python scripts/llm_eval.py --project /path/to/project

Requirements:
    pip install anthropic   (or: uv sync --extra dev)
    ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

_QUESTION_TEMPLATES = [
    "How does `{name}` work? Walk me through its logic.",
    "What does `{name}` do and when is it called?",
    "Explain the implementation of `{name}` in this codebase.",
    "Where is `{name}` defined and what is its purpose?",
]

_SYSTEM_ANSWER = (
    "You are an expert software engineer. Answer questions about codebases concisely "
    "and accurately. If you don't know or aren't sure, say so — do not invent details."
)

_SYSTEM_JUDGE = """\
You are an expert code reviewer evaluating LLM responses to coding questions.
You have the ground truth: the actual source code the question was generated from.

Score each response (1–5):
  correctness  — accurately describes the real implementation (5=fully correct, 1=wrong)
  completeness — covers the key logic and behavior (5=comprehensive, 1=superficial)
  hallucination — avoids invented code/APIs/behavior (5=no hallucination, 1=heavily invented)

Output ONLY valid JSON, no extra text:
{"baseline": {"correctness": N, "completeness": N, "hallucination": N}, \
"rag": {"correctness": N, "completeness": N, "hallucination": N}, \
"reasoning": "one sentence"}\
"""

_CHUNK_TYPES = {"function", "class", "method"}


@dataclass
class EvalResult:
    name: str
    file: str
    language: str
    chunk_type: str
    query: str
    baseline: str = ""
    rag: str = ""
    context_files: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)
    reasoning: str = ""
    error: str | None = None


# ── Runtime ───────────────────────────────────────────────────────────────────


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
    rt: Any, project: str | None, max_samples: int, seed: int
) -> list[dict]:
    rt.dense._open_or_create_table()
    rows = rt.dense._table.search().limit(max_samples * 20).to_list()
    named = [
        r
        for r in rows
        if r.get("name")
        and r.get("chunk_type") in _CHUNK_TYPES
        and (not project or r.get("file_path", "").startswith(project))
    ]
    if len(named) > max_samples:
        rng = random.Random(seed)
        rng.shuffle(named)
        named = named[:max_samples]
    return named


# ── Retrieval ─────────────────────────────────────────────────────────────────


def _retrieve_context(rt: Any, query: str, q_vec: Any, k: int) -> list[dict]:
    """Full pipeline: RRF + PPR, same as _stage_full in benchmark.py."""
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
    return candidates[:k]


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        path = c.get("file_path", "?")
        lang = c.get("language", "")
        start = c.get("start_line", "?")
        end = c.get("end_line", "?")
        code = (c.get("raw_content") or c.get("snippet", ""))[:800]
        parts.append(f"### {path} (lines {start}–{end})\n```{lang}\n{code}\n```")
    return "\n\n".join(parts)


# ── Claude calls via claude -p ────────────────────────────────────────────────


def _claude(system: str, user: str) -> str:
    """Call Claude Code CLI in print mode. Uses subscription credits, no API key needed."""
    prompt = f"{system}\n\n---\n\n{user}"
    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "text"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[:300] or "claude -p failed")
    return result.stdout.strip()


# ── Eval ──────────────────────────────────────────────────────────────────────


def _eval_one(
    chunk: dict,
    rt: Any,
    k: int,
    executor: concurrent.futures.ThreadPoolExecutor,
    template_idx: int,
) -> EvalResult:
    name = chunk["name"]
    query = _QUESTION_TEMPLATES[template_idx % len(_QUESTION_TEMPLATES)].format(name=name)

    # raw_content is actual source code; semantic_text is the embedded description
    ground_truth = (chunk.get("raw_content") or chunk.get("snippet", ""))[:2000]

    result = EvalResult(
        name=name,
        file=chunk.get("file_path", ""),
        language=chunk.get("language", ""),
        chunk_type=chunk.get("chunk_type", ""),
        query=query,
    )

    try:
        q_vec = rt.embedder.encode_query(query)
        context_chunks = _retrieve_context(rt, query, q_vec, k)
        context_text = _format_context(context_chunks)
        result.context_files = [c.get("file_path", "") for c in context_chunks]

        rag_user = f"Using this codebase context:\n\n{context_text}\n\n---\n\n{query}"

        # Baseline and RAG calls in parallel (two claude -p subprocesses)
        f_baseline = executor.submit(_claude, _SYSTEM_ANSWER, query)
        f_rag = executor.submit(_claude, _SYSTEM_ANSWER, rag_user)
        result.baseline = f_baseline.result(timeout=120)
        result.rag = f_rag.result(timeout=120)

        # Judge (sequential — needs both responses first)
        judge_user = (
            f"Ground truth:\nFile: {result.file}\n"
            f"```{result.language}\n{ground_truth}\n```\n\n"
            f"Question: {query}\n\n"
            f"Response A (baseline — no codebase context):\n{result.baseline}\n\n"
            f"Response B (RAG — retrieved context injected):\n{result.rag}"
        )
        raw = _claude(_SYSTEM_JUDGE, judge_user)
        # Extract JSON from response (may have surrounding text)
        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end])
        result.scores = {"baseline": parsed["baseline"], "rag": parsed["rag"]}
        result.reasoning = parsed.get("reasoning", "")

    except json.JSONDecodeError as exc:
        result.error = f"judge parse error: {exc}"
    except Exception as exc:
        result.error = str(exc)

    return result


# ── Metrics ───────────────────────────────────────────────────────────────────


def _avg(results: list[EvalResult], condition: str) -> dict[str, float]:
    scored = [r for r in results if condition in r.scores and not r.error]
    if not scored:
        return {}
    keys = ["correctness", "completeness", "hallucination"]
    return {k: sum(r.scores[condition][k] for r in scored) / len(scored) for k in keys}


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM response quality benchmark")
    parser.add_argument("--project", nargs="+", type=str, help="Project root path(s)")
    parser.add_argument("--samples", type=int, default=30, help="Questions per project")
    parser.add_argument("--k", type=int, default=10, help="Context chunks for RAG condition")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, help="Save raw results to JSON file")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not shutil.which("claude"):
        print("Error: 'claude' CLI not found. Install Claude Code first.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print("Loading runtime...", flush=True)
    t0 = time.monotonic()
    rt = _build_runtime(args.data_dir)
    if not args.quiet:
        print(f"Runtime loaded in {time.monotonic() - t0:.1f}s\n")

    all_results: list[EvalResult] = []
    projects = args.project or [None]

    for project in projects:
        chunks = _get_named_chunks(rt, project, args.samples, args.seed)
        if not chunks:
            label = project or "(all)"
            print(f"No named chunks for {label}. Run index_codebase first.")
            continue

        if not args.quiet:
            label = project or "(all indexed projects)"
            print(f"Project : {label}")
            print(f"Samples : {len(chunks)}  k={args.k}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            for i, chunk in enumerate(chunks):
                if not args.quiet:
                    print(f"  [{i + 1:3d}/{len(chunks)}] {chunk['name'][:45]:<45}", end="\r", flush=True)
                result = _eval_one(chunk, rt, args.k, executor, i)
                all_results.append(result)
                if result.error and not args.quiet:
                    print(f"\n  ! {chunk['name']}: {result.error}")

        if not args.quiet:
            print(f"  [{len(chunks):3d}/{len(chunks)}] done{' ' * 50}")

    # ── Results table ────────────────────────────────────────────────────────
    b_avg = _avg(all_results, "baseline")
    r_avg = _avg(all_results, "rag")
    scored = sum(1 for r in all_results if not r.error)

    print(f"\n{'─' * 60}")
    print(f"{'Metric':<18} {'Baseline':>10} {'RAG':>10} {'Δ (abs)':>8} {'Δ (%)':>8}")
    print(f"{'─' * 60}")
    for metric in ["correctness", "completeness", "hallucination"]:
        b = b_avg.get(metric, 0.0)
        r = r_avg.get(metric, 0.0)
        delta = r - b
        pct = (delta / b * 100) if b else 0.0
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<18} {b:>10.2f} {r:>10.2f} {sign}{delta:>+7.2f} {sign}{pct:>7.1f}%")
    print(f"{'─' * 60}")

    composite_b = sum(b_avg.values()) / len(b_avg) if b_avg else 0.0
    composite_r = sum(r_avg.values()) / len(r_avg) if r_avg else 0.0
    delta_c = composite_r - composite_b
    pct_c = (delta_c / composite_b * 100) if composite_b else 0.0
    sign = "+" if delta_c >= 0 else ""
    print(f"{'composite':<18} {composite_b:>10.2f} {composite_r:>10.2f} {sign}{delta_c:>+7.2f} {sign}{pct_c:>7.1f}%")
    print(f"\nScored: {scored}/{len(all_results)}  via: claude -p")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps([asdict(r) for r in all_results], indent=2))
        print(f"Raw results → {out}")


if __name__ == "__main__":
    main()
