# Benchmarks

Three questions, three benchmarks:

| #   | Question                                        | Method                                        |
| --- | ----------------------------------------------- | --------------------------------------------- |
| 1   | Does the right chunk appear in top-k?           | Recall@k / MRR@k on named-symbol queries      |
| 2   | Does retrieved context improve Claude's answer? | Claude-as-judge, baseline vs. RAG (1–5 scale) |
| 3   | Can Claude find code it doesn't know exists?    | Description-only queries (no symbol names)    |

All commands to reproduce are at the [bottom](#reproducing-these-benchmarks).

## 1. Retrieval quality

Synthetic golden set: named functions/classes sampled from the live index.
Query template: `"implementation of {name}"` / `"how does {name} work"`.
Metric: fraction of queries where the exact chunk appears in top-k (Recall@k), plus MRR@k.

### reporag (246 chunks, AST strategy)

| Stage     |  Recall@5 | Recall@10 |    MRR@10 | ms/query |
| --------- | --------: | --------: | --------: | -------: |
| dense     |     0.875 |     0.900 |     0.791 |        6 |
| bm25      |     0.875 |     0.875 |     0.765 |       <1 |
| rrf       |     0.875 |     0.925 |     0.778 |        6 |
| rrf + ppr |     0.875 |     0.925 |     0.768 |        8 |
| **full**  | **0.900** | **0.900** | **0.814** |      430 |

`full` = RRF fusion → Reverse PPR hub re-ranking → cross-encoder rerank. Reranker cost (~430 ms) trades latency for top-1 precision; disable with `rerank=false` in `query_code`.

### Django (869 files / ~11k chunks, 100 samples)

Validates chunking strategy at scale. PPR requires a dense dependency graph to help — sparse heuristic graphs (857 edges / 11k chunks) show no benefit or slight regression.

| Stage     | Strategy | Recall@5 | Recall@10 | MRR@10 | ms/query |
| --------- | -------- | -------: | --------: | -----: | -------: |
| dense     | ast      |    0.770 |     0.790 |  0.625 |       35 |
| dense     | hybrid   |    0.690 |     0.740 |  0.574 |       37 |
| bm25      | ast      |    0.610 |     0.660 |  0.497 |        1 |
| bm25      | hybrid   |    0.600 |     0.640 |  0.495 |        1 |
| rrf       | ast      |    0.750 | **0.850** |  0.634 |       37 |
| rrf       | hybrid   |    0.670 |     0.780 |  0.566 |       38 |
| rrf + ppr | ast      |    0.750 |     0.840 |  0.617 |       41 |
| rrf + ppr | hybrid   |    0.670 |     0.780 |  0.566 |       43 |

AST consistently outperforms hybrid by 5–7 pp on Recall@10 for **named-symbol queries** (e.g. "implementation of QuerySet"). Hybrid window chunks dilute the dense vector space when exact AST chunks compete against overlapping windows.

> **Caveat:** both tables measure exact named-chunk recall — structurally biased toward AST. The cited paper (arXiv:2605.04763) where sliding window wins tests _code completion at cursor positions_, not NL→code architectural queries. For queries like "how does the ORM query execution flow work?" or "where is middleware processing defined?", hybrid may outperform AST by providing import context and cross-function windows that AST drops — this is untested.

**Strategy guidance:**

| Strategy          | Best for                                      | Trade-off                                             |
| ----------------- | --------------------------------------------- | ----------------------------------------------------- |
| `"ast"` (default) | Symbol lookup, exact function/class retrieval | Smallest index, lowest latency                        |
| `"hybrid"`        | Architectural/contextual queries (untested)   | −5–7 pp recall on symbol lookup, ~2× indexing latency |
| `"sliding"`       | Pure window coverage                          | No symbol-level precision                             |

## 2. LLM response quality

Retrieval recall measures _whether_ the right chunk is returned. This benchmark measures _whether Claude gives a better answer_ when that chunk is injected as context.

**Method:** for each sampled function/class, generate a question (`"How does {name} work?"`), call Claude twice — once with no context (baseline), once with retrieved chunks injected (RAG) — then use Claude as judge to score both responses on correctness, completeness, and hallucination avoidance (1–5 each). Uses `claude -p` (Claude Code CLI, no API key required).

| Codebase           | Files | Chunks | Scored | Baseline |  RAG | Composite Δ |
| ------------------ | ----: | -----: | -----: | -------: | ---: | ----------: |
| reporag            |    39 |    266 |  17/30 |     3.98 | 4.16 |   **+4.4%** |
| private Go project |    74 |    488 |  30/30 |     2.24 | 3.84 |  **+71.3%** |
| Django             | 2,955 |   ~45k |  24/30 |     2.35 | 4.25 |  **+81.1%** |

![LLM response quality: baseline vs RAG](https://github.com/user-attachments/assets/f338658b-c62c-4314-aad9-361736c98cf2)

_"Discovery" = description-only queries (anti-duplication use case), see [§3](#3-discovery-mode-anti-duplication)._

<details>
<summary>Per-metric breakdown</summary>

**reporag** — well-known patterns (MCP, Python, RAG); baseline already strong. RAG helps most on completeness — actual source makes answers more thorough.

| Metric        | Baseline |      RAG |     Δ (%) |
| ------------- | -------: | -------: | --------: |
| correctness   |     4.18 |     4.29 |     +2.8% |
| completeness  |     3.94 |     4.24 |     +7.5% |
| hallucination |     3.82 |     3.94 |     +3.1% |
| **composite** | **3.98** | **4.16** | **+4.4%** |

**Private Go project** — baseline near 1: Claude cannot answer without context. Hallucination score is high at baseline because Claude hedges rather than inventing. RAG raises correctness/completeness ~2.5×.

| Metric        | Baseline |      RAG |      Δ (%) |
| ------------- | -------: | -------: | ---------: |
| correctness   |     1.27 |     3.57 |    +181.6% |
| completeness  |     1.13 |     3.53 |    +211.8% |
| hallucination |     4.33 |     4.43 |      +2.3% |
| **composite** | **2.24** | **3.84** | **+71.3%** |

**Django** — largest correctness/completeness gains. Hallucination slightly regresses with RAG: injecting 10 chunks causes Claude to synthesize across chunks in ways that diverge from the single ground-truth function. Baseline hedges cleanly; RAG over-extends retrieved context.

| Metric        | Baseline |      RAG |      Δ (%) |
| ------------- | -------: | -------: | ---------: |
| correctness   |     1.33 |     4.25 |    +218.8% |
| completeness  |     1.17 |     4.21 |    +260.7% |
| hallucination |     4.54 |     4.29 |      −5.5% |
| **composite** | **2.35** | **4.25** | **+81.1%** |

</details>

**Takeaway:** the less Claude knows about a codebase from training, the larger the RAG gain. On private or large codebases, reporag delivers **+71–81% composite quality improvement** — turning unusable baselines (1.1–1.3) into competent answers (3.8–4.2).

## 3. Discovery mode (anti-duplication)

The benchmarks above test **named-symbol lookup** — Claude already knows what it's looking for ("how does `selfCorrectLoop` work"). The real failure mode in multi-file features is the opposite: Claude doesn't know a function/pattern already exists, so it reimplements it.

**Method:** strip the function name from `semantic_text`, leaving only the description (parameters/docstring). Use that as the query — simulates "I need to implement X" without knowing any names. Grep can't help here (no name to search for); semantic retrieval is the only option.

**Retrieval recall** (private Go project, 488 chunks, 50 samples):

| Stage          |  Recall@5 | Recall@10 | MRR@10 | ms/query |
| -------------- | --------: | --------: | -----: | -------: |
| grep-discovery |     0.660 |     0.760 |  0.390 |   1598.7 |
| dense          |     1.000 |     1.000 |  0.911 |     88.3 |
| bm25           |     1.000 |     1.000 |  0.910 |     13.7 |
| **full**       | **1.000** | **1.000** |  0.907 |    120.8 |

![Discovery mode Recall@10 by stage](https://github.com/user-attachments/assets/70ea1c4c-1e27-40d8-8cff-3370a1de9e07)

RAG vs grep-discovery Recall@10: **+0.240**. Grep word-overlap on raw content gets close (0.760) but is 16× slower and misses 1 in 4. RAG hits every case.

**LLM response quality** (same project, 5 samples → 15 `claude -p` calls):

| Metric        | Baseline |      RAG |       Δ (%) |
| ------------- | -------: | -------: | ----------: |
| correctness   |     1.00 |     4.20 |     +320.0% |
| completeness  |     1.20 |     4.00 |     +233.3% |
| hallucination |     3.20 |     5.00 |      +56.2% |
| **composite** | **1.80** | **4.40** | **+144.4%** |

4/5 samples: baseline either refused, hallucinated the wrong language, or invented a different implementation. RAG returned the **exact existing function verbatim with file:line location**, ready to import instead of reimplement. The 1 miss was a query with no real signal (`"Parameters: result types.ExecutionResult."` — no docstring, no useful description).

> **Caveat:** queries are derived from `semantic_text` (leave-one-out), so this is optimistic vs. true unknown-unknowns — it validates the embedding space is dense enough for description→code retrieval, not a blind real-world test. n=5 for the LLM eval is small.

**Takeaway:** when Claude is told "before writing new code, call `find_existing(task=...)`," semantic retrieval reliably surfaces existing implementations from a one-sentence description — the core anti-duplication use case.

## Reproducing these benchmarks

```bash
# Retrieval quality (named-symbol lookup)
devenv shell -- python scripts/benchmark.py --samples 100 --k 5 10
devenv shell -- python scripts/benchmark.py --project /path/to/project --stages dense rrf full
devenv shell -- python scripts/benchmark.py --filter-chunk-types function --quiet

# LLM response quality (named-symbol lookup) — index project first via index_codebase
devenv shell -- python scripts/llm_eval.py --project /path/to/project --samples 30 --output results.json

# Discovery mode (anti-duplication)
devenv shell -- python scripts/benchmark.py --project /path/to/project --mode discovery --samples 50
devenv shell -- python scripts/llm_eval.py --project /path/to/project --mode discovery --samples 5 --output results.json
```
