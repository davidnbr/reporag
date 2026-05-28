"""
RAG MCP Server — entry point.

Exposes 9 tools via MCP stdio transport:
  index_codebase, query_code, get_symbol, remember, recall,
  summarize_project, get_architecture, project_status, ask_project

Compatible with: agy, Claude Code, Cursor, any MCP client.
Zero API cost at query time. All computation local.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp.server.stdio
from mcp import types
from mcp.server import Server

from codebrain.config import get_config, Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Lazily-initialized shared state for the MCP server."""

    config: Config = field(default_factory=get_config)
    embedder: Any = None
    dense: Any = None
    bm25: Any = None
    graph_db: Any = None
    graph: Any = None      # networkx.DiGraph
    reranker: Any = None
    memory: Any = None
    chunker: Any = None

    def _data_dir(self) -> Path:
        return Path(self.config.data_dir).expanduser()

    def initialize(self) -> None:
        """Load all components. Called once at server startup."""
        data = self._data_dir()
        data.mkdir(parents=True, exist_ok=True)

        from codebrain.indexer.embedder import Embedder
        from codebrain.retrieval.dense import DenseIndex
        from codebrain.retrieval.sparse import BM25Index
        from codebrain.retrieval.reranker import CrossEncoderReranker
        from codebrain.indexer.graph_builder import GraphDB
        from codebrain.memory.store import MemoryStore
        from codebrain.indexer.chunker import ChunkIndexer

        self.embedder = Embedder(
            model=self.config.embed_model,
            backend=self.config.embed_backend,  # type: ignore[arg-type]
            ollama_url=self.config.ollama_url,
        )
        self.dense = DenseIndex(data, dim=self.embedder.dim)
        self.bm25 = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.reranker = CrossEncoderReranker(self.config.reranker_model)
        self.graph_db = GraphDB(data / "dependency_graph.db")
        self.memory = MemoryStore(data / "memory.db")
        self.chunker = ChunkIndexer(
            data, self.embedder, self.dense, self.bm25,
            chunk_strategy=self.config.chunk_strategy,
            chunk_window_lines=self.config.chunk_window_lines,
            chunk_overlap_lines=self.config.chunk_overlap_lines,
        )

        # Try loading persisted BM25 index
        bm25_path = data / "bm25"
        if (bm25_path / "retriever.pkl").exists():
            try:
                self.bm25 = BM25Index.load(bm25_path, k1=self.config.bm25_k1, b=self.config.bm25_b)
                logger.info("BM25 index loaded from disk")
            except Exception as exc:
                logger.warning("Could not load BM25 index: %s", exc)

        # Load graph
        self.reload_graph()
        logger.info("RAG MCP runtime initialized (data_dir=%s)", data)

    def reload_graph(self) -> None:
        """Reload NetworkX graph from SQLite after re-indexing."""
        if self.graph_db is not None:
            self.graph = self.graph_db.load_networkx_graph()
            logger.info("Graph loaded: %d nodes, %d edges", self.graph.number_of_nodes(), self.graph.number_of_edges())


_runtime = Runtime()

server = Server("codebrain")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="index_codebase",
            description=(
                "Parse, embed, and graph-index a codebase. Uses tree-sitter for AST chunking, "
                "SCIP for compiler-grade dependency graph (with heuristic fallback), "
                "and local embeddings. Run once, then incrementally on changes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to project root"},
                    "incremental": {"type": "boolean", "default": True, "description": "Skip unchanged files"},
                    "languages": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these languages"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Directory names to exclude"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="query_code",
            description=(
                "Retrieve code context via hybrid RAG: dense (cosine) + sparse (BM25) retrieval, "
                "RRF fusion (k=60), Reverse Personalized PageRank hub ranking, and cross-encoder reranking. "
                "Returns top-k chunks with subgraph neighbors in a single call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language or code query"},
                    "k": {"type": "integer", "default": 10, "description": "Number of results"},
                    "rerank": {"type": "boolean", "description": "Enable cross-encoder reranking (default: rerank_by_default from config, off by default)"},
                    "languages": {"type": "array", "items": {"type": "string"}, "description": "Filter by language (e.g. ['go', 'python'])"},
                    "project": {"type": "string", "description": "Restrict results to this absolute project root path"},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_symbol",
            description="Exact symbol lookup by name from the dependency graph. Returns file, line, and type.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Symbol name (class, function, method)"},
                    "language": {"type": "string", "description": "Optional language filter"},
                    "fuzzy": {"type": "boolean", "default": False, "description": "Use semantic search instead of exact match"},
                },
                "required": ["name"],
            },
        ),
        types.Tool(
            name="remember",
            description="Store a decision, discovery, or pattern in the persistent project memory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Knowledge to store"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Searchable tags"},
                    "category": {
                        "type": "string",
                        "enum": ["decision", "discovery", "pattern", "architecture", "note", "general"],
                        "default": "general",
                    },
                },
                "required": ["content"],
            },
        ),
        types.Tool(
            name="recall",
            description="Search the persistent project memory by keyword and optional tags.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter by tags"},
                    "category": {"type": "string", "description": "Filter by category"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="summarize_project",
            description=(
                "Return a structured overview of an indexed project: description from README, "
                "tech stack, entry points, top components by graph centrality, and public API symbols."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Absolute path to project root"},
                },
                "required": ["project"],
            },
        ),
        types.Tool(
            name="get_architecture",
            description=(
                "Return dependency topology for an indexed project: nodes classified as hub/utility/bridge/leaf, "
                "edges within the project, layer grouping by top-level path segment, and role summary counts."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Absolute path to project root"},
                },
                "required": ["project"],
            },
        ),
        types.Tool(
            name="project_status",
            description=(
                "Return project health: TODOs/FIXMEs, stub functions, test-to-source ratio, "
                "and git activity (files changed in 30 days, last commit). "
                "Health is 'good', 'needs-attention', or 'stale'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Absolute path to project root"},
                },
                "required": ["project"],
            },
        ),
        types.Tool(
            name="ask_project",
            description=(
                "Natural language router: routes questions to summarize_project, get_architecture, "
                "or project_status based on keywords, falling back to query_code for semantic search. "
                "Use this when you don't know which specific tool to call."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language question about the project"},
                    "project": {"type": "string", "description": "Absolute path to project root"},
                },
                "required": ["query", "project"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    from codebrain.tools import index, query, symbol, memory as mem_tools

    try:
        if name == "index_codebase":
            result = await index.run(arguments, _runtime)
        elif name == "query_code":
            result = await query.run(arguments, _runtime)
        elif name == "get_symbol":
            result = await symbol.run(arguments, _runtime)
        elif name == "remember":
            result = await mem_tools.run_remember(arguments, _runtime)
        elif name == "recall":
            result = await mem_tools.run_recall(arguments, _runtime)
        elif name == "summarize_project":
            from codebrain.tools import summarize
            result = await summarize.run(arguments, _runtime)
        elif name == "get_architecture":
            from codebrain.tools import architecture
            result = await architecture.run(arguments, _runtime)
        elif name == "project_status":
            from codebrain.tools import status
            result = await status.run(arguments, _runtime)
        elif name == "ask_project":
            from codebrain.tools import ask
            result = await ask.run(arguments, _runtime)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _serve() -> None:
    _runtime.initialize()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
