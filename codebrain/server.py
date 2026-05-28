"""
RAG MCP Server — entry point.

Exposes 10 tools via MCP stdio transport:
  index_codebase, index_status, query_code, get_symbol, remember, recall,
  summarize_project, get_architecture, project_status, ask_project

Compatible with: agy, Claude Code, Cursor, any MCP client.
Zero API cost at query time. All computation local.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp.server.stdio
from mcp import types
from mcp.server import Server

from codebrain.config import Config, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class IndexTask:
    task_id: str
    project: str
    started_at: float
    total_files: int
    incremental: bool = True
    indexed_files: int = 0
    indexed_chunks: int = 0
    skipped_files: int = 0
    graph_edges_scip: int = 0
    graph_edges_heuristic: int = 0
    status: str = "running"   # running | done | error
    error: str | None = None
    finished_at: float | None = None


class _ChangeHandler:
    """Watchdog event handler: debounces file changes and triggers incremental reindex."""

    def __init__(self, runtime: Runtime, loop: asyncio.AbstractEventLoop) -> None:
        self._runtime = runtime
        self._loop = loop
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # watchdog calls these from its observer thread
    def dispatch(self, event: Any) -> None:  # noqa: ANN401
        if event.is_directory:
            return
        src = getattr(event, "src_path", None)
        if src:
            self._enqueue(src)
        # FileMovedEvent has dest_path too
        dest = getattr(event, "dest_path", None)
        if dest:
            self._enqueue(dest)

    def _enqueue(self, path: str) -> None:
        from codebrain.indexer.ast_parser import detect_language
        if not detect_language(Path(path)):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(
                self._runtime.config.watch_debounce_s, self._flush
            )
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending)
            self._pending.clear()
        if not paths:
            return
        # Find the project root for changed files
        projects: set[str] = set()
        for p in paths:
            project = self._find_project(p)
            if project:
                projects.add(project)
        for project in projects:
            logger.info("File watcher: triggering incremental reindex for %s", project)
            asyncio.run_coroutine_threadsafe(
                self._reindex(project), self._loop
            )

    def _find_project(self, file_path: str) -> str | None:
        fp = Path(file_path)
        for proj in self._runtime.watched_projects:
            try:
                fp.relative_to(proj)
                return proj
            except ValueError:
                continue
        return None

    async def _reindex(self, project: str) -> None:
        from codebrain.tools.index import run as index_run
        await index_run({"path": project, "incremental": True}, self._runtime)


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
    # Background indexing
    index_tasks: dict[str, IndexTask] = field(default_factory=dict)
    watched_projects: set[str] = field(default_factory=set)
    _watcher: Any = None       # watchdog Observer
    _watcher_handlers: dict[str, Any] = field(default_factory=dict)  # project → handler
    _loop: Any = None
    index_sem: Any = None      # asyncio.Semaphore — prevents concurrent index runs

    def _data_dir(self) -> Path:
        return Path(self.config.data_dir).expanduser()

    def initialize(self) -> None:
        """Load all components. Called once at server startup."""
        data = self._data_dir()
        data.mkdir(parents=True, exist_ok=True)

        from codebrain.indexer.chunker import ChunkIndexer
        from codebrain.indexer.embedder import Embedder
        from codebrain.indexer.graph_builder import GraphDB
        from codebrain.memory.store import MemoryStore
        from codebrain.retrieval.dense import DenseIndex
        from codebrain.retrieval.reranker import CrossEncoderReranker
        from codebrain.retrieval.sparse import BM25Index

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

        bm25_path = data / "bm25"
        if (bm25_path / "retriever.pkl").exists():
            try:
                self.bm25 = BM25Index.load(bm25_path, k1=self.config.bm25_k1, b=self.config.bm25_b)
                logger.info("BM25 index loaded from disk")
            except Exception as exc:
                logger.warning("Could not load BM25 index: %s", exc)

        self.reload_graph()
        logger.info("codebrain runtime initialized (data_dir=%s)", data)

    def reload_graph(self) -> None:
        """Reload NetworkX graph from SQLite after re-indexing."""
        if self.graph_db is not None:
            self.graph = self.graph_db.load_networkx_graph()
            logger.info(
                "Graph loaded: %d nodes, %d edges",
                self.graph.number_of_nodes(), self.graph.number_of_edges(),
            )

    def _start_watcher(self, project: str) -> None:
        """Attach a watchdog observer to project dir. No-op if watchdog not installed."""
        if project in self._watcher_handlers:
            return  # already watching
        try:
            from watchdog.observers import Observer
        except ImportError:
            logger.debug("watchdog not installed — install codebrain[watch] to enable file watching")
            return

        if self._watcher is None:
            self._watcher = Observer()
            self._watcher.daemon = True
            self._watcher.start()

        handler = _ChangeHandler(self, self._loop)
        self._watcher.schedule(handler, project, recursive=True)
        self._watcher_handlers[project] = handler
        logger.info("File watcher active for %s", project)

    async def _auto_index(self) -> None:
        """Kick off background indexing for all auto_index_paths from config."""
        from codebrain.tools.index import run as index_run
        for path_str in self.config.auto_index_paths:
            path = Path(path_str).expanduser().resolve()
            if path.exists() and path.is_dir():
                logger.info("Auto-indexing %s", path)
                await index_run({"path": str(path), "incremental": True}, self)
            else:
                logger.warning("auto_index_paths: %s does not exist, skipping", path_str)


_runtime = Runtime()

server = Server("codebrain")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="index_codebase",
            description=(
                "Parse, embed, and graph-index a codebase. Returns immediately with a task_id — "
                "indexing runs in background with progressive availability (first batch queryable "
                "within seconds). Uses tree-sitter AST chunking, SCIP dependency graph, and local "
                "embeddings. Call index_status to track progress."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to project root"},
                    "incremental": {"type": "boolean", "default": True, "description": "Skip unchanged files (mtime + hash check)"},
                    "languages": {"type": "array", "items": {"type": "string"}, "description": "Restrict to these languages"},
                    "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Directory names to exclude"},
                },
                "required": ["path"],
            },
        ),
        types.Tool(
            name="index_status",
            description=(
                "Query the status of a background index task. Returns progress percentage, "
                "files indexed, chunks indexed, elapsed time, and ETA. "
                "Omit task_id to list all tasks."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID returned by index_codebase. Omit to list all tasks."},
                },
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
    from codebrain.tools import index, query, symbol
    from codebrain.tools import memory as mem_tools

    try:
        if name == "index_codebase":
            result = await index.run(arguments, _runtime)
        elif name == "index_status":
            from codebrain.tools import index_status
            result = await index_status.run(arguments, _runtime)
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
    _runtime._loop = asyncio.get_event_loop()
    _runtime.index_sem = asyncio.Semaphore(1)  # serialise concurrent index runs

    if _runtime.config.auto_index_paths:
        asyncio.create_task(_runtime._auto_index())

    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(_serve())


if __name__ == "__main__":
    main()
