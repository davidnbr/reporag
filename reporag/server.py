"""
RAG MCP Server — entry point.

Exposes 11 tools over MCP:
  index_codebase, index_status, query_code, find_existing, get_symbol, remember,
  recall, summarize_project, get_architecture, project_status, ask_project

Two transports:
  * stdio (default) — one server process spawned per client.
  * streamable HTTP (`reporag serve`) — one shared daemon many clients connect to,
    so a single embedding model and index semaphore are reused across sessions.

Compatible with: agy, Claude Code, Cursor, any MCP client.
Zero API cost at query time. All computation local.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mcp.server.stdio
from mcp import types
from mcp.server import Server

from reporag.config import Config, get_config

# MCP speaks JSON-RPC over stdout; any stray byte on fd 1 corrupts the stream.
# Silence ML library progress bars (HF downloads, tqdm encode batches) which
# would otherwise spam the channel. _serve() additionally isolates fd 1 so even
# native/trust_remote_code prints cannot reach the protocol stream.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TQDM_DISABLE", "1")

# MCP clients (e.g. Cursor) surface every stderr line as an "[error]" entry in
# their log panel, so default to WARNING to keep it quiet. Override with
# REPORAG_LOG_LEVEL=INFO/DEBUG when debugging. Noisy third-party loggers that
# log per-request or per-HTTP-call are clamped regardless. We route warnings
# through logging (captureWarnings) and mute them at the logger level rather
# than swallowing them globally, so they stay recoverable via REPORAG_LOG_LEVEL.
_log_level = getattr(
    logging, os.environ.get("REPORAG_LOG_LEVEL", "WARNING").upper(), logging.WARNING
)
logging.basicConfig(level=_log_level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logging.captureWarnings(True)  # route warnings.warn() through logging so we can mute it
logger = logging.getLogger(__name__)
logger.setLevel(_log_level)

# Quiet third-party chatter only in the default quiet mode. When the operator
# explicitly opts into INFO/DEBUG, let everything through so nothing is hidden
# while debugging — clamping here would otherwise override the requested level.
if _log_level > logging.DEBUG:
    for _noisy in ("mcp", "httpx", "httpcore", "urllib3"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)
    for _silent in (
        "py.warnings",
        "sentence_transformers",
        "transformers",
        "transformers_modules",
        "huggingface_hub",
        "huggingface_hub.utils._http",
    ):
        logging.getLogger(_silent).setLevel(logging.ERROR)


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
    status: str = "running"  # running | done | error
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
        from reporag.indexer.ast_parser import detect_language

        if not detect_language(Path(path)):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self._runtime.config.watch_debounce_s, self._flush)
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
            asyncio.run_coroutine_threadsafe(self._reindex(project), self._loop)

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
        from reporag.tools.index import run as index_run

        await index_run({"path": project, "incremental": True}, self._runtime)


@dataclass
class Runtime:
    """Lazily-initialized shared state for the MCP server."""

    config: Config = field(default_factory=get_config)
    embedder: Any = None
    dense: Any = None
    bm25: Any = None
    graph_db: Any = None
    graph: Any = None  # networkx.DiGraph
    reranker: Any = None
    memory: Any = None
    chunker: Any = None
    # Background indexing
    index_tasks: dict[str, IndexTask] = field(default_factory=dict)
    watched_projects: set[str] = field(default_factory=set)
    _watcher: Any = None  # watchdog Observer
    _watcher_handlers: dict[str, Any] = field(default_factory=dict)  # project -> handler
    _loop: Any = None
    index_sem: Any = None  # asyncio.Semaphore — prevents concurrent index runs

    def _data_dir(self) -> Path:
        return Path(self.config.data_dir).expanduser()

    def initialize(self) -> None:
        """Load all components. Called once at server startup."""
        data = self._data_dir()
        data.mkdir(parents=True, exist_ok=True)

        from reporag.indexer.chunker import ChunkIndexer
        from reporag.indexer.embedder import Embedder
        from reporag.indexer.graph_builder import GraphDB
        from reporag.memory.store import MemoryStore
        from reporag.retrieval.dense import DenseIndex
        from reporag.retrieval.reranker import CrossEncoderReranker
        from reporag.retrieval.sparse import BM25Index

        self.embedder = Embedder(
            model=self.config.embed_model,
            backend=self.config.embed_backend,  # type: ignore[arg-type]
            ollama_url=self.config.ollama_url,
        )
        logger.info(
            "Embed model configured: %s (dim=%d)", self.config.embed_model, self.embedder.dim
        )
        self.dense = DenseIndex(data, dim=self.embedder.dim)
        self.bm25 = BM25Index(k1=self.config.bm25_k1, b=self.config.bm25_b)
        self.reranker = CrossEncoderReranker(self.config.reranker_model)
        self.graph_db = GraphDB(data / "dependency_graph.db")
        self.memory = MemoryStore(data / "memory.db")
        self.chunker = ChunkIndexer(
            data,
            self.embedder,
            self.dense,
            self.bm25,
            chunk_strategy=self.config.chunk_strategy,
            chunk_window_lines=self.config.chunk_window_lines,
            chunk_overlap_lines=self.config.chunk_overlap_lines,
            graph_db=self.graph_db,
        )

        bm25_path = data / "bm25"
        if (bm25_path / "retriever.pkl").exists():
            try:
                self.bm25 = BM25Index.load(bm25_path, k1=self.config.bm25_k1, b=self.config.bm25_b)
                logger.info("BM25 index loaded from disk")
            except Exception as exc:
                logger.warning("Could not load BM25 index: %s", exc)

        self.reload_graph()
        logger.info("reporag runtime initialized (data_dir=%s)", data)

    def reload_graph(self) -> None:
        """Reload NetworkX graph from SQLite after re-indexing."""
        if self.graph_db is not None:
            self.graph = self.graph_db.load_networkx_graph()
            logger.info(
                "Graph loaded: %d nodes, %d edges",
                self.graph.number_of_nodes(),
                self.graph.number_of_edges(),
            )

    def _start_watcher(self, project: str) -> None:
        """Attach a watchdog observer to project dir. No-op if watchdog not installed."""
        if project in self._watcher_handlers:
            return  # already watching
        try:
            from watchdog.observers import Observer
        except ImportError:
            logger.debug("watchdog not installed — install reporag[watch] to enable file watching")
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
        from reporag.tools.index import run as index_run

        for path_str in self.config.auto_index_paths:
            path = Path(path_str).expanduser().resolve()
            if path.exists() and path.is_dir():
                logger.info("Auto-indexing %s", path)
                await index_run({"path": str(path), "incremental": True}, self)
            else:
                logger.warning("auto_index_paths: %s does not exist, skipping", path_str)


_runtime = Runtime()

server = Server("reporag")


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
                    "incremental": {
                        "type": "boolean",
                        "default": True,
                        "description": "Skip unchanged files (mtime + hash check)",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Restrict to these languages",
                    },
                    "exclude_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Directory names to exclude",
                    },
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
                    "task_id": {
                        "type": "string",
                        "description": "Task ID returned by index_codebase. Omit to list all tasks.",
                    },
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
                    "rerank": {
                        "type": "boolean",
                        "description": "Enable cross-encoder reranking (default: rerank_by_default from config, off by default)",
                    },
                    "languages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by language (e.g. ['go', 'python'])",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict results to this absolute project root path "
                            "(default: the current project)"
                        ),
                    },
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
                    "name": {
                        "type": "string",
                        "description": "Symbol name (class, function, method)",
                    },
                    "language": {"type": "string", "description": "Optional language filter"},
                    "fuzzy": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use semantic search instead of exact match",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict results to this absolute project root path "
                            "(default: the current project)"
                        ),
                    },
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
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Searchable tags",
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "decision",
                            "discovery",
                            "pattern",
                            "architecture",
                            "note",
                            "general",
                        ],
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
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by tags",
                    },
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
                    "query": {
                        "type": "string",
                        "description": "Natural language question about the project",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Absolute path to project root (default: the current project)"
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="find_existing",
            description=(
                "Before implementing new code, surface existing functions, classes, and patterns "
                "that already handle the described functionality. Call this at the start of any "
                "implementation task to prevent duplicating logic that already exists. "
                "Returns ranked existing code with reuse hints."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Description of what you are about to implement",
                    },
                    "project": {
                        "type": "string",
                        "description": (
                            "Restrict to this absolute project root path "
                            "(default: the current project)"
                        ),
                    },
                    "k": {
                        "type": "integer",
                        "default": 10,
                        "description": "Number of results to return",
                    },
                },
                "required": ["task"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    from reporag.tools import index, query, symbol
    from reporag.tools import memory as mem_tools

    try:
        if name == "index_codebase":
            result = await index.run(arguments, _runtime)
        elif name == "index_status":
            from reporag.tools import index_status

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
            from reporag.tools import summarize

            result = await summarize.run(arguments, _runtime)
        elif name == "get_architecture":
            from reporag.tools import architecture

            result = await architecture.run(arguments, _runtime)
        elif name == "project_status":
            from reporag.tools import status

            result = await status.run(arguments, _runtime)
        elif name == "ask_project":
            from reporag.tools import ask

            result = await ask.run(arguments, _runtime)
        elif name == "find_existing":
            from reporag.tools import find_existing

            result = await find_existing.run(arguments, _runtime)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _startup() -> None:
    """Initialize shared runtime state common to every transport.

    The ``index_sem`` created here is process-global: a single HTTP daemon shared
    by multiple clients therefore serialises *all* indexing across those clients,
    so two sessions can never spin up concurrent embed runs that saturate CPU/GPU.
    """
    _runtime.initialize()
    _runtime._loop = asyncio.get_running_loop()
    _runtime.index_sem = asyncio.Semaphore(1)  # serialise concurrent index runs

    from reporag import setup

    setup._auto_setup_hooks()

    if _runtime.config.auto_index_paths:
        asyncio.create_task(_runtime._auto_index())


def _shutdown() -> None:
    """Tear down background resources. Safe to call once per process."""
    if _runtime._watcher is not None:
        _runtime._watcher.stop()
        _runtime._watcher.join(timeout=2)


async def _serve() -> None:
    import sys
    from io import TextIOWrapper

    import anyio

    # Hand the transport a private copy of the real stdout, then redirect fd 1
    # to stderr. After this, every other write path (Python prints, ML library
    # output, native extensions) lands on stderr and can never corrupt the
    # JSON-RPC frames the MCP client parses on stdout.
    real_stdout_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    transport_stdout = anyio.wrap_file(
        TextIOWrapper(os.fdopen(real_stdout_fd, "wb"), encoding="utf-8")
    )

    await _startup()

    try:
        async with mcp.server.stdio.stdio_server(stdout=transport_stdout) as (
            read_stream,
            write_stream,
        ):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        _shutdown()


async def _serve_http(host: str, port: int) -> None:
    """Run a long-lived streamable-HTTP daemon shared by every MCP client.

    Unlike stdio (one process spawned per client), a single daemon holds one
    embedding model in memory and one ``index_sem``, so concurrent sessions reuse
    the same instance instead of each loading their own model and indexing at once.
    The fd-1 isolation done in ``_serve`` is stdio-only and deliberately omitted:
    HTTP carries JSON-RPC over the socket, so stdout is free for normal logging.
    """
    import contextlib
    from collections.abc import AsyncGenerator

    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    await _startup()

    manager = StreamableHTTPSessionManager(app=server, stateless=False)

    async def handle_mcp(scope: Any, receive: Any, send: Any) -> None:  # noqa: ANN401
        await manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
        async with manager.run():
            try:
                yield
            finally:
                _shutdown()

    app = Starlette(routes=[Mount("/mcp", app=handle_mcp)], lifespan=lifespan)
    logger.warning("reporag HTTP daemon listening on http://%s:%d/mcp", host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    await uvicorn.Server(config).serve()


def _http_addr() -> tuple[str, int]:
    """Resolve the shared-daemon bind address from the environment.

    Loopback by default so the daemon is reachable only from the machine running
    the IDE terminals — never exposed on the network.
    """
    host = os.environ.get("REPORAG_HTTP_HOST", "127.0.0.1")
    port = int(os.environ.get("REPORAG_HTTP_PORT", "7800"))
    return host, port


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Return True if a TCP listener is already accepting on ``host:port``."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _ensure_daemon(host: str, port: int, wait_s: float = 30.0) -> None:
    """Guarantee exactly one shared daemon is running on this machine.

    Every client (Claude, Cursor, ...) spawns a thin bridge; the first one to run
    starts the daemon, the rest reuse it. A ``flock`` on ``daemon.lock`` serialises
    the check-and-spawn so simultaneous launches can't race into two daemons, and
    the fixed port bind is the final backstop — a second ``serve`` would fail to
    bind and exit. The daemon is detached (``start_new_session``) so it outlives
    the bridge that spawned it and keeps serving the other clients.
    """
    import fcntl
    import subprocess
    import time

    if _port_open(host, port):
        return

    data_dir = _runtime._data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / "daemon.lock"
    log_path = data_dir / "daemon.log"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        # Double-checked: another bridge may have started the daemon while we
        # waited for the lock.
        if _port_open(host, port):
            return

        logger.warning("reporag: no shared daemon on %s:%d — starting one", host, port)
        with open(log_path, "ab") as log:
            subprocess.Popen(
                ["reporag", "serve", "--host", host, "--port", str(port)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )

        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if _port_open(host, port):
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"reporag daemon did not come up on {host}:{port} within {wait_s}s; "
            f"see {log_path}"
        )


async def _serve_bridge() -> None:
    """Thin stdio↔HTTP proxy that every IDE spawns in place of a full server.

    The IDE still launches ``reporag`` per session exactly as before, but this
    process is cheap: it loads no embedding model. It ensures the single shared
    daemon is up, then pumps JSON-RPC ``SessionMessage`` frames verbatim between
    the client's stdio and the daemon's HTTP transport. Because it forwards
    without interpreting, the full tool set, notifications, and streaming all pass
    through untouched — and all sessions share one model and one index semaphore.
    """
    import anyio
    from mcp.client.streamable_http import streamable_http_client

    host, port = _http_addr()
    _ensure_daemon(host, port)
    url = f"http://{host}:{port}/mcp"

    async def _pump(source: Any, sink: Any) -> None:  # noqa: ANN401
        async for message in source:
            if isinstance(message, Exception):
                logger.warning("reporag bridge: dropping transport error: %r", message)
                continue
            await sink.send(message)

    async with mcp.server.stdio.stdio_server() as (client_read, client_write):
        async with streamable_http_client(url) as (daemon_read, daemon_write, _):
            async with anyio.create_task_group() as tg:

                async def client_to_daemon() -> None:
                    await _pump(client_read, daemon_write)
                    tg.cancel_scope.cancel()

                async def daemon_to_client() -> None:
                    await _pump(daemon_read, client_write)
                    tg.cancel_scope.cancel()

                tg.start_soon(client_to_daemon)
                tg.start_soon(daemon_to_client)


def _cmd_status() -> None:
    import argparse
    import sys

    from reporag.projects import all_projects, get

    p = argparse.ArgumentParser(prog="reporag status")
    p.add_argument("--project", type=str, help="Check a specific project path")
    args = p.parse_args(sys.argv[2:])

    if args.project:
        info = get(args.project)
        result: dict = dict(info) if info else {"chunks": 0, "files": 0, "indexed": False}
        result["project"] = args.project
        result.setdefault("indexed", info is not None)
    else:
        result = all_projects()

    print(json.dumps(result, indent=2))


def _cmd_serve() -> None:
    """Run reporag as a shared streamable-HTTP daemon.

    Both host and port fall back to the ``REPORAG_HTTP_HOST`` / ``REPORAG_HTTP_PORT``
    env vars so a systemd unit can set them without CLI flags.
    """
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="reporag serve")
    p.add_argument(
        "--host",
        default=os.environ.get("REPORAG_HTTP_HOST", "127.0.0.1"),
        help="bind address (default: 127.0.0.1; loopback keeps the daemon local)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("REPORAG_HTTP_PORT", "7800")),
        help="bind port (default: 7800)",
    )
    args = p.parse_args(sys.argv[2:])

    asyncio.run(_serve_http(args.host, args.port))


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        cmd = sys.argv[1]
        if cmd == "status":
            _cmd_status()
            return
        if cmd == "serve":
            _cmd_serve()
            return
        if cmd == "stdio":
            # Legacy direct mode: this process is itself the full server, loading
            # its own model. Use for debugging or single-client setups that don't
            # want the shared daemon. REPORAG_NO_DAEMON=1 selects it by default.
            asyncio.run(_serve())
            return
        if cmd in ("setup", "setup-hooks"):
            from reporag import setup

            if cmd == "setup-hooks":
                setup.cli_hooks()  # backward-compat alias
            else:
                setup.cli()
            return

    # Default: thin bridge that auto-spawns and shares one machine-wide daemon.
    # Opt out (revert to a self-contained per-client server) with REPORAG_NO_DAEMON=1.
    if os.environ.get("REPORAG_NO_DAEMON") == "1":
        asyncio.run(_serve())
    else:
        asyncio.run(_serve_bridge())


if __name__ == "__main__":
    main()
