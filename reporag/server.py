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

from reporag.config import Config, get_config

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
    _watcher_handlers: dict[str, Any] = field(default_factory=dict)  # project → handler
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
                        "description": "Restrict results to this absolute project root path",
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
                    "project": {"type": "string", "description": "Absolute path to project root"},
                },
                "required": ["query", "project"],
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
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        result = {"error": str(exc)}

    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def _serve() -> None:
    _runtime.initialize()
    _runtime._loop = asyncio.get_running_loop()
    _runtime.index_sem = asyncio.Semaphore(1)  # serialise concurrent index runs

    _auto_setup_hooks()

    if _runtime.config.auto_index_paths:
        asyncio.create_task(_runtime._auto_index())

    try:
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    finally:
        if _runtime._watcher is not None:
            _runtime._watcher.stop()
            _runtime._watcher.join(timeout=2)


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


def _setup_hooks_impl(claude_dir: Path, verbose: bool = False) -> bool:
    """Install reporag Claude Code hooks. Returns True if settings changed."""
    import shutil

    pkg_hooks = Path(__file__).parent / "hooks"
    if not pkg_hooks.exists():
        if verbose:
            print(f"Hook scripts not found at {pkg_hooks}")
        return False

    hooks_dir = claude_dir / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)

    installed: list[Path] = []
    for hook_file in sorted(pkg_hooks.glob("reporag-*.py")):
        dest = hooks_dir / hook_file.name
        if not dest.exists() or hook_file.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(hook_file, dest)
            dest.chmod(0o755)
            if verbose:
                print(f"  copied → {dest}")
        elif verbose:
            print(f"  up-to-date → {dest}")
        installed.append(dest)

    if not installed:
        return False

    settings_path = claude_dir / "settings.json"
    try:
        settings: dict = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    except json.JSONDecodeError:
        msg = f"  warning: {settings_path} contains invalid JSON — skipping to avoid data loss. Fix manually and re-run."
        if verbose:
            print(msg)
        else:
            logger.warning(msg)
        return False

    hooks_cfg = settings.setdefault("hooks", {})
    up_hooks: list = hooks_cfg.setdefault("UserPromptSubmit", [])

    def _hook_command(h: dict) -> str | None:
        entries = h.get("hooks") or []
        return entries[0].get("command") if entries else None

    changed = False
    for dest in installed:
        command = str(dest)
        already = any(_hook_command(h) == command for h in up_hooks)
        if not already:
            up_hooks.append({"matcher": ".*", "hooks": [{"type": "command", "command": command}]})
            changed = True

    if changed:
        settings_path.write_text(json.dumps(settings, indent=2))
        if verbose:
            print(f"  updated → {settings_path}")

    return changed


def _auto_setup_hooks() -> None:
    """Called on MCP server start — installs hooks automatically on first connection."""
    try:
        claude_dir = Path.home() / ".claude"
        changed = _setup_hooks_impl(claude_dir, verbose=False)
        if changed:
            logger.info("reporag: Claude Code hooks installed automatically (restart Claude Code to activate)")
    except Exception:
        pass  # never break the MCP server


def _cmd_setup_hooks() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="reporag setup-hooks")
    p.add_argument("--claude-dir", default="~/.claude", help="Claude config directory")
    args = p.parse_args(sys.argv[2:])

    claude_dir = Path(args.claude_dir).expanduser()
    _setup_hooks_impl(claude_dir, verbose=True)
    print("\nDone. Restart Claude Code to activate hooks.")


_MCP_CONFIG_BLOCK = {
    "command": "uvx",
    "args": [
        "--from",
        "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
        "reporag",
    ],
    "env": {"REPORAG_DATA_DIR": "~/.local/share/reporag"},
}

_CURSORRULES_SNIPPET = """\
# reporag — local RAG code search
# When answering questions about code in this project:
# 1. Call query_code to retrieve relevant context before answering
# 2. Call index_codebase if the project has not been indexed yet
# 3. Call get_symbol for exact function/class lookups
Use the reporag MCP tools proactively for all code questions.
"""


def _setup_cursor_impl(cursor_dir: Path, verbose: bool = False) -> bool:
    """Write ~/.cursor/mcp.json and install Cursor rules. Returns True if changed."""
    cursor_dir.mkdir(parents=True, exist_ok=True)
    mcp_path = cursor_dir / "mcp.json"

    try:
        mcp: dict = json.loads(mcp_path.read_text()) if mcp_path.exists() else {}
    except json.JSONDecodeError:
        msg = f"  warning: {mcp_path} contains invalid JSON — skipping to avoid data loss. Fix manually and re-run."
        if verbose:
            print(msg)
        else:
            logger.warning(msg)
        return False

    servers = mcp.setdefault("mcpServers", {})
    changed = "reporag" not in servers or servers["reporag"] != _MCP_CONFIG_BLOCK
    servers["reporag"] = _MCP_CONFIG_BLOCK
    if changed:
        mcp_path.write_text(json.dumps(mcp, indent=2))
        if verbose:
            print(f"  written → {mcp_path}")
    elif verbose:
        print(f"  unchanged → {mcp_path}")

    # Cursor ≥0.50 global rules dir
    rules_dir = cursor_dir / "rules"
    rules_path = rules_dir / "reporag.mdc"
    if not rules_path.exists():
        rules_dir.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(
            "---\ndescription: Use reporag MCP for code questions\nalwaysApply: true\n---\n"
            + _CURSORRULES_SNIPPET
        )
        if verbose:
            print(f"  written → {rules_path}")
        changed = True
    elif verbose:
        print(f"  exists  → {rules_path} (not overwritten)")

    if verbose:
        print(
            "\nFor older Cursor versions, add to your project's .cursorrules:\n"
            + _CURSORRULES_SNIPPET
        )

    return changed


def _cmd_setup() -> None:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="reporag setup",
        description="Configure reporag for Claude Code, Cursor, or both.",
    )
    p.add_argument(
        "--client",
        choices=["claude", "cursor", "all"],
        default="all",
        help="Which AI client to configure (default: all)",
    )
    p.add_argument("--claude-dir", default="~/.claude")
    p.add_argument("--cursor-dir", default="~/.cursor")
    args = p.parse_args(sys.argv[2:])

    clients = ["claude", "cursor"] if args.client == "all" else [args.client]

    for client in clients:
        print(f"\n[{client}]")
        if client == "claude":
            _setup_hooks_impl(Path(args.claude_dir).expanduser(), verbose=True)
            print("  Restart Claude Code to activate hooks.")
        else:
            _setup_cursor_impl(Path(args.cursor_dir).expanduser(), verbose=True)
            print("  Restart Cursor to activate.")


def main() -> None:
    import sys

    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        cmd = sys.argv[1]
        if cmd == "status":
            _cmd_status()
            return
        if cmd in ("setup", "setup-hooks"):
            if cmd == "setup-hooks":
                _cmd_setup_hooks()  # backward-compat alias
            else:
                _cmd_setup()
            return

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
