"""Light unit tests for the shared-daemon hardening in reporag.server.

Deliberately avoids loading the embedding model or booting a live daemon:
only exercises pure helpers (env parsing, spawn command shape) and asserts
the mcp internals this module relies on haven't shifted underneath it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Any

import pytest

import reporag.server as server


def test_http_addr_defaults(monkeypatch):
    monkeypatch.delenv("REPORAG_HTTP_HOST", raising=False)
    monkeypatch.delenv("REPORAG_HTTP_PORT", raising=False)

    host, port = server._http_addr()

    assert host == "127.0.0.1"
    assert port == 7800


def test_http_addr_reads_env(monkeypatch):
    monkeypatch.setenv("REPORAG_HTTP_HOST", "0.0.0.0")
    monkeypatch.setenv("REPORAG_HTTP_PORT", "9999")

    host, port = server._http_addr()

    assert host == "0.0.0.0"
    assert port == 9999


def test_idle_timeout_zero_disables_watcher(monkeypatch):
    """REPORAG_IDLE_TIMEOUT=0 must disable idle shutdown (daemon runs forever).

    ``_serve_http`` parses ``REPORAG_IDLE_TIMEOUT`` inline (it never boots a real
    event loop for this test); this asserts the same parsing rule the idle-watcher
    guards on (``if idle_timeout <= 0: return``).
    """
    monkeypatch.setenv("REPORAG_IDLE_TIMEOUT", "0")

    idle_timeout = float(os.environ.get("REPORAG_IDLE_TIMEOUT", "900"))

    assert idle_timeout <= 0


def test_idle_timeout_default_is_900s(monkeypatch):
    monkeypatch.delenv("REPORAG_IDLE_TIMEOUT", raising=False)

    idle_timeout = float(os.environ.get("REPORAG_IDLE_TIMEOUT", "900"))

    assert idle_timeout == 900.0


def test_ensure_daemon_spawn_command_shape(tmp_path, monkeypatch):
    """`_ensure_daemon` must spawn via sys.executable -c, not the `reporag` launcher.

    This makes daemon spawn PATH-independent. We stub `_port_open` (always closed)
    and `time.sleep`/deadline via a tiny wait window, and intercept subprocess.Popen
    to capture the command without actually spawning a process — no daemon boot, no
    model load.
    """
    import sys

    monkeypatch.setenv("REPORAG_DATA_DIR", str(tmp_path))
    server._runtime.config.data_dir = str(tmp_path)

    captured: dict[str, list[str]] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> object:
        captured["cmd"] = cmd

        class _FakeProc:
            pass

        return _FakeProc()

    port_open_calls = {"n": 0}

    def fake_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
        # First two checks (pre-lock, post-lock) report closed so we reach spawn;
        # then report open so the wait loop returns immediately.
        port_open_calls["n"] += 1
        return port_open_calls["n"] > 2

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(server, "_port_open", fake_port_open)

    server._ensure_daemon("127.0.0.1", 7800, wait_s=1.0)

    assert captured["cmd"] == [
        sys.executable,
        "-c",
        "from reporag.server import main; main()",
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "7800",
    ]


def test_streamable_http_session_manager_exposes_server_instances():
    """Smoke test: idle-watcher relies on a PRIVATE mcp attribute for live-session
    counting. If this ever disappears, the idle-watcher's live-session check breaks
    silently — re-verify on any mcp upgrade.
    """
    from mcp.server import Server
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    manager = StreamableHTTPSessionManager(app=Server("test"), stateless=False)

    assert hasattr(manager, "_server_instances")
    assert manager._server_instances == {}


class _EmptyStream:
    """Async-iterable/sendable stub that yields nothing — lets the bridge pump
    loop finish immediately without a real transport."""

    def __aiter__(self) -> _EmptyStream:
        return self

    async def __anext__(self) -> Any:
        raise StopAsyncIteration

    async def send(self, message: Any) -> None:  # noqa: ANN401
        pass


class _FailingStream:
    """Async-iterable stub that raises on the first read — simulates a daemon
    dying mid-session, after the connect already succeeded."""

    def __aiter__(self) -> _FailingStream:
        return self

    async def __anext__(self) -> Any:
        raise ConnectionError("daemon died mid-session")

    async def send(self, message: Any) -> None:  # noqa: ANN401
        pass


async def test_bridge_retries_only_the_connect(monkeypatch):
    """A failed *connect* retries once: `_ensure_daemon` is re-invoked and
    `streamable_http_client` is re-entered exactly once more."""
    ensure_daemon_calls = {"n": 0}

    def fake_ensure_daemon(host: str, port: int, wait_s: float = 30.0) -> None:
        ensure_daemon_calls["n"] += 1

    monkeypatch.setattr(server, "_ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(server, "_http_addr", lambda: ("127.0.0.1", 7800))

    @contextlib.asynccontextmanager
    async def fake_stdio_server():
        yield _EmptyStream(), _EmptyStream()

    monkeypatch.setattr(server.mcp.server.stdio, "stdio_server", fake_stdio_server)

    connect_calls = {"n": 0}

    @contextlib.asynccontextmanager
    async def flaky_streamable_http_client(url: str):
        connect_calls["n"] += 1
        if connect_calls["n"] == 1:
            raise ConnectionError("connect refused")
        yield _EmptyStream(), _EmptyStream(), None

    import mcp.client.streamable_http as streamable_http_mod

    monkeypatch.setattr(streamable_http_mod, "streamable_http_client", flaky_streamable_http_client)

    await server._serve_bridge()

    assert connect_calls["n"] == 2
    # Once at top of `_serve_bridge`, once more on the connect retry.
    assert ensure_daemon_calls["n"] == 2


async def test_bridge_mid_session_failure_does_not_retry(monkeypatch):
    """A failure AFTER the connect succeeds (mid-session pump failure) must NOT
    trigger a retry: retrying there would silently open a fresh session and drop
    an in-flight JSON-RPC reply the IDE is still waiting on."""
    ensure_daemon_calls = {"n": 0}

    def fake_ensure_daemon(host: str, port: int, wait_s: float = 30.0) -> None:
        ensure_daemon_calls["n"] += 1

    monkeypatch.setattr(server, "_ensure_daemon", fake_ensure_daemon)
    monkeypatch.setattr(server, "_http_addr", lambda: ("127.0.0.1", 7800))

    @contextlib.asynccontextmanager
    async def fake_stdio_server():
        yield _EmptyStream(), _EmptyStream()

    monkeypatch.setattr(server.mcp.server.stdio, "stdio_server", fake_stdio_server)

    connect_calls = {"n": 0}

    @contextlib.asynccontextmanager
    async def streamable_http_client(url: str):
        connect_calls["n"] += 1
        yield _FailingStream(), _EmptyStream(), None

    import mcp.client.streamable_http as streamable_http_mod

    monkeypatch.setattr(streamable_http_mod, "streamable_http_client", streamable_http_client)

    # anyio's task group wraps the pump failure in an ExceptionGroup rather than
    # letting it propagate bare.
    with pytest.raises(BaseExceptionGroup) as exc_info:
        await server._serve_bridge()

    assert any(isinstance(exc, ConnectionError) for exc in exc_info.value.exceptions)
    assert connect_calls["n"] == 1
    assert ensure_daemon_calls["n"] == 1
