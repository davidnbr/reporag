# Client setup

reporag works with any MCP-compatible client. Below are per-client config files and the proactive-usage rules/hooks reporag installs for each.

## Automatic setup

```bash
# Configure Claude Code, Cursor, and Codex CLI at once:
uvx --from "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git" \
    reporag setup --client all

# Or configure individually:
reporag setup --client claude   # Claude Code only
reporag setup --client cursor   # Cursor only
reporag setup --client codex    # Codex CLI only
```

Then restart the client.

> Each client launches `reporag` as a thin bridge that connects to one shared,
> auto-spawned daemon per machine (loopback `127.0.0.1:7800`) — so multiple editor
> windows reuse a single embedding model instead of one per client. This is transparent;
> the config below is unchanged. See [CONFIGURATION.md § Shared daemon](CONFIGURATION.md#shared-daemon).

## Manual config

### Claude Code

`reporag setup --client claude` installs only the proactive-use **hooks** (into
`~/.claude/settings.json`) — it does **not** register the MCP server. Register it yourself
with the `claude mcp add` CLI (recommended — it writes the correct file for you):

```bash
claude mcp add --scope user --env REPORAG_DATA_DIR=~/.local/share/reporag \
  reporag -- uvx --from "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git" reporag
```

`--scope user` makes reporag available across all your projects (stored in `~/.claude.json`).
Use `--scope project` to share it with a team via a committed `.mcp.json`, or omit `--scope`
for the default local scope (current project only, also in `~/.claude.json`).

To edit JSON directly instead, MCP servers live in **`~/.claude.json`** (user/local scope) or
a project-root **`.mcp.json`** — there is no `~/.claude/.mcp.json`. The entry format:

```json
{
  "mcpServers": {
    "reporag": {
      "command": "uvx",
      "args": [
        "--from",
        "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
        "reporag"
      ],
      "env": {
        "REPORAG_DATA_DIR": "~/.local/share/reporag"
      }
    }
  }
}
```

Use `reporag[ml]` instead if you have an NVIDIA GPU and want CUDA-accelerated embedding.

### agy / antigravity (`~/.gemini/antigravity/mcp_config.json`)

Same format as Claude Code above.

### Cursor (`~/.cursor/mcp.json`)

```json
{
  "mcpServers": {
    "reporag": {
      "command": "uvx",
      "args": [
        "--from",
        "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
        "reporag"
      ],
      "env": { "REPORAG_DATA_DIR": "~/.local/share/reporag" }
    }
  }
}
```

### Codex CLI (`~/.codex/config.toml`)

```toml
[mcp_servers.reporag]
command = "uvx"
args = ["--from", "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git", "reporag"]
env = { REPORAG_DATA_DIR = "~/.local/share/reporag" }
enabled = true
startup_timeout_sec = 30
tool_timeout_sec = 120
```

Honors `CODEX_HOME` (defaults to `~/.codex`). Restart Codex to activate.

### Any other MCP client

```json
{
  "command": "uvx",
  "args": [
    "--from",
    "reporag[ml-cpu] @ git+https://github.com/davidnbr/reporag.git",
    "reporag"
  ],
  "env": { "REPORAG_DATA_DIR": "~/.local/share/reporag" }
}
```

## How each client is wired for proactive use

### Claude Code hooks

Two `UserPromptSubmit` hooks ship inside the package and are **installed automatically** the first time the MCP server connects — no manual step needed. They install into `~/.claude/hooks/` and register in `~/.claude/settings.json`.

**`reporag-autoindex`** — fires on every prompt. If the current directory is not indexed:

```
[reporag] /path/to/project has not been indexed yet.
Call index_codebase with path="/path/to/project" to enable code search.
```

Claude automatically calls `index_codebase` before answering.

**`reporag-hint`** — fires on code-related prompts. If the project is indexed:

```
[reporag] /path/to/project is indexed (285 chunks).
Use query_code to retrieve relevant context before answering.
```

Claude proactively calls `query_code` before answering.

Both hooks read `~/.local/share/reporag/projects.json` — no ML imports, < 5 ms overhead.

### Cursor

Writes `~/.cursor/mcp.json` with the reporag server config and creates `~/.cursor/rules/reporag.mdc` (Cursor ≥0.50 global rules, `alwaysApply: true`) instructing Cursor to use `query_code` and `index_codebase` proactively.

For older Cursor versions, add to your project's `.cursorrules`:

```
Use the reporag MCP tools for all code questions:
query_code before answering, index_codebase if not indexed, get_symbol for lookups.
```

### Codex CLI

Writes `~/.codex/config.toml` with the reporag MCP server registration plus `UserPromptSubmit` (`reporag-hint`), `PreToolUse` (`reporag-dupcheck`, matcher `apply_patch`), and `SessionStart` (`reporag-autoindex`) command hooks — using a marker-delimited managed block (`# >>> reporag managed (do not edit) >>>` … `# <<< reporag managed <<<`) so hand-edited comments, other `[mcp_servers.*]` tables, and top-level settings survive untouched. The assembled file is validated with `tomllib` before writing; an invalid or conflicting existing file is left untouched and reported, never corrupted.

Codex hooks emit `{"hookSpecificOutput": {...}}` JSON (set via `REPORAG_HOOK_FORMAT=codex` in the hook command) since Codex doesn't read plain stdout as context the way Claude Code does. Re-running setup is idempotent — byte-identical output, no rewrite.

## Additional CLI

```bash
reporag status --project /path/to/project              # check if a project is indexed
reporag setup-hooks [--claude-dir ~/.claude]           # reinstall Claude Code hooks only
reporag setup --client codex [--codex-dir ~/.codex]    # reinstall Codex config only
```
