"""MCP tool: get_architecture — dependency topology with role classification."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def _classify_role(out_deg: int, in_deg: int) -> str:
    if out_deg > 5:
        return "hub"          # imports many — orchestrator/controller
    if in_deg > 3 and out_deg <= 2:
        return "utility"      # imported by many — shared library
    if out_deg >= 2 and in_deg >= 2:
        return "bridge"       # middle layer
    return "leaf"             # standalone module


async def run(
    arguments: dict[str, Any],
    runtime: Runtime,  # type: ignore[name-defined]  # noqa: F821
) -> dict[str, Any]:
    project = arguments.get("project", "").strip()
    if not project:
        return {"error": "project is required"}

    root = str(Path(project).expanduser().resolve())

    if runtime.graph is None or runtime.graph.number_of_nodes() == 0:
        return {"error": "No graph loaded. Run index_codebase first."}

    # ── project subgraph ─────────────────────────────────────────────────────
    project_nodes = [n for n in runtime.graph.nodes() if n.startswith(root)]
    if not project_nodes:
        return {
            "project": root,
            "nodes": [],
            "edges": [],
            "layers": {},
            "message": "No nodes in graph for this project.",
        }

    # ── classify nodes ───────────────────────────────────────────────────────
    nodes: list[dict[str, Any]] = []
    for node in project_nodes[:200]:
        out_deg = runtime.graph.out_degree(node)
        in_deg = runtime.graph.in_degree(node)
        rel = node[len(root):].lstrip("/")
        nodes.append({
            "file": rel,
            "role": _classify_role(out_deg, in_deg),
            "out_degree": out_deg,
            "in_degree": in_deg,
        })

    nodes.sort(key=lambda n: n["out_degree"] + n["in_degree"], reverse=True)

    # ── edges within project ─────────────────────────────────────────────────
    edges: list[dict[str, str]] = []
    for src, dst in runtime.graph.edges():
        if src.startswith(root) and dst.startswith(root):
            edges.append({
                "from": src[len(root):].lstrip("/"),
                "to": dst[len(root):].lstrip("/"),
            })
        if len(edges) >= 100:
            break

    # ── layer grouping by top-level path segment ──────────────────────────────
    layers: dict[str, list[str]] = {}
    for n in nodes:
        parts = n["file"].split("/")
        layer = parts[0] if len(parts) > 1 else "root"
        layers.setdefault(layer, []).append(n["file"])

    # ── role summary ─────────────────────────────────────────────────────────
    role_counts: dict[str, int] = {}
    for n in nodes:
        role_counts[n["role"]] = role_counts.get(n["role"], 0) + 1

    return {
        "project": root,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "role_summary": role_counts,
        "nodes": nodes,
        "edges": edges,
        "layers": layers,
    }
