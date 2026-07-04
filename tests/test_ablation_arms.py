"""Tests for the ablation-arm MCP surface (kg-only / rag-only filtering).

These assert the wiring that the Table 5 / Figure 7 2x2 ablation depends on:
the reasoning/validation functions are exposed as first-class MCP tools (via
``reasoning_endpoints``), and ``FastApiMCP``'s ``include_operations`` filter
restricts the surface to exactly the arm's tool set. They need only FastAPI +
fastapi-mcp (no transportations-validator / vector store), so they run in CI
regardless of whether the heavy reasoning deps are installed.
"""

import pytest

pytest.importorskip("fastapi_mcp", reason="fastapi-mcp not installed")

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from hcm_mcp_server.core import reasoning_endpoints

# Mirrors ablation/arms.py KG_TOOLS and the kg-only launcher's KG_OPERATIONS.
KG_OPERATIONS = [
    "validate_design_full",
    "propagate_change",
    "diagnose_failure",
    "repair_design",
    "repair_freeway",
    "reconcile_codes",
    "inverse_design",
]


def _app():
    app = FastAPI()
    app.include_router(reasoning_endpoints.router)
    return app


def _tool_names(app, include_operations):
    mcp = FastApiMCP(app, include_operations=include_operations)
    mcp.mount()
    return sorted(t.name for t in mcp.tools)


def test_reasoning_router_exposes_all_kg_tools_as_first_class():
    """Each KG function is a dedicated MCP tool, not behind generic call_tool."""
    names = _tool_names(_app(), KG_OPERATIONS)
    assert names == sorted(KG_OPERATIONS)


def test_kg_only_filter_excludes_retrieval_and_compute():
    """The kg-only surface is exactly the 7 KG tools — no query_hcm, no analysis."""
    names = _tool_names(_app(), KG_OPERATIONS)
    assert "query_hcm" not in names
    assert "chapter15_complete" not in names
    assert len(names) == 7


def test_include_operations_is_precise():
    """A single-operation filter yields exactly that one tool."""
    names = _tool_names(_app(), ["propagate_change"])
    assert names == ["propagate_change"]
