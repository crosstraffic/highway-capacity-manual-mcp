"""First-class REST + MCP endpoints for the reasoning & validation layers.

The reasoning (``propagate_change``, ``diagnose_failure``, ``repair_design``,
``repair_freeway``, ``reconcile_codes``, ``inverse_design``) and full-corpus
``validate_design_full`` functions are registered in ``function_registry.yaml``
but, unlike the Chapter 12/15 and research functions, previously had no
dedicated route — they were reachable only through the generic ``call_tool``
dispatcher. That made them invisible to ``FastApiMCP`` as individual tools.

This router gives each one a dedicated endpoint (and therefore a first-class
MCP tool with its own schema), so the knowledge-graph capabilities can be
exposed — or isolated for ablation — by operation id. Each endpoint resolves
the implementation from the registry (single source of truth) and dispatches
it, awaiting async functions and calling sync ones directly.
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from hcm_mcp_server.core.dependencies import get_function_registry
from hcm_mcp_server.core.registry import FunctionRegistry

router = APIRouter()


async def _invoke(registry: FunctionRegistry, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a registered function and run it (async or sync)."""
    function_impl = registry.get_function(name)
    if function_impl is None:
        raise HTTPException(status_code=404, detail=f"Function '{name}' not available")
    try:
        if asyncio.iscoroutinefunction(function_impl):
            return await function_impl(data)
        return function_impl(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{name} failed: {e}")


# ─── Request models (mirror function_registry.yaml parameter schemas) ───────

class PropagateChangeRequest(BaseModel):
    root: str = Field(..., description="The changed parameter (rust_field) to propagate from")
    facility_type: Optional[str] = Field(None, description="Restrict to a facility (TwoLaneHighway, BasicFreeway)")


class DiagnoseFailureRequest(BaseModel):
    target: str = Field(..., description="The failing parameter (rust_field) to trace causes for")
    facility_type: Optional[str] = Field(None, description="Restrict to a facility (TwoLaneHighway, BasicFreeway)")


class RepairDesignRequest(BaseModel):
    design: Dict[str, Any] = Field(..., description="Current design inputs (rust_fields: lane_width, shoulder_width, apd, volume, spl, grade, phv, phf, length, passing_type)")
    goal_los: str = Field("C", description="Target LOS letter — repair goal is 'no worse than this'")
    immutable: Optional[List[str]] = Field(None, description="Override which parameters are fixed site conditions")
    allow_demand_changes: bool = Field(False, description="Also consider demand-side levers")


class RepairFreewayRequest(BaseModel):
    design: Dict[str, Any] = Field(..., description="Current design inputs (rust_fields: bffs, lw, lane_count, demand_flow_i, lc_r, trd, grade, length, p_t, phf)")
    goal_los: str = Field("D", description="Target LOS letter — repair goal is 'no worse than this'")
    immutable: Optional[List[str]] = Field(None, description="Override which parameters are fixed site conditions")
    allow_demand_changes: bool = Field(False, description="Also consider demand-side levers")


class ReconcileCodesRequest(BaseModel):
    scenario: Optional[str] = Field(None, description="Name of a constructed conflict scenario (alternative to inline claims)")
    claims: Optional[List[Dict[str, Any]]] = Field(None, description="Inline competing rule claims (seed-shaped rule dicts)")
    parameter: Optional[str] = Field(None, description="The parameter under contention")
    value: Optional[float] = Field(None, description="A value to render a compliance verdict for")
    context: Optional[Dict[str, Any]] = Field(None, description="Applicability conditions (e.g. {jurisdiction: state})")


class InverseDesignRequest(BaseModel):
    site: Dict[str, Any] = Field(..., description="Fixed site conditions (demand, grade, speed limit, …)")
    goal_los: str = Field("C", description="Target LOS letter")
    facility_type: str = Field("TwoLaneHighway", description="Facility type")
    design_parameters: Optional[List[str]] = Field(None, description="Levers to discover geometry over (auto-discovered if omitted)")
    bounds: Optional[Dict[str, Any]] = Field(None, description="Override (min, max) bounds per design parameter")


class ValidateDesignFullRequest(BaseModel):
    design: Dict[str, Any] = Field(..., description="Flat parameter -> value map, e.g. {lane_width: 9.0, facility_type: TwoLaneHighway}")
    context: Optional[Dict[str, Any]] = Field(None, description="Optional conditions, e.g. {terrain_type: mountainous, jurisdiction: state}")


# ─── Reasoning endpoints (KG / causal-graph layer) ──────────────────────────

@router.post("/reason/propagate-change", tags=["reasoning"], operation_id="propagate_change")
async def propagate_change(
    request: PropagateChangeRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Forward-chain: downstream parameters affected by a changed root parameter."""
    return await _invoke(registry, "propagate_change", request.model_dump(exclude_none=True))


@router.post("/reason/diagnose-failure", tags=["reasoning"], operation_id="diagnose_failure")
async def diagnose_failure(
    request: DiagnoseFailureRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Backward-chain: upstream causes of a failed target parameter."""
    return await _invoke(registry, "diagnose_failure", request.model_dump(exclude_none=True))


@router.post("/reason/repair-design", tags=["reasoning"], operation_id="repair_design")
async def repair_design(
    request: RepairDesignRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Abductive repair: minimal compliant fix for a failing Two-Lane Highway (HCM Ch.15). Every candidate is re-executed and proved compliant."""
    return await _invoke(registry, "repair_design", request.model_dump(exclude_none=True))


@router.post("/reason/repair-freeway", tags=["reasoning"], operation_id="repair_freeway")
async def repair_freeway(
    request: RepairFreewayRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Abductive repair: minimal compliant fix for a failing Basic Freeway (HCM Ch.12). Keep grade/length on the library's heavy-vehicle PCE grid."""
    return await _invoke(registry, "repair_freeway", request.model_dump(exclude_none=True))


@router.post("/reason/reconcile-codes", tags=["reasoning"], operation_id="reconcile_codes")
async def reconcile_codes(
    request: ReconcileCodesRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Defeasible adjudication of overlapping/conflicting code provisions about one parameter, with an argument trace."""
    return await _invoke(registry, "reconcile_codes", request.model_dump(exclude_none=True))


@router.post("/reason/inverse-design", tags=["reasoning"], operation_id="inverse_design")
async def inverse_design(
    request: InverseDesignRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Goal-directed synthesis: feasible geometries that reach a target LOS, each validated by forward execution (HCM Ch.15 PoC)."""
    return await _invoke(registry, "inverse_design", request.model_dump(exclude_none=True))


# ─── Validation endpoint (full rule corpus, in-process, no database) ────────

@router.post("/validate/design-full", tags=["validation"], operation_id="validate_design_full")
async def validate_design_full(
    request: ValidateDesignFullRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Validate a design against the FULL rule corpus with citations and clarifications (in-process, no database)."""
    return await _invoke(registry, "validate_design_full", request.model_dump(exclude_none=True))
