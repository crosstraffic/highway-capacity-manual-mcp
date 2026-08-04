"""Reasoning functions — the X-KG reasoning layer.

Where ``chapter15.py`` *computes* HCM analyses, these functions *reason* over the knowledge graph and the verified executable substrate:

* ``propagate_change`` / ``diagnose_failure`` — forward/backward chaining over the AFFECTS graph (which downstream parameters a change touches; which upstream causes could explain a failure).
* ``repair_design`` / ``repair_freeway`` — abductive design repair: the minimal compliant fix for a failing Two-Lane Highway (HCM Ch.15) or Basic Freeway (HCM Ch.12). Every candidate is re-executed through ``transportations-library`` before it is returned — proposals are *proved* compliant, not asserted.
* ``reconcile_codes`` — defeasible adjudication of overlapping/conflicting code provisions, with an argument trace.
* ``inverse_design`` — goal-directed synthesis: feasible geometries that reach a target LOS, each validated by forward execution.

All graph and bounds data come from the ``transportations-validator`` seed corpus; the reasoning layer is database-free (it reads seed JSON and executes the Rust library). These functions are registered in ``function_registry.yaml`` under the ``reasoning`` category and follow the same ``(data: dict) -> dict`` convention as the Chapter 15 functions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from transportations_validator.validators.executors import (
    BasicFreewayExecutor,
    TwoLaneHighwayExecutor,
)
from transportations_validator.validators.forward_chain import (
    backward_chain,
    forward_chain,
    load_relationships_from_seed,
)
from transportations_validator.validators.inverse import inverse_design
from transportations_validator.validators.reconcile import (
    load_conflict_scenarios,
    reconcile,
)
from transportations_validator.validators.repair import (
    load_parameter_bounds,
    los_no_worse_than,
    repair_design,
)

# Derived quantities are evidence of compliance, never repair levers.
_DERIVED = frozenset({
    "ffs", "avg_speed", "percent_followers", "followers_density", "flow_rate",
    "capacity", "speed", "density", "vc_ratio", "los",
})

# Site conditions held fixed by default (per facility) — demand and terrain are facts of the site; repairs come from geometry and access/ramp management.
_DEFAULT_IMMUTABLE = {
    "TwoLaneHighway": {"volume", "grade", "phv", "phf", "spl", "length", "passing_type"},
    "BasicFreeway": {"demand_flow_i", "grade", "length", "p_t", "bffs", "lane_count", "phf"},
}

_SUPPORTED = ("TwoLaneHighway", "BasicFreeway")


@lru_cache(maxsize=1)
def _relationships() -> list[dict[str, Any]]:
    return load_relationships_from_seed()


def _executor(facility_type: str):
    if facility_type == "BasicFreeway":
        return BasicFreewayExecutor()
    if facility_type == "TwoLaneHighway":
        return TwoLaneHighwayExecutor()
    raise ValueError(
        f"Executable reasoning is wired for {' and '.join(_SUPPORTED)}, "
        f"not '{facility_type}'."
    )


# ─── Forward / backward chaining ────────────────────────────────────────────


def propagate_change_function(data: dict[str, Any]) -> dict[str, Any]:
    """Forward-chain: downstream parameters affected by a changed root."""
    try:
        result = forward_chain(
            _relationships(), data["root"], facility_type=data.get("facility_type")
        )
        return {"success": True, **result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


def diagnose_failure_function(data: dict[str, Any]) -> dict[str, Any]:
    """Backward-chain: upstream causes of a failed target parameter."""
    try:
        result = backward_chain(
            _relationships(), data["target"], facility_type=data.get("facility_type")
        )
        return {"success": True, **result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


# ─── Abductive design repair ────────────────────────────────────────────────


def _run_repair(
    facility_type: str,
    design: dict[str, Any],
    goal_los: str,
    immutable: list[str] | None,
    allow_demand_changes: bool,
) -> dict[str, Any]:
    executor = _executor(facility_type)
    bounds = load_parameter_bounds(facility_type)
    if immutable is None:
        site = set(_DEFAULT_IMMUTABLE.get(facility_type, set()))
        if allow_demand_changes:
            site.discard("phf")
            site.discard("volume")
            site.discard("demand_flow_i")
        immutable = list(site)
    imm = frozenset(immutable) | _DERIVED | {"los"}
    goal_letter = goal_los.upper()
    result = repair_design(
        _relationships(),
        target="los",
        design=dict(design),
        executor=executor,
        goal=los_no_worse_than(goal_letter),
        bounds=bounds,
        facility_type=facility_type,
        immutable=imm,
        goal_description=f"facility LOS no worse than {goal_letter}",
    )
    return {"success": True, **result.to_dict()}


def repair_design_function(data: dict[str, Any]) -> dict[str, Any]:
    """Minimal compliant fix for a failing Two-Lane Highway (HCM Ch.15).

    ``data``: ``design`` (rust_field inputs), ``goal_los`` (default "C"), optional ``immutable`` (override site conditions) and ``allow_demand_changes``.
    """
    try:
        return _run_repair(
            "TwoLaneHighway",
            data["design"],
            data.get("goal_los", "C"),
            data.get("immutable"),
            data.get("allow_demand_changes", False),
        )
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


def repair_freeway_function(data: dict[str, Any]) -> dict[str, Any]:
    """Minimal compliant fix for a failing Basic Freeway (HCM Ch.12).

    ``data``: ``design`` (rust_field inputs incl. bffs, lw, lane_count, demand_flow_i), ``goal_los`` (default "D"), optional ``immutable`` and ``allow_demand_changes``. NOTE: transportations-library >=0.3.0 interpolates the heavy-vehicle grade/length grid, so formerly off-grid inputs evaluate normally; the "non-evaluable" error path remains only as a guard for unexpected library errors.
    """
    try:
        return _run_repair(
            "BasicFreeway",
            data["design"],
            data.get("goal_los", "D"),
            data.get("immutable"),
            data.get("allow_demand_changes", False),
        )
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


# ─── Defeasible multi-jurisdiction reconciliation ───────────────────────────


def reconcile_codes_function(data: dict[str, Any]) -> dict[str, Any]:
    """Adjudicate competing code provisions about one parameter.

    ``data``: either ``scenario`` (a constructed conflict-scenario name) or an inline ``claims`` list, plus optional ``parameter``, ``value``, ``context``.
    """
    try:
        claims: list[dict[str, Any]] = []
        parameter = data.get("parameter")
        context: dict[str, str] = {}

        scenario_name = data.get("scenario")
        if scenario_name is not None:
            scenarios = load_conflict_scenarios()
            scenario = scenarios.get(scenario_name)
            if scenario is None:
                return {
                    "success": False,
                    "error": f"Unknown conflict scenario '{scenario_name}'. "
                             f"Available: {sorted(scenarios)}",
                }
            claims = list(scenario.get("claims", []))
            parameter = parameter or scenario.get("parameter")
            context.update(scenario.get("default_context", {}))

        if data.get("claims") is not None:
            claims = list(data["claims"])
        if not claims:
            return {"success": False,
                    "error": "Provide either 'scenario' or a non-empty 'claims' list."}
        context.update(data.get("context") or {})

        result = reconcile(
            claims, parameter=parameter, value=data.get("value"), context=context
        )
        return {"success": True, **result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}


# ─── Goal-directed inverse design ───────────────────────────────────────────


def inverse_design_function(data: dict[str, Any]) -> dict[str, Any]:
    """Synthesize feasible geometries reaching a target LOS (HCM Ch.15 PoC).

    ``data``: ``site`` (fixed conditions), ``goal_los``, ``facility_type`` (default "TwoLaneHighway"), optional ``design_parameters`` and ``bounds``.
    """
    try:
        facility_type = data.get("facility_type", "TwoLaneHighway")
        executor = _executor(facility_type)
        bounds = load_parameter_bounds(facility_type)
        if data.get("bounds"):
            bounds.update(
                {k: (float(lo), float(hi)) for k, (lo, hi) in data["bounds"].items()}
            )
        goal_letter = data.get("goal_los", "C").upper()
        result = inverse_design(
            _relationships(),
            target="los",
            site=dict(data["site"]),
            executor=executor,
            goal=los_no_worse_than(goal_letter),
            bounds=bounds,
            design_parameters=data.get("design_parameters"),
            facility_type=facility_type,
            goal_description=f"reach LOS {goal_letter}",
        )
        return {"success": True, **result.to_dict()}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "error": str(e)}
