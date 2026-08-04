"""General facility-analysis interface — the unified surface that replaces per-step per-chapter tools.

One dispatching tool (``analyze_facility``) takes a facility type plus its inputs and returns the complete analysis by delegating to the verified transportations-library executor for that facility. Two discovery tools (``list_facility_types``, ``describe_facility_inputs``) let a caller — human or LLM — find what facilities exist and construct a valid input without reading the Rust bindings.

Coverage grows one table row at a time: an adapter is a (input model, runner) pair in ``FACILITIES``. Facilities the library can execute but which have no adapter yet are listed with status ``adapter_pending`` so the surface never overstates what it covers. The per-step chapter functions in ``chapter12.py``/``chapter15.py`` are unchanged — the ablation server variants import them directly and their behavior is frozen with the paper.
"""

from typing import Any, Dict

from pydantic import BaseModel

from hcm_mcp_server.core.models import BasicFreewaysInput, TwoLaneHighwaysInput
from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
from hcm_mcp_server.functions.chapter15 import complete_highway_analysis_function


def _run_basic_freeway(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return complete_freeway_analysis_function({"freeway_data": inputs})


def _run_two_lane_highway(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return complete_highway_analysis_function({"highway_data": inputs})


# The dispatch table. Keys are the public facility_type strings; each available
# entry carries its pydantic input model and a runner returning the standard
# {"success": ..., ...} dict. Pending entries document the library class that
# will back them so wiring one is a self-contained task.
FACILITIES: Dict[str, Dict[str, Any]] = {
    "BasicFreeway": {
        "chapter": 12,
        "description": "Basic freeway segment (HCM Ch. 12): FFS, capacity, speed, density, LOS",
        "library_class": "BasicFreeways",
        "input_model": BasicFreewaysInput,
        "run": _run_basic_freeway,
    },
    "TwoLaneHighway": {
        "chapter": 15,
        "description": "Two-lane highway facility (HCM Ch. 15): per-segment follower density and facility LOS",
        "library_class": "TwoLaneHighways",
        "input_model": TwoLaneHighwaysInput,
        "run": _run_two_lane_highway,
    },
    # ---- library-backed, adapter pending ----
    "FreewayFacility": {"chapter": 10, "library_class": "FreewayFacility", "description": "Freeway facility core methodology (HCM Ch. 10)"},
    "FreewayReliability": {"chapter": 11, "library_class": "FreewayReliability", "description": "Freeway reliability and strategy assessment (HCM Ch. 11)"},
    "WeavingSegment": {"chapter": 13, "library_class": "WeavingSegment", "description": "Freeway weaving segment (HCM Ch. 13)"},
    "RampSegment": {"chapter": 14, "library_class": "RampSegment", "description": "Freeway merge/diverge segment (HCM Ch. 14)"},
    "UrbanFacility": {"chapter": 16, "library_class": "UrbanFacility", "description": "Urban street facility (HCM Ch. 16)"},
    "UrbanReliability": {"chapter": 17, "library_class": "UrbanReliability", "description": "Urban street reliability and ATDM (HCM Ch. 17)"},
    "UrbanSegment": {"chapter": 18, "library_class": "UrbanSegment", "description": "Urban street segment (HCM Ch. 18)"},
    "SignalizedIntersection": {"chapter": 19, "library_class": "SignalizedIntersection", "description": "Signalized intersection (HCM Ch. 19)"},
    "TWSC": {"chapter": 20, "library_class": "Twsc", "description": "Two-way stop-controlled intersection (HCM Ch. 20)"},
    "AWSC": {"chapter": 21, "library_class": "Awsc", "description": "All-way stop-controlled intersection (HCM Ch. 21)"},
    "Roundabout": {"chapter": 22, "library_class": "Roundabouts", "description": "Roundabout (HCM Ch. 22)"},
    "RampTerminal": {"chapter": 23, "library_class": "Interchange", "description": "Ramp terminals and alternative intersections (HCM Ch. 23)"},
    "OffStreetPedBike": {"chapter": 24, "library_class": "OffStreetBicycleFacility", "description": "Off-street pedestrian and bicycle facilities (HCM Ch. 24)"},
}


def _status(entry: Dict[str, Any]) -> str:
    return "available" if "run" in entry else "adapter_pending"


def _describe_model(model: type[BaseModel]) -> list[dict[str, Any]]:
    """Flatten a pydantic model into field descriptions, recursing into nested list-of-model fields (e.g. two-lane segments)."""
    fields = []
    for name, info in model.model_fields.items():
        entry: Dict[str, Any] = {
            "name": name,
            "type": str(info.annotation).replace("typing.", ""),
            "required": info.is_required(),
            "description": info.description or "",
        }
        if not info.is_required():
            entry["default"] = repr(info.default) if not repr(info.default).startswith("<") else "factory"
        args = getattr(info.annotation, "__args__", ())
        nested = next((a for a in args if isinstance(a, type) and issubclass(a, BaseModel)), None)
        if nested is not None:
            entry["item_fields"] = _describe_model(nested)
        fields.append(entry)
    return fields


def analyze_facility_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the complete HCM analysis for one facility: dispatch on facility_type and delegate to the library executor."""
    facility_type = data.get("facility_type")
    if not facility_type or facility_type not in FACILITIES:
        return {
            "success": False,
            "error": f"Unknown facility_type {facility_type!r}. Valid types: {sorted(FACILITIES)}. Call list_facility_types for status and describe_facility_inputs for input schemas.",
        }
    entry = FACILITIES[facility_type]
    if "run" not in entry:
        return {
            "success": False,
            "error": f"facility_type {facility_type!r} is library-backed ({entry['library_class']}, HCM Ch. {entry['chapter']}) but its MCP adapter is pending. Currently available: {sorted(k for k, v in FACILITIES.items() if 'run' in v)}.",
            "status": "adapter_pending",
        }
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        return {"success": False, "error": "Missing 'inputs' object. Call describe_facility_inputs for the schema of this facility_type."}
    try:
        entry["input_model"](**inputs)
    except Exception as e:  # surface pydantic's message as a clean error
        return {"success": False, "error": f"Invalid inputs for {facility_type}: {e}"}
    result = entry["run"](inputs)
    if isinstance(result, dict):
        result.setdefault("facility_type", facility_type)
        result.setdefault("chapter", entry["chapter"])
    return result


def list_facility_types_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """List every facility type the library can execute, with adapter status."""
    rows = [
        {
            "facility_type": name,
            "chapter": entry["chapter"],
            "description": entry["description"],
            "status": _status(entry),
        }
        for name, entry in sorted(FACILITIES.items(), key=lambda kv: kv[1]["chapter"])
    ]
    return {
        "success": True,
        "facility_types": rows,
        "available": sorted(k for k, v in FACILITIES.items() if "run" in v),
        "adapter_pending": sorted(k for k, v in FACILITIES.items() if "run" not in v),
    }


def describe_facility_inputs_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Describe the input schema for a facility type so a caller can construct a valid analyze_facility request."""
    facility_type = data.get("facility_type")
    if not facility_type or facility_type not in FACILITIES:
        return {"success": False, "error": f"Unknown facility_type {facility_type!r}. Valid types: {sorted(FACILITIES)}."}
    entry = FACILITIES[facility_type]
    if "run" not in entry:
        return {
            "success": False,
            "status": "adapter_pending",
            "error": f"{facility_type} has no adapter yet; its input model is not defined. Library class: {entry['library_class']} (HCM Ch. {entry['chapter']}).",
        }
    return {
        "success": True,
        "facility_type": facility_type,
        "chapter": entry["chapter"],
        "fields": _describe_model(entry["input_model"]),
    }
