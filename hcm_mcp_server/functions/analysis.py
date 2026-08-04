"""General facility-analysis interface — the unified surface that replaces per-step per-chapter tools.

One dispatching tool (``analyze_facility``) takes a facility type plus its inputs and returns the complete analysis by delegating to the verified transportations-library executor for that facility. One discovery tool (``describe_facility_inputs``) serves both discovery modes: without a facility_type it lists every library-backed facility with adapter status, and with one it returns the field schema needed to construct a valid request — so a caller, human or LLM, never reads the Rust bindings.

Coverage grows one table row at a time: an adapter is a (input model, runner) pair in ``FACILITIES``. Facilities the library can execute but which have no adapter yet are listed with status ``adapter_pending`` so the surface never overstates what it covers. The per-step chapter functions in ``chapter12.py``/``chapter15.py`` are unchanged — the ablation server variants import them directly and their behavior is frozen with the paper.
"""

import json
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

import transportations_library as tl
from hcm_mcp_server.core.models import BasicFreewaysInput, TwoLaneHighwaysInput
from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
from hcm_mcp_server.functions.chapter15 import complete_highway_analysis_function


def _run_basic_freeway(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return complete_freeway_analysis_function({"freeway_data": inputs})


def _run_two_lane_highway(inputs: Dict[str, Any]) -> Dict[str, Any]:
    return complete_highway_analysis_function({"highway_data": inputs})


def _snapshot(obj: Any, names: list[str]) -> Dict[str, Any]:
    """Read a list of accessors off a PyO3 object into a plain dict. Bindings are inconsistent about methods vs getter properties, and some accessors raise before their pipeline step has run — read what exists, call what is callable, skip what raises."""
    out: Dict[str, Any] = {}
    for name in names:
        try:
            value = getattr(obj, name)
            if callable(value):
                value = value()
        except Exception:
            continue
        out[name] = value
    return out


def _guarded_run(fn):
    """Turn any library validation error (ValueError from a PyO3 constructor or step method) into a clean error dict."""
    def wrapper(inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return fn(inputs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001 — PyO3 panics are BaseException
            return {"success": False, "error": str(e)}
    return wrapper


class WeavingSegmentInput(BaseModel):
    """HCM Chapter 13 freeway weaving segment. Every field optional; the library fills HCM defaults."""
    weaving_type: Optional[str] = Field(default=None, description="one_sided or two_sided")
    facility_type: Optional[str] = Field(default=None, description="freeway or multilane (C-D road)")
    length_short: Optional[float] = Field(default=None, description="Short length L_S in feet")
    num_lanes: Optional[int] = Field(default=None, description="Total lanes N in the weaving segment")
    num_weaving_lanes: Optional[int] = Field(default=None, description="Weaving lanes N_WL")
    ffs: Optional[float] = Field(default=None, description="Free-flow speed in mi/h")
    v_ff: Optional[float] = Field(default=None, description="Freeway-to-freeway demand in veh/h")
    v_fr: Optional[float] = Field(default=None, description="Freeway-to-ramp demand in veh/h")
    v_rf: Optional[float] = Field(default=None, description="Ramp-to-freeway demand in veh/h")
    v_rr: Optional[float] = Field(default=None, description="Ramp-to-ramp demand in veh/h")
    phf: Optional[float] = Field(default=None, description="Peak hour factor")
    heavy_vehicle_pct: Optional[float] = Field(default=None, description="Heavy-vehicle proportion (decimal)")
    terrain: Optional[str] = Field(default=None, description="level / rolling / mountainous")
    lc_rf: Optional[int] = Field(default=None, description="Minimum ramp-to-freeway lane changes")
    lc_fr: Optional[int] = Field(default=None, description="Minimum freeway-to-ramp lane changes")
    lc_rr: Optional[int] = Field(default=None, description="Minimum ramp-to-ramp lane changes")
    interchange_density: Optional[float] = Field(default=None, description="Interchange density in int/mi")
    basic_freeway_capacity: Optional[float] = Field(default=None, description="Basic freeway capacity in pc/h/ln")
    caf: Optional[float] = Field(default=None, description="Capacity adjustment factor")
    saf: Optional[float] = Field(default=None, description="Speed adjustment factor")
    version: Optional[str] = Field(default=None, description="HCM edition, e.g. 7.0 or 7.1")
    nw_rf: Optional[int] = Field(default=None, description="HCM 7.1: ramp-to-freeway weaving lanes")
    nw_fr: Optional[int] = Field(default=None, description="HCM 7.1: freeway-to-ramp weaving lanes")
    nw_rr: Optional[int] = Field(default=None, description="HCM 7.1: ramp-to-ramp weaving lanes")


class RampSegmentInput(BaseModel):
    """HCM Chapter 14 freeway merge/diverge (ramp) segment. Every field optional; the library fills HCM defaults."""
    ramp_type: Optional[str] = Field(default=None, description="OnRamp or OffRamp")
    ramp_side: Optional[str] = Field(default=None, description="Right or Left")
    ramp_lanes: Optional[int] = Field(default=None, description="Number of ramp lanes")
    freeway_lanes: Optional[int] = Field(default=None, description="Directional freeway lanes")
    freeway_ffs: Optional[float] = Field(default=None, description="Freeway free-flow speed in mi/h")
    ramp_ffs: Optional[float] = Field(default=None, description="Ramp free-flow speed in mi/h")
    accel_lane_length: Optional[float] = Field(default=None, description="Acceleration lane length in feet")
    accel_lane_length2: Optional[float] = Field(default=None, description="Second acceleration lane length in feet")
    decel_lane_length: Optional[float] = Field(default=None, description="Deceleration lane length in feet")
    decel_lane_length2: Optional[float] = Field(default=None, description="Second deceleration lane length in feet")
    freeway_demand: Optional[float] = Field(default=None, description="Freeway demand upstream of the ramp in veh/h")
    ramp_demand: Optional[float] = Field(default=None, description="Ramp demand in veh/h")
    phf: Optional[float] = Field(default=None, description="Peak hour factor")
    heavy_vehicle_pct: Optional[float] = Field(default=None, description="Freeway heavy-vehicle proportion (decimal)")
    ramp_heavy_vehicle_pct: Optional[float] = Field(default=None, description="Ramp heavy-vehicle proportion (decimal)")
    terrain: Optional[str] = Field(default=None, description="level / rolling / mountainous")
    adjacent_upstream: Optional[str] = Field(default=None, description="Adjacent upstream ramp: None / OnRamp / OffRamp")
    upstream_distance: Optional[float] = Field(default=None, description="Distance to upstream ramp in feet")
    upstream_ramp_flow: Optional[float] = Field(default=None, description="Upstream ramp demand in veh/h")
    adjacent_downstream: Optional[str] = Field(default=None, description="Adjacent downstream ramp: None / OnRamp / OffRamp")
    downstream_distance: Optional[float] = Field(default=None, description="Distance to downstream ramp in feet")
    downstream_ramp_flow: Optional[float] = Field(default=None, description="Downstream ramp demand in veh/h")
    caf: Optional[float] = Field(default=None, description="Capacity adjustment factor")
    saf: Optional[float] = Field(default=None, description="Speed adjustment factor")
    version: Optional[str] = Field(default=None, description="HCM edition, e.g. 7.0 or 7.1")


_WEAVING_ACCESSORS = [
    "speed_avg", "speed_weaving", "speed_nonweaving", "density", "capacity",
    "vc_ratio", "flow_weaving", "flow_nonweaving", "flow_total", "volume_ratio",
    "lc_min", "lc_all", "l_max", "is_weaving", "version",
]

_RAMP_ACCESSORS = [
    "density", "speed_ramp", "speed_avg", "speed_outer", "v12", "vr12",
    "flow_freeway", "flow_ramp", "capacity_freeway", "capacity_ramp",
    "vc_ratio", "demand_exceeds_capacity", "exceeds_max_desirable", "version",
]


@_guarded_run
def _run_weaving(inputs: Dict[str, Any]) -> Dict[str, Any]:
    seg = tl.WeavingSegment(**{k: v for k, v in inputs.items() if v is not None})
    los = seg.run_analysis()
    return {"success": True, "level_of_service": los, "results": _snapshot(seg, _WEAVING_ACCESSORS)}


@_guarded_run
def _run_ramp(inputs: Dict[str, Any]) -> Dict[str, Any]:
    seg = tl.RampSegment(**{k: v for k, v in inputs.items() if v is not None})
    los = seg.run_analysis()
    return {"success": True, "level_of_service": los, "results": _snapshot(seg, _RAMP_ACCESSORS)}


def _json_config_runner(cls_name: str, headline: list[str], run_method: str = "analyze"):
    """Runner for facilities whose binding takes one JSON config string. The Rust from_json/serde constructor is the input validator; to_json after the run carries the full result."""
    @_guarded_run
    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        obj = getattr(tl, cls_name)(json.dumps(inputs))
        getattr(obj, run_method)()
        result: Dict[str, Any] = {"success": True, "results": json.loads(obj.to_json())}
        result.update(_snapshot(obj, headline))
        return result
    return run


_run_twsc = _json_config_runner("Twsc", ["intersection_delay", "rank1_major_delay", "approach_delays"])
_run_awsc = _json_config_runner("Awsc", ["intersection_los", "intersection_delay"])
_run_roundabout = _json_config_runner("Roundabouts", ["intersection_los", "intersection_delay"])
_run_signalized = _json_config_runner("SignalizedIntersection", ["intersection_los", "intersection_delay"])
_run_urban_segment = _json_config_runner("UrbanSegment", ["los", "travel_speed", "running_time"])
_run_freeway_facility = _json_config_runner("FreewayFacility", ["facility_los", "los", "vc_ratio"], run_method="run_analysis")
_run_urban_facility = _json_config_runner("UrbanFacility", ["los", "poorest_segment_los", "critical_vc_ratio", "perception_score"])
_run_ramp_terminal = _json_config_runner("Interchange", ["interchange_los", "interchange_ett"])


def _json_config_snapshot_runner(cls_name: str, accessors: list[str], run_method: str = "run"):
    """Runner for JSON-config facilities whose binding has no to_json (the reliability engines): the result is a snapshot of their accessor set."""
    @_guarded_run
    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        obj = getattr(tl, cls_name)(json.dumps(inputs))
        getattr(obj, run_method)()
        return {"success": True, "results": _snapshot(obj, accessors)}
    return run


_run_freeway_reliability = _json_config_snapshot_runner("FreewayReliability", [
    "reliability_rating", "tti_mean", "misery_index", "semi_std_dev",
    "expected_vhd", "num_scenarios", "num_observations", "free_flow_travel_time_min",
])
_run_urban_reliability = _json_config_snapshot_runner("UrbanReliability", [
    "reliability_rating", "tti_mean", "total_vhd", "num_scenarios",
    "num_incidents", "num_weather_events", "mean_travel_time_s", "base_free_flow_travel_time_s",
])


class PedestrianWalkwayInput(BaseModel):
    """HCM Ch. 24 exclusive pedestrian facility (walkway or stairwell)."""
    total_walkway_width: float = Field(description="Total walkway width in feet")
    fixed_object_width: float = Field(description="Total fixed-object/shy width to subtract in feet")
    pedestrian_demand: Optional[float] = Field(default=None, description="Hourly pedestrian demand in p/h (alternative to peak_15min_volume)")
    peak_15min_volume: Optional[float] = Field(default=None, description="Peak 15-minute pedestrian volume in p")
    phf: Optional[float] = Field(default=None, description="Peak hour factor")
    pedestrian_speed: Optional[float] = Field(default=None, description="Average pedestrian speed in ft/min")
    facility_type: str = Field(description="walkway or stairwell")
    flow_type: str = Field(description="random or platoon")


class SharedUsePathPedestrianInput(BaseModel):
    """HCM Ch. 24 pedestrian LOS on a shared-use path (governed by bicycle events)."""
    bicycle_demand_same_direction: Optional[float] = Field(default=None, description="Same-direction bicycle demand in bikes/h")
    bicycle_demand_opposing: Optional[float] = Field(default=None, description="Opposing bicycle demand in bikes/h")
    phf: Optional[float] = Field(default=None, description="Peak hour factor")
    pedestrian_speed: Optional[float] = Field(default=None, description="Pedestrian speed in mi/h")
    bicycle_speed: Optional[float] = Field(default=None, description="Bicycle speed in mi/h")
    bicycle_flow_rate_same_direction: Optional[float] = Field(default=None, description="Direct same-direction bicycle flow rate override in bikes/h")
    bicycle_flow_rate_opposing: Optional[float] = Field(default=None, description="Direct opposing bicycle flow rate override in bikes/h")
    is_one_way: bool = Field(default=False, description="One-way path")


class OffStreetBicycleInput(BaseModel):
    """HCM Ch. 24 bicycle LOS (BLOS) on an off-street or shared-use path."""
    path_width: float = Field(description="Path width in feet")
    segment_length: float = Field(description="Segment length in miles")
    has_centerline: bool = Field(description="Path has a marked centerline")
    two_way_demand: Optional[float] = Field(default=None, description="Two-way path user demand in users/h")
    directional_split: Optional[float] = Field(default=None, description="Subject-direction share of demand (decimal)")
    phf: Optional[float] = Field(default=None, description="Peak hour factor")
    subject_demand: Optional[float] = Field(default=None, description="Direct subject-direction demand override in users/h")
    opposing_demand: Optional[float] = Field(default=None, description="Direct opposing demand override in users/h")
    is_one_way: bool = Field(default=False, description="One-way path")
    mode_splits: Optional[list[float]] = Field(default=None, description="Mode split proportions in HCM order: bicycle, pedestrian, runner, inline skater, child bicyclist")
    mode_speeds: Optional[list[float]] = Field(default=None, description="Mode mean speeds in mi/h, same order")
    mode_speed_sds: Optional[list[float]] = Field(default=None, description="Mode speed standard deviations in mi/h, same order")


def _kwargs_runner(cls_name: str, model: type[BaseModel], accessors: list[str]):
    """Runner for keyword-constructor facilities (the Ch. 24 ped/bike classes): analyze() returns the LOS letter directly."""
    @_guarded_run
    def run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        obj = getattr(tl, cls_name)(**model(**inputs).model_dump())
        los = obj.analyze()
        return {"success": True, "level_of_service": los, "results": _snapshot(obj, accessors)}
    return run


_run_ped_walkway = _kwargs_runner("ExclusivePedestrianFacility", PedestrianWalkwayInput, [
    "effective_width", "unit_flow_rate", "pedestrian_space", "vc_ratio", "flow_rate_15min",
])
_run_sup_pedestrian = _kwargs_runner("SharedUsePathPedestrian", SharedUsePathPedestrianInput, [
    "passing_events", "meeting_events", "total_events",
])
_run_offstreet_bicycle = _kwargs_runner("OffStreetBicycleFacility", OffStreetBicycleInput, [
    "blos_score", "effective_lanes", "active_passings_per_minute", "meetings_per_minute",
    "delayed_passings_per_minute", "total_probability_delayed_passing",
])


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
    "WeavingSegment": {
        "chapter": 13,
        "description": "Freeway weaving segment (HCM Ch. 13): component flows, lane changing, speeds, density, LOS",
        "library_class": "WeavingSegment",
        "input_model": WeavingSegmentInput,
        "run": _run_weaving,
    },
    "RampSegment": {
        "chapter": 14,
        "description": "Freeway merge/diverge segment (HCM Ch. 14): ramp influence-area flows, capacity checks, density, LOS",
        "library_class": "RampSegment",
        "input_model": RampSegmentInput,
        "run": _run_ramp,
    },
    "TWSC": {
        "chapter": 20,
        "description": "Two-way stop-controlled intersection (HCM Ch. 20): movement capacities, delays, queue lengths",
        "library_class": "Twsc",
        "input_example": {
            "demand": {"v2": 240.0, "v3": 40.0, "v4": 160.0, "v5": 300.0, "v7": 40.0, "v9": 120.0},
            "geometry": {"is_three_leg": True, "major_lanes_per_direction": 1, "major_right_turn_eb": "Shared", "major_right_turn_wb": "Shared", "minor_lanes_nb": "SingleShared"},
            "phf": None, "analysis_period_h": 0.25, "heavy_vehicle_pct": 10.0,
        },
        "run": _run_twsc,
    },
    "AWSC": {
        "chapter": 21,
        "description": "All-way stop-controlled intersection (HCM Ch. 21): departure headways, capacities, delays, LOS",
        "library_class": "Awsc",
        "input_example": {
            "eb": {"lanes": [{"volume_left": 50.0, "volume_through": 300.0, "volume_right": 0.0}], "heavy_vehicle_pct": 2.0},
            "wb": {"lanes": [{"volume_left": 0.0, "volume_through": 300.0, "volume_right": 100.0}], "heavy_vehicle_pct": 2.0},
            "nb": {"lanes": [], "heavy_vehicle_pct": 0.0},
            "sb": {"lanes": [{"volume_left": 100.0, "volume_through": 0.0, "volume_right": 50.0}], "heavy_vehicle_pct": 2.0},
            "phf": 0.95, "analysis_period_h": 0.25,
        },
        "run": _run_awsc,
    },
    "Roundabout": {
        "chapter": 22,
        "description": "Roundabout (HCM Ch. 22): entry capacities, control delays, queue lengths, LOS",
        "library_class": "Roundabouts",
        "input_example": {
            "nb": {"v_u": 30.0, "v_l": 105.0, "v_t": 210.0, "v_r": 50.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "None", "n_ped": 50.0},
            "sb": {"v_u": 20.0, "v_l": 175.0, "v_t": 95.0, "v_r": 580.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "NonYielding", "n_ped": 0.0},
            "eb": {"v_u": 50.0, "v_l": 190.0, "v_t": 280.0, "v_r": 85.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "None", "n_ped": 0.0},
            "wb": {"v_u": 20.0, "v_l": 110.0, "v_t": 395.0, "v_r": 610.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "Yielding", "n_ped": 0.0},
            "phf": 0.94, "analysis_period_h": 0.25,
        },
        "run": _run_roundabout,
    },
    "FreewayFacility": {
        "chapter": 10,
        "description": "Freeway facility core methodology (HCM Ch. 10): segment-by-period speed, density, and LOS matrices over the study period",
        "library_class": "FreewayFacility",
        "input_example": {"_note": "Segment-by-analysis-period facility config; see the library's FreewayFacilities example cases for the full shape. The serde constructor rejects malformed config with a descriptive error."},
        "run": _run_freeway_facility,
    },
    "UrbanSegment": {
        "chapter": 18,
        "description": "Urban street segment (HCM Ch. 18): running time, through delay, travel speed, LOS",
        "library_class": "UrbanSegment",
        "input_example": {"_note": "Segment config (geometry, signal timing at the boundary, demand); see the library's UrbanSegments example cases for the full shape."},
        "run": _run_urban_segment,
    },
    "SignalizedIntersection": {
        "chapter": 19,
        "description": "Signalized intersection (HCM Ch. 19): lane-group capacities, delays, approach and intersection LOS",
        "library_class": "SignalizedIntersection",
        "input_example": {"_note": "Intersection config (phasing, lane groups, demand); see the library's Signalized example cases for the full shape."},
        "run": _run_signalized,
    },
    "FreewayReliability": {
        "chapter": 11,
        "description": "Freeway reliability analysis (HCM Ch. 11): scenario generation and travel-time reliability measures (TTI, misery index, reliability rating)",
        "library_class": "FreewayReliability",
        "input_example": {"_note": "Reliability config (seed facility, demand/weather/incident scenario inputs); see the library's FreewayReliability example cases for the full shape."},
        "run": _run_freeway_reliability,
    },
    "UrbanFacility": {
        "chapter": 16,
        "description": "Urban street facility (HCM Ch. 16): aggregated travel speed, perception score, facility LOS",
        "library_class": "UrbanFacility",
        "input_example": {"_note": "Facility config (ordered segment configs with boundary signal timing); see the library's UrbanFacilities example cases for the full shape."},
        "run": _run_urban_facility,
    },
    "UrbanReliability": {
        "chapter": 17,
        "description": "Urban street reliability (HCM Ch. 17): scenario-based travel-time reliability measures",
        "library_class": "UrbanReliability",
        "input_example": {"_note": "Reliability config (base facility, demand patterns, weather, incidents); see the library's UrbanReliability example cases for the full shape."},
        "run": _run_urban_reliability,
    },
    "RampTerminal": {
        "chapter": 23,
        "description": "Ramp terminals and alternative intersections (HCM Ch. 23): O-D delays, experienced travel time, interchange LOS",
        "library_class": "Interchange",
        "input_example": {"_note": "Interchange config (signalized ramp-terminal geometry, phasing, O-D demand); see the library's RampTerminals example cases for the full shape."},
        "run": _run_ramp_terminal,
    },
    "PedestrianWalkway": {
        "chapter": 24,
        "description": "Exclusive pedestrian facility (HCM Ch. 24): walkway or stairwell pedestrian space and LOS",
        "library_class": "ExclusivePedestrianFacility",
        "input_model": PedestrianWalkwayInput,
        "run": _run_ped_walkway,
    },
    "SharedUsePathPedestrian": {
        "chapter": 24,
        "description": "Pedestrian LOS on a shared-use path (HCM Ch. 24): bicycle passing/meeting events govern",
        "library_class": "SharedUsePathPedestrian",
        "input_model": SharedUsePathPedestrianInput,
        "run": _run_sup_pedestrian,
    },
    "OffStreetBicycle": {
        "chapter": 24,
        "description": "Bicycle LOS on an off-street or shared-use path (HCM Ch. 24): BLOS score from passing and meeting events",
        "library_class": "OffStreetBicycleFacility",
        "input_model": OffStreetBicycleInput,
        "run": _run_offstreet_bicycle,
    },
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
            "error": f"Unknown facility_type {facility_type!r}. Valid types: {sorted(FACILITIES)}. Call describe_facility_inputs (no arguments) for the list with status, or with a facility_type for its input schema.",
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
    if "input_model" in entry:
        try:
            entry["input_model"](**inputs)
        except Exception as e:  # surface pydantic's message as a clean error
            return {"success": False, "error": f"Invalid inputs for {facility_type}: {e}"}
    # JSON-config facilities (input_example instead of input_model) are
    # validated by the library's from_json constructor inside the runner.
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
    """Describe the input schema for a facility type so a caller can construct a valid analyze_facility request; called without a facility_type, list every facility type with adapter status instead."""
    facility_type = data.get("facility_type")
    if not facility_type:
        return list_facility_types_function(data)
    if facility_type not in FACILITIES:
        return {"success": False, "error": f"Unknown facility_type {facility_type!r}. Valid types: {sorted(FACILITIES)}."}
    entry = FACILITIES[facility_type]
    if "run" not in entry:
        return {
            "success": False,
            "status": "adapter_pending",
            "error": f"{facility_type} has no adapter yet; its input model is not defined. Library class: {entry['library_class']} (HCM Ch. {entry['chapter']}).",
        }
    out: Dict[str, Any] = {
        "success": True,
        "facility_type": facility_type,
        "chapter": entry["chapter"],
    }
    if "input_model" in entry:
        out["fields"] = _describe_model(entry["input_model"])
    else:
        out["input_format"] = "structured config validated by the compute library; pass the same shape as the example"
        out["example"] = entry["input_example"]
    return out
