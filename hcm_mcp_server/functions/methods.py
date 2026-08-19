"""One analysis tool per HCM method the compute library implements.

Where ``analysis.py`` offers a single dispatching tool (``analyze_facility`` plus a ``facility_type``), this module offers a named tool per method, so an LLM sees the coverage in the tool list itself instead of having to discover it through a string argument. Both surfaces call the same verified ``transportations_library`` executors; neither replaces the other, and the ten tools of the published ablation surface are untouched.

The input to every tool here is the library's own fixture-schema JSON — the shape of the files under ``transportations-library/tests/ExampleCases/hcm/`` — never a second flattened schema invented for the MCP layer. Each method ships the example-problem fixture that validates it under ``hcm_mcp_server/data/examples/<method>.json``, and ``describe_method`` serves that fixture together with a key sketch so a caller can construct a request without reading the Rust bindings.

Three of the library's constructors (``WeavingSegment``, ``RampSegment``, ``BasicFreeways``) take keyword arguments rather than a JSON string, and their fixtures spell enums in PascalCase (``"OneSided"``, ``"OffRamp"``, ``"TwoLane"``). The fixture-to-keyword mapping here is ported from the library's own Python integration tests, which are the canonical readers of those files.

Domain refusals are the library's own: an off-domain specific-upgrade grade, a mixed-flow grade outside the digitised truck curves, a malformed serde config. Every runner surfaces the library's message verbatim in ``error`` rather than substituting one of its own.
"""

import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List

import transportations_library as tl

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "examples"


def _guarded(fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
    """Turn any library failure into a clean error dict. PyO3 raises Python ``ValueError`` for serde and domain rejections but a ``PanicException`` (a BaseException, not an Exception) for an internal panic, so the catch has to be wide."""
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:  # noqa: BLE001
            return {"success": False, "error": str(e)}
    return wrapper


def _snapshot(obj: Any, names: List[str]) -> Dict[str, Any]:
    """Read a list of accessors off a PyO3 object into a plain dict, skipping the ones that need arguments or raise before their step has run."""
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


def _config(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the fixture-schema config out of a tool-call payload. ``{"config": {...}}`` is the documented form; a bare fixture object is accepted too, because an LLM that has just been handed an example fixture tends to pass it directly.

    ``_source`` is dropped on the way through. It is the provenance line the shipped examples carry and the describe tool serves, never an engine input, and the mixed-flow configs are ``deny_unknown_fields`` so passing it back would be rejected.
    """
    config = data.get("config")
    if config is None:
        # A bare fixture object, minus the arguments that belong to the tool
        # rather than to the analysis. `method` has to come out here or it would
        # ride into a serde config that rejects unknown fields.
        bare = {k: v for k, v in data.items() if k not in ("config", "mode", "method")}
        config = bare or None
    if not isinstance(config, dict):
        raise ValueError("Missing 'config' object in the method's fixture schema. Call hcm_describe with this method name for the shape and a worked example fixture.")
    return _strip_provenance(config)


def _dispatch(runner: Callable, data: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
    """Extract the config and run, turning a missing or malformed payload into the same error dict shape a failed analysis returns. Callers of an MCP tool should never have to tell a raised exception from a refusal."""
    try:
        config = _config(data)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    return runner(config, **kwargs)


def _example(method: str) -> Dict[str, Any]:
    """Load a method's shipped example-problem fixture."""
    return json.loads((EXAMPLES_DIR / f"{method}.json").read_text())


def _strip_provenance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the ``_source`` provenance string the shipped fixtures carry. The library's serde configs ignore unknown fields, so this is cosmetic for the analysis and keeps echoed inputs readable."""
    return {k: v for k, v in config.items() if k != "_source"}


# ── Fixture-to-keyword adapters ──────────────────────────────────────────────
# Ported from the library's own Python integration tests, which are the
# canonical readers of these fixture files (test_chapter12/13/14_integration.py).

_WEAVING_TYPE = {"OneSided": "one_sided", "TwoSided": "two_sided"}
_WEAVING_FACILITY = {"Freeway": "freeway", "MultilaneOrCD": "multilane"}
_RAMP_TYPE = {"OnRamp": "on_ramp", "OffRamp": "off_ramp", "MajorMerge": "major_merge", "MajorDiverge": "major_diverge"}
_RAMP_LANES = {"OneLane": 1, "TwoLane": 2}
_ADJACENT = {"None": "none", "OnRamp": "on_ramp", "OffRamp": "off_ramp"}


def _edition_7_1(segment: Any) -> Dict[str, Any]:
    """The HCM 7.1 result block, present only when the segment ran under 7.1. Its equations differ from the 7th Edition ones rather than adjusting them, so the two sets of numbers are reported side by side instead of merged."""
    try:
        payload = segment.analysis_v7_1()
    except Exception:
        return {}
    return {"analysis_v7_1": json.loads(payload)} if payload else {}


def _weaving_segment(config: Dict[str, Any]) -> Any:
    d = config
    kwargs = dict(
        weaving_type=_WEAVING_TYPE[d["weaving_type"]],
        facility_type=_WEAVING_FACILITY[d["facility_type"]],
        length_short=d["length_short"],
        num_lanes=d["num_lanes"],
        num_weaving_lanes=d["num_weaving_lanes"],
        ffs=d["ffs"],
        lc_rf=d["lc_rf"],
        lc_fr=d["lc_fr"],
        lc_rr=d["lc_rr"],
        interchange_density=d["interchange_density"],
        basic_freeway_capacity=d["basic_freeway_capacity"],
    )
    # The service-volume fixtures carry geometry only: demand, PHF and the
    # adjustment factors belong to the operational problem, not to the segment
    # template a service-flow solver sweeps.
    for key in ("v_ff", "v_fr", "v_rf", "v_rr", "phf", "heavy_vehicle_pct", "caf", "saf", "nw_rf", "nw_fr", "nw_rr"):
        if d.get(key) is not None:
            kwargs[key] = d[key]
    if d.get("terrain") is not None:
        kwargs["terrain"] = d["terrain"].lower()
    if d.get("version") is not None:
        kwargs["version"] = d["version"]
    return tl.WeavingSegment(**kwargs)


def _ramp_segment(config: Dict[str, Any]) -> Any:
    d = config
    kwargs = dict(
        ramp_type=_RAMP_TYPE[d["ramp_type"]],
        ramp_side=d["ramp_side"].lower(),
        ramp_lanes=_RAMP_LANES[d["ramp_lanes"]] if isinstance(d["ramp_lanes"], str) else d["ramp_lanes"],
        freeway_lanes=d["freeway_lanes"],
        freeway_ffs=d["freeway_ffs"],
        ramp_ffs=d["ramp_ffs"],
        freeway_demand=d["freeway_demand"],
        ramp_demand=d["ramp_demand"],
        phf=d["phf"],
        heavy_vehicle_pct=d["heavy_vehicle_pct"],
        terrain=d["terrain"].lower(),
        adjacent_upstream=_ADJACENT[d["adjacent_upstream"]],
        adjacent_downstream=_ADJACENT[d["adjacent_downstream"]],
        caf=d["caf"],
        saf=d["saf"],
    )
    for key in ("accel_lane_length", "accel_lane_length2", "decel_lane_length", "decel_lane_length2",
                "ramp_heavy_vehicle_pct", "upstream_distance", "upstream_ramp_flow",
                "downstream_distance", "downstream_ramp_flow", "version"):
        if d.get(key) is not None:
            kwargs[key] = d[key]
    return tl.RampSegment(**kwargs)


# ── Chapter 10: freeway facilities ───────────────────────────────────────────

_FACILITY_MATRICES = ["speed", "density_veh", "los", "capacity", "demand", "volume_served", "dc_ratio", "queue_length_ft"]
_FACILITY_SUMMARY = ["num_segments", "num_periods", "total_length_mi", "oversaturated", "overall_speed", "overall_density_veh"]


@_guarded
def _run_freeway_facility(config: Dict[str, Any]) -> Dict[str, Any]:
    fac = tl.FreewayFacility(json.dumps(config))
    fac.run_analysis()
    results = _snapshot(fac, _FACILITY_SUMMARY)
    results.update(_snapshot(fac, _FACILITY_MATRICES))
    results["facility_los"] = [fac.facility_los(p) for p in range(fac.num_periods)]
    results["facility_speed"] = [fac.facility_speed(p) for p in range(fac.num_periods)]
    results["facility_density_veh"] = [fac.facility_density_veh(p) for p in range(fac.num_periods)]
    return {"success": True, "results": results}


def analyze_freeway_facility_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 10 freeway facilities, run through the Chapter 25 computational engine.

    Input is the freeway-facility fixture schema: an ordered list of segments (basic, weaving, merge, diverge) and a demand matrix over the 15-minute analysis periods of the study period. Worked example: ``analyze_freeway_facility`` ships HCM Chapter 25, Example Problem 1 (Exhibits 25-43 through 25-52), an eleven-segment undersaturated urban freeway over five analysis periods.

    Results are segment-by-period matrices indexed ``[segment][period]``: ``speed`` in mi/h, ``density_veh`` in veh/mi/ln, ``capacity``, ``demand`` and ``volume_served`` in veh/h, ``dc_ratio`` the demand-to-capacity ratio, ``queue_length_ft`` the back-of-queue extent where a segment is oversaturated, and ``los`` the per-cell letter. ``facility_los``, ``facility_speed`` and ``facility_density_veh`` are one value per analysis period, aggregated across the whole facility by Equations 10-1 through 10-3. ``oversaturated`` reports whether any cell queued, which is what selects the oversaturated (Chapter 25 Section 3) path.
    """
    return _dispatch(_run_freeway_facility, data)


_ML_MATRICES = ["ml_speed", "ml_capacity", "ml_density_veh", "ml_dc_ratio", "ml_friction_active", "gp_speed", "gp_capacity", "gp_density_veh"]


@_guarded
def _run_managed_lanes(config: Dict[str, Any]) -> Dict[str, Any]:
    fac = tl.ManagedLaneFacility(json.dumps(config))
    fac.run_analysis()
    results = _snapshot(fac, ["num_segments", "num_periods"])
    results.update(_snapshot(fac, _ML_MATRICES))
    results["facility_los"] = [fac.facility_los(p) for p in range(fac.num_periods)]
    results["facility_speed"] = [fac.facility_speed(p) for p in range(fac.num_periods)]
    results["facility_density_veh"] = [fac.facility_density_veh(p) for p in range(fac.num_periods)]
    return {"success": True, "results": results}


def analyze_managed_lanes_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 10 managed-lane freeway facility (Steps A-9, A-13, A-14 and A-17 of the Chapter 25 procedure).

    Input is the managed-lane facility fixture schema: a general-purpose lane group and a parallel managed-lane lane group over the same segments and analysis periods, plus the separation type (marking, buffer or barrier) that decides whether adjacent-friction applies. Worked example: ``analyze_managed_lanes`` ships HCM Chapter 25, Example Problem 5 (Exhibits 25-81 through 25-87), a marking-separated continuous-access HOV lane alongside the Example Problem 1 facility.

    Results carry both lane groups. ``ml_*`` and ``gp_*`` are ``[segment][period]`` matrices for the managed and general-purpose lanes: speed in mi/h, capacity in veh/h, density in veh/mi/ln, ``ml_dc_ratio`` the managed-lane demand-to-capacity ratio. ``ml_friction_active`` flags the cells where adjacent general-purpose density exceeded the 35 pc/mi/ln threshold and the Equation 12-18/12-19 speed reduction was applied. ``facility_los``, ``facility_speed`` and ``facility_density_veh`` are per-period values for the two lane groups combined by lane-mile weighting.
    """
    return _dispatch(_run_managed_lanes, data)


@_guarded
def _run_planning_facility(config: Dict[str, Any]) -> Dict[str, Any]:
    fac = tl.PlanningFacility(json.dumps(config))
    fac.run_analysis()
    results = _snapshot(fac, ["num_sections", "total_length_mi"])
    # The planning binding exposes per-period accessors but no period count, and
    # an out-of-range period PANICS in Rust rather than raising, so the count has
    # to come from to_json() -- never from probing until the accessor refuses.
    state = json.loads(fac.to_json())
    periods = len(state.get("facility_results", []))
    results["num_periods"] = periods
    results["facility_results"] = state.get("facility_results")
    results["section_results"] = state.get("section_results")
    for accessor in ("facility_los", "facility_speed", "facility_density"):
        results[accessor] = [getattr(fac, accessor)(p) for p in range(periods)]
    for accessor in ("section_speed", "section_density", "dc_ratio"):
        results[accessor] = [[getattr(fac, accessor)(s, p) for p in range(periods)] for s in range(fac.num_sections)]
    return {"success": True, "results": results}


def analyze_planning_facility_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 25, Section 6 planning-level freeway facility method.

    Input is the planning-facility fixture schema: the facility described as sections rather than the full operational segment set, with demand by analysis period. Worked example: ``analyze_planning_facility`` ships HCM Chapter 25, Example Problem 6 (Exhibits 25-88 through 25-96), the planning-level treatment of the Example Problem 1 facility.

    Results are ``[section][period]`` matrices for ``section_speed`` (mi/h), ``section_density`` (veh/mi/ln) and ``dc_ratio``, plus per-period ``facility_speed``, ``facility_density`` and ``facility_los``. This is a screening method: it trades the segment-level detail of ``hcm_analyze_freeway_facility`` for far fewer inputs, and its results are not interchangeable with the operational ones.
    """
    return _dispatch(_run_planning_facility, data)


# ── Chapter 11: freeway reliability ──────────────────────────────────────────

_FREEWAY_RELIABILITY = [
    "reliability_rating", "tti_mean", "misery_index", "semi_std_dev", "expected_vhd",
    "num_scenarios", "num_observations", "free_flow_travel_time_min",
]


@_guarded
def _run_freeway_reliability(config: Dict[str, Any]) -> Dict[str, Any]:
    rel = tl.FreewayReliability(json.dumps(config))
    rel.run()
    results = _snapshot(rel, _FREEWAY_RELIABILITY)
    results["tti_percentile"] = {str(p): rel.tti_percentile(p) for p in (50, 80, 95)}
    return {"success": True, "results": results}


def analyze_freeway_reliability_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 11 freeway travel-time reliability.

    Input is the freeway-reliability fixture schema: a seed freeway facility (the Chapter 10 shape), the reliability reporting period, and the demand, weather and incident inputs that drive scenario generation. Worked example: ``analyze_freeway_reliability`` ships HCM Chapter 25, Example Problem 7 (Exhibits 25-97 through 25-105), a reliability evaluation of an existing freeway facility.

    ``num_scenarios`` and ``num_observations`` describe the generated scenario set. ``tti_mean`` is the mean travel time index over all scenario-periods and ``tti_percentile`` gives the 50th, 80th and 95th percentile TTI (the 95th is the planning time index). ``misery_index`` is the mean of the worst 5% of travel times, ``semi_std_dev`` the semi-standard deviation about free-flow, ``reliability_rating`` the percentage of vehicle-miles travelled at a TTI at or below 1.33, and ``expected_vhd`` the expected vehicle-hours of delay over the reporting period. The published Chapter 25 values come from FREEVAL's own Monte Carlo stream, so central measures reproduce within a band rather than exactly.
    """
    return _dispatch(_run_freeway_reliability, data)


# ── Chapter 12: basic freeway and multilane segments ─────────────────────────

_BASIC_FREEWAY_KEYS = ("bffs", "lane_width", "lane_count", "lc_r", "lc_l", "trd", "apd", "grade",
                       "terrain_type", "speed_limit", "phf", "p_t", "sut_percentage",
                       "demand_flow_i", "length", "highway_type", "city_type")


@_guarded
def _run_basic_freeway(config: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(config)
    d.setdefault("lane_width", d.get("lw"))
    kwargs = {k: d[k] for k in _BASIC_FREEWAY_KEYS if d.get(k) is not None}
    seg = tl.BasicFreeways(**kwargs)
    los = seg.run_operational_analysis()
    results = _snapshot(seg, ["ffs", "capacity", "adjusted_capacity", "speed", "density", "vc_ratio", "e_t", "f_hv", "lane_count"])
    results["level_of_service"] = los
    return {"success": True, "level_of_service": los, "results": results}


def analyze_basic_freeway_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 12 basic freeway and multilane highway segments.

    Input is the basic-freeway fixture schema (``bffs``, ``lw``/``lane_width``, ``lane_count``, lateral clearances, ``trd``, ``grade``, ``terrain_type``, ``speed_limit``, ``phf``, ``p_t``, ``sut_percentage``, ``demand_flow_i``, ``length``, ``highway_type``). Worked example: ``analyze_basic_freeway`` ships HCM Chapter 26, Example Problem 1 (four-lane freeway segment, published FFS 60.8 mi/h, capacity 2,308 pc/h/ln, density 18.8 pc/mi/ln, LOS C).

    ``ffs`` is the adjusted free-flow speed in mi/h, ``capacity`` and ``adjusted_capacity`` are pc/h/ln, ``speed`` is space mean speed from the speed-flow curve in mi/h, ``density`` is pc/mi/ln, ``vc_ratio`` the volume-to-capacity ratio, and ``level_of_service`` the Exhibit 12-15 letter. ``e_t`` and ``f_hv`` expose the heavy-vehicle equivalence and adjustment that converted demand to passenger-car units.

    ``sut_percentage`` selects the heavy-vehicle exhibit: 0 (the default) reads the general-terrain Exhibit 12-25 and makes grade and length irrelevant, while 30, 50 and 70 select the specific-upgrade Exhibits 12-26, 12-27 and 12-28. The library interpolates those over grades up to 6% and refuses inputs outside that domain with its own message.
    """
    return _dispatch(_run_basic_freeway, data)


# ── Chapter 13: weaving segments ─────────────────────────────────────────────

_WEAVING_ACCESSORS = [
    "speed_avg", "speed_weaving", "speed_nonweaving", "density", "capacity", "vc_ratio",
    "flow_weaving", "flow_nonweaving", "flow_total", "volume_ratio", "lc_min", "lc_all",
    "l_max", "is_weaving", "version",
]


@_guarded
def _run_weaving(config: Dict[str, Any]) -> Dict[str, Any]:
    seg = _weaving_segment(config)
    los = seg.run_analysis()
    results = _snapshot(seg, _WEAVING_ACCESSORS)
    results.update(_edition_7_1(seg))
    return {"success": True, "level_of_service": los, "results": results}


def analyze_weaving_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 13 freeway weaving segments, in both editions the library carries.

    Input is the weaving fixture schema (``weaving_type`` OneSided/TwoSided, ``facility_type`` Freeway/MultilaneOrCD, ``length_short`` in feet, ``num_lanes``, ``num_weaving_lanes``, ``ffs``, the four component demands ``v_ff``/``v_fr``/``v_rf``/``v_rr``, ``phf``, ``heavy_vehicle_pct``, ``terrain``, the minimum lane changes ``lc_rf``/``lc_fr``/``lc_rr``, ``interchange_density``, ``basic_freeway_capacity``, ``caf`` and ``saf``). Worked example: ``analyze_weaving`` ships HCM Chapter 27, Example Problem 1 (major weaving segment, published weaving flow 1,995 pc/h, capacity 8,038 pc/h, LOS C).

    Set ``"version": "7.1"`` to run the HCM 7.1 weaving procedure, which adds the per-movement weaving-lane counts ``nw_rf``, ``nw_fr`` and ``nw_rr``; omitting ``version`` runs HCM 7 (the library's default).

    ``flow_weaving``, ``flow_nonweaving`` and ``flow_total`` are pc/h and ``volume_ratio`` their ratio. ``lc_min`` and ``lc_all`` are minimum and total lane changes per hour, ``l_max`` the maximum weaving length in feet beyond which the segment behaves as a basic segment (``is_weaving`` false). ``speed_weaving``, ``speed_nonweaving`` and ``speed_avg`` are mi/h, ``density`` pc/mi/ln, and ``level_of_service`` the Exhibit 13-6 letter.
    """
    return _dispatch(_run_weaving, data)


# ── Chapter 14: merge and diverge segments ───────────────────────────────────

_RAMP_ACCESSORS = [
    "density", "speed_ramp", "speed_avg", "speed_outer", "v12", "vr12", "p_f",
    "flow_freeway", "flow_ramp", "capacity_freeway", "capacity_ramp", "vc_ratio",
    "demand_exceeds_capacity", "exceeds_max_desirable", "version",
]


@_guarded
def _run_merge_diverge(config: Dict[str, Any]) -> Dict[str, Any]:
    seg = _ramp_segment(config)
    los = seg.run_analysis()
    results = _snapshot(seg, _RAMP_ACCESSORS)
    results.update(_edition_7_1(seg))
    return {"success": True, "level_of_service": los, "results": results}


def analyze_merge_diverge_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 14 freeway merge and diverge (ramp) segments, in both editions the library carries.

    Input is the merge/diverge fixture schema (``ramp_type`` OnRamp/OffRamp/MajorMerge/MajorDiverge, ``ramp_side``, ``ramp_lanes`` OneLane/TwoLane, ``freeway_lanes``, ``freeway_ffs``, ``ramp_ffs``, acceleration or deceleration lane lengths in feet, ``freeway_demand`` and ``ramp_demand`` in veh/h, ``phf``, ``heavy_vehicle_pct``, ``terrain``, the adjacent-ramp fields, ``caf`` and ``saf``). Worked example: ``analyze_merge_diverge`` ships HCM Chapter 28, Example Problem 2, first off-ramp (two adjacent single-lane right-hand off-ramps on a six-lane freeway).

    Set ``"version": "7.1"`` to run the HCM 7.1 merge/diverge procedure; omitting ``version`` runs HCM 7.

    ``v12`` is the flow entering lanes 1 and 2 in pc/h and ``vr12`` the flow in the ramp influence area; ``p_f`` is the proportion of freeway flow remaining in those lanes. ``flow_freeway`` and ``flow_ramp`` are pc/h, ``capacity_freeway`` and ``capacity_ramp`` their capacities, and ``vc_ratio`` the governing ratio. ``density`` is the ramp-influence-area density in pc/mi/ln that sets ``level_of_service`` by Exhibit 14-3. ``demand_exceeds_capacity`` and ``exceeds_max_desirable`` are the Chapter 14 capacity and desirable-flow checks; when either is set, the density and LOS describe a condition the method does not fully cover.
    """
    return _dispatch(_run_merge_diverge, data)


# ── Chapter 15: two-lane highways ────────────────────────────────────────────

@_guarded
def _run_two_lane_highway(config: Dict[str, Any]) -> Dict[str, Any]:
    hwy = tl.TwoLaneHighways(
        segments=[_two_lane_segment(s) for s in config["segments"]],
        lane_width=config.get("lane_width", 12.0),
        shoulder_width=config.get("shoulder_width", 6.0),
        apd=config.get("apd", 0.0),
        pmhvfl=config.get("pmhvfl", 0.0),
        l_de=config.get("l_de", 0.0),
    )
    # The Chapter 15 engine is stateful and its step order is load-bearing. The
    # passing-lane branch (passing_type 2) uses a different follower-density
    # step than the passing-constrained and passing-zone branches, and the
    # adjustment step only applies to the latter. This mirrors the sequence in
    # chapter15.py, which is the surface the ablation servers use.
    segments = []
    facility_speed = total_length = weighted_spl = 0.0
    for i in range(hwy.num_segments):
        hwy.identify_vertical_class(i)
        _, _, capacity = hwy.determine_demand_flow(i)
        hwy.determine_vertical_alignment(i)
        hwy.determine_free_flow_speed(i)
        avg_speed, _ = hwy.estimate_average_speed(i)
        hwy.estimate_percent_followers(i)
        segment = hwy.segments[i]
        if segment.passing_type == 2:
            fd, fd_mid = hwy.determine_follower_density_pl(i)
        else:
            fd = hwy.determine_follower_density_pc_pz(i)
            fd_mid = None
            adjusted = hwy.determine_adjustment_to_follower_density(i)
            if adjusted > 0.0:
                fd = adjusted
        facility_speed += avg_speed * segment.length
        total_length += segment.length
        weighted_spl += segment.spl * segment.length
        segments.append({
            "segment_index": i,
            "passing_type": segment.passing_type,
            "length_mi": segment.length,
            "free_flow_speed": segment.ffs,
            "average_speed": avg_speed,
            "percent_followers": segment.percent_followers,
            "follower_density": fd,
            "follower_density_mid": fd_mid,
            "capacity": capacity,
            "level_of_service": hwy.determine_segment_los(i, avg_speed, int(capacity)),
        })
    # Equation 15-39 over the ADJUSTED segment densities, which is what the
    # library's own River Falls gate test uses. Exhibit 15-6 then splits on the
    # POSTED SPEED LIMIT, not on the computed average speed -- the second
    # argument is a speed limit. Length-weighting it reduces to the common limit
    # when every segment is posted the same, which is the usual case.
    facility_fd = hwy.determine_facility_follower_density()
    facility_spl = weighted_spl / total_length if total_length else 0.0
    facility_los = hwy.determine_facility_los(facility_fd, facility_spl)
    return {
        "success": True,
        "level_of_service": facility_los,
        "results": {
            "facility_follower_density": facility_fd,
            "facility_level_of_service": facility_los,
            "facility_average_speed": facility_speed / total_length if total_length else 0.0,
            "facility_posted_speed_limit": facility_spl,
            "total_length": hwy.total_length,
            "num_segments": hwy.num_segments,
            "segments": segments,
        },
    }


def _two_lane_segment(s: Dict[str, Any]) -> Any:
    """Build one Chapter 15 segment from the fixture schema. Lengths are miles at the segment level and FEET inside a subsegment, and ``phv``/``sup_ele`` are percents, which is what the fixture files encode."""
    kwargs = {k: s[k] for k in (
        "passing_type", "length", "grade", "spl", "is_hc", "volume", "volume_op",
        "flow_rate", "flow_rate_o", "capacity", "ffs", "avg_speed", "vertical_class",
        "phf", "phv", "pf", "fd", "fd_mid", "hor_class",
    ) if s.get(k) is not None}
    subsegments = s.get("subsegments") or []
    if subsegments:
        kwargs["subsegments"] = [
            tl.SubSegment(**{k: ss[k] for k in ("length", "avg_speed", "hor_class", "design_rad", "central_angle", "sup_ele") if ss.get(k) is not None})
            for ss in subsegments
        ]
    return tl.Segment(**kwargs)


def analyze_two_lane_highway_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 15 two-lane highway facilities, taking the library's fixture schema.

    This is the fixture-schema entry point. The per-step ``chapter15_*`` tools and the ``TwoLaneHighway`` branch of ``analyze_facility`` are unchanged and remain the published surface; this tool exists so a caller can hand over an example-case file unmodified.

    Input is the two-lane fixture schema: ``segments`` (each with ``passing_type`` 0/1/2 for passing-constrained, passing-zone or passing-lane, ``length`` in MILES, ``grade``, ``spl``, ``volume``, ``volume_op``, ``phf``, ``phv``, ``vertical_class``, ``is_hc`` and an optional ``subsegments`` list whose ``length`` is in FEET), plus facility-level ``lane_width``, ``shoulder_width``, ``apd``, ``pmhvfl`` and ``l_de``. Worked example: ``analyze_two_lane_highway`` ships HCM Chapter 15, Example Problem 1.

    Two units in this schema are easy to get wrong and fail silently. ``spl`` is the POSTED speed limit, not free-flow speed (base FFS is derived as 1.14 x spl), and ``phv`` and ``sup_ele`` are PERCENTS, so 5% is 5.0 and not 0.05. A fractional ``phv`` lands in the lowest lookup bucket and produces a plausible but wrong follower density. Horizontal-curve subsegments are also ignored unless the segment sets ``is_hc`` true.

    ``facility_follower_density`` is followers per mile per lane over the whole facility, the Chapter 15 service measure, and ``facility_level_of_service`` its Exhibit 15-6 letter. ``segments`` carries the per-segment LOS in facility order, and ``summary`` the library's own per-segment record of flow rate, free-flow speed, average speed, percent followers and follower density.
    """
    return _dispatch(_run_two_lane_highway, data)


@_guarded
def _run_bicycle_los(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "results": json.loads(tl.analyze_bicycle_los(json.dumps(config)))}


def analyze_bicycle_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 15, Section 4 bicycle mode on a two-lane or multilane highway segment.

    The bicycle method's input set is not the motorized method's, which is why this is a separate method rather than a mode argument on ``analyze_two_lane_highway``: pavement rating and on-highway parking govern the score, and segment length does not enter it at all.

    Input is the bicycle-LOS fixture schema: ``lane_width`` and ``shoulder_width`` in feet, ``speed_limit`` in mi/h, ``num_lanes`` (directional through lanes, 1 for a two-lane highway), ``pavement_condition`` on the FHWA 5-point scale where 1 is very poor and 5 very good, ``hourly_volume`` in veh/h, ``phf``, and ``heavy_vehicle_pct`` and ``pct_on_highway_parking``. All nine are required; none of them defaults. Worked example: ``analyze_bicycle_los`` ships the HCM two-lane highway widening example, a 12 ft lane with a 2 ft shoulder at a posted 50 mi/h whose published BLOS score is 5.90 at LOS F, and 3.58 at LOS D after the project widens the shoulder to 6 ft, raises the limit to 55 mi/h and repaves.

    ``heavy_vehicle_pct`` and ``pct_on_highway_parking`` are DECIMALS here, 0.05 for 5%, which is the opposite of the ``phv`` percent convention in the same chapter's motorized schema. A percent passed as a percent does not raise; it drives the score far past the LOS F threshold.

    ``flow_rate_outside_lane`` is the Step 2 directional flow in the outside lane in veh/h, ``effective_width`` the Step 3 effective width of that lane in feet (which shoulder width moves in steps, at 4 ft and 8 ft, rather than continuously), ``effective_speed_factor`` the Step 4 term from the posted limit, and ``blos_score`` the Equation 15-47 score whose Exhibit 15-7 letter is ``los``. The score runs the opposite way to a grade: lower is better, and a project succeeds by reducing it.

    Equation 15-46 takes ln(speed_limit - 20), so a posted limit of 20 mi/h or below has no defined effective speed factor. The library returns ``effective_speed_factor`` and ``blos_score`` as null there while still reporting an ``los`` letter from the raw value, and that letter is meaningless. Treat a null score as the answer, not the letter beside it.
    """
    return _dispatch(_run_bicycle_los, data)


# ── Chapters 16-18: urban streets ────────────────────────────────────────────

_URBAN_FACILITY = ["los", "poorest_segment_los", "travel_speed_mph", "base_free_flow_speed_mph",
                   "critical_vc_ratio", "perception_score", "spatial_stop_rate", "num_segments",
                   "length_ft", "segment_travel_speeds", "spillback_flags"]


@_guarded
def _run_urban_facility(config: Dict[str, Any], mode: str = "aggregate") -> Dict[str, Any]:
    fac = tl.UrbanFacility(json.dumps(config))
    if mode == "analyze":
        fac.analyze()
    elif mode == "aggregate":
        fac.aggregate()
    else:
        raise ValueError(f"mode must be 'aggregate' or 'analyze', got {mode!r}")
    return {"success": True, "level_of_service": fac.los, "results": _snapshot(fac, _URBAN_FACILITY)}


def analyze_urban_facility_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 16 urban street facilities.

    Input is the urban-facility fixture schema: an ordered list of urban street segments in the Chapter 18 shape, in one direction of travel, with the boundary intersection timing that links them. Worked example: ``analyze_urban_facility`` ships HCM Chapter 29, Example Problem 1 (Exhibit 29-49), the eastbound direction of a five-segment urban street.

    The optional ``mode`` argument selects between the library's two entry points, which are alternatives rather than stages. ``aggregate`` (the default, and what the shipped example needs) combines per-segment measures the config already supplies, which is how the published example problem is set up. ``analyze`` instead runs the Chapter 18 segment engine over each segment first and then aggregates, which is what a config carrying raw segment inputs needs.

    ``travel_speed_mph`` is the facility travel speed aggregated over the segments by Equations 16-2 through 16-4 and ``base_free_flow_speed_mph`` the length-weighted base free-flow speed; their ratio sets ``los`` through Exhibit 16-6. ``poorest_segment_los`` reports the worst individual segment, which the HCM asks analysts to check alongside the facility letter. ``critical_vc_ratio`` is the largest through-movement volume-to-capacity ratio at any boundary intersection, ``spatial_stop_rate`` stops per mile, ``perception_score`` the traveller-perception score, and ``spillback_flags`` marks segments where a downstream queue reached the upstream intersection, which invalidates the automobile method for those segments.
    """
    return _dispatch(_run_urban_facility, data, mode=data.get("mode", "aggregate"))


_URBAN_RELIABILITY = ["reliability_rating", "tti_mean", "total_vhd", "num_scenarios",
                      "num_incidents", "num_weather_events", "mean_travel_time_s",
                      "base_free_flow_travel_time_s"]


@_guarded
def _run_urban_reliability(config: Dict[str, Any]) -> Dict[str, Any]:
    rel = tl.UrbanReliability(json.dumps(config))
    rel.run()
    results = _snapshot(rel, _URBAN_RELIABILITY)
    results["tti_percentile"] = {str(p): rel.tti_percentile(p) for p in (50, 80, 95)}
    return {"success": True, "results": results}


def analyze_urban_reliability_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 17 urban street travel-time reliability.

    Input is the urban-reliability fixture schema: a base urban street facility in the Chapter 16 shape, the reliability reporting period (study hours, days of week, weeks of year), and the demand-pattern, weather and incident inputs that generate scenarios. Worked example: ``analyze_urban_reliability`` ships HCM Chapter 29, Example Problem 4 (Exhibit 29-73), a three-mile Lincoln, Nebraska principal arterial over weekday mornings for one year.

    ``num_scenarios`` is the generated scenario count (the published example generates 3,120), with ``num_incidents`` and ``num_weather_events`` describing how many drew a non-base condition. ``tti_mean`` is the mean travel time index and ``tti_percentile`` the 50th, 80th and 95th percentiles, the last being the planning time index. ``reliability_rating`` is the percentage of trips at a TTI at or below 1.33 and ``total_vhd`` the vehicle-hours of delay over the reporting period. Scenario generation is a seeded Monte Carlo process and the HCM notes the streams are software-specific, so central measures reproduce within a band rather than exactly.
    """
    return _dispatch(_run_urban_reliability, data)


_URBAN_SEGMENT = ["los", "travel_speed_mph", "running_time_s", "running_speed_mph",
                  "through_delay_s", "base_free_flow_speed_mph", "free_flow_speed_mph",
                  "proportion_arriving_green", "spatial_stop_rate", "full_stop_rate",
                  "perception_score", "vc_ratio", "segment_length_ft", "through_demand_veh_h",
                  "access_point_delay_total_s", "speed_limit_mph"]


@_guarded
def _run_urban_segment(config: Dict[str, Any]) -> Dict[str, Any]:
    seg = tl.UrbanSegment(json.dumps(config))
    seg.analyze()
    return {"success": True, "level_of_service": seg.los, "results": _snapshot(seg, _URBAN_SEGMENT)}


def analyze_urban_segment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 18 urban street segments.

    Input is the urban-segment fixture schema: segment geometry and speed limit, the access points along it, the through demand, and the signal timing at the boundary intersection. Worked example: ``analyze_urban_segment`` ships HCM Chapter 30, Example Problem 1 eastbound (Exhibit 30-36), whose published base FFS is 40.78 mi/h and running time 33.54 s.

    ``base_free_flow_speed_mph`` comes from Exhibit 18-11 and ``free_flow_speed_mph`` applies the signal-spacing adjustment. ``running_time_s`` is segment running time excluding control delay, split out as ``running_speed_mph``, with ``access_point_delay_total_s`` the delay from mid-segment access points. ``through_delay_s`` is control delay at the boundary intersection and ``proportion_arriving_green`` the platoon-arrival term that drives it. ``travel_speed_mph`` divided by ``base_free_flow_speed_mph`` sets ``los`` through Exhibit 18-8. ``spatial_stop_rate`` is stops per mile and ``full_stop_rate`` stops per vehicle.
    """
    return _dispatch(_run_urban_segment, data)


# ── Chapters 19-22: intersections ────────────────────────────────────────────

@_guarded
def _run_signalized(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.SignalizedIntersection(json.dumps(config))
    ix.analyze()
    results = {
        "intersection_delay_s": ix.intersection_delay_s,
        "intersection_los": ix.intersection_los,
        "critical_vc_ratio": ix.critical_vc_ratio,
        "cycle_length_s": ix.cycle_length_s,
        "num_lane_groups": ix.num_lane_groups,
        "lane_groups": json.loads(ix.lane_groups_json()),
    }
    approaches = {}
    for direction in ("EB", "WB", "NB", "SB"):
        try:
            approaches[direction] = {"delay_s": ix.approach_delay_s(direction), "los": ix.approach_los(direction)}
        except Exception:
            continue
    results["approaches"] = approaches
    return {"success": True, "level_of_service": ix.intersection_los, "results": results}


def analyze_signalized_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 19 signalized intersections, motorized vehicle mode.

    Input is the signalized fixture schema: lane groups by approach with their demand, saturation flow adjustment inputs and turn treatments, plus the phasing, cycle length and detector settings. Worked example: ``analyze_signalized`` ships HCM Chapter 31, Example Problem 1 (Exhibit 31-81), whose published intersection control delay is 45.9 s/veh at LOS D.

    ``intersection_delay_s`` is volume-weighted control delay in s/veh and ``intersection_los`` its Exhibit 19-8 letter; ``approaches`` carries the same pair per approach. ``critical_vc_ratio`` is the critical-movement volume-to-capacity ratio, which can flag an over-capacity intersection even when the delay-based letter looks acceptable. ``cycle_length_s`` is the cycle actually used, which for an actuated controller is the one the library estimated rather than an input. ``lane_groups`` carries the full per-lane-group record: saturation flow rate, capacity, volume-to-capacity ratio, uniform and incremental delay, queue measures and LOS.
    """
    return _dispatch(_run_signalized, data)


@_guarded
def _run_twsc(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.Twsc(json.dumps(config))
    ix.analyze()
    results = json.loads(ix.to_json())
    results.update(_snapshot(ix, ["intersection_delay", "rank1_major_delay", "approach_delays", "phf", "heavy_vehicle_pct", "analysis_period_h"]))
    return {"success": True, "results": results}


def analyze_twsc_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 20 two-way STOP-controlled intersections, vehicular modes.

    Input is the TWSC fixture schema: the numbered-movement demand block (``v1`` through ``v12`` in the HCM's movement numbering), the geometry block (leg count, lanes per direction, shared or exclusive turn lanes on each approach), ``phf``, ``analysis_period_h`` and ``heavy_vehicle_pct``. Worked example: ``analyze_twsc`` ships HCM Chapter 32, TWSC Example Problem 1, a three-leg intersection whose published movement 4 capacity is 1,238 veh/h at 8.3 s/veh, LOS A.

    Results carry the per-movement chain: conflicting flow rate, potential capacity from Equation 20-18, the movement capacity after rank-based impedance, control delay in s/veh, the Exhibit 20-2 LOS letter and the 95th-percentile queue in vehicles. ``intersection_delay`` is the volume-weighted delay over the movements the HCM includes and ``approach_delays`` the same by approach. ``rank1_major_delay`` is the delay to major-street through vehicles caused by queued left-turners, which the HCM reports separately because it does not receive an LOS letter.

    This is the vehicular procedure of Section 4. The pedestrian mode is a separate method with its own tool, ``hcm_analyze_twsc_pedestrian``.
    """
    return _dispatch(_run_twsc, data)


@_guarded
def _run_twsc_pedestrian(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "results": json.loads(tl.analyze_twsc_pedestrian(json.dumps(config)))}


def analyze_twsc_pedestrian_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 20, Section 5 pedestrian mode at a TWSC or midblock crossing (Steps 1 through 7, Equations 20-76 through 20-99).

    This is the pedestrian method proper, not the Section 4 pedestrian-impedance adjustment that ``hcm_analyze_twsc`` already applies to vehicular capacity.

    Input is the pedestrian-crossing fixture schema: ``stages`` (one entry per crossing stage, each with ``crossing_length_ft``, ``conflicting_flow_veh_h`` and ``through_lanes``), ``walk_speed_fps``, ``startup_clearance_s``, ``motorist_yield_rate`` as a decimal, ``peak_hour_volume_veh_h`` with ``k_factor`` (or ``aadt_veh`` directly) for Equation 20-95, the ``has_rrfb``, ``has_marked_crosswalk`` and ``has_median_refuge`` treatment indicators, and the optional platooning inputs ``pedestrian_platooning``, ``crosswalk_width_ft`` and ``pedestrian_flow_p_h``. Worked example: ``analyze_twsc_pedestrian`` ships the library's Chapter 20 pedestrian example case.

    Results give the per-stage intermediates (critical headway, the probability of a blocked lane, the delay before an adequate gap), the total pedestrian delay in s/ped, the satisfaction probabilities and average proportion dissatisfied, and the Exhibit 20-3 LOS letter.
    """
    return _dispatch(_run_twsc_pedestrian, data)


@_guarded
def _run_awsc(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.Awsc(json.dumps(config))
    ix.analyze()
    results = json.loads(ix.to_json())
    results.update(_snapshot(ix, ["intersection_delay", "intersection_los", "iterations", "phf", "analysis_period_h"]))
    return {"success": True, "level_of_service": ix.intersection_los, "results": results}


def analyze_awsc_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 21 all-way STOP-controlled intersections.

    Input is the AWSC fixture schema: one block per approach (``eb``, ``wb``, ``nb``, ``sb``), each carrying its lanes with left, through and right volumes plus ``heavy_vehicle_pct``, together with ``phf`` and ``analysis_period_h``. An approach with an empty ``lanes`` list is a missing leg. Worked example: ``analyze_awsc`` ships HCM Chapter 32, AWSC Example Problem 1, a single-lane three-leg intersection whose published eastbound departure headway is 4.97 s at 13.0 s/veh, LOS B.

    The AWSC method solves departure headways iteratively, so ``iterations`` reports how many passes the degree-of-utilization loop needed to converge. Per lane the results carry the departure headway in seconds, the service time, the degree of utilization, capacity, control delay, LOS and the 95th-percentile queue. ``intersection_delay`` is the volume-weighted delay in s/veh and ``intersection_los`` its Exhibit 21-8 letter, with the same pair reported per approach.
    """
    return _dispatch(_run_awsc, data)


@_guarded
def _run_roundabout(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.Roundabouts(json.dumps(config))
    ix.analyze()
    results = json.loads(ix.to_json())
    results.update(_snapshot(ix, ["intersection_delay", "intersection_los", "phf", "analysis_period_h"]))
    return {"success": True, "level_of_service": ix.intersection_los, "results": results}


def analyze_roundabout_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 22 roundabouts.

    Input is the roundabout fixture schema: one block per approach (``nb``, ``sb``, ``eb``, ``wb``) with the U-turn, left, through and right demands ``v_u``/``v_l``/``v_t``/``v_r``, ``heavy_vehicle_pct``, the ``entry_lanes``, ``circulating_lanes`` and ``exiting_lanes`` counts, the ``bypass`` treatment (None, Yielding or NonYielding) and pedestrian volume ``n_ped``, together with ``phf`` and ``analysis_period_h``. Worked example: ``analyze_roundabout`` ships HCM Chapter 33, Example Problem 1, a single-lane roundabout with bypass lanes whose published northbound entry capacity is 597 veh/h at 22.6 s/veh, LOS C.

    Per entry lane the results carry the demand flow rate, capacity from the Chapter 22 exponential model, the volume-to-capacity ratio, control delay in s/veh, LOS and the 95th-percentile queue in vehicles. Circulating flow is reported in passenger-car equivalents per approach, since that is the term the capacity model consumes. Bypass lanes are reported separately because a yielding bypass receives its own delay and LOS while a non-yielding one does not. ``intersection_delay`` and ``intersection_los`` aggregate over the entries by Exhibit 22-8.
    """
    return _dispatch(_run_roundabout, data)


# ── Chapter 23: interchange ramp terminals and alternative intersections ─────

@_guarded
def _run_ramp_terminal(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.Interchange(json.dumps(config))
    ix.analyze()
    results = json.loads(ix.to_json())
    results.update(_snapshot(ix, ["interchange_ett", "interchange_los", "cycle_length", "peak_hour_factor"]))
    od = {}
    for movement in ix.get_od_movements():
        demand, delay, edtt, ett, los = ix.get_od_result(movement)
        od[movement] = {"demand_veh_h": demand, "control_delay_s": delay, "edtt_s": edtt, "ett_s": ett, "los": los}
    results["od_movements"] = od
    return {"success": True, "level_of_service": ix.interchange_los, "results": results}


def analyze_ramp_terminal_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 23, Part B interchange ramp terminals.

    Input is the ramp-terminal fixture schema: the interchange form (diamond, DDI, parclo and so on), the two signalized ramp-terminal intersections with their lane groups and phasing, the segment between them, and the origin-destination demand across the interchange. Worked example: ``analyze_ramp_terminal`` ships HCM Chapter 34, Example Problem 1 (Exhibit 34-16), a diamond interchange whose published interchange ETT is about 50.7 s/veh at LOS C.

    Chapter 23 grades an interchange on experienced travel time rather than delay alone. Per O-D movement the results give ``control_delay_s`` at the signals, ``edtt_s`` the extra distance travel time from the path the movement is forced to take, ``ett_s`` their sum, and the Exhibit 23-10 LOS letter. ``interchange_ett`` is the demand-weighted ETT over the O-D movements and ``interchange_los`` its letter, taken from weighted ETT alone. ``get_queue_storage_ratio`` values in the lane-group record flag turn bays where the queue exceeds the available storage.

    Alternative intersection forms are Part C of the same chapter and have their own tools, ``hcm_analyze_alternative_intersection`` and ``hcm_analyze_displaced_left_turn``.
    """
    return _dispatch(_run_ramp_terminal, data)


@_guarded
def _run_alternative_intersection(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.AlternativeIntersection(json.dumps(config))
    movements = {}
    for label in ix.get_movements():
        result = ix.get_movement_result(label)
        movements[label] = result
    approaches = {}
    for approach in {label.split()[0] for label in ix.get_movements()}:
        try:
            approaches[approach] = ix.get_approach_ett(approach)
        except Exception:
            continue
    return {
        "success": True,
        "results": {
            "intersection_ett": ix.intersection_ett,
            "movements": movements,
            "approach_ett": approaches,
        },
    }


def analyze_alternative_intersection_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 23, Part C alternative intersections: restricted crossing U-turn (RCUT) and median U-turn (MUT).

    Input is the alternative-intersection fixture schema: the ``form``, and per movement the junction journey it makes (the sequence of signalized or STOP-controlled junctions it passes through) with the extra distance travel time each leg adds. Worked example: ``analyze_alternative_intersection`` ships HCM Chapter 34, Example Problem 13 (Exhibits 34-126 through 34-129), a three-legged RCUT with STOP signs whose junction control delays are computed by the Chapter 20 gap-acceptance procedure.

    Part C grades these forms on experienced travel time assembled over a movement's whole journey, because a movement that is not permitted to turn directly pays its cost in distance rather than in delay at one stop line. Per movement the results give the accumulated control delay, the extra distance travel time, ``ett_s`` and the Exhibit 23-13 LOS letter. ``approach_ett`` and ``intersection_ett`` are the demand-weighted aggregations of Equations 23-60 through 23-62.

    Displaced left-turn intersections use a different aggregation and have their own tool, ``hcm_analyze_displaced_left_turn``.
    """
    return _dispatch(_run_alternative_intersection, data)


@_guarded
def _run_displaced_left_turn(config: Dict[str, Any]) -> Dict[str, Any]:
    ix = tl.DisplacedLeftTurn(json.dumps(config))
    return {"success": True, "level_of_service": ix.los, "results": {"intersection_ett": ix.intersection_ett, "los": ix.los}}


def analyze_displaced_left_turn_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 23, Part C displaced left-turn (DLT) intersections.

    Input is the DLT block of the alternative-intersection fixture schema: the ``form`` (partial or full), the per-junction (flow, control delay) cells of Exhibit 34-145, and the O-D demand total. Worked example: ``analyze_displaced_left_turn`` ships HCM Chapter 34, Example Problem 16 (Exhibits 34-139 through 34-145), a partial DLT whose published weighted control delay is 28.5 s/veh at LOS C.

    A DLT is graded by the volume-weighted control delay of Equation 23-69 across every junction a driver passes, which is why the input is a cell table rather than an approach description. ``intersection_ett`` is that weighted value in s/veh and ``los`` its Exhibit 23-13 letter. The signal-offset computation that pairs the displaced left-turn crossover with the main intersection (Equations 23-63 through 23-68) is a separate library function and is not part of this tool's output.
    """
    return _dispatch(_run_displaced_left_turn, data)


# ── Chapter 24: off-street pedestrian and bicycle facilities ─────────────────

_PED_WALKWAY = ["effective_width", "unit_flow_rate", "pedestrian_space", "vc_ratio", "flow_rate_15min"]


@_guarded
def _run_pedestrian_walkway(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("total_walkway_width", "fixed_object_width", "pedestrian_demand", "peak_15min_volume",
            "phf", "pedestrian_speed", "facility_type", "flow_type")
    facility = tl.ExclusivePedestrianFacility(**{k: config[k] for k in keys if config.get(k) is not None})
    los = facility.analyze()
    return {"success": True, "level_of_service": los, "results": _snapshot(facility, _PED_WALKWAY)}


def analyze_pedestrian_walkway_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 24 exclusive pedestrian facilities: walkways and stairwells.

    Input is the exclusive-path fixture schema: ``total_walkway_width`` and ``fixed_object_width`` in feet, either ``peak_15min_volume`` with ``phf`` or an hourly ``pedestrian_demand``, ``pedestrian_speed`` in ft/min, ``facility_type`` (walkway or stairwell) and ``flow_type`` (random or platoon). Worked example: ``analyze_pedestrian_walkway`` ships HCM Chapter 35, Example Problem 1, exclusive-path half, whose published effective width is 5 ft, unit flow rate 1.33 p/ft/min and pedestrian space 180 ft2/p at LOS A.

    ``effective_width`` is walkway width less the shy distance around fixed objects, in feet. ``unit_flow_rate`` is pedestrians per foot of effective width per minute and ``pedestrian_space`` its reciprocal in ft2 per pedestrian, which is the Chapter 24 service measure and sets the LOS letter through Exhibit 24-1 (walkways) or 24-2 (stairwells). ``flow_type`` matters because platoon flow is graded on a different band than random flow. ``vc_ratio`` compares the unit flow rate to capacity.
    """
    return _dispatch(_run_pedestrian_walkway, data)


@_guarded
def _run_shared_use_path_pedestrian(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("bicycle_demand_same_direction", "bicycle_demand_opposing", "phf", "pedestrian_speed",
            "bicycle_speed", "bicycle_flow_rate_same_direction", "bicycle_flow_rate_opposing", "is_one_way")
    path = tl.SharedUsePathPedestrian(**{k: config[k] for k in keys if config.get(k) is not None})
    los = path.analyze()
    return {"success": True, "level_of_service": los, "results": _snapshot(path, ["passing_events", "meeting_events", "total_events"])}


def analyze_shared_use_path_pedestrian_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 24 pedestrian level of service on a shared-use path.

    Input is the shared-use-path fixture schema: ``bicycle_demand_same_direction`` and ``bicycle_demand_opposing`` in bikes/h, ``phf``, ``pedestrian_speed`` and ``bicycle_speed`` in mi/h, and ``is_one_way``. Worked example: ``analyze_shared_use_path_pedestrian`` ships HCM Chapter 35, Example Problem 1, shared-use-path half, whose published event counts are 90 passing, 151 meeting and 166 total per hour at LOS E.

    On a shared-use path the pedestrian service measure is not space but interference from bicycles. ``passing_events`` counts same-direction bicycles overtaking the pedestrian per hour, ``meeting_events`` opposing bicycles encountered, and ``total_events`` the weighted sum that sets the LOS letter through Exhibit 24-4, with meetings weighted at half a passing because they are less disruptive. Pedestrian demand does not enter the calculation at all, which is why the input carries only bicycle volumes.
    """
    return _dispatch(_run_shared_use_path_pedestrian, data)


_BICYCLE = ["blos_score", "effective_lanes", "active_passings_per_minute", "meetings_per_minute",
            "delayed_passings_per_minute", "total_probability_delayed_passing", "subject_flow_rates"]


@_guarded
def _run_offstreet_bicycle(config: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("path_width", "segment_length", "has_centerline", "two_way_demand", "directional_split",
            "phf", "subject_demand", "opposing_demand", "is_one_way", "mode_splits", "mode_speeds", "mode_speed_sds")
    facility = tl.OffStreetBicycleFacility(**{k: config[k] for k in keys if config.get(k) is not None})
    los = facility.analyze()
    return {"success": True, "level_of_service": los, "results": _snapshot(facility, _BICYCLE)}


def analyze_offstreet_bicycle_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 24 bicycle level of service on an off-street or shared-use path.

    Input is the bicycle-facility fixture schema: ``path_width`` in feet, ``segment_length`` in miles, ``has_centerline``, ``two_way_demand`` in users/h with ``directional_split``, ``phf``, ``is_one_way``, and the user-group mix as three parallel lists in HCM order (bicycle, pedestrian, runner, inline skater, child bicyclist): ``mode_splits`` as proportions, ``mode_speeds`` and ``mode_speed_sds`` in mi/h. Worked example: ``analyze_offstreet_bicycle`` ships HCM Chapter 35, Example Problem 2, whose published active passing rate is 2.42 passings/min and BLOS score 2.69 at LOS D.

    ``subject_flow_rates`` is the directional demand split across the five user groups in users/h. ``active_passings_per_minute`` and ``meetings_per_minute`` come from the speed distribution of that mix, since a bicyclist's experience is set by how often faster and slower path users are encountered. ``effective_lanes`` reflects whether path width and a centerline allow simultaneous passing, and ``delayed_passings_per_minute`` with ``total_probability_delayed_passing`` measures how often a pass is blocked. ``blos_score`` is the Equation 24-15 score, converted to the LOS letter through Exhibit 24-5, and it runs the opposite way to a grade: a lower score is better.
    """
    return _dispatch(_run_offstreet_bicycle, data)


# ── Chapters 25 and 26: mixed flow ───────────────────────────────────────────

@_guarded
def _run_mixed_flow(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "results": json.loads(tl.analyze_mixed_flow(json.dumps(config)))}


def analyze_mixed_flow_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 26 mixed-flow model for a single grade.

    Input is the mixed-flow fixture schema: ``ffs`` in mi/h, ``length`` in miles, ``grade`` in percent, ``v_mix`` in veh/h/ln, the truck shares ``p_sut`` and ``p_tt`` as decimals, and an optional ``caf_ao``. Worked example: ``analyze_mixed_flow`` ships HCM Chapter 26, Example Problem 5, mixed-flow half, whose published mixed-flow capacity is 1,725 veh/h/ln, speed 47.3 mi/h and density 31.7 veh/mi/ln.

    The mixed-flow model replaces the passenger-car-equivalent approach on grades steep enough that trucks govern the traffic stream. ``caf_t_mix``, ``caf_g_mix`` and ``caf_mix`` are the truck, grade and combined capacity adjustment factors, applied to ``capacity_ao`` to give ``capacity_mix`` in veh/h/ln. ``tau_sut_kin`` and ``tau_tt_kin`` are the kinematic travel rates for single-unit and tractor trailers in s/mi, ``ffs_mix`` and ``saf_mix`` the mixed-flow free-flow speed and its adjustment, ``bp_mix`` the breakpoint, ``s_calib_cap`` and ``s_calib_90cap`` the calibration speeds, and ``phi_mix`` the exponent of the speed-flow curve. ``s_mix`` in mi/h and ``d_mix`` in veh/mi/ln are the answers; both come back null with ``oversaturated`` true when demand exceeds mixed-flow capacity.

    The truck speed curves are digitised from the published exhibits at the grades and speeds the exhibits cover. A grade and entry-speed combination outside that digitised set is refused with the library's own message rather than extrapolated, and that refusal is a statement about the published data, not about the input being invalid engineering.
    """
    return _dispatch(_run_mixed_flow, data)


@_guarded
def _run_composite_grade(config: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "results": json.loads(tl.analyze_composite_grade(json.dumps(config)))}


def analyze_composite_grade_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 25 mixed-flow model for a composite grade.

    Input is the composite-grade fixture schema: ``ffs`` in mi/h, ``v_mix`` in veh/h/ln, the truck shares ``p_sut`` and ``p_tt`` as decimals, an optional ``caf_ao``, and ``segments``, a list of ``length`` (miles) and ``grade`` (percent) pairs in the order a vehicle meets them. Worked example: ``analyze_composite_grade`` ships HCM Chapter 25, Example Problem 11, a three-segment composite grade whose published overall mixed-flow speed is 55.6 mi/h over 4.5 miles.

    A composite grade is not the average of its parts: a truck enters each segment at the speed it left the last one, so the order matters and the governing capacity may be set by a segment that is not the steepest. ``segments`` carries per-segment ``capacity_mix``, ``s_mix``, ``tau_mix`` (travel rate in s/mi) and ``travel_time``. ``governing_segment`` names the one-based segment that sets ``capacity_mix`` for the whole grade. ``total_travel_time`` in seconds over ``total_length`` in miles gives ``s_mix_overall``, and the spot and space mean speeds are reported separately for automobiles, single-unit trucks and tractor trailers.

    As with the single-grade model, a chain that reaches a grade or entry speed outside the digitised truck curves is refused with the library's own message rather than extrapolated.
    """
    return _dispatch(_run_composite_grade, data)


# ── Chapters 27 and 28: service volumes ──────────────────────────────────────

@_guarded
def _run_ramp_service_volumes(config: Dict[str, Any]) -> Dict[str, Any]:
    seg = _ramp_segment(config["segment"])
    ramp_fraction = config.get("ramp_fraction")
    fixed_freeway_vf = config.get("fixed_freeway_vf")
    if (ramp_fraction is None) == (fixed_freeway_vf is None):
        raise ValueError("provide exactly one of 'ramp_fraction' (Case 1, solves the approaching freeway flow) or 'fixed_freeway_vf' (Case 2, solves the ramp flow)")
    f_hv = config["f_hv"]
    f_p = config.get("f_p", 1.0)
    phf = config["phf"]
    rows = []
    for density in config.get("target_densities", [10.0, 20.0, 28.0, 35.0]):
        kwargs = {"ramp_fraction": ramp_fraction} if ramp_fraction is not None else {"fixed_freeway_vf": fixed_freeway_vf}
        sfi = tl.ramp_service_flow_rate_ideal(seg, density, **kwargs)
        if sfi is None:
            rows.append({"target_density": density, "sfi": None, "sf": None, "sv": None,
                         "unachievable": True})
            continue
        sf, sv = tl.ramp_service_volumes(sfi, f_hv, f_p, phf)
        rows.append({"target_density": density, "sfi": sfi, "sf": sf, "sv": sv, "unachievable": False})
    return {"success": True, "results": {
        "basis": "ramp_fraction" if ramp_fraction is not None else "fixed_freeway_vf",
        "service_volumes": rows,
    }}


def analyze_ramp_service_volumes_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 28 service flow rates and service volumes for a merge or diverge segment.

    This inverts the Chapter 14 operational method: instead of asking what density a demand produces, it asks what demand holds a segment at a target LOS density. Input is a ``segment`` in the merge/diverge fixture schema (geometry only; its demand fields are not used), ``target_densities`` (the Exhibit 14-3 thresholds 10, 20, 28 and 35 pc/mi/ln for LOS A through D), the heavy-vehicle adjustment ``f_hv``, the driver-population factor ``f_p``, ``phf``, and exactly one basis. Worked example: ``analyze_ramp_service_volumes`` ships HCM Chapter 28, Example Problem 5, a one-lane right-side on-ramp on a six-lane freeway.

    Give ``ramp_fraction`` for Case 1, where ramp demand is a fixed share of freeway demand and the solver returns the approaching freeway flow. Give ``fixed_freeway_vf`` instead for Case 2, where the freeway flow is held constant and the solver returns the ramp flow. Providing both or neither is refused.

    Per target density the results give ``sfi``, the ideal-conditions service flow rate in pc/h, then ``sf`` the prevailing service flow rate and ``sv`` the service volume, both in veh/h, where SF = SFI x f_HV x f_p and SV = SF x PHF. ``unachievable`` marks a LOS that cannot be reached at all under the given basis, which happens in Case 2 when the fixed freeway flow already puts the influence-area density above the target with zero ramp demand.

    The library ships no fixture file for this example problem, so the shipped example transcribes the segment from the library's own Chapter 28 service-volume test.
    """
    return _dispatch(_run_ramp_service_volumes, data)


@_guarded
def _run_weaving_service_volumes(config: Dict[str, Any]) -> Dict[str, Any]:
    seg = _weaving_segment(config["segment"])
    split = tuple(config["split"])
    if len(split) != 4:
        raise ValueError("'split' must be four demand fractions (ff, rf, fr, rr) summing to 1")
    f_hv = config["f_hv"]
    phf = config["phf"]
    k_factor = config["k_factor"]
    d_factor = config["d_factor"]
    rows = []
    for density in config.get("target_densities", [10.0, 20.0, 28.0, 35.0]):
        sfi = tl.service_flow_rate_ideal(seg, split, density)
        sfi_out, sf, sv, dsv = tl.service_volumes(sfi, f_hv, phf, k_factor, d_factor)
        rows.append({"target_density": density, "sfi": sfi_out, "sf": sf, "sv": sv, "dsv": dsv})
    return {"success": True, "results": {"service_volumes": rows}}


def analyze_weaving_service_volumes_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 27 service flow rates, service volumes and daily service volumes for a weaving segment.

    The Chapter 13 counterpart of ``hcm_analyze_ramp_service_volumes``: it solves for the total weaving-segment flow that holds the segment at a target LOS density. Input is a ``segment`` in the weaving fixture schema (geometry only), ``split``, the four demand fractions (freeway-to-freeway, ramp-to-freeway, freeway-to-ramp, ramp-to-ramp) summing to 1, ``target_densities``, ``f_hv``, ``phf``, and the ``k_factor`` and ``d_factor`` that convert an hourly volume to a daily one. Worked example: ``analyze_weaving_service_volumes`` ships HCM Chapter 27, Example Problem 5.

    Per target density the results give ``sfi`` in pc/h, ``sf`` the prevailing service flow rate and ``sv`` the service volume in veh/h, and ``dsv`` the daily service volume in veh/day, where SF = SFI x f_HV, SV = SF x PHF and DSV = SV / (K x D). The movement split matters because weaving intensity, not total flow alone, is what limits the segment.

    The library ships no fixture file for this example problem, so the shipped example transcribes the segment from the library's own Chapter 27 service-volume test.
    """
    return _dispatch(_run_weaving_service_volumes, data)


# ── Chapters 18 and 19: non-automobile modes ─────────────────────────────────
# Each of these is a distinct HCM method with its own service measure, not a
# variant of the automobile procedure in the same chapter, so each gets its own
# tool. The library exposes them as JSON-in/JSON-out functions.

def _json_function_runner(symbol: str):
    @_guarded
    def run(config: Dict[str, Any]) -> Dict[str, Any]:
        return {"success": True, "results": json.loads(getattr(tl, symbol)(json.dumps(config)))}
    return run


_run_pedestrian_segment = _json_function_runner("analyze_pedestrian_segment")
_run_bicycle_segment = _json_function_runner("analyze_bicycle_segment")
_run_transit_segment = _json_function_runner("analyze_transit_segment")
_run_signalized_pedestrian = _json_function_runner("analyze_signalized_pedestrian")
_run_signalized_bicycle = _json_function_runner("analyze_signalized_bicycle")


def analyze_pedestrian_segment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 18, Section 4 pedestrian mode on an urban street segment.

    Input is the pedestrian-segment schema: segment ``length_ft`` and ``num_through_lanes``, the mid-segment vehicle flow rate, the pedestrian flow rate, the sidewalk and buffer widths with the separation shares (``prop_fence``, ``prop_window``, ``prop_building``), the outside-lane, bike-lane, shoulder and parking-lane widths, ``motor_running_speed`` and ``free_flow_walk_speed``, the three pedestrian delay terms, ``ped_los_score_intersection`` from the boundary intersection, and the crossing inputs. Worked example: ``analyze_pedestrian_segment`` ships HCM Chapter 30, Example Problem 2, whose published link score is 2.35 and segment score 3.62 at LOS D.

    ``effective_sidewalk_width`` and ``flow_per_width`` give ``pedestrian_space`` in ft2 per pedestrian and ``walking_speed`` the achieved speed. ``f_w``, ``f_v`` and ``f_s`` are the width, volume and speed factors of the link score. ``link_score`` grades walking along the segment and ``crossing_score`` grades crossing it; ``segment_score`` combines them with the boundary intersection score, and ``segment_los`` is its Exhibit 18-10 letter. These scores run the opposite way to a grade: lower is better.
    """
    return _dispatch(_run_pedestrian_segment, data)


def analyze_bicycle_segment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 18, Section 5 bicycle mode on an urban street segment.

    Input is the bicycle-segment schema: segment ``length_ft`` and ``num_through_lanes``, the mid-segment vehicle flow rate and ``pct_heavy_vehicles``, ``prop_parking_occupied``, the outside-lane, bike-lane, shoulder and parking-lane widths, ``curb_present`` and ``median_divided``, ``num_access_points_right``, ``pavement_condition`` on the HCM's 0-to-5 scale, the motor and bicycle running speeds, ``bicycle_control_delay`` at the boundary and ``bicycle_los_score_intersection``. Worked example: ``analyze_bicycle_segment`` ships HCM Chapter 30, Example Problem 3, whose published link score is 3.62 at LOS D and segment score 2.88 at LOS C.

    ``effective_width`` is the usable width after parking occupancy and the ``f_w``, ``f_v``, ``f_s`` and ``f_p`` factors are its width, volume, speed and pavement terms. ``link_score`` with ``link_los`` grades riding along the segment; ``f_c`` carries the crossing effect and ``segment_score`` with ``segment_los`` combines link and boundary intersection through Exhibit 18-10. Lower scores are better.
    """
    return _dispatch(_run_bicycle_segment, data)


def analyze_transit_segment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 18, Section 6 transit mode on an urban street segment.

    Input is the transit-segment schema: segment ``length_ft``, ``num_transit_stops``, ``motor_running_speed``, ``dwell_time_s``, ``transit_frequency`` in vehicles per hour, ``g_c_ratio`` and ``near_side_signalized_stop`` at the boundary, the acceleration and deceleration rates, ``reentry_delay_s``, ``through_delay_s``, ``passenger_load_factor``, the stop-amenity shares, ``passenger_trip_length``, ``on_time_performance``, ``base_travel_time_rate`` and ``ped_los_score_link``. Worked example: ``analyze_transit_segment`` ships HCM Chapter 30, Example Problem 4, whose published travel speed is 11.3 mi/h and segment score 2.83 at LOS C.

    ``running_speed`` less the acceleration/deceleration, passenger-service and stop delays gives ``running_time`` and ``travel_speed`` in mi/h. ``headway_factor``, ``perceived_travel_time_rate`` and ``travel_time_factor`` build the ``wait_ride_score``, which with the pedestrian link score gives ``segment_score`` and its Exhibit 18-10 letter ``segment_los``. Transit LOS is graded on the passenger's experience of waiting and riding, not on the vehicle's delay, which is why frequency and amenities enter the score.
    """
    return _dispatch(_run_transit_segment, data)


def analyze_signalized_pedestrian_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 19, Section 5 pedestrian mode at a signalized intersection.

    Input is the pedestrian-intersection schema: ``cycle_length_s``, ``walk_setting_s``, ``lanes_crossed``, the turning volumes that conflict with the crossing (``v_rtor``, ``v_lt_perm``, ``num_rtci``), ``crossed_street_volume_sum`` and ``crossed_street_lanes``, and ``speed_85_mph``. Worked example: ``analyze_signalized_pedestrian`` ships HCM Chapter 31, Example Problem 2, whose published pedestrian delay is 29.8 s/ped and LOS score 2.37 at LOS B.

    ``effective_walk_s`` is the usable walk time, ``delay`` the pedestrian delay in s/ped and ``n15_per_lane`` the corner circulation measure. ``f_w``, ``f_v``, ``f_s`` and ``f_delay`` are the width, volume, speed and delay terms of ``los_score``, whose Exhibit 19-9 letter is ``los``. Lower scores are better.
    """
    return _dispatch(_run_signalized_pedestrian, data)


def analyze_signalized_bicycle_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 19, Section 6 bicycle mode at a signalized intersection.

    Input is the bicycle-intersection schema: ``saturation_flow`` for the bicycle lane, ``effective_green_s`` and ``cycle_length_s``, ``bicycle_flow``, ``cross_street_width_ft`` and ``total_width_ft``, the motorized ``v_left``, ``v_through`` and ``v_right`` volumes, and ``num_through_lanes``. Worked example: ``analyze_signalized_bicycle`` ships HCM Chapter 31, Example Problem 3, whose published bicycle capacity is 800 bicycles/h at 23.0 s/bicycle and LOS score 2.45, LOS B.

    ``capacity`` is the bicycle-lane capacity in bicycles per hour and ``delay`` the bicycle control delay in s/bicycle. ``f_w`` and ``f_v`` are the cross-section and volume terms of ``los_score``, whose Exhibit 19-9 letter is ``los``. Lower scores are better.
    """
    return _dispatch(_run_signalized_bicycle, data)


@_guarded
def _run_two_stage_crossing(config: Dict[str, Any]) -> Dict[str, Any]:
    # This one returns a bare float rather than a JSON document.
    return {"success": True, "results": {"delay_s": tl.signalized_two_stage_crossing_delay(json.dumps(config))}}


def analyze_two_stage_crossing_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """HCM Chapter 19 two-stage pedestrian crossing delay at a signalized intersection.

    Input is the two-stage-crossing schema: ``cycle_length_s``, the walk settings for the two stages ``walk_setting_x_s`` and ``walk_setting_y_s``, ``first_stage_distance_ft``, ``walk_speed_fps``, and the two stage start times within the cycle ``walk_start_x_s`` and ``walk_start_y_s``. Worked example: ``analyze_two_stage_crossing`` ships HCM Chapter 31, Example Problem 4, whose published delay is 78.0 s/ped.

    ``delay_s`` is the total pedestrian delay in s/ped for the two-stage crossing. It is not the sum of two independent one-stage delays: a pedestrian who reaches the median refuge part-way through the cycle waits for the second stage's own walk indication, so the relative timing of the two stage starts is what governs the answer.
    """
    return _dispatch(_run_two_stage_crossing, data)


# ── The method table ─────────────────────────────────────────────────────────
# Every entry is a method the compute library implements, its HCM chapter, the
# library symbol behind it, and the example-problem fixture shipped alongside.
# Registry names are the keys; the served tool name is `hcm_<key>` (the registry
# prefixes non-analysis sections with their section name).

METHODS: Dict[str, Dict[str, Any]] = {
    "analyze_freeway_facility": {"chapter": 10, "library": "FreewayFacility", "function": analyze_freeway_facility_function,
                                 "title": "Freeway facility (Chapter 25 engine)"},
    "analyze_managed_lanes": {"chapter": 10, "library": "ManagedLaneFacility", "function": analyze_managed_lanes_function,
                              "title": "Managed-lane freeway facility"},
    "analyze_planning_facility": {"chapter": 25, "library": "PlanningFacility", "function": analyze_planning_facility_function,
                                  "title": "Planning-level freeway facility"},
    "analyze_freeway_reliability": {"chapter": 11, "library": "FreewayReliability", "function": analyze_freeway_reliability_function,
                                    "title": "Freeway travel-time reliability"},
    "analyze_basic_freeway": {"chapter": 12, "library": "BasicFreeways", "function": analyze_basic_freeway_function,
                              "title": "Basic freeway and multilane segment"},
    "analyze_weaving": {"chapter": 13, "library": "WeavingSegment", "function": analyze_weaving_function,
                        "title": "Freeway weaving segment", "editions": ["7", "7.1"]},
    "analyze_merge_diverge": {"chapter": 14, "library": "RampSegment", "function": analyze_merge_diverge_function,
                              "title": "Freeway merge and diverge segment", "editions": ["7", "7.1"]},
    "analyze_two_lane_highway": {"chapter": 15, "library": "TwoLaneHighways", "function": analyze_two_lane_highway_function,
                                 "title": "Two-lane highway facility"},
    "analyze_bicycle_los": {"chapter": 15, "library": "analyze_bicycle_los", "function": analyze_bicycle_los_function,
                            "title": "Two-lane and multilane highway segment, bicycle mode"},
    "analyze_urban_facility": {"chapter": 16, "library": "UrbanFacility", "function": analyze_urban_facility_function,
                               "title": "Urban street facility"},
    "analyze_urban_reliability": {"chapter": 17, "library": "UrbanReliability", "function": analyze_urban_reliability_function,
                                  "title": "Urban street travel-time reliability"},
    "analyze_urban_segment": {"chapter": 18, "library": "UrbanSegment", "function": analyze_urban_segment_function,
                              "title": "Urban street segment, automobile mode"},
    "analyze_pedestrian_segment": {"chapter": 18, "library": "analyze_pedestrian_segment", "function": analyze_pedestrian_segment_function,
                                   "title": "Urban street segment, pedestrian mode"},
    "analyze_bicycle_segment": {"chapter": 18, "library": "analyze_bicycle_segment", "function": analyze_bicycle_segment_function,
                                "title": "Urban street segment, bicycle mode"},
    "analyze_transit_segment": {"chapter": 18, "library": "analyze_transit_segment", "function": analyze_transit_segment_function,
                                "title": "Urban street segment, transit mode"},
    "analyze_signalized": {"chapter": 19, "library": "SignalizedIntersection", "function": analyze_signalized_function,
                           "title": "Signalized intersection, automobile mode"},
    "analyze_signalized_pedestrian": {"chapter": 19, "library": "analyze_signalized_pedestrian", "function": analyze_signalized_pedestrian_function,
                                      "title": "Signalized intersection, pedestrian mode"},
    "analyze_signalized_bicycle": {"chapter": 19, "library": "analyze_signalized_bicycle", "function": analyze_signalized_bicycle_function,
                                   "title": "Signalized intersection, bicycle mode"},
    "analyze_two_stage_crossing": {"chapter": 19, "library": "signalized_two_stage_crossing_delay", "function": analyze_two_stage_crossing_function,
                                   "title": "Two-stage pedestrian crossing delay"},
    "analyze_twsc": {"chapter": 20, "library": "Twsc", "function": analyze_twsc_function,
                     "title": "Two-way STOP-controlled intersection, vehicular"},
    "analyze_twsc_pedestrian": {"chapter": 20, "library": "analyze_twsc_pedestrian", "function": analyze_twsc_pedestrian_function,
                                "title": "TWSC and midblock crossing, pedestrian mode"},
    "analyze_awsc": {"chapter": 21, "library": "Awsc", "function": analyze_awsc_function,
                     "title": "All-way STOP-controlled intersection"},
    "analyze_roundabout": {"chapter": 22, "library": "Roundabouts", "function": analyze_roundabout_function,
                           "title": "Roundabout"},
    "analyze_ramp_terminal": {"chapter": 23, "library": "Interchange", "function": analyze_ramp_terminal_function,
                              "title": "Interchange ramp terminals (Part B)"},
    "analyze_alternative_intersection": {"chapter": 23, "library": "AlternativeIntersection", "function": analyze_alternative_intersection_function,
                                         "title": "RCUT and MUT alternative intersections (Part C)"},
    "analyze_displaced_left_turn": {"chapter": 23, "library": "DisplacedLeftTurn", "function": analyze_displaced_left_turn_function,
                                    "title": "Displaced left-turn intersection (Part C)"},
    "analyze_pedestrian_walkway": {"chapter": 24, "library": "ExclusivePedestrianFacility", "function": analyze_pedestrian_walkway_function,
                                   "title": "Exclusive pedestrian walkway or stairwell"},
    "analyze_shared_use_path_pedestrian": {"chapter": 24, "library": "SharedUsePathPedestrian", "function": analyze_shared_use_path_pedestrian_function,
                                           "title": "Shared-use path, pedestrian mode"},
    "analyze_offstreet_bicycle": {"chapter": 24, "library": "OffStreetBicycleFacility", "function": analyze_offstreet_bicycle_function,
                                  "title": "Off-street path, bicycle mode"},
    "analyze_mixed_flow": {"chapter": 26, "library": "analyze_mixed_flow", "function": analyze_mixed_flow_function,
                           "title": "Mixed-flow model, single grade"},
    "analyze_composite_grade": {"chapter": 25, "library": "analyze_composite_grade", "function": analyze_composite_grade_function,
                                "title": "Mixed-flow model, composite grade"},
    "analyze_weaving_service_volumes": {"chapter": 27, "library": "service_flow_rate_ideal", "function": analyze_weaving_service_volumes_function,
                                        "title": "Weaving segment service volumes"},
    "analyze_ramp_service_volumes": {"chapter": 28, "library": "ramp_service_flow_rate_ideal", "function": analyze_ramp_service_volumes_function,
                                     "title": "Merge and diverge service volumes"},
}


# ── Dry-run validation ───────────────────────────────────────────────────────
# A caller iterating on a config should not have to pay for a full analysis to
# find out a field is wrong. For most methods the library's deserialiser IS the
# validator: constructing the object runs serde (or the keyword constructor's
# range checks) and raises with the library's own message, without running a
# single equation. Methods reached through a bare JSON function have no such
# split -- validation happens inside the analysis -- and those say so rather
# than quietly running the analysis and calling it a validation. The one
# exception is analyze_bicycle_los, whose bare JSON function is shadowed by a
# BicycleLOS class holding the same input set and computing nothing, so the
# parse can be run there.


def _json_class_validator(cls_name: str) -> Callable[[Dict[str, Any]], None]:
    """Deserialize a JSON-config class without running it. serde rejects unknown shapes, missing fields and bad enum spellings here."""
    def validate(config: Dict[str, Any]) -> None:
        getattr(tl, cls_name)(json.dumps(config))
    return validate


def _kwargs_validator(builder: Callable[[Dict[str, Any]], Any]) -> Callable[[Dict[str, Any]], None]:
    """Build a keyword-constructor object without running it."""
    def validate(config: Dict[str, Any]) -> None:
        builder(config)
    return validate


def _validate_basic_freeway(config: Dict[str, Any]) -> None:
    d = dict(config)
    d.setdefault("lane_width", d.get("lw"))
    tl.BasicFreeways(**{k: d[k] for k in _BASIC_FREEWAY_KEYS if d.get(k) is not None})


def _validate_two_lane_highway(config: Dict[str, Any]) -> None:
    """Chapter 15 is the one method with a real rule-level validator behind the constructor. ``tl.validate_input`` returns the Exhibit 15-8 range violations that a constructor would happily accept, which is exactly the class of error that otherwise produces a plausible wrong follower density."""
    segments = config.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("'segments' must be a non-empty list of Chapter 15 segments")
    errors: List[str] = []
    for index, segment in enumerate(segments):
        _two_lane_segment(segment)
        reported = tl.validate_input(
            lane_width=config.get("lane_width"),
            shoulder_width=config.get("shoulder_width"),
            passing_type=segment.get("passing_type"),
            hor_class=segment.get("hor_class"),
            grade=segment.get("grade"),
            phf=segment.get("phf"),
            phv=segment.get("phv"),
            spl=segment.get("spl"),
        )
        errors += [f"segment {index}: {message}" for message in reported]
    if errors:
        raise ValueError("; ".join(errors))


_BICYCLE_LOS_KEYS = ("lane_width", "shoulder_width", "speed_limit", "num_lanes", "pavement_condition",
                     "hourly_volume", "phf", "heavy_vehicle_pct", "pct_on_highway_parking")


def _validate_bicycle_los(config: Dict[str, Any]) -> None:
    """The bicycle mode reaches the engine through a bare JSON function, but the same input set is also a ``BicycleLOS`` constructor that computes nothing, so a dry run here is a real parse rather than the analysis under another name. The missing-field check is spelled out because serde names only the first field it misses, and a config short three fields is worth learning about in one round trip."""
    missing = [k for k in _BICYCLE_LOS_KEYS if config.get(k) is None]
    if missing:
        raise ValueError(
            f"missing required field(s) {', '.join(repr(k) for k in missing)}: BicycleLOS has no defaults, "
            "every one of the nine inputs enters Equation 15-47"
        )
    tl.BicycleLOS(*(config[k] for k in _BICYCLE_LOS_KEYS))


def _validate_ramp_service_volumes(config: Dict[str, Any]) -> None:
    _ramp_segment(config["segment"])
    if (config.get("ramp_fraction") is None) == (config.get("fixed_freeway_vf") is None):
        raise ValueError("provide exactly one of 'ramp_fraction' (Case 1) or 'fixed_freeway_vf' (Case 2)")
    for key in ("f_hv", "phf"):
        if config.get(key) is None:
            raise ValueError(f"missing required field {key!r}")


def _validate_weaving_service_volumes(config: Dict[str, Any]) -> None:
    _weaving_segment(config["segment"])
    if len(config.get("split") or ()) != 4:
        raise ValueError("'split' must be four demand fractions (ff, rf, fr, rr) summing to 1")
    for key in ("f_hv", "phf", "k_factor", "d_factor"):
        if config.get(key) is None:
            raise ValueError(f"missing required field {key!r}")


# method -> (validator, what the validator actually checked). A None validator
# means the library offers no step between "parse" and "compute" for that
# method, and hcm_validate reports that instead of pretending otherwise.
_VALIDATORS: Dict[str, Any] = {
    "analyze_freeway_facility": (_json_class_validator("FreewayFacility"), "serde deserialisation of the facility config"),
    "analyze_managed_lanes": (_json_class_validator("ManagedLaneFacility"), "serde deserialisation of the managed-lane facility config"),
    "analyze_planning_facility": (_json_class_validator("PlanningFacility"), "serde deserialisation of the planning facility config"),
    "analyze_freeway_reliability": (_json_class_validator("FreewayReliability"), "serde deserialisation of the reliability config"),
    "analyze_basic_freeway": (_validate_basic_freeway, "the BasicFreeways keyword constructor's range and enum checks"),
    "analyze_weaving": (_kwargs_validator(_weaving_segment), "the WeavingSegment keyword constructor's range and enum checks"),
    "analyze_merge_diverge": (_kwargs_validator(_ramp_segment), "the RampSegment keyword constructor's range and enum checks"),
    "analyze_two_lane_highway": (_validate_two_lane_highway, "the Segment/SubSegment constructors plus tl.validate_input's Exhibit 15-8 parameter ranges"),
    "analyze_bicycle_los": (_validate_bicycle_los, "the nine required BicycleLOS inputs and their types, through the constructor (no range check: Equation 15-46 is undefined at a posted speed limit of 20 mi/h or below and the library does not refuse it)"),
    "analyze_urban_facility": (_json_class_validator("UrbanFacility"), "serde deserialisation of the urban facility config"),
    "analyze_urban_reliability": (_json_class_validator("UrbanReliability"), "serde deserialisation of the urban reliability config"),
    "analyze_urban_segment": (_json_class_validator("UrbanSegment"), "serde deserialisation of the urban segment config"),
    "analyze_signalized": (_json_class_validator("SignalizedIntersection"), "serde deserialisation of the intersection config"),
    "analyze_twsc": (_json_class_validator("Twsc"), "serde deserialisation of the TWSC config"),
    "analyze_awsc": (_json_class_validator("Awsc"), "serde deserialisation of the AWSC config"),
    "analyze_roundabout": (_json_class_validator("Roundabouts"), "serde deserialisation of the roundabout config"),
    "analyze_ramp_terminal": (_json_class_validator("Interchange"), "serde deserialisation of the interchange config"),
    "analyze_alternative_intersection": (_json_class_validator("AlternativeIntersection"), "serde deserialisation of the alternative-intersection config"),
    "analyze_displaced_left_turn": (_json_class_validator("DisplacedLeftTurn"), "serde deserialisation of the DLT config"),
    "analyze_pedestrian_walkway": (_kwargs_validator(lambda c: tl.ExclusivePedestrianFacility(**{
        k: c[k] for k in ("total_walkway_width", "fixed_object_width", "pedestrian_demand", "peak_15min_volume",
                          "phf", "pedestrian_speed", "facility_type", "flow_type") if c.get(k) is not None})),
        "the ExclusivePedestrianFacility keyword constructor"),
    "analyze_shared_use_path_pedestrian": (_kwargs_validator(lambda c: tl.SharedUsePathPedestrian(**{
        k: c[k] for k in ("bicycle_demand_same_direction", "bicycle_demand_opposing", "phf", "pedestrian_speed",
                          "bicycle_speed", "bicycle_flow_rate_same_direction", "bicycle_flow_rate_opposing",
                          "is_one_way") if c.get(k) is not None})),
        "the SharedUsePathPedestrian keyword constructor"),
    "analyze_offstreet_bicycle": (_kwargs_validator(lambda c: tl.OffStreetBicycleFacility(**{
        k: c[k] for k in ("path_width", "segment_length", "has_centerline", "two_way_demand", "directional_split",
                          "phf", "subject_demand", "opposing_demand", "is_one_way", "mode_splits", "mode_speeds",
                          "mode_speed_sds") if c.get(k) is not None})),
        "the OffStreetBicycleFacility keyword constructor"),
    "analyze_weaving_service_volumes": (_validate_weaving_service_volumes, "the WeavingSegment constructor plus the split and factor requirements"),
    "analyze_ramp_service_volumes": (_validate_ramp_service_volumes, "the RampSegment constructor plus the Case 1 / Case 2 basis requirement"),
}

_NO_SEPARATE_VALIDATION = (
    "this method is a single JSON entry point in the compute library: parsing, range checking and computation happen "
    "in one call, so there is no step to run short of the analysis itself. Call hcm_analyze; a bad config comes back "
    "as the same error message this tool would have returned."
)


# ── Shared helpers for the describe capability ───────────────────────────

def _sketch(value: Any, depth: int = 0) -> Any:
    """Reduce an example config to a key/type sketch. Lists collapse to their first element so a fifty-segment facility describes as one segment, and nesting stops at depth 4 so a deep config stays readable in a tool response."""
    if depth >= 4:
        return "..."
    if isinstance(value, dict):
        return {k: _sketch(v, depth + 1) for k, v in value.items() if k != "_source"}
    if isinstance(value, list):
        if not value:
            return []
        sketched = _sketch(value[0], depth + 1)
        return [sketched, f"... {len(value)} items"] if len(value) > 1 else [sketched]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if value is None:
        return "null (optional)"
    return "string"


def _method_row(name: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "method": name,
        "chapter": entry["chapter"],
        "title": entry["title"],
        "library_symbol": entry["library"],
    }
    if "editions" in entry:
        row["hcm_editions"] = entry["editions"]
    return row


# ── The three capability tools ───────────────────────────────────────────────
# The MCP surface is capability-shaped, matching the ten tools of the published
# surface: one tool to run a method, one to discover it, one to dry-run a config.
# The 33 per-method functions above stay as the implementation and keep their
# REST routes; they are not advertised as separate tools, because 33 near-
# identical schemas cost every caller context and blunt tool selection.


def _resolve(method: Any) -> str:
    """Accept a method id with or without the tool prefix a caller may have copied from a route or an older tool name."""
    if not isinstance(method, str) or not method:
        raise ValueError(
            f"Missing 'method'. Call hcm_describe with no arguments for the catalog of {len(METHODS)} methods."
        )
    name = method[4:] if method.startswith("hcm_") else method
    name = name.replace("-", "_")
    if name not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Call hcm_describe with no arguments for the catalog of {len(METHODS)} methods."
        )
    return name


def analyze_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run one HCM method. Takes the method id and that method's config in the compute library's example-case (fixture) schema."""
    try:
        name = _resolve(data.get("method"))
    except ValueError as e:
        return {"success": False, "error": str(e)}
    entry = METHODS[name]
    payload = {"config": data.get("config")}
    if "mode" in data:
        payload["mode"] = data["mode"]
    result = entry["function"](payload)
    if isinstance(result, dict):
        result.setdefault("method", name)
        result.setdefault("chapter", entry["chapter"])
    return result


def describe_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """With a method, its input schema sketch, result-field meanings and worked-example fixture; without one, the catalog."""
    method = data.get("method")
    if not method:
        return {
            "success": True,
            "methods": [
                dict(_method_row(name, entry), summary=entry["title"])
                for name, entry in sorted(METHODS.items(), key=lambda kv: (kv[1]["chapter"], kv[0]))
            ],
            "total_count": len(METHODS),
            "usage": "Pass one of these ids as 'method' to hcm_describe for its input schema and worked example, then to hcm_analyze to run it.",
        }
    try:
        name = _resolve(method)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    entry = METHODS[name]
    example = _example(name)
    row = _method_row(name, entry)
    validator = _VALIDATORS.get(name)
    row.update({
        "success": True,
        "docstring": (entry["function"].__doc__ or "").strip(),
        "input_format": "Pass the example's shape as hcm_analyze's 'config' argument. This is the compute library's own fixture schema, the same files its example-problem tests read, so an example case can be handed over unmodified.",
        "input_sketch": _sketch(example),
        "example": example,
        "example_source": example.get("_source") or example.get("_comment") or example.get("description") or "See the library's example cases.",
        "dry_run_validation": validator[1] if validator else None,
    })
    return row


def validate_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse and validate a config without running the analysis."""
    try:
        name = _resolve(data.get("method"))
        config = _config(data)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    entry = _VALIDATORS.get(name)
    if entry is None:
        return {
            "success": True,
            "method": name,
            "chapter": METHODS[name]["chapter"],
            "valid": None,
            "validated": False,
            "reason": _NO_SEPARATE_VALIDATION,
        }
    validator, checked = entry
    try:
        validator(config)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 -- PyO3 raises outside Exception
        return {
            "success": True,
            "method": name,
            "chapter": METHODS[name]["chapter"],
            "valid": False,
            "validated": True,
            "checked": checked,
            "error": str(e),
        }
    return {
        "success": True,
        "method": name,
        "chapter": METHODS[name]["chapter"],
        "valid": True,
        "validated": True,
        "checked": checked,
    }


def method_catalog_lines() -> List[str]:
    """The method list as it appears in hcm_analyze's tool description. This text is in every caller's context window, so it is one compact line per method and nothing more."""
    return [
        f"{name} (Ch.{entry['chapter']}): {entry['title']}"
        for name, entry in sorted(METHODS.items(), key=lambda kv: (kv[1]["chapter"], kv[0]))
    ]
