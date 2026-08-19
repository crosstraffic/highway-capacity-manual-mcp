"""The unified analysis surface: analyze_facility dispatch, discovery tools, and the legacy registry split."""

from pathlib import Path

from hcm_mcp_server.core.registry import FunctionRegistry
from hcm_mcp_server.functions import analysis

FREEWAY_INPUTS = {
    "bffs": 70.0, "lw": 12.0, "lane_count": 3, "lc_r": 6, "trd": 1,
    "demand_flow_i": 4000.0, "phf": 0.95, "p_t": 0.10, "grade": 2.0, "length": 1.0,
}

TWO_LANE_INPUTS = {
    "lane_width": 12.0, "shoulder_width": 6.0, "apd": 5.0,
    "segments": [{
        "passing_type": 0, "length": 2.0, "grade": 2.0, "spl": 60.0,
        "volume": 650.0, "phv": 0.05, "phf": 0.94, "vertical_class": 1,
    }],
}


class TestAnalyzeFacility:
    def test_basic_freeway_runs_complete_analysis(self):
        r = analysis.analyze_facility_function({"facility_type": "BasicFreeway", "inputs": FREEWAY_INPUTS})
        assert r["success"] is True
        assert r["facility_type"] == "BasicFreeway"
        assert r["chapter"] == 12
        assert r["results"]["level_of_service"] in list("ABCDEF")

    def test_two_lane_highway_runs_complete_analysis(self):
        r = analysis.analyze_facility_function({"facility_type": "TwoLaneHighway", "inputs": TWO_LANE_INPUTS})
        assert r["success"] is True
        assert r["facility_type"] == "TwoLaneHighway"
        assert r["chapter"] == 15

    def test_unknown_facility_is_a_clean_error_listing_valid_types(self):
        r = analysis.analyze_facility_function({"facility_type": "Hyperloop", "inputs": {}})
        assert r["success"] is False
        assert "BasicFreeway" in r["error"]

    def test_every_facility_type_has_an_adapter(self):
        assert all("run" in entry for entry in analysis.FACILITIES.values())

    def test_missing_inputs_is_a_clean_error(self):
        r = analysis.analyze_facility_function({"facility_type": "BasicFreeway"})
        assert r["success"] is False
        assert "inputs" in r["error"]

    def test_invalid_inputs_surface_validation_message(self):
        r = analysis.analyze_facility_function({
            "facility_type": "BasicFreeway",
            "inputs": {"lane_count": "three"},
        })
        assert r["success"] is False
        assert "Invalid inputs" in r["error"]


class TestNewAdapters:
    """Each adapter is pinned to the published values of a library example case (the same HCM example problems the Rust integration suite asserts)."""

    def test_weaving_reproduces_hcm_ch27_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "WeavingSegment", "inputs": {
            "weaving_type": "one_sided", "facility_type": "freeway", "length_short": 1500.0,
            "num_lanes": 4, "num_weaving_lanes": 3, "ffs": 65.0,
            "v_ff": 1815.0, "v_fr": 692.0, "v_rf": 1037.0, "v_rr": 1297.0,
            "phf": 0.91, "heavy_vehicle_pct": 0.05, "terrain": "level",
            "lc_rf": 0, "lc_fr": 1, "lc_rr": 0, "interchange_density": 0.8,
            "basic_freeway_capacity": 2350.0, "caf": 1.0, "saf": 1.0,
        }})
        assert r["success"] is True
        assert r["level_of_service"] == "C"
        assert abs(r["results"]["speed_avg"] - 53.1) < 0.5
        assert abs(r["results"]["density"] - 26.3) < 0.5

    def test_ramp_reproduces_hcm_ch14_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "RampSegment", "inputs": {
            "ramp_type": "OnRamp", "ramp_side": "Right", "ramp_lanes": 1,
            "freeway_lanes": 2, "freeway_ffs": 60.0, "ramp_ffs": 45.0,
            "accel_lane_length": 740.0, "freeway_demand": 2500.0, "ramp_demand": 535.0,
            "phf": 0.90, "heavy_vehicle_pct": 0.05, "terrain": "Level",
            "adjacent_upstream": "None", "adjacent_downstream": "None", "caf": 1.0, "saf": 1.0,
        }})
        assert r["success"] is True
        assert r["level_of_service"] == "D"
        assert abs(r["results"]["density"] - 28.2) < 0.5

    def test_roundabout_reproduces_hcm_ch33_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "Roundabout", "inputs": {
            "nb": {"v_u": 30.0, "v_l": 105.0, "v_t": 210.0, "v_r": 50.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "None", "n_ped": 50.0},
            "sb": {"v_u": 20.0, "v_l": 175.0, "v_t": 95.0, "v_r": 580.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "NonYielding", "n_ped": 0.0},
            "eb": {"v_u": 50.0, "v_l": 190.0, "v_t": 280.0, "v_r": 85.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "None", "n_ped": 0.0},
            "wb": {"v_u": 20.0, "v_l": 110.0, "v_t": 395.0, "v_r": 610.0, "heavy_vehicle_pct": 2.0, "entry_lanes": 1, "circulating_lanes": 1, "exiting_lanes": 1, "bypass": "Yielding", "n_ped": 0.0},
            "phf": 0.94, "analysis_period_h": 0.25,
        }})
        assert r["success"] is True
        assert r["intersection_los"] == "C"

    def test_awsc_reproduces_hcm_ch32_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "AWSC", "inputs": {
            "eb": {"lanes": [{"volume_left": 50.0, "volume_through": 300.0, "volume_right": 0.0}], "heavy_vehicle_pct": 2.0},
            "wb": {"lanes": [{"volume_left": 0.0, "volume_through": 300.0, "volume_right": 100.0}], "heavy_vehicle_pct": 2.0},
            "nb": {"lanes": [], "heavy_vehicle_pct": 0.0},
            "sb": {"lanes": [{"volume_left": 100.0, "volume_through": 0.0, "volume_right": 50.0}], "heavy_vehicle_pct": 2.0},
            "phf": 0.95, "analysis_period_h": 0.25,
        }})
        assert r["success"] is True
        assert r["intersection_los"] == "B"
        assert abs(r["intersection_delay"] - 12.8) < 0.5

    def test_twsc_runs_hcm_ch32_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "TWSC", "inputs": {
            "demand": {"v2": 240.0, "v3": 40.0, "v4": 160.0, "v5": 300.0, "v7": 40.0, "v9": 120.0},
            "geometry": {"is_three_leg": True, "major_lanes_per_direction": 1, "major_right_turn_eb": "Shared", "major_right_turn_wb": "Shared", "minor_lanes_nb": "SingleShared"},
            "phf": None, "analysis_period_h": 0.25, "heavy_vehicle_pct": 10.0,
        }})
        assert r["success"] is True
        assert r["intersection_delay"] >= 0
        assert "movements" in r["results"]

    def test_signalized_reproduces_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "signalized_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "SignalizedIntersection", "inputs": case})
        assert r["success"] is True
        assert r["intersection_los"] == "D"

    def test_urban_segment_reproduces_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "urbansegments_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "UrbanSegment", "inputs": case})
        assert r["success"] is True
        assert r["los"] == "C"

    def test_freeway_facility_runs_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "freewayfacilities_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "FreewayFacility", "inputs": case})
        assert r["success"] is True
        matrix = r["los"]
        assert len(matrix) == 11 and len(matrix[0]) == 5
        assert all(cell in list("ABCDEF") for row in matrix for cell in row)

    def test_freeway_reliability_runs_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "freewayreliability_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "FreewayReliability", "inputs": case})
        assert r["success"] is True
        assert abs(r["results"]["reliability_rating"] - 84.2) < 0.5
        assert abs(r["results"]["tti_mean"] - 1.241) < 0.01

    def test_urban_facility_runs_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "urbanfacilities_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "UrbanFacility", "inputs": case})
        assert r["success"] is True
        assert r["los"] == "A"
        assert r["poorest_segment_los"] == "B"

    def test_urban_reliability_runs_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "urbanreliability_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "UrbanReliability", "inputs": case})
        assert r["success"] is True
        assert abs(r["results"]["reliability_rating"] - 98.8) < 0.5
        assert abs(r["results"]["tti_mean"] - 1.545) < 0.01

    def test_ramp_terminal_runs_library_example_1(self):
        import json as _json
        case = _json.loads((Path(__file__).parent / "data" / "rampterminals_case1.json").read_text())
        r = analysis.analyze_facility_function({"facility_type": "RampTerminal", "inputs": case})
        assert r["success"] is True
        assert r["interchange_los"] == "C"
        # Engine value, re-anchored when the pin moved with transportations-library
        # 0.3.1 (interchange aggregate LOS from weighted ETT only, d2 on lane-group
        # capacity). The library's own Chapter 23 test asserts 50.7 +-0.5 for this
        # fixture; the published Exhibit 34-16 figure is 52.4.
        assert abs(r["interchange_ett"] - 50.7) < 0.5

    def test_pedestrian_walkway_reproduces_hcm_ch35_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "PedestrianWalkway", "inputs": {
            "total_walkway_width": 5.0, "fixed_object_width": 0.0,
            "peak_15min_volume": 100.0, "phf": 0.83, "pedestrian_speed": 240.0,
            "facility_type": "walkway", "flow_type": "random",
        }})
        assert r["success"] is True
        assert r["level_of_service"] == "A"
        assert abs(r["results"]["pedestrian_space"] - 180.0) < 2.0

    def test_shared_use_path_pedestrian_reproduces_hcm_ch35_example_1(self):
        r = analysis.analyze_facility_function({"facility_type": "SharedUsePathPedestrian", "inputs": {
            "bicycle_demand_same_direction": 100.0, "bicycle_demand_opposing": 100.0,
            "phf": 0.83, "pedestrian_speed": 4.0, "bicycle_speed": 16.0, "is_one_way": False,
        }})
        assert r["success"] is True
        assert r["level_of_service"] == "E"
        assert abs(r["results"]["total_events"] - 166.0) < 2.0

    def test_offstreet_bicycle_reproduces_hcm_ch35_example_2(self):
        r = analysis.analyze_facility_function({"facility_type": "OffStreetBicycle", "inputs": {
            "path_width": 10.0, "segment_length": 3.0, "has_centerline": False,
            "two_way_demand": 340.0, "directional_split": 0.5, "phf": 0.90, "is_one_way": False,
            "mode_splits": [0.55, 0.20, 0.10, 0.10, 0.05],
            "mode_speeds": [12.8, 3.4, 6.5, 10.1, 7.9],
            "mode_speed_sds": [3.4, 0.6, 1.2, 2.7, 1.9],
        }})
        assert r["success"] is True
        assert r["level_of_service"] == "D"
        assert abs(r["results"]["blos_score"] - 2.69) < 0.02

    def test_bad_json_config_is_a_clean_error(self):
        r = analysis.analyze_facility_function({"facility_type": "Roundabout", "inputs": {"nonsense": True}})
        assert r["success"] is False
        assert "error" in r


class TestDiscovery:
    def test_describe_without_type_lists_all_core_motorized_chapters(self):
        r = analysis.describe_facility_inputs_function({})
        assert r["success"] is True
        chapters = {row["chapter"] for row in r["facility_types"]}
        assert set(range(10, 25)) <= chapters
        assert {"BasicFreeway", "TwoLaneHighway", "WeavingSegment", "RampSegment", "TWSC", "AWSC", "Roundabout"} <= set(r["available"])
        for row in r["facility_types"]:
            assert row["status"] in ("available", "adapter_pending")

    def test_describe_basic_freeway_fields(self):
        r = analysis.describe_facility_inputs_function({"facility_type": "BasicFreeway"})
        assert r["success"] is True
        names = {f["name"] for f in r["fields"]}
        assert {"bffs", "lw", "lane_count", "demand_flow_i", "sut_percentage"} <= names

    def test_describe_two_lane_recurses_into_segments(self):
        r = analysis.describe_facility_inputs_function({"facility_type": "TwoLaneHighway"})
        assert r["success"] is True
        seg = next(f for f in r["fields"] if f["name"] == "segments")
        item_names = {f["name"] for f in seg["item_fields"]}
        assert {"passing_type", "length", "grade", "spl"} <= item_names

    def test_describe_json_config_facility_serves_an_example(self):
        r = analysis.describe_facility_inputs_function({"facility_type": "RampTerminal"})
        assert r["success"] is True
        assert "example" in r

    def test_every_facility_type_is_describable(self):
        for name in analysis.FACILITIES:
            r = analysis.describe_facility_inputs_function({"facility_type": name})
            assert r["success"] is True, name
            assert "fields" in r or "example" in r, name


class TestRegistryLegacySplit:
    REGISTRY = Path(__file__).resolve().parent.parent / "function_registry.yaml"

    def test_default_surface_is_the_ten_general_tools_plus_the_per_method_family(self):
        # The ten general tools are the published ablation surface and are frozen
        # (tests/test_frozen_surface.py). Everything else in the default registry
        # is the additive per-method family, which lives under the hcm_ prefix.
        reg = FunctionRegistry(self.REGISTRY)
        names = set(reg.get_all_functions())
        assert {"analyze_facility", "describe_facility_inputs", "query_hcm"} <= names
        assert len({n for n in names if not n.startswith("hcm_")}) == 10
        assert not any(n.startswith("chapter12_") or n.startswith("chapter15_") for n in names)
        assert "search_hcm_by_chapter" not in names

    def test_legacy_flag_restores_per_step_families(self):
        reg = FunctionRegistry(self.REGISTRY, include_legacy=True)
        names = set(reg.get_all_functions())
        assert "chapter12_complete_analysis" in names
        assert "chapter15_determine_segment_los" in names
        assert "search_hcm_by_chapter" in names
        assert "analyze_facility" in names


class TestServedSurface:
    """The MCP tool surface is FastApiMCP over route operation ids, not the registry — so the served app must expose the new routes, keep every legacy REST route resolvable, and default the MCP mount to the public tool set."""

    def test_public_operations_are_routes_and_default_mcp_surface(self):
        import mcp_server_fastapi as srv

        op_ids = {getattr(r, "operation_id", None) for r in srv.app.routes}
        missing = set(srv.PUBLIC_OPERATIONS) - op_ids
        assert not missing, f"PUBLIC_OPERATIONS without a backing route: {missing}"
        assert srv._mcp_kwargs.get("include_operations") is not None

    def test_app_registry_resolves_every_routed_function(self):
        # Every registry.get_function(name) lookup made by a route handler must
        # resolve on the app's include_legacy registry — a None here means a
        # live REST endpoint that can only 404.
        import re
        root = Path(__file__).resolve().parent.parent
        reg = FunctionRegistry(root / "function_registry.yaml", include_legacy=True)
        src_dir = root / "hcm_mcp_server" / "core"
        looked_up = set()
        for f in ("endpoints.py", "reasoning_endpoints.py"):
            looked_up |= set(re.findall(r"get_function\(\"([a-z0-9_]+)\"\)", (src_dir / f).read_text()))
        unresolved = {n for n in looked_up if reg.get_function(n) is None}
        assert not unresolved, f"routes whose implementation is missing from the legacy registry: {unresolved}"
