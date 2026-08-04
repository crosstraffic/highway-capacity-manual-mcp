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

    def test_pending_facility_reports_adapter_pending(self):
        r = analysis.analyze_facility_function({"facility_type": "WeavingSegment", "inputs": {}})
        assert r["success"] is False
        assert r["status"] == "adapter_pending"
        assert "WeavingSegment" in r["error"]

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


class TestDiscovery:
    def test_describe_without_type_lists_all_core_motorized_chapters(self):
        r = analysis.describe_facility_inputs_function({})
        assert r["success"] is True
        chapters = {row["chapter"] for row in r["facility_types"]}
        assert set(range(10, 25)) <= chapters
        assert "BasicFreeway" in r["available"]
        assert "TwoLaneHighway" in r["available"]
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

    def test_describe_pending_facility_says_so(self):
        r = analysis.describe_facility_inputs_function({"facility_type": "Roundabout"})
        assert r["success"] is False
        assert r["status"] == "adapter_pending"


class TestRegistryLegacySplit:
    REGISTRY = Path(__file__).resolve().parent.parent / "function_registry.yaml"

    def test_default_surface_is_ten_general_tools(self):
        reg = FunctionRegistry(self.REGISTRY)
        names = set(reg.get_all_functions())
        assert {"analyze_facility", "describe_facility_inputs", "query_hcm"} <= names
        assert len(names) == 10
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
