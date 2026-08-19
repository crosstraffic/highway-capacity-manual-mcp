"""The paper-surface guard.

The ten public tools and the twenty-two legacy per-step tools in this repository are the tool surface the published ESWA ablation experiment ran against. Their names, their JSON schemas, their descriptions and their behaviour are part of the experimental record, so adding coverage must never move them. Every assertion in this file exists to fail loudly when one does.

``tests/data/paper_surface.json`` is the snapshot. If a test here fails, the first question is not how to update the snapshot but why a frozen tool changed: a renamed tool, a reworded description or a new required argument all change what an LLM under test could see and do, and re-running the paper's numbers against a moved surface would not reproduce them. Update the snapshot only when Rei has decided the experiment is being re-run.

The one thing deliberately not frozen here is the numeric output of the compute library. A library upgrade legitimately moves those values (the 0.3.1 interchange corrections did), so the representative-response test pins the response SHAPE exactly and the numbers at the library's own tolerance.
"""

import json
from pathlib import Path

import pytest

from hcm_mcp_server.core.registry import FunctionRegistry

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "function_registry.yaml"
SNAPSHOT = json.loads((Path(__file__).parent / "data" / "paper_surface.json").read_text())


def live_surface(include_legacy: bool):
    """The registry's view of every tool, reduced to the fields that make up the contract."""
    registry = FunctionRegistry(REGISTRY_FILE, include_legacy=include_legacy)
    return {
        name: {
            "module": info["module"],
            "function": info["function"].__name__,
            "description": info["description"],
            "category": info.get("category"),
            "chapter": info.get("chapter"),
            "step": info.get("step"),
            "parameters": info["parameters"],
        }
        for name, info in registry.get_all_functions().items()
    }


class TestPublicSurfaceIsFrozen:
    def test_the_ten_paper_tools_are_all_present(self):
        live = live_surface(include_legacy=False)
        missing = set(SNAPSHOT["public"]) - set(live)
        assert not missing, (
            f"the published ablation surface lost tools: {sorted(missing)}. "
            "These are the tools the ESWA experiment called; removing or renaming one invalidates the paper's runs."
        )

    def test_no_paper_tool_changed_name_schema_or_description(self):
        live = live_surface(include_legacy=False)
        for name, frozen in SNAPSHOT["public"].items():
            assert live[name] == frozen, (
                f"the published tool {name!r} changed. Diff against tests/data/paper_surface.json. "
                "Name, JSON schema and description are all part of what the ablation measured, so this is a revert, not a snapshot update."
            )

    def test_new_tools_are_additive_only(self):
        """Coverage may grow the tool list; it may not shrink or rewrite it."""
        live = live_surface(include_legacy=False)
        added = set(live) - set(SNAPSHOT["public"])
        assert set(SNAPSHOT["public"]) <= set(live)
        # Everything new must be in the new namespace, so no addition can collide
        # with, or be mistaken for, a tool the paper used.
        assert all(name.startswith("hcm_") for name in added), sorted(added)


class TestLegacySurfaceIsFrozen:
    """The per-step chapter families the ablation server variants load with include_legacy."""

    def test_every_legacy_tool_is_present_and_unchanged(self):
        live = live_surface(include_legacy=True)
        for name, frozen in SNAPSHOT["legacy"].items():
            assert name in live, f"legacy tool {name!r} disappeared from the include_legacy registry"
            assert live[name] == frozen, f"legacy tool {name!r} changed; see tests/data/paper_surface.json"

    def test_legacy_families_are_complete(self):
        live = live_surface(include_legacy=True)
        assert len([n for n in live if n.startswith("chapter12_")]) == 7
        assert len([n for n in live if n.startswith("chapter15_")]) == 12
        assert {"search_hcm_by_chapter", "get_hcm_section", "summarize_hcm_content"} <= set(live)

    def test_legacy_tools_stay_out_of_the_default_surface(self):
        live = live_surface(include_legacy=False)
        assert not any(n.startswith(("chapter12_", "chapter15_")) for n in live)
        assert "search_hcm_by_chapter" not in live


class TestServedSurfaceIsFrozen:
    def test_public_operations_is_exactly_the_paper_surface_in_order(self):
        import mcp_server_fastapi as srv

        assert srv.PUBLIC_OPERATIONS == [
            "analyze_facility",
            "describe_facility_inputs",
            "query_hcm",
            "propagate_change",
            "diagnose_failure",
            "repair_design",
            "repair_freeway",
            "reconcile_codes",
            "inverse_design",
            "validate_design_full",
        ]

    def test_the_default_mcp_mount_offers_only_the_paper_surface(self):
        """The full-coverage tools reach MCP only when a server opts in with HCM_MCP_FULL_COVERAGE.

        This is the load-bearing one for the experiment. mcp_server_fastapi.py is the `ct` ablation arm and it reads this default, so a change that silently widened the default surface would change what that arm measures rather than extend the server.
        """
        import mcp_server_fastapi as srv

        assert srv._mcp_kwargs["include_operations"] == srv.PUBLIC_OPERATIONS

    def test_the_opt_in_list_is_disjoint_from_the_paper_surface(self):
        import mcp_server_fastapi as srv
        from hcm_mcp_server.core import endpoints

        assert not set(endpoints.FULL_COVERAGE_OPERATIONS) & set(srv.PUBLIC_OPERATIONS)

    def test_every_paper_operation_still_has_a_backing_route(self):
        import mcp_server_fastapi as srv

        operations = {getattr(route, "operation_id", None) for route in srv.app.routes}
        assert set(srv.PUBLIC_OPERATIONS) <= operations


class TestRepresentativeResponseIsUnchanged:
    """One old tool, called end to end, with its response pinned.

    A schema snapshot catches a renamed argument but not a rewritten response, so this pins what ``analyze_facility`` actually hands back. The case is HCM Chapter 26, Example Problem 1 (the ``analyze_basic_freeway`` example fixture), so the numbers below are published values at the compute library's own Chapter 12 tolerances rather than a self-referential recording of whatever the code happens to return.
    """

    @pytest.fixture(scope="class")
    def response(self):
        inputs = json.loads(
            (Path(__file__).resolve().parent.parent / "hcm_mcp_server" / "data" / "examples" / "analyze_basic_freeway.json").read_text()
        )
        inputs.pop("_source", None)
        registry = FunctionRegistry(REGISTRY_FILE)
        impl = registry.get_function("analyze_facility")
        assert impl is not None
        return impl({"facility_type": "BasicFreeway", "inputs": inputs})

    def test_top_level_shape(self, response):
        assert set(response) == {"success", "analysis_type", "results", "facility_type", "chapter"}
        assert response["success"] is True
        assert response["analysis_type"] == "complete_hcm_chapter12"
        assert response["facility_type"] == "BasicFreeway"
        assert response["chapter"] == 12

    def test_results_shape(self, response):
        assert set(response["results"]) == {
            "inputs", "free_flow_speed", "capacity", "adjusted_capacity",
            "speed", "density", "vc_ratio", "level_of_service",
        }
        assert response["results"]["inputs"] == {
            "bffs": 75.4, "lane_width": 11.0, "lane_count": 2,
            "grade": 0.0, "demand_flow_i": 2000.0, "heavy_vehicle_pct": 0.05,
        }

    def test_published_values(self, response):
        # HCM Chapter 26, Example Problem 1: FFS 60.8 mi/h, capacity 2,308
        # pc/h/ln, density 18.8 pc/mi/ln, LOS C. Tolerances as the library
        # asserts them (FFS and density to 0.05, capacity to the integer).
        r = response["results"]
        assert r["free_flow_speed"] == pytest.approx(60.8, abs=0.05)
        assert round(r["capacity"]) == 2308
        assert r["adjusted_capacity"] == pytest.approx(r["capacity"])
        assert r["speed"] == pytest.approx(60.8, abs=0.1)
        assert r["density"] == pytest.approx(18.8, abs=0.05)
        assert r["level_of_service"] == "C"
