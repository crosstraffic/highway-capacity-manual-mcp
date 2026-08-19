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
                f"the published tool {name!r} changed. Diff against tests/data/paper_surface.json.\n"
                "If the differing field is 'function' and the live value is '<lambda>', this is NOT an edit to the tool: "
                "the registry degraded a failed module import to a placeholder. Look for a 'Could not import module' "
                "line in the captured stdout and fix the missing dependency; see TestEveryRegisteredToolImported below.\n"
                "Otherwise: name, JSON schema and description are all part of what the ablation measured, so this is a "
                "revert, not a snapshot update."
            )

    def test_new_tools_are_additive_only(self):
        """Coverage may grow the tool list; it may not shrink or rewrite it."""
        live = live_surface(include_legacy=False)
        added = set(live) - set(SNAPSHOT["public"])
        assert set(SNAPSHOT["public"]) <= set(live)
        # Full HCM coverage is three capability tools, matching the capability
        # shape of the ten frozen ones, not one tool per method. Pinning the
        # exact set keeps a later change from quietly reintroducing a
        # method-per-tool surface and tripling every caller's context cost.
        assert added == {"hcm_analyze", "hcm_describe", "hcm_validate"}, sorted(added)


class TestEveryRegisteredToolImported:
    """No registered tool may be a placeholder.

    ``FunctionRegistry.register_function`` degrades a failed module import into a lambda that answers every call with ``{"success": false, "error": "Function ... not found"}``. Nothing raises, the tool keeps its name, description and schema, and the server starts clean — so a tool can ship completely broken and look healthy.

    That is exactly how ``validation_validate_design_full`` shipped broken: ``simpleeval`` is imported unconditionally by the validator's rule engine but was declared by neither the validator's base requirements nor this project's, so it was present only in developer venvs that had picked it up by accident. Every clean install served a stub. The frozen-surface snapshot caught it by accident, through the placeholder's ``<lambda>`` name, and its message pointed at the wrong cause.

    These tests make the real cause loud. They are deliberately tests rather than a change to the registry's degradation behaviour: the server's runtime semantics are part of what the ablation measured, and a registry that started raising on import failure could change how an arm behaves under a partial install.
    """

    def test_no_registered_tool_is_an_import_placeholder(self):
        registry = FunctionRegistry(REGISTRY_FILE, include_legacy=True)
        broken = {
            name: info["module"]
            for name, info in registry.get_all_functions().items()
            if info["function"].__name__ == "<lambda>" or info.get("available") is False
        }
        assert not broken, (
            f"these tools resolved to the registry's import placeholder and answer every call with an error: {broken}. "
            "A missing dependency is the usual cause; the captured stdout carries the 'Could not import module' line "
            "naming it. Declare it in pyproject rather than relying on it being present by accident."
        )

    def test_every_module_named_in_the_registry_imports(self):
        """Checked directly rather than through the registry, so a module that fails to import is reported with its real exception instead of a swallowed warning."""
        import importlib

        import yaml

        config = yaml.safe_load(REGISTRY_FILE.read_text())
        modules = {
            entry["module"]
            for block in ("functions", "legacy_functions")
            for section in config.get(block, {}).values()
            for entry in section.values()
        }
        failures = {}
        for module in sorted(modules):
            try:
                importlib.import_module(module)
            except Exception as e:  # noqa: BLE001 -- the point is to report it
                failures[module] = f"{type(e).__name__}: {e}"
        assert not failures, f"registry modules that do not import: {failures}"

    def test_the_full_corpus_validator_actually_runs(self):
        """The tool whose breakage started this. Its own suite uses ``importorskip``, so when the import broke that suite skipped rather than failed -- the second reason a stub shipped unnoticed. This one does not skip."""
        import asyncio

        registry = FunctionRegistry(REGISTRY_FILE)
        impl = registry.get_function("validation_validate_design_full")
        assert impl is not None
        result = asyncio.run(impl({"design": {"lane_width": 8.0, "facility_type": "TwoLaneHighway"}}))
        assert result["success"] is True, result.get("error")
        assert result["error_count"] >= 1, "a 8 ft lane should violate the Exhibit 15-8 range"
        assert any(v.get("citation") for v in result["violations"]), "violations must carry citations"


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
