"""Tests for the X-KG reasoning functions (hcm_mcp_server.functions.reasoning).

The reasoning layer needs newer dependencies than are published yet: ``transportations-validator`` with the repair/reconcile/inverse/executors modules, and ``transportations-library`` with the BasicFreeways PyO3 binding. Until those are released, install both editable for local dev:

    uv pip install -e ../transportations-validator -e ../transportations-library

If the reasoning module can't import (older validator), the whole module is skipped; the BasicFreeway tests additionally skip when the library lacks the BasicFreeways binding — so CI stays green until the deps ship.
"""

import pytest

reasoning = pytest.importorskip(
    "hcm_mcp_server.functions.reasoning",
    reason=(
        "reasoning layer needs transportations-validator>=0.2.0 + "
        "transportations-library>=0.1.12 (BasicFreeways) — install editable for dev"
    ),
)

try:
    import transportations_library as _tl

    _HAVE_BASICFREEWAY = hasattr(_tl, "BasicFreeways")
except Exception:  # pragma: no cover
    _HAVE_BASICFREEWAY = False

needs_basicfreeway = pytest.mark.skipif(
    not _HAVE_BASICFREEWAY,
    reason="transportations-library build lacks the BasicFreeways PyO3 binding",
)


class TestChaining:
    def test_propagate_change_finds_downstream(self):
        result = reasoning.propagate_change_function(
            {"root": "lane_width", "facility_type": "TwoLaneHighway"}
        )
        assert result["success"] is True
        assert len(result["chain"]) > 0

    def test_diagnose_failure_finds_upstream_causes(self):
        result = reasoning.diagnose_failure_function(
            {"target": "los", "facility_type": "BasicFreeway"}
        )
        assert result["success"] is True
        assert len(result["chain"]) > 0

    def test_missing_root_is_a_clean_error(self):
        result = reasoning.propagate_change_function({"facility_type": "TwoLaneHighway"})
        assert result["success"] is False
        assert "error" in result


class TestRepair:
    def test_repair_design_twolane_widens_lane(self):
        # 9 ft lanes, no shoulder, 650 veh/h -> LOS D; repair reaches LOS C.
        result = reasoning.repair_design_function({
            "design": {
                "passing_type": 0, "length": 2.0, "grade": 2.0, "spl": 60.0,
                "volume": 650.0, "phv": 0.08, "phf": 0.94,
                "lane_width": 9.0, "shoulder_width": 0.0, "apd": 20.0,
            },
            "goal_los": "C",
        })
        assert result["success"] is True
        assert result["baseline_evaluated"]["los"] == "D"
        assert result["repaired"] is True

    @needs_basicfreeway
    def test_repair_freeway_basicfreeway_widens_lane(self):
        # 10 ft lanes, 25% trucks, 3000 veh/h -> LOS E; repair reaches LOS D.
        result = reasoning.repair_freeway_function({
            "design": {
                "bffs": 70.0, "lw": 10.0, "lane_count": 2, "lc_r": 6, "trd": 1,
                "demand_flow_i": 3000.0, "phf": 0.95, "p_t": 0.25,
                "grade": 2.0, "length": 0.625,
            },
            "goal_los": "D",
        })
        assert result["success"] is True
        assert result["baseline_evaluated"]["los"] == "E"
        assert result["repaired"] is True
        changed = {
            c["parameter"]
            for p in result["proposals"] if p["compliant"]
            for c in p["changes"]
        }
        assert "lw" in changed

    @needs_basicfreeway
    def test_repair_freeway_off_grid_is_clean_error(self):
        # Off the heavy-vehicle PCE grid -> non-evaluable, not a crash.
        result = reasoning.repair_freeway_function({
            "design": {
                "bffs": 70.0, "lw": 10.0, "lane_count": 2,
                "demand_flow_i": 3000.0, "grade": 3.7, "length": 0.5, "p_t": 0.25,
            },
            "goal_los": "D",
        })
        assert result["success"] is False
        assert "non-evaluable" in result["error"]


class TestReconcile:
    def test_reconcile_scenario_produces_trace(self):
        result = reasoning.reconcile_codes_function(
            {"scenario": "lane_width_state_trunk", "value": 11.0}
        )
        assert result["success"] is True
        assert len(result["trace_lines"]) > 0

    def test_unknown_scenario_is_a_clean_error(self):
        result = reasoning.reconcile_codes_function({"scenario": "does_not_exist"})
        assert result["success"] is False
        assert "Unknown conflict scenario" in result["error"]

    def test_no_scenario_or_claims_is_a_clean_error(self):
        result = reasoning.reconcile_codes_function({})
        assert result["success"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
