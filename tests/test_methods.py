"""The per-method HCM surface: every hcm_analyze_* tool driven through the MCP call path against its example-problem fixture.

Each tool is called the way ``/tools/call`` calls it — ``registry.get_function(name)(arguments)`` — so a test failing here means either the analysis is wrong or the registry wiring is, and the two are not confused.

Every expected value and tolerance below is cribbed from the compute library's own test suite for that example problem (``transportations-library/tests/test_chapter*_integration.py`` for the Python-tested chapters, ``tests/chapter*_integration.rs`` for the ones the Rust side pins). Nothing here is an invented tolerance: where the library asserts +-0.5 mi/h, so does this file, and where the library documents a computed-versus-published gap the value asserted is the library's computed one with the published one in the comment. The published-value provenance lives in the fixtures' own ``_source`` lines.
"""

import json
from pathlib import Path

import pytest

from hcm_mcp_server.core.registry import FunctionRegistry
from hcm_mcp_server.functions import methods

REGISTRY_FILE = Path(__file__).resolve().parent.parent / "function_registry.yaml"
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def registry():
    return FunctionRegistry(REGISTRY_FILE)


def call(registry, tool, **arguments):
    """Invoke a tool exactly as the /tools/call endpoint does."""
    impl = registry.get_function(tool)
    assert impl is not None, f"{tool} is not registered"
    return impl(arguments)


def run_example(registry, method, **extra):
    """Run a method against its own shipped example-problem fixture."""
    result = call(registry, f"hcm_{method}", config=methods._example(method), **extra)
    assert result.get("success") is True, result.get("error")
    return result


# ── Chapter 10: freeway facilities ───────────────────────────────────────────

class TestFreewayFacility:
    """HCM Chapter 25, Example Problem 1 (Exhibits 25-43 through 25-52)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_freeway_facility")["results"]

    def test_structure(self, r):
        assert r["num_segments"] == 11
        assert r["num_periods"] == 5
        assert r["total_length_mi"] == pytest.approx(6.0, abs=0.01)
        assert r["oversaturated"] is False

    def test_period_1_matrices(self, r):
        # Exhibit 25-49 speeds, 25-50 densities, 25-51 LOS letters.
        speeds = [60.0, 53.9, 59.7, 56.1, 60.0, 48.0, 59.9, 53.4, 53.4, 56.0, 59.7]
        densities = [25.0, 30.6, 27.6, 29.4, 26.0, 27.2, 27.1, 33.2, 33.2, 31.6, 28.1]
        letters = ["C", "C", "D", "C", "D", "C", "D", "D", "D", "D", "D"]
        for i, (s, d, letter) in enumerate(zip(speeds, densities, letters)):
            assert r["speed"][i][0] == pytest.approx(s, abs=0.5), i
            assert r["density_veh"][i][0] == pytest.approx(d, abs=0.5), i
            assert r["los"][i][0] == letter, i

    def test_facility_performance(self, r):
        # Exhibit 25-52.
        for p, (speed, density, letter) in enumerate([
            (57.6, 27.5, "D"), (56.6, 31.3, "D"), (55.0, 34.8, "E"),
            (57.9, 27.5, "D"), (58.4, 21.4, "C"),
        ]):
            assert r["facility_speed"][p] == pytest.approx(speed, abs=0.5), p
            assert r["facility_density_veh"][p] == pytest.approx(density, abs=0.5), p
            assert r["facility_los"][p] == letter, p
        assert r["overall_speed"] == pytest.approx(56.9, abs=0.5)
        assert r["overall_density_veh"] == pytest.approx(28.4, abs=0.5)


class TestManagedLanes:
    """HCM Chapter 25, Example Problem 5 (Exhibits 25-81 through 25-87)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_managed_lanes")["results"]

    def test_ml_capacity_and_demand(self, r):
        # Exhibit 25-81: 1,614 veh/h uniform. Exhibit 25-82: vd/c by period.
        assert r["ml_capacity"][0][0] == pytest.approx(1614.0, abs=3.0)
        for p, e in enumerate([0.62, 0.68, 0.72, 0.64, 0.52]):
            assert r["ml_dc_ratio"][0][p] == pytest.approx(e, abs=0.005), p

    def test_adjacent_friction(self, r):
        # Exhibit 25-83: unaffected 59.3 mi/h in period 1; segments 8-9 drop to
        # 53.5 in period 2 where adjacent GP density passes 35 pc/mi/ln.
        assert r["ml_speed"][0][0] == pytest.approx(59.3, abs=0.3)
        assert r["ml_speed"][7][1] == pytest.approx(53.5, abs=0.4)
        assert r["ml_speed"][7][2] == pytest.approx(52.1, abs=0.4)
        assert r["ml_friction_active"][7][2] is True
        assert r["ml_friction_active"][0][0] is False

    def test_combined_facility(self, r):
        # Exhibit 25-87.
        for p, (speed, letter) in enumerate([(58.0, "C"), (57.5, "D"), (56.7, "D"), (58.2, "C"), (58.7, "C")]):
            assert r["facility_speed"][p] == pytest.approx(speed, abs=0.6), p
            assert r["facility_los"][p] == letter, p


class TestPlanningFacility:
    """HCM Chapter 25, Example Problem 6 (Exhibits 25-88 through 25-96)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_planning_facility")["results"]

    def test_dc_ratios(self, r):
        # Exhibit 25-91, by period then section.
        expected = [
            [0.72, 0.86, 0.74, 0.65, 0.76, 0.91, 0.79],
            [0.80, 0.96, 0.82, 0.72, 0.85, 1.02, 0.88],
            [0.72, 0.86, 0.74, 0.65, 0.76, 0.93, 0.80],
            [0.64, 0.77, 0.66, 0.58, 0.68, 0.81, 0.70],
        ]
        assert r["num_sections"] == 7
        assert r["num_periods"] == 4
        for p, row in enumerate(expected):
            for i, e in enumerate(row):
                assert r["dc_ratio"][i][p] == pytest.approx(e, abs=0.01), (i, p)

    def test_facility_performance(self, r):
        # Exhibit 25-96: period 2 is the oversaturated one.
        for p, (speed, density, letter) in enumerate([
            (58.9, 29.2, "D"), (56.6, 33.7, "F"), (58.8, 29.4, "D"), (59.8, 25.5, "C"),
        ]):
            assert r["facility_speed"][p] == pytest.approx(speed, abs=0.6), p
            assert r["facility_density"][p] == pytest.approx(density, abs=0.8), p
            assert r["facility_los"][p] == letter, p


# ── Chapter 11: freeway reliability ──────────────────────────────────────────

class TestFreewayReliability:
    """HCM Chapter 25, Example Problem 7 (Exhibits 25-97 through 25-105).

    The published measures come from FREEVAL's own Monte Carlo stream, so the library checks the central ones inside documented bands rather than exactly. Those are the bands used here.
    """

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_freeway_reliability")["results"]

    def test_scenario_set(self, r):
        assert r["num_scenarios"] == 240
        assert r["num_observations"] == 240 * 12
        assert r["free_flow_travel_time_min"] == pytest.approx(6.0, abs=0.01)
        assert r["expected_vhd"] > 0.0

    def test_reliability_measures(self, r):
        assert r["tti_percentile"]["50"] == pytest.approx(1.04, abs=0.01)  # published 1.03
        assert r["tti_mean"] == pytest.approx(1.24, abs=0.05)  # published 1.30
        assert r["reliability_rating"] == pytest.approx(84.2, abs=1.5)  # published 90.8
        assert r["tti_percentile"]["95"] >= r["tti_percentile"]["80"] >= r["tti_percentile"]["50"] >= 1.0
        assert r["misery_index"] >= r["tti_mean"]


# ── Chapter 12: basic freeway ────────────────────────────────────────────────

class TestBasicFreeway:
    """HCM Chapter 26, Example Problem 1 (four-lane freeway segment)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_basic_freeway")

    def test_published_values(self, result):
        r = result["results"]
        assert r["ffs"] == pytest.approx(60.8, abs=0.05)
        assert round(r["capacity"]) == 2308
        assert r["density"] == pytest.approx(18.8, abs=0.05)
        assert r["speed"] == pytest.approx(60.8, abs=0.1)
        assert round(r["density"] * r["speed"]) == pytest.approx(1142, abs=2)
        assert result["level_of_service"] == "C"


# ── Chapter 13: weaving, both editions ───────────────────────────────────────

class TestWeaving:
    """HCM Chapter 27, Example Problem 1 (major weaving segment)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_weaving")

    def test_published_values(self, result):
        r = result["results"]
        assert r["flow_weaving"] == pytest.approx(1995.0, abs=5.0)
        assert r["flow_nonweaving"] == pytest.approx(3591.0, abs=5.0)
        assert r["flow_total"] == pytest.approx(5586.0, abs=5.0)
        assert r["volume_ratio"] == pytest.approx(0.357, abs=0.002)
        assert r["lc_min"] == pytest.approx(798.0, abs=5.0)
        assert r["l_max"] == pytest.approx(4639.0, abs=5.0)
        assert r["is_weaving"] is True
        assert r["capacity"] == pytest.approx(8038.0, abs=10.0)
        assert r["lc_all"] == pytest.approx(1926.0, abs=8.0)
        assert r["speed_weaving"] == pytest.approx(54.2, abs=0.5)
        assert r["speed_nonweaving"] == pytest.approx(52.5, abs=0.5)
        assert r["speed_avg"] == pytest.approx(53.1, abs=0.5)
        assert r["density"] == pytest.approx(26.3, abs=0.5)
        assert result["level_of_service"] == "C"

    def test_default_edition_is_hcm_7(self, result):
        assert result["results"]["version"] == "7"
        assert "analysis_v7_1" not in result["results"]

    def test_hcm_7_1_edition(self, registry):
        # 7.1 replaces the single num_weaving_lanes count with per-movement
        # weaving-lane counts, so the 7th-Edition fixture gains nw_rf and nw_fr.
        config = dict(methods._example("analyze_weaving"), version="7.1", nw_rf=2, nw_fr=1)
        result = call(registry, "hcm_analyze_weaving", config=config)
        assert result["success"] is True
        a = result["results"]["analysis_v7_1"]
        assert a["class"] == "Complex"
        assert a["speed_basic"] == pytest.approx(65.0, abs=1e-9)
        assert a["weaving_intensity"] == pytest.approx(0.006336, abs=5e-6)
        assert a["speed_impedance"] == pytest.approx(5.68, abs=0.02)
        assert a["speed_avg"] == pytest.approx(59.32, abs=0.02)
        assert a["capacity_per_lane"] == pytest.approx(1866.0, abs=2.0)
        assert a["dc_ratio"] == pytest.approx(0.75, abs=0.005)
        assert a["density"] == pytest.approx(23.6, abs=0.1)
        assert a["los"] == "C"

    def test_the_two_editions_disagree(self, registry, result):
        """7.1 replaces the weaving equations rather than adjusting them, so the same segment must not land on the same density."""
        config = dict(methods._example("analyze_weaving"), version="7.1", nw_rf=2, nw_fr=1)
        v71 = call(registry, "hcm_analyze_weaving", config=config)
        assert abs(result["results"]["density"] - v71["results"]["analysis_v7_1"]["density"]) > 1.0


# ── Chapter 14: merge and diverge, both editions ─────────────────────────────

class TestMergeDiverge:
    """HCM Chapter 28, Example Problem 2, first off-ramp."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_merge_diverge")

    def test_published_values(self, result):
        r = result["results"]
        assert r["flow_freeway"] == pytest.approx(5093.0, abs=5.0)
        assert r["flow_ramp"] == pytest.approx(340.0, abs=2.0)
        assert r["p_f"] == pytest.approx(0.617, abs=0.002)
        assert r["v12"] == pytest.approx(3273.0, abs=6.0)
        assert r["capacity_freeway"] == pytest.approx(6900.0)
        assert r["capacity_ramp"] == pytest.approx(2000.0)
        assert r["demand_exceeds_capacity"] is False
        assert r["exceeds_max_desirable"] is False
        assert r["density"] == pytest.approx(27.9, abs=0.5)
        assert r["speed_ramp"] == pytest.approx(52.9, abs=0.5)
        assert r["speed_outer"] == pytest.approx(62.6, abs=0.5)
        assert r["speed_avg"] == pytest.approx(56.0, abs=0.5)
        assert result["level_of_service"] == "C"

    def test_hcm_7_1_edition(self, registry):
        # Chapter 28 Example Problem 1 (on-ramp), the case the library pins its
        # 7.1 merge/diverge values on.
        config = dict(json.loads((DATA / "mergediverge_ep1.json").read_text()), version="7.1")
        result = call(registry, "hcm_analyze_merge_diverge", config=config)
        assert result["success"] is True
        a = result["results"]["analysis_v7_1"]
        assert a["flow_freeway"] == pytest.approx(2918.0, abs=2.0)
        assert a["flow_ramp"] == pytest.approx(624.0, abs=1.0)
        assert a["speed_basic"] == pytest.approx(59.47, abs=0.02)
        assert a["speed_impedance"] == pytest.approx(4.37, abs=0.02)
        assert a["speed_avg"] == pytest.approx(55.10, abs=0.03)
        assert a["capacity_per_lane"] == pytest.approx(1882.0, abs=3.0)
        assert a["density"] == pytest.approx(32.1, abs=0.1)
        assert a["los"] == "E"


# ── Chapter 15: two-lane highways ────────────────────────────────────────────

class TestTwoLaneHighway:
    """HCM Chapter 15, Example Problem 1 (level straight passing-constrained segment)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_two_lane_highway")

    def test_published_values(self, result):
        r = result["results"]
        segment = r["segments"][0]
        assert round(segment["free_flow_speed"], 2) == 56.83
        assert round(segment["percent_followers"], 1) == 67.7
        # Step 11 aggregates the ADJUSTED segment densities (Equation 15-39) and
        # Exhibit 15-6 splits on the posted limit, not the computed speed. Both
        # are the library's own facility convention, pinned in its integration
        # test at 10.092 / LOS D for this fixture.
        assert round(r["facility_follower_density"], 3) == 10.092
        assert r["facility_posted_speed_limit"] == pytest.approx(50.0)
        assert result["level_of_service"] == "D"


# ── Chapters 16-18: urban streets ────────────────────────────────────────────

class TestUrbanFacility:
    """HCM Chapter 29, Example Problem 1 (Exhibit 29-49), eastbound."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_urban_facility")

    def test_published_values(self, result):
        r = result["results"]
        assert r["num_segments"] == 5
        assert r["length_ft"] == pytest.approx(5280.0)
        assert r["base_free_flow_speed_mph"] == pytest.approx(40.1, abs=0.05)
        assert r["travel_speed_mph"] == pytest.approx(22.6, abs=0.6)
        assert r["spatial_stop_rate"] == pytest.approx(1.83, abs=0.15)
        assert r["critical_vc_ratio"] <= 1.0
        assert result["level_of_service"] == "C"
        assert r["poorest_segment_los"] == "D"

    def test_unknown_mode_is_refused(self, registry):
        result = call(registry, "hcm_analyze_urban_facility",
                      config=methods._example("analyze_urban_facility"), mode="wishful")
        assert result["success"] is False
        assert "aggregate" in result["error"]


class TestUrbanReliability:
    """HCM Chapter 29, Example Problem 4 (Exhibit 29-73)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_urban_reliability")["results"]

    def test_published_values(self, r):
        # Published 3,120 scenarios (4/h x 3 h x 5 days x 52 weeks), exact.
        assert r["num_scenarios"] == 3120
        assert r["base_free_flow_travel_time_s"] == pytest.approx(262.9, abs=10.0)
        # The HCM notes the seeded Monte Carlo streams are software-specific, so
        # the library checks these at band level (published mean TTI 1.69/1.64,
        # PTI 2.98/2.61, reliability rating 93.2/94.1).
        assert 1.1 <= r["tti_mean"] <= 2.6
        assert 1.3 <= r["tti_percentile"]["95"] <= 5.0
        assert r["tti_percentile"]["80"] <= r["tti_percentile"]["95"]
        assert 70.0 <= r["reliability_rating"] <= 100.0
        assert r["total_vhd"] > 0.0
        assert r["num_incidents"] > 50
        assert r["num_weather_events"] > 50


class TestUrbanSegment:
    """HCM Chapter 30, Example Problem 1 (Exhibit 30-36), eastbound."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_urban_segment")

    def test_published_values(self, result):
        r = result["results"]
        assert r["base_free_flow_speed_mph"] == pytest.approx(40.78, abs=0.01)
        assert r["free_flow_speed_mph"] == pytest.approx(39.33, abs=0.01)
        assert r["running_time_s"] == pytest.approx(33.54, abs=0.5)
        assert r["running_speed_mph"] == pytest.approx(36.59, abs=0.5)
        assert r["through_delay_s"] == pytest.approx(18.310, abs=0.5)
        assert r["travel_speed_mph"] == pytest.approx(23.67, abs=0.5)
        assert r["full_stop_rate"] == pytest.approx(0.547, abs=0.01)
        assert r["spatial_stop_rate"] == pytest.approx(1.61, abs=0.01)
        assert r["vc_ratio"] == pytest.approx(0.52, abs=0.005)
        assert r["perception_score"] == pytest.approx(2.53, abs=0.01)
        assert result["level_of_service"] == "C"


class TestUrbanSegmentNonAutoModes:
    """HCM Chapter 30, Example Problems 2, 3 and 4."""

    def test_pedestrian_mode(self, registry):
        r = run_example(registry, "analyze_pedestrian_segment")["results"]
        assert r["link_score"] == pytest.approx(2.35, abs=0.02)
        assert r["pedestrian_space"] == pytest.approx(32.0, abs=0.3)
        assert r["segment_score"] == pytest.approx(3.62, abs=0.03)
        assert r["segment_los"] == "D"

    def test_bicycle_mode(self, registry):
        r = run_example(registry, "analyze_bicycle_segment")["results"]
        assert r["link_score"] == pytest.approx(3.62, abs=0.02)
        assert r["link_los"] == "D"
        assert r["segment_score"] == pytest.approx(2.88, abs=0.02)
        assert r["segment_los"] == "C"

    def test_transit_mode(self, registry):
        r = run_example(registry, "analyze_transit_segment")["results"]
        assert r["travel_speed"] == pytest.approx(11.3, abs=0.1)
        assert r["wait_ride_score"] == pytest.approx(2.47, abs=0.02)
        assert r["segment_score"] == pytest.approx(2.83, abs=0.03)
        assert r["segment_los"] == "C"


# ── Chapters 19-22: intersections ────────────────────────────────────────────

class TestSignalized:
    """HCM Chapter 31, Example Problem 1 (Exhibit 31-81)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_signalized")

    def test_published_values(self, result):
        r = result["results"]
        assert r["intersection_delay_s"] == pytest.approx(45.9, abs=0.5)
        assert result["level_of_service"] == "D"
        for direction, (delay, los) in {"EB": (32.4, "C"), "WB": (37.0, "D"),
                                        "NB": (70.0, "E"), "SB": (19.6, "B")}.items():
            assert r["approaches"][direction]["delay_s"] == pytest.approx(delay, abs=0.5), direction
            assert r["approaches"][direction]["los"] == los, direction

    def test_lane_groups(self, result):
        r = result["results"]
        assert r["num_lane_groups"] == 12
        assert len(r["lane_groups"]) == 12
        nb_through = next(g for g in r["lane_groups"] if g["direction"] == "NB" and g["kind"] == "ExclusiveThrough")
        assert nb_through["vc_ratio"] == pytest.approx(1.05, abs=0.02)
        assert nb_through["los"] == "F"


class TestSignalizedNonAutoModes:
    """HCM Chapter 31, Example Problems 2, 3 and 4."""

    def test_pedestrian_mode(self, registry):
        r = run_example(registry, "analyze_signalized_pedestrian")["results"]
        assert r["delay"] == pytest.approx(29.8, abs=0.1)
        assert r["los_score"] == pytest.approx(2.37, abs=0.02)
        assert r["los"] == "B"

    def test_bicycle_mode(self, registry):
        r = run_example(registry, "analyze_signalized_bicycle")["results"]
        assert r["capacity"] == pytest.approx(800.0, abs=1.0)
        assert r["delay"] == pytest.approx(23.0, abs=0.1)
        assert r["los_score"] == pytest.approx(2.45, abs=0.01)
        assert r["los"] == "B"

    def test_two_stage_crossing_delay(self, registry):
        r = run_example(registry, "analyze_two_stage_crossing")["results"]
        assert r["delay_s"] == pytest.approx(78.0, abs=0.5)


class TestTwsc:
    """HCM Chapter 32, TWSC Example Problem 1 (three-leg intersection)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_twsc")["results"]

    def test_intersection_and_approach_delays(self, r):
        assert r["intersection_delay"] == pytest.approx(4.1, abs=0.5)
        d_eb, d_wb, d_nb, _ = r["approach_delays"]
        assert d_eb == pytest.approx(0.0, abs=0.5)
        assert d_wb == pytest.approx(2.9, abs=0.5)
        assert d_nb == pytest.approx(14.9, abs=0.5)

    def test_config_is_echoed(self, r):
        assert r["geometry"]["is_three_leg"] is True
        assert r["heavy_vehicle_pct"] == pytest.approx(10.0)


class TestTwscPedestrian:
    """HCM Chapter 32, TWSC Example Problem 2, scenario B: marked crosswalk with a median refuge, two stages, 50% motorist yield rate."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_twsc_pedestrian")["results"]

    def test_stage_intermediates(self, r):
        assert len(r["stages"]) == 2
        s = r["stages"][0]
        assert s["critical_headway"] == pytest.approx(6.0, abs=0.05)
        assert s["prob_blocked_lane"] == pytest.approx(0.508, abs=0.001)
        assert s["prob_delayed_crossing"] == pytest.approx(0.758, abs=0.001)
        assert s["gap_delay"] == pytest.approx(7.2, abs=0.05)
        assert s["average_short_headway"] == pytest.approx(2.3, abs=0.05)
        assert s["yield_events"] == 4

    def test_delay_and_satisfaction(self, r):
        assert r["delay"] == pytest.approx(6.0, abs=0.5)
        assert r["odds_satisfied_no_delay"] == pytest.approx(13.44, abs=0.05)
        assert r["prob_yield_first_event"] == pytest.approx(0.314, abs=0.001)
        assert r["prob_non_delayed"] == pytest.approx(0.481, abs=0.001)
        assert r["proportion_dissatisfied"] == pytest.approx(0.207, abs=0.001)
        assert r["los"] == "C"


class TestAwsc:
    """HCM Chapter 32, AWSC Example Problem 1 (single-lane, three-leg)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_awsc")

    def test_published_values(self, result):
        assert result["results"]["intersection_delay"] == pytest.approx(12.8, abs=0.5)
        assert result["level_of_service"] == "B"
        assert result["results"]["iterations"] >= 1


class TestRoundabout:
    """HCM Chapter 33, Example Problem 1 (single-lane with bypass lanes)."""

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_roundabout")

    def test_published_values(self, result):
        assert result["results"]["intersection_delay"] == pytest.approx(17.5, abs=0.5)
        assert result["level_of_service"] == "C"
        assert result["results"]["wb"]["bypass"] == "Yielding"


# ── Chapter 23: ramp terminals and alternative intersections ────────────────

class TestRampTerminal:
    """HCM Chapter 34, Example Problem 1 (diamond interchange, Exhibit 34-16).

    Nine of the ten O-D ETTs sit below the published figures because of the 0.3.1 interchange corrections (aggregate LOS from weighted ETT only, d2 on lane-group capacity). The engine values below are the library's own assertions; the published ones are in its test comments.
    """

    @pytest.fixture(scope="class")
    def result(self, registry):
        return run_example(registry, "analyze_ramp_terminal")

    def test_od_results(self, result):
        expected = {
            "A": (233.0, 47.7, "C"), "B": (227.0, 41.8, "C"), "C": (173.0, 52.7, "C"),
            "D": (206.0, 65.7, "D"), "E": (107.0, 98.9, "E"), "F": (89.0, 40.0, "C"),
            "G": (150.0, 32.7, "C"), "H": (236.0, 81.8, "D"), "I": (761.0, 49.8, "C"),
            "J": (650.0, 36.9, "C"),
        }
        od = result["results"]["od_movements"]
        for movement, (demand, ett, los) in expected.items():
            assert od[movement]["demand_veh_h"] == pytest.approx(demand, abs=1.0), movement
            assert od[movement]["ett_s"] == pytest.approx(ett, abs=1.0), movement
            assert od[movement]["los"] == los, movement

    def test_interchange_aggregate(self, result):
        assert result["results"]["interchange_ett"] == pytest.approx(50.7, abs=0.5)
        assert result["level_of_service"] == "C"


class TestAlternativeIntersection:
    """HCM Chapter 34, Example Problem 13 (three-legged RCUT with STOP signs, Exhibits 34-126 through 34-129)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_alternative_intersection")["results"]

    def test_movement_ett_and_los(self, r):
        expected = {"EB L": (55.2, "E"), "EB R": (22.9, "C"), "NB L": (13.0, "B"),
                    "NB T": (0.0, "A"), "SB T": (0.0, "A"), "SB R": (0.0, "A")}
        for label, (ett, los) in expected.items():
            _, _, got_ett, got_los = r["movements"][label]
            assert got_ett == pytest.approx(ett, abs=1.0), label
            assert got_los == los, label

    def test_intersection_ett(self, r):
        assert r["intersection_ett"] > 0.0


class TestDisplacedLeftTurn:
    """HCM Chapter 34, Example Problem 16 (partial DLT, Exhibits 34-139 through 34-145)."""

    def test_published_values(self, registry):
        result = run_example(registry, "analyze_displaced_left_turn")
        assert result["results"]["intersection_ett"] == pytest.approx(28.5, abs=0.1)
        assert result["level_of_service"] == "C"


# ── Chapter 24: off-street pedestrian and bicycle ───────────────────────────

class TestOffStreetPedBike:
    """HCM Chapter 35, Example Problems 1 and 2."""

    def test_exclusive_walkway(self, registry):
        result = run_example(registry, "analyze_pedestrian_walkway")
        r = result["results"]
        # Published: W_E = 5 ft, v_p = 1.33 p/ft/min, A_p = 180 ft2/p -> LOS A.
        assert r["effective_width"] == pytest.approx(5.0, abs=1e-9)
        assert r["unit_flow_rate"] == pytest.approx(1.33, abs=0.005)
        assert r["pedestrian_space"] == pytest.approx(180.0, abs=0.5)
        assert result["level_of_service"] == "A"

    def test_shared_use_path_pedestrian(self, registry):
        result = run_example(registry, "analyze_shared_use_path_pedestrian")
        r = result["results"]
        # Published: F_p = 90, F_m = 151, F = 166 events/h -> LOS E.
        assert r["passing_events"] == pytest.approx(90.0, abs=0.5)
        assert r["meeting_events"] == pytest.approx(151.0, abs=0.5)
        assert r["total_events"] == pytest.approx(166.0, abs=0.5)
        assert result["level_of_service"] == "E"

    def test_offstreet_bicycle(self, registry):
        result = run_example(registry, "analyze_offstreet_bicycle")
        r = result["results"]
        # Published Step 1 directional flow rates, then A_T = 2.42, M_T = 8.33.
        assert r["subject_flow_rates"][0] == pytest.approx(104.0, abs=0.5)
        assert r["subject_flow_rates"][1] == pytest.approx(38.0, abs=0.5)
        assert r["active_passings_per_minute"] == pytest.approx(2.42, abs=0.01)
        assert r["meetings_per_minute"] == pytest.approx(8.33, abs=0.03)
        assert r["effective_lanes"] == 2
        assert r["total_probability_delayed_passing"] == pytest.approx(0.8334, abs=0.002)
        assert r["delayed_passings_per_minute"] == pytest.approx(1.82, abs=0.01)
        assert r["blos_score"] == pytest.approx(2.69, abs=0.01)
        assert result["level_of_service"] == "D"


# ── Chapters 25 and 26: mixed flow ───────────────────────────────────────────

class TestMixedFlow:
    """HCM Chapter 26, Example Problem 5, mixed-flow half."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_mixed_flow")["results"]

    def test_capacity(self, r):
        assert r["caf_t_mix"] == pytest.approx(0.135, abs=0.001)
        assert r["caf_g_mix"] == pytest.approx(0.131, abs=0.001)
        assert r["caf_mix"] == pytest.approx(0.734, abs=0.001)
        assert r["capacity_ao"] == pytest.approx(2350.0)
        assert r["capacity_mix"] == pytest.approx(1725.0, abs=2.0)
        assert r["oversaturated"] is False

    def test_speed_and_density(self, r):
        assert r["tau_sut_kin"] == pytest.approx(71.1, abs=0.5)
        assert r["tau_tt_kin"] == pytest.approx(92.2, abs=0.5)
        assert r["ffs_mix"] == pytest.approx(60.1, abs=0.1)
        assert r["saf_mix"] == pytest.approx(0.92, abs=0.01)
        assert r["phi_mix"] == pytest.approx(4.07, abs=0.1)
        # The example's own Step 8 density is 31.7, not the 32.6 its comparison
        # paragraph quotes; the library asserts the self-consistent one.
        assert r["s_mix"] == pytest.approx(47.3, abs=0.3)
        assert r["d_mix"] == pytest.approx(31.7, abs=0.3)

    def test_oversaturation_returns_null_speed(self, registry):
        config = dict(methods._example("analyze_mixed_flow"), v_mix=2000.0)
        r = call(registry, "hcm_analyze_mixed_flow", config=config)["results"]
        assert r["oversaturated"] is True
        assert r["s_mix"] is None and r["d_mix"] is None


class TestCompositeGrade:
    """HCM Chapter 25, Example Problem 11 (three-segment composite grade)."""

    @pytest.fixture(scope="class")
    def r(self, registry):
        return run_example(registry, "analyze_composite_grade")["results"]

    def test_capacity_and_governing_segment(self, r):
        for i, want in enumerate([1875.0, 1934.0, 1746.0]):
            assert r["segments"][i]["capacity_mix"] == pytest.approx(want, abs=2.0), i
        assert r["governing_segment"] == 2
        assert r["capacity_mix"] == pytest.approx(1746.0, abs=2.0)

    def test_segment_speeds_and_travel_times(self, r):
        for i, (speed, time) in enumerate([(57.7, 93.6), (58.7, 122.7), (47.9, 75.2)]):
            assert r["segments"][i]["s_mix"] == pytest.approx(speed, abs=0.3), i
            assert r["segments"][i]["travel_time"] == pytest.approx(time, abs=0.7), i

    def test_overall(self, r):
        assert r["total_length"] == pytest.approx(4.5)
        # 291.5 s, not the 294 s of the example's Step 7 prose.
        assert r["total_travel_time"] == pytest.approx(291.5, abs=1.5)
        assert r["s_mix_overall"] == pytest.approx(55.6, abs=0.3)


# ── Chapters 27 and 28: service volumes ──────────────────────────────────────

class TestServiceVolumes:
    def test_ramp_case_1_solves_the_freeway_flow(self, registry):
        """HCM Chapter 28, Example Problem 5, Case 1: ramp demand is 10% of freeway demand."""
        rows = {row["target_density"]: row for row in
                run_example(registry, "analyze_ramp_service_volumes")["results"]["service_volumes"]}
        assert rows[10.0]["sfi"] == pytest.approx(1979.0, abs=6.0)
        assert rows[28.0]["sfi"] == pytest.approx(5280.0, abs=6.0)
        assert rows[10.0]["sf"] == pytest.approx(1858.0, abs=3.0)
        assert rows[10.0]["sv"] == pytest.approx(1616.0, abs=3.0)

    def test_ramp_case_2_marks_unachievable_levels(self, registry):
        """Case 2 holds the freeway flow fixed, and LOS A and B are then out of reach at zero ramp demand."""
        config = dict(methods._example("analyze_ramp_service_volumes"))
        config.pop("ramp_fraction")
        config["fixed_freeway_vf"] = 4000.0 / (0.87 * config["f_hv"])
        rows = {row["target_density"]: row for row in
                call(registry, "hcm_analyze_ramp_service_volumes", config=config)["results"]["service_volumes"]}
        assert rows[20.0]["unachievable"] is True
        assert rows[20.0]["sfi"] is None
        assert rows[28.0]["sfi"] == pytest.approx(772.0, abs=3.0)
        assert rows[35.0]["sfi"] == pytest.approx(1726.0, abs=3.0)
        assert rows[28.0]["sf"] == pytest.approx(725.0, abs=3.0)
        assert rows[28.0]["sv"] == pytest.approx(631.0, abs=3.0)

    def test_ramp_basis_must_be_exactly_one(self, registry):
        config = dict(methods._example("analyze_ramp_service_volumes"), fixed_freeway_vf=4896.0)
        result = call(registry, "hcm_analyze_ramp_service_volumes", config=config)
        assert result["success"] is False
        assert "exactly one" in result["error"]

    def test_weaving_service_volumes(self, registry):
        """HCM Chapter 27, Example Problem 5."""
        rows = {row["target_density"]: row for row in
                run_example(registry, "analyze_weaving_service_volumes")["results"]["service_volumes"]}
        row = rows[28.0]
        assert (row["sfi"] // 100) * 100 == pytest.approx(4300.0, abs=100.0)
        assert row["sf"] == pytest.approx(row["sfi"] * 0.952, abs=0.01)
        assert row["sv"] == pytest.approx(row["sf"] * 0.93, abs=0.01)
        assert row["dsv"] == pytest.approx(row["sv"] / (0.08 * 0.55), abs=0.01)


# ── Domain refusals and malformed input ──────────────────────────────────────

class TestRefusals:
    """A refusal must carry the compute library's own message, not a substitute. These are statements about what the published HCM data covers, and rewording them would hide why an input was rejected."""

    def test_mixed_flow_refuses_an_undigitised_grade(self, registry):
        # The truck speed curves are digitised at -5, 0, 2, 3 and 5% only.
        config = dict(methods._example("analyze_mixed_flow"), grade=7.0)
        result = call(registry, "hcm_analyze_mixed_flow", config=config)
        assert result["success"] is False
        assert "digitised" in result["error"]
        assert "7" in result["error"]

    def test_mixed_flow_refuses_off_domain_truck_proportions(self, registry):
        config = dict(methods._example("analyze_mixed_flow"), p_tt=0.99)
        result = call(registry, "hcm_analyze_mixed_flow", config=config)
        assert result["success"] is False
        assert "truck proportions" in result["error"]

    def test_composite_grade_refuses_an_entry_speed_off_the_ladder(self, registry):
        # Reversing the segment order enters a grade at a speed with no curve.
        config = dict(methods._example("analyze_composite_grade"))
        config["segments"] = list(reversed(config["segments"]))
        result = call(registry, "hcm_analyze_composite_grade", config=config)
        assert result["success"] is False
        assert "2.5 mi/h" in result["error"]

    def test_basic_freeway_refuses_a_steep_specific_upgrade(self, registry):
        # Exhibits 12-26/27/28 stop at 6%; beyond that the HCM sends the analyst
        # to the Chapter 25/26 mixed-flow model instead.
        config = dict(methods._example("analyze_basic_freeway"), grade=8.0, sut_percentage=50, p_t=0.06)
        result = call(registry, "hcm_analyze_basic_freeway", config=config)
        assert result["success"] is False
        assert "mixed-flow model" in result["error"]

    def test_basic_freeway_refuses_an_untabulated_truck_mix(self, registry):
        config = dict(methods._example("analyze_basic_freeway"), grade=2.5, sut_percentage=40, p_t=0.06)
        result = call(registry, "hcm_analyze_basic_freeway", config=config)
        assert result["success"] is False
        assert "30%, 50%, and 70%" in result["error"]

    def test_malformed_config_is_a_clean_error(self, registry):
        result = call(registry, "hcm_analyze_roundabout", config={"nonsense": True})
        assert result["success"] is False
        assert result["error"]

    def test_malformed_json_config_is_a_clean_error(self, registry):
        result = call(registry, "hcm_analyze_signalized", config={"cycle_length_s": "ninety"})
        assert result["success"] is False
        assert result["error"]

    def test_missing_config_names_the_describe_tool(self, registry):
        result = call(registry, "hcm_analyze_weaving")
        assert result["success"] is False
        assert "hcm_describe_method" in result["error"]


# ── The describe companion ───────────────────────────────────────────────────

class TestDescribeMethod:
    def test_without_a_method_lists_every_method(self, registry):
        result = call(registry, "hcm_describe_method")
        assert result["success"] is True
        assert result["total_count"] == len(methods.METHODS)
        chapters = {row["chapter"] for row in result["methods"]}
        # Every computational chapter 10-24, plus the supplemental chapters that
        # carry the mixed-flow model (25, 26) and the service-volume solvers
        # (27, 28).
        assert set(range(10, 25)) <= chapters
        assert {25, 26, 27, 28} <= chapters
        for row in result["methods"]:
            assert row["tool"] == f"hcm_{row['method']}"
            assert row["title"] and row["library_symbol"]

    def test_every_method_is_describable_and_serves_its_fixture(self, registry):
        for method in methods.METHODS:
            result = call(registry, "hcm_describe_method", method=method)
            assert result["success"] is True, method
            assert result["example"], method
            assert result["input_sketch"], method
            assert result["docstring"].startswith("HCM Chapter"), method
            assert "_source" not in result["input_sketch"], method

    def test_accepts_the_served_tool_name(self, registry):
        by_tool = call(registry, "hcm_describe_method", method="hcm_analyze_weaving")
        by_method = call(registry, "hcm_describe_method", method="analyze_weaving")
        assert by_tool == by_method

    def test_editions_are_advertised_where_they_exist(self, registry):
        assert call(registry, "hcm_describe_method", method="analyze_weaving")["hcm_editions"] == ["7", "7.1"]
        assert call(registry, "hcm_describe_method", method="analyze_merge_diverge")["hcm_editions"] == ["7", "7.1"]
        assert "hcm_editions" not in call(registry, "hcm_describe_method", method="analyze_signalized")

    def test_unknown_method_is_a_clean_error(self, registry):
        result = call(registry, "hcm_describe_method", method="analyze_teleportation")
        assert result["success"] is False
        assert "hcm_describe_method" in result["error"]

    def test_the_sketch_collapses_long_lists(self, registry):
        sketch = call(registry, "hcm_describe_method", method="analyze_freeway_facility")["input_sketch"]
        segments = sketch["segments"]
        assert len(segments) == 2 and segments[1].startswith("... ")


class TestMethodTableIntegrity:
    def test_every_method_ships_its_example_fixture(self):
        for method in methods.METHODS:
            path = methods.EXAMPLES_DIR / f"{method}.json"
            assert path.exists(), f"{method} has no example fixture"
            json.loads(path.read_text())

    def test_every_method_is_registered_and_routed(self, registry):
        import mcp_server_fastapi as srv

        operations = {getattr(route, "operation_id", None) for route in srv.app.routes}
        for method in methods.METHODS:
            assert registry.get_function(f"hcm_{method}") is not None, method
            assert f"hcm_{method}" in operations, method
        assert registry.get_function("hcm_describe_method") is not None
        assert "hcm_describe_method" in operations

    def test_every_method_documents_its_chapter_and_results(self):
        for method, entry in methods.METHODS.items():
            doc = entry["function"].__doc__ or ""
            assert doc.startswith("HCM Chapter"), method
            assert "Worked example" in doc or "worked example" in doc, method
