"""Tests for the HCM Chapter 12 basic-freeway computation tools.

These need ``transportations-library>=0.1.12`` (the BasicFreeways binding with the Ch.12 step methods). The module is skipped cleanly when that binding is absent (e.g. an older library is installed).
"""

import pytest

try:
    import transportations_library as _tl

    _HAVE_BASICFREEWAY = hasattr(_tl, "BasicFreeways") and hasattr(
        _tl.BasicFreeways, "estimate_density"
    )
except Exception:  # pragma: no cover
    _HAVE_BASICFREEWAY = False

pytestmark = pytest.mark.skipif(
    not _HAVE_BASICFREEWAY,
    reason="transportations-library>=0.1.12 with the BasicFreeways Ch.12 binding required",
)

# 25% trucks, 2% grade, 3100 veh/h. No SUT mix is given (sut_percentage defaults to 0),
# so E_T comes from the general-terrain exhibit (12-25, level) = 2.0, not a specific-upgrade
# table. At 12 ft lanes the segment holds LOS D (FFS 66.78, density 33.9); dropping to 10 ft
# lanes lowers FFS to 60.18 and pushes density to 36.04 → LOS E. Demand is 3100, not the
# pre-fix 3000: at 3000 the corrected general-terrain default already yields LOS D at 10 ft,
# so NARROW would no longer degrade. See tests/unit/test_repair.py for the full hand derivation.
GOOD = {"bffs": 70.0, "lw": 12.0, "lane_count": 2, "lc_r": 6, "trd": 1,
        "demand_flow_i": 3100.0, "phf": 0.95, "p_t": 0.25, "grade": 2.0, "length": 0.625}
NARROW = {**GOOD, "lw": 10.0}


class TestCompleteAnalysis:
    def test_good_lanes_los_d(self):
        from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
        out = complete_freeway_analysis_function({"freeway_data": GOOD})
        assert out["success"] is True
        assert out["results"]["level_of_service"] == "D"

    def test_narrow_lanes_degrade_to_los_e(self):
        """The lw -> FFS -> density -> LOS chain: 10ft lanes drop FFS and LOS."""
        from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
        good = complete_freeway_analysis_function({"freeway_data": GOOD})["results"]
        narrow = complete_freeway_analysis_function({"freeway_data": NARROW})["results"]
        assert narrow["free_flow_speed"] < good["free_flow_speed"]
        assert narrow["level_of_service"] == "E"


class TestSteps:
    def test_step_chain_matches_complete(self):
        from hcm_mcp_server.functions import chapter12 as C
        d = {"freeway_data": NARROW}
        assert C.determine_free_flow_speed_function(d)["success"]
        assert C.estimate_capacity_function(d)["capacity"] > 0
        assert C.estimate_demand_volume_function(d)["flow_rate"] > 0
        assert C.calculate_speed_function(d)["speed"] > 0
        assert C.estimate_density_function(d)["density"] > 0
        assert C.determine_segment_los_function(d)["level_of_service"] == "E"

    def test_off_domain_specific_upgrade_returns_clean_error(self):
        """The specific-upgrade exhibits (12-26/27/28) are tabulated only to a 6%
        grade; a steeper grade with an explicit SUT mix is off-domain and the
        library raises, which _guarded turns into a clean error dict. At the
        default sut_percentage=0 grade is irrelevant (general terrain), so the
        error path is only reachable with a specific-upgrade mix."""
        from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
        out = complete_freeway_analysis_function(
            {"freeway_data": {**NARROW, "sut_percentage": 30, "grade": 7.0}}
        )
        assert out["success"] is False
        assert "error" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
