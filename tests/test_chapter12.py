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

GOOD = {"bffs": 70.0, "lw": 12.0, "lane_count": 2, "lc_r": 6, "trd": 1,
        "demand_flow_i": 3000.0, "phf": 0.95, "p_t": 0.25, "grade": 2.0, "length": 0.625}
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

    def test_off_grid_inputs_return_clean_error(self):
        from hcm_mcp_server.functions.chapter12 import complete_freeway_analysis_function
        out = complete_freeway_analysis_function(
            {"freeway_data": {**NARROW, "grade": 3.7, "length": 0.5}}
        )
        assert out["success"] is False
        assert "error" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
