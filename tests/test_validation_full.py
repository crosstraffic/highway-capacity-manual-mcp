"""Full-corpus validation tool (validate_design_full).

Runs the complete rule engine in-process over the bundled seed corpus — no database. Needs ``transportations-validator>=0.2.0`` (with the seed-backed engine) and ``sqlalchemy``; skipped cleanly if an older dependency is installed.
"""

import asyncio

import pytest

validation = pytest.importorskip(
    "hcm_mcp_server.functions.validation",
    reason="needs transportations-validator>=0.2.0 (seed engine) + sqlalchemy",
)

V = validation.validate_design_full_function


class TestValidateDesignFull:
    def test_multi_authority_violations_with_citations(self):
        r = asyncio.run(V({"design": {"lane_width": 8.0, "facility_type": "TwoLaneHighway"}}))
        assert r["success"] is True
        assert r["is_valid"] is False
        cites = " ".join(v["citation"] or "" for v in r["violations"])
        assert "HCM" in cites
        assert "AASHTO" in cites

    def test_valid_value_has_no_errors(self):
        r = asyncio.run(V({"design": {"lane_width": 11.0, "facility_type": "TwoLaneHighway"}}))
        lw_errors = [
            v for v in r["violations"]
            if v["parameter"] == "lane_width" and v["severity"] == "error"
        ]
        assert lw_errors == []

    def test_terrain_conditional_grade_flips(self):
        """The same 6% grade: compliant in mountainous terrain, a violation in
        level terrain — the full engine's context-gated rules, in-process."""
        mtn = asyncio.run(V({
            "design": {"grade": 6.0, "facility_type": "TwoLaneHighway"},
            "context": {"terrain_type": "mountainous"},
        }))
        lvl = asyncio.run(V({
            "design": {"grade": 6.0, "facility_type": "TwoLaneHighway"},
            "context": {"terrain_type": "level"},
        }))
        grade_mtn = [v for v in mtn["violations"] if "Grade" in v["rule"]]
        grade_lvl = [v for v in lvl["violations"] if "Grade" in v["rule"]]
        assert grade_mtn == []
        assert len(grade_lvl) >= 1

    def test_unknown_terrain_asks_rather_than_assumes(self):
        r = asyncio.run(V({"design": {"grade": 6.0, "facility_type": "TwoLaneHighway"}}))
        asks = " ".join(
            (c.get("message") or "") + str(c.get("question") or "")
            for c in r["clarifications"]
        ).lower()
        assert "terrain" in asks


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
