"""Minimal HCM Chapter 15 calculator MCP server.

Exposes only the forward operational-analysis tools (no RAG, so no torch /
chromadb / faiss dependencies). Intended for the cross-interface consistency
experiment: an LLM calls these tools and the results are compared against the
hard-coded library (Path A). Runs the same verified core as everything else.

Run:  PORT=8765 PYTHONPATH=. python mcp_calc_only.py
Then point an MCP client at  http://localhost:8765/mcp
"""
import os

from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from hcm_mcp_server.core.models import TwoLaneHighwaysInput
from hcm_mcp_server.functions.chapter15 import (
    complete_highway_analysis_function,
    determine_facility_los_function,
)

app = FastAPI(title="HCM Chapter 15 Calculator (minimal)")


@app.post("/analysis/chapter15/complete", operation_id="chapter15_complete")
def chapter15_complete(highway_data: TwoLaneHighwaysInput):
    """Run the full HCM Chapter 15 two-lane highway facility analysis.

    Returns per-segment average travel speed, percent followers, follower
    density, and LOS, plus the facility length-weighted speed, follower
    density, and LOS.

    IMPORTANT input conventions:
    - Segment length is in MILES. Subsegment length is ALSO in MILES.
    - For a segment with horizontal curves, provide ALL subsegments tiling the
      full segment in order: both tangents (design_rad = 0) and curves
      (design_rad > 0 with sup_ele set). The subsegment lengths must sum to the
      segment length. Do not supply curves only; the tangent portions are
      explicit subsegments here.
    - The horizontal-curve class is derived from radius and superelevation
      automatically; you do not need to set hor_class.
    """
    return complete_highway_analysis_function({"highway_data": highway_data.model_dump()})


@app.post("/analysis/chapter15/facility", operation_id="chapter15_facility")
def chapter15_facility(highway_data: TwoLaneHighwaysInput):
    """Return only the facility-level roll-up: length-weighted average travel
    speed, length-weighted follower density, and facility LOS. Same input
    conventions as chapter15_complete."""
    return determine_facility_los_function({"highway_data": highway_data.model_dump()})


mcp = FastApiMCP(app)
mcp.mount()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8765")))
