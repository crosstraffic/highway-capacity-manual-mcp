"""KG-only ablation launcher for the HCM-LLM MCP server.

Exposes ONLY the knowledge-graph / validator reasoning tools — no retrieval
(``query_hcm`` et al.) and no raw Chapter 12/15 computation. This is the ``kg``
arm of the Table 5 / Figure 7 2x2 ablation; it isolates "structured rules +
causal graph" from "manual text" (``rag`` arm) and "the full system" (``ct``).

Tools exposed (7, all first-class via reasoning_endpoints):
    validate_design_full, propagate_change, diagnose_failure,
    repair_design, repair_freeway, reconcile_codes, inverse_design

It sets the env vars the shared app honors, then runs the same FastAPI app:
- HCM_MCP_INCLUDE_OPS  restricts the MCP surface to these 7 operation ids.
- HCM_ENABLE_RAG=false skips Chroma + sentence-transformers (not needed here),
  so this arm runs with no vector store present.

Run:  python mcp_server_kg_only.py   (PORT defaults to 8001 to coexist with the
full server on 8000). The reasoning layer is in-process and database-free.
"""

import os

KG_OPERATIONS = [
    "validate_design_full",
    "propagate_change",
    "diagnose_failure",
    "repair_design",
    "repair_freeway",
    "reconcile_codes",
    "inverse_design",
]

# Must be set BEFORE importing the app (the MCP mount + lifespan read them).
os.environ["HCM_MCP_INCLUDE_OPS"] = ",".join(KG_OPERATIONS)
os.environ.setdefault("HCM_ENABLE_RAG", "false")
os.environ.setdefault("PORT", "8001")

if __name__ == "__main__":
    import uvicorn

    from mcp_server_fastapi import app

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host=host, port=port)
