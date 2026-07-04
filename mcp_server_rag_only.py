"""RAG-only ablation launcher for the HCM-LLM MCP server.

Exposes ONLY the retrieval tool (``query_hcm``) — no KG reasoning and no
Chapter 12/15 computation. This is the ``rag`` arm of the Table 5 / Figure 7
2x2 ablation; it isolates "manual text retrieval" from "structured rules"
(``kg`` arm) and "the full system" (``ct``).

Matches the ablation harness's ``rag`` arm in ``ablation/arms.py`` exactly
(``query_hcm`` only). Requires the vector store, so RAG stays enabled.

Run:  python mcp_server_rag_only.py   (PORT defaults to 8002 to coexist with
the full server on 8000 and the kg-only server on 8001).
"""

import os

RAG_OPERATIONS = ["query_hcm"]

# Must be set BEFORE importing the app (the MCP mount reads it).
os.environ["HCM_MCP_INCLUDE_OPS"] = ",".join(RAG_OPERATIONS)
os.environ.setdefault("HCM_ENABLE_RAG", "true")
os.environ.setdefault("PORT", "8002")

if __name__ == "__main__":
    import uvicorn

    from mcp_server_fastapi import app

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host=host, port=port)
