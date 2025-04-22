import os
import pickle
import numpy as np
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

import faiss
from sentence_transformers import SentenceTransformer
from mcp.server.fastmcp import FastMCP, Context

# from fastapi import FastAPI
# from fastapi_mcp import FastApiMCP

load_dotenv()

### Let HCM put in the database
### Connect with RAG
### Use MCP server to extract information 
### Let it run with Claude Desktop (MCP Client)
### Connect MCP server to my website
mcp = FastMCP("Demo")
# app = FastAPI()

# ---------- APP STATE & LIFESPAN ----------
@dataclass
class AppState:
    embedding_model: Any
    faiss_index: Any
    metadata: list

@asynccontextmanager
async def lifespan(server: FastMCP):
    print("🔧 Loading embedding model and FAISS index...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("hcm_index.faiss")
    with open("hcm_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    yield AppState(embedding_model=model, faiss_index=index, metadata=metadata)

# mcp = FastMCP("HCM-LLM", lifespan=lifespan, debug=True)
mcp = FastApiMCP(
    app,
    name=os.environ.get("PUBLIC_MY_API_KEY"),
    description="My API",
    base_url="http://localhost:5000"
)

@mcp.resource("manual://hcm/chapter15")
def get_hcm_chap15() -> str:
    return "Freeway facilities are defined as..."

@mcp.tool()
def query_hcm(question: str, top_k: int = 5, ctx: Context = None) -> str:
    """Query the HCM content using FAISS and sentence-transformer embeddings."""
    model = ctx.request_context.lifespan_context.embedding_model
    index = ctx.request_context.lifespan_context.faiss_index
    metadata = ctx.request_context.lifespan_context.metadata

    query_vec = model.encode([question]).astype("float32")
    D, I = index.search(query_vec, k=top_k)

    results = []
    for idx in I[0]:
        result = metadata[idx]
        results.append(f"[{result['chapter']}]\n{result['text'].strip()}")

    return "\n\n".join(results)

@mcp.prompt("summarize://manual/section")
def summarize_section(section_text: str) -> str:
    return f"Summarize the following HCM section:\n\n{section_text}"

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # mcp.run(
    #     host="127.0.0.1",
    #     port=5000,
    #     transport="sse",
    # )
    # mcp.mount()