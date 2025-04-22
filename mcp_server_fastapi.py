import pickle
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import json
import time
import asyncio
from dotenv import load_dotenv
from contextlib import asynccontextmanager

import faiss
from sentence_transformers import SentenceTransformer

from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("hcm_index.faiss")
    with open("hcm_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    app.state.embedding_model = model
    app.state.faiss_index = index
    app.state.metadata = metadata

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], # your SvelteKit frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryHCMRequest(BaseModel):
    question: str
    top_k: int = 5

class Message(BaseModel):
    role: str  # 'user', 'assistant', or 'system'
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    created: int
    object: str = "chat.completion"
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int]

async def format_sse(data: str) -> str:
    """Format data for SSE."""
    return f"data: {data}\n\n"

async def stream_completion(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    
    # Extract the last user message
    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    
    # In a real implementation, you would call your LLM here
    words = f"This is a streaming response to: {last_user_msg}".split()
    
    # Generate an id for this completion
    completion_id = f"chatcmpl-{int(time.time())}"
    
    # Send chunks of the response
    for i, word in enumerate(words):
        await asyncio.sleep(0.1)  # Simulate thinking time
        
        chunk = {
            "id": completion_id,
            "model": request.model,
            "created": int(time.time()),
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": word + " "},
                    "finish_reason": None
                }
            ]
        }
        
        yield await format_sse(json.dumps(chunk))
    
    # Send final chunk with finish_reason
    final_chunk = {
        "id": completion_id,
        "model": request.model,
        "created": int(time.time()),
        "object": "chat.completion.chunk",
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }
        ]
    }
    
    yield await format_sse(json.dumps(final_chunk))
    
    # Signal end of stream
    yield await format_sse("[DONE]")

# @app.post("/mcp/tools/query-hcm", tags=["tools"], operation_id="mcp_post_query_hcm")
# async def mcp_query_hcm(request: dict):
#     """MCP SDK compatible endpoint for querying HCM"""
#     data = QueryHCMRequest(**request)
    
#     model = app.state.embedding_model
#     index = app.state.faiss_index
#     metadata = app.state.metadata

#     query_vec = model.encode([data.question]).astype("float32")
#     D, I = index.search(query_vec, k=data.top_k)

#     results = []
#     for idx in I[0]:
#         result = metadata[idx]
#         results.append(f"[{result['chapter']}]\n{result['text'].strip()}")

#     return {"results": results}

# @app.get("/mcp/resources/manual/hcm/chapter15", tags=["resources"], operation_id="mcp_get_chap15")
# async def mcp_get_hcm_chapter15():
#     """MCP SDK compatible endpoint for getting HCM chapter 15"""
#     return {"content": "Freeway facilities are defined as..."}

# @app.post("/mcp/prompts/summarize-section", tags=["prompts"], operation_id="mcp_post_summary")
# async def mcp_summarize_section(request: dict):
#     """MCP SDK compatible endpoint for summarization prompt"""
#     section_text = request.get("section_text", "")
#     return {
#         "prompt": f"Summarize the following HCM section:\n\n{section_text}"
#     }

# Legacy endpoints for backward compatibility
@app.post("/chat/completions", tags=["chat"], operation_id="post_chat_completion")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion."""
    
    # Handle streaming response
    if request.stream:
        return StreamingResponse(
            stream_completion(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    
    # Non-streaming response
    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        model=request.model,
        created=int(time.time()),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": f"This is a response to: {last_user_msg}"
                },
                finish_reason="stop"
            )
        ],
        usage={
            "prompt_tokens": len(str(request.messages)),
            "completion_tokens": 20,
            "total_tokens": len(str(request.messages)) + 20
        }
    )
    
    return response

@app.get("/resources/manual/hcm/chapter15", tags=["resources"], operation_id="get_chap15")
async def get_hcm_chap15():
    return {"content": "Freeway facilities are defined as..."}

@app.post("/tools/query-hcm", tags=["tools"], operation_id="post_query_hcm")
async def query_hcm(data: QueryHCMRequest):
    """Query the HCM content using FAISS and sentence-transformer embeddings."""
    model = app.state.embedding_model
    index = app.state.faiss_index
    metadata = app.state.metadata

    query_vec = model.encode([data.question]).astype("float32")
    D, I = index.search(query_vec, k=data.top_k)

    results = []
    for idx in I[0]:
        result = metadata[idx]
        results.append(f"[{result['chapter']}]\n{result['text'].strip()}")

    return {"results": results}

@app.post("/prompts/summarize-section", tags=["prompts"], operation_id="post_summary")
async def summarize_section(section_text: str = Body(...)):
    return {
        "prompt": "Summarize the following HCM section:\n\n{section_text}"
    }

@app.get("/mcp/discovery")
async def mcp_discovery():
    """MCP SDK discovery endpoint"""
    return {
        "name": "HCM-LLM",
        "version": "1.0.0",
        "capabilities": {
            "tools": [
                {
                    "name": "chat-completions",
                    "description": "Generate chat completions with LLM models"
                },
                {
                    "name": "query-hcm",
                    "description": "Query the Highway Capacity Manual database"
                }
            ],
            "resources": [
                {
                    "uri": "manual/hcm/chapter15",
                    "description": "Highway Capacity Manual Chapter 15 content"
                }
            ],
            "prompts": [
                {
                    "name": "summarize-section",
                    "description": "Generates a prompt to summarize an HCM section"
                }
            ]
        }
    }


mcp = FastApiMCP(
    app,
    name="HCM-LLM",
    description="My API",
    base_url="http://localhost:5000",
    describe_all_responses=True,
    describe_full_response_schema=True  
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)