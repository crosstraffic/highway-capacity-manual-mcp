import os
from typing import AsyncGenerator
import json
import time
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP
from fastapi.middleware.cors import CORSMiddleware

from hcm_mcp_server.core.registry import FunctionRegistry
from hcm_mcp_server.core.models import *
from hcm_mcp_server.core.endpoints import create_endpoints

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chroma_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    client = chromadb.PersistentClient(path=chroma_path, settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name="hcm_documents")

    # Store in app state
    app.state.embedding_model = model
    app.state.chroma_client = client
    app.state.chroma_collection = collection

    # Initialize function registry
    registry_path = Path("function_registry.yaml")
    app.state.function_registry = FunctionRegistry(registry_path)

    yield

app = FastAPI(
    title="HCM-LLM MCP Server",
    description="Highway Capacity Manual API with Transportation Analysis",
    version="0.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create endpoints dynamically based on registry
create_endpoints(app)

# MCP Protocol Endpoints
# @app.get("/mcp")
# async def mcp_root():
#     """MCP root endpoint."""
#     return {
#         "name": "HCM-LLM",
#         "version": "0.2.0",
#         "protocol_version": "1.0.0",
#         "capabilities": ["tools", "resources", "prompts"]
#     }

# @app.post("/mcp")
# async def mcp_handle_request(request: Dict[str, Any]):
#     """Handle MCP protocol requests."""
#     try:
#         method = request.get("method")
#         params = request.get("params", {})
#         req_id = request.get("id")

#         def wrap_result(result):
#             return {
#                 "jsonrpc": "2.0",
#                 "id": req_id,
#                 "result": result
#             }

#         def wrap_error(message, code=400):
#             return {
#                 "jsonrpc": "2.0",
#                 "id": req_id,
#                 "error": {
#                     "code": code,
#                     "message": message
#                 }
#             }

        
#         if method == "tools/list":
#             registry = app.state.function_registry
#             if not registry:
#                 return {"tools": []}
                
#             tools = []
#             for name, info in registry.get_all_functions().items():
#                 tools.append({
#                     "name": name,
#                     "description": info["description"],
#                     "inputSchema": info["parameters"]
#                 })
#             return {"tools": tools}
            
#         elif method == "tools/call":
#             tool_name = params.get("name")
#             arguments = params.get("arguments", {})
            
#             registry = app.state.function_registry
#             if not registry:
#                 raise HTTPException(status_code=500, detail="Registry not available")
                
#             function_impl = registry.get_function(tool_name)
#             if not function_impl:
#                 raise HTTPException(status_code=404, detail=f"Tool {tool_name} not found")
            
#             # Execute function
#             if asyncio.iscoroutinefunction(function_impl):
#                 result = await function_impl(arguments)
#             else:
#                 result = function_impl(arguments)
            
#             return {"content": [{"type": "text", "text": json.dumps(result)}]}
            
#         elif method == "resources/list":
#             return {
#                 "resources": [
#                     {
#                         "uri": "hcm://chapter/15",
#                         "name": "HCM Chapter 15",
#                         "description": "Two-Lane Highways"
#                     }
#                 ]
#             }
#         elif method == "initialize":
#             return wrap_result({
#                 "name": "HCM-LLM",
#                 "version": "0.2.0",
#                 "protocol_version": "1.0.0",
#                 "capabilities": ["tools", "resources", "prompts"]
#             })
#         else:
#             raise HTTPException(status_code=400, detail=f"Unknown method: {method}")
            
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# MCP discovery endpoint
@app.get("/mcp/discovery")
async def mcp_discovery():
    """MCP SDK discovery endpoint."""
    registry = app.state.function_registry

    tools = []
    if registry:
        for name, func_info in registry.get_all_functions().items():
            tools.append({
                "name": name,
                "description": func_info["description"],
                "category": func_info.get("category", "general")
            })
    
    return {
        "name": "HCM-LLM",
        "version": "0.2.0",
        "capabilities": {
            "tools": tools,
            # "resources": [
            #     {
            #         "uri": "manual/hcm/chapter15",
            #         "description": "Highway Capacity Manual Chapter 15 content"
            #     }
            # ],
            # "prompts": [
            #     {
            #         "name": "summarize-section",
            #         "description": "Generates a prompt to summarize an HCM section"
            #     }
            # ]
        }
    }

async def stream_completion(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""
    # Extract the last user message
    last_user_msg = next((m.content for m in reversed(request.messages) if m.role == "user"), "")
    words = f"This is a streaming response to: {last_user_msg}".split()
    
    # Generate an id for this completion
    completion_id = f"chatcmpl-{int(time.time())}"
    
    # Send chunks of the response
    for i, word in enumerate(words):
        await asyncio.sleep(0.1) # Simulate thinking time
        
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
        
        yield await f"data: {json.dumps(chunk)}\n\n"
    
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
    
    yield await f"data: {json.dumps(final_chunk)}\n\n"
    # Signal end of stream
    yield await "data: [DONE]\n\n"

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

mcp = FastApiMCP(
    app,
    name="HCM-LLM",
    description="Highway Capacity Manual API with Transportation Analysis",
    describe_all_responses=True,
    describe_full_response_schema=True  
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    print("RUN SERVER")

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")

    uvicorn.run(app, host=host, port=port)