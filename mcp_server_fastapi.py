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
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP
from fastapi.middleware.cors import CORSMiddleware

from hcm_mcp_server.core.registry import FunctionRegistry
from hcm_mcp_server.core.models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice
from hcm_mcp_server.core import endpoints
# from hcm_mcp_server.core.endpoints import create_endpoints
# from hcm_mcp_server.functions.research import create_research_tools

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

app.include_router(endpoints.router)

# MCP discovery endpoint
@app.get("/mcp/discovery")
async def mcp_discovery():
    """MCP SDK discovery endpoint."""
    registry = app.state.function_registry

    tools = []
    resources = []
    prompts = []

    if registry:
        # Build tools from registry
        for name, func_info in registry.get_all_functions().items():
            tools.append({
                "name": name,
                "description": func_info["description"],
                "category": func_info.get("category", "general"),
                "chapter": func_info.get("chapter", None),
                "step": func_info.get("step", None),
                "inputSchema": func_info.get("parameters", {}),
            })

        # Add HCM resources
        chapters = registry.list_chapters()
        for chapter in chapters:
            if chapter: # Skip None chapter
                resources.append({
                    "uri": f"hcm://chapter/{chapter}",
                    "name": f"HCM Chapter {chapter}",
                    "description": f"Highway Capacity Manual Chapter {chapter} content",
                    "mimetype": "text/plain"
                })

        # Add research specific resources
        resources.extend([
            {
                "uri": "hcm://database/search",
                "name": "HCM Database Search",
                "description": "Searchable HCM database content",
                "mimetype": "application/json"
            },
            {
                "uri": "hcm://analytics/database",
                "name": "HCM Database Analytics",
                "description": "Database statistics and capabilities",
                "mimetype": "application/json"
            }
        ])

        # Add prompts
        categories = registry.list_categories()
        for category in categories:
            if category == "research":
                prompts.extend([
                    {
                        "name": "hcm-research-query",
                        "description": "Generate an effective HCM database query",
                        "arguments": [{
                                "name": "topic",
                                "description": "Research topic or question",
                                "required": True
                            },
                            {
                                "name": "chapter",
                                "description": "Specific chapter to focus on (optional)",
                                "required": False,
                            }
                        ]
                    },
                    {
                        "name": "hcm-content-summarize",
                        "description": "Summarize HCM content for a given topic",
                        "arguments": [{
                                "name": "topic",
                                "description": "Topic to summarize",
                                "required": True
                            },
                            {
                                "name": "detail_level",
                                "description": "Level of detail for summary (brief, detailed, comprehensive)",
                                "required": False,
                            }
                        ]
                    }
                ])
            elif category == "transportation":
                prompts.append({
                    "name": "highway-analysis-setup",
                    "description": "Generate a prompt for highway capacity analysis",
                    "arguments": [{
                            "name": "highway_type",
                            "description": "Type of highway facility",
                            "required": True
                        },
                        {
                            "name": "analysis_scope",
                            "description": "Scope of analysis to perform (e.g., speed, LOS, facility)",
                            "required": True,
                        }
                    ]
                })
    
    return {
        "name": "HCM-LLM",
        "version": "0.2.0",
        "description": "Highway Capacity Manual Analysis and research API with LLM integration",
        "author": "CrossTraffic",
        "license": "MIT",
        "capabilities": {
            "tools": {
                "listChanged": True,
                "supportsProgress": False
            },
            "resources": {
                "supportsSubscribe": False,
                "listChanged": True
            },
            "prompts": {
                "listChanged": True
            }
        },
        "tools": tools,
        "resources": resources,
        "prompts": prompts,
        "metadata": {
            "total_functions": len(tools),
            "categories": list(registry.list_categories()) if registry else [],
            "chapters": list(registry.list_chapters()) if registry else [],
            "last_updated": time.time()
        }
    }

async def stream_completion(request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
    """Generate streaming chat completion response."""

    system_instruction = {
        "role": "system",
        "content": (
            "You are an expert transportation engineer specializing in transportation analysis "
            "Always use retrieved context from the database to answer questions accurately "
            "and explain calculations when relevant. "
            "Use the HCM version 7th database to provide detailed, accurate responses. "
            "If the question is about a specific chapter, focus on that chapter's content. "
            "If the question is about transportation analysis, provide detailed calculations and explanations. "
            "If the question is about research, provide relevant HCM database content and analysis. "
            "If you don't know the answer, say 'I don't know."
        )
    }
    messages = [system_instruction] + request.messages

    # Extract the last user message
    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
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

    # Add instruction prompt
    system_instruction = {
        "role": "system",
        "content": (
            "You are an expert transportation engineer specializing in transportation analysis "
            "Always use retrieved context from the database to answer questions accurately "
            "and explain calculations when relevant. "
            "Use the HCM version 7th database to provide detailed, accurate responses. "
            "If the question is about a specific chapter, focus on that chapter's content. "
            "If the question is about transportation analysis, provide detailed calculations and explanations. "
            "If the question is about research, provide relevant HCM database content and analysis. "
            "If you don't know the answer, say 'I don't know."
        )
    }
    
    messages = [system_instruction] + request.messages
    
    # Non-streaming response
    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")
    
    response = ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time())}",
        model=request.model,
        created=int(time.time()),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": f"(HCM Response) This is a response to: {last_user_msg}"
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

    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port)