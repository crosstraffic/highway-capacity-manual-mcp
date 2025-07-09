import pickle
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
import json
import time
import asyncio
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from sentence_transformers import SentenceTransformer

import chromadb
from fastapi import Body, FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from fastapi_mcp import FastApiMCP
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from transportations_library import SubSegment, Segment, TwoLaneHighways

# load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="hcm_documents")

    app.state.embedding_model = model
    app.state.chroma_client = client
    app.state.chroma_collection = collection

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryHCMRequest(BaseModel):
    question: str
    top_k: int = 5

class Message(BaseModel):
    role: str # 'user', 'assistant', or 'system'
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

class SubSegmentInput(BaseModel):
    length: float = Field(default=0.0, description="Length of the sub-segment in miles")
    avg_speed: float = Field(default=0.0, description="Average travel speed in mph")
    hor_class: int = Field(default=1, description="Horizontal alignment class (1-5)")
    design_rad: float = Field(default=0.0, description="Design radius in feet")
    central_angle: float = Field(default=0.0, description="Central angle in degrees")
    sup_ele: float = Field(default=0.0, description="Superelevation rate as decimal")

class SegmentInput(BaseModel):
    passing_type: int = Field(default=0, description="Passing type (0: passing constrained, 1: passing zone, 2: passing lane)")
    length: float = Field(default=0.0, description="Length of the segment in miles")
    grade: float = Field(default=0.0, description="Grade of the segment as a percentage")
    spl: float = Field(default=0.0, description="Speed limit in mph")
    is_hc: bool = Field(default=False, description="Is horizontal curve present")
    volume: float = Field(default=0.0, description="Traffic volume in vehicles per hour")
    volume_op: float = Field(default=0.0, description="Opposing traffic volume in vehicles per hour")
    flow_rate: float = Field(default=0.0, description="Flow rate in vehicles per hour")
    flow_rate_o: float = Field(default=0.0, description="Opposing flow rate in vehicles per hour")
    capacity: int = Field(default=0, description="Capacity in vehicles per hour")
    ffs: float = Field(default=0.0, description="Free flow speed in mph")
    avg_speed: float = Field(default=0.0, description="Average speed in mph")
    vertical_class: int = Field(default=1, description="Vertical alignment class (1-5)")
    subsegments: List[SubSegmentInput] = Field(default_factory=list, description="List of sub-segments")
    phf: float = Field(default=0.0, description="Peak hour factor")
    phv: float = Field(default=0.0, description="Percent heavy vehicles")
    pf: float = Field(default=0.0, description="Parcent followers")
    fd: float = Field(default=0.0, description="Follower density")
    fd_mid: float = Field(default=0.0, description="Mid-segment follower density")
    hor_class: int = Field(default=1, description="Horizontal class (1-5)")

class TwoLaneHighwaysInput(BaseModel):
    segments: List[SegmentInput] = Field(default_factory=list, description="List of highway segments")
    lane_width: float = Field(default=12.0, description="Lane width in feet")
    shoulder_width: float = Field(default=10.0, description="Shoulder width in feet")
    apd: float = Field(default=0.0, description="Access point density per mile")
    pmhvfl: float = Field(default=0.0, description="Percent heavy vehicles following")
    l_de: float = Field(default=0.0, description="Effective distance to passing lane")

class ListToolsRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter tools by category")

class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]

class ToolCallRequest(BaseModel):
    function: FunctionCall

# TODO: Read from Yaml
AVAILABLE_FUNCTIONS = {
    "calc_twolanehighway": {
        "function": None, # Will be set below
        "description": "Calculate two-lane highway capacity and level of service analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "passing_type": {"type": "integer", "description": "Passing type"},
                            "length": {"type": "number", "description": "Segment length in miles"},
                            "grade": {"type": "number", "description": "Grade percentage"},
                            "spl": {"type": "number", "description": "Speed limit in mph"},
                            "volume": {"type": "number", "description": "Traffic volume"},
                            "subsegments": {"type": "array", "items": {"type": "object"}}
                        }
                    }
                },
                "lane_width": {"type": "number", "description": "Lane width in feet"},
                "shoulder_width": {"type": "number", "description": "Shoulder width in feet"}
            }
        },
        "category": "transportation"
    },
    "query_hcm": {
        "function": None, # Will be set below
        "description": "Query the Highway Capacity Manual database for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to search in HCM database"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
            }
        },
        "category": "research"
    },
    "analyze_highway_segment": {
        "function": None, # Will be set below
        "description": "Analyze a single highway segment for detailed performance metrics",
        "parameters": {
            "type": "object",
            "properties": {
                "segment": {
                    "type": "object",
                    "description": "Highway segment data",
                    "properties": {
                        "passing_type": {"type": "integer"},
                        "length": {"type": "number"},
                        "grade": {"type": "number"},
                        "spl": {"type": "number"},
                        "volume": {"type": "number"}
                    }
                }
            }
        },
        "category": "transportation"
    }
}

# async def format_sse(data: str) -> str:
#     """Format data for SSE."""
#     return f"data: {data}\n\n"

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
        
        # yield await format_sse(json.dumps(chunk))
        yield await json.dumps(chunk)
    
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
    
    # yield await format_sse(json.dumps(final_chunk))
    yield await json.dumps(final_chunk)
    
    # Signal end of stream
    # yield await format_sse("[DONE]")
    yield await "[DONE]"

# @app.post("/tools/calc/twolanehighway", tags=["tools"], operation_id="post_calc_twolanehighway")
def calc_twolanehighway_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate two-lane highway parameters."""
    try:
        # Convert dict to TwoLaneHighwaysInput
        highway_input = TwoLaneHighwaysInput(**data)

        # Convert to Python objects
        py_segments = []
        for seg in highway_input.segments:
            subsegments = [
                SubSegment(
                    length=sub.length,
                    avg_speed=sub.avg_speed,
                    hor_class=sub.hor_class,
                    design_rad=sub.design_rad,
                    central_angle=sub.central_angle,
                    sup_ele=sub.sup_ele
                ) for sub in seg.subsegments
            ]
            py_segments.append(Segment(
                passing_type=seg.passing_type,
                length=seg.length,
                grade=seg.grade,
                spl=seg.spl,
                is_hc=seg.is_hc,
                volume=seg.volume,
                volume_op=seg.volume_op,
                flow_rate=seg.flow_rate,
                flow_rate_o=seg.flow_rate_o,
                capacity=seg.capacity,
                ffs=seg.ffs,
                avg_speed=seg.avg_speed,
                vertical_class=seg.vertical_class,
                subsegments=subsegments,
                phf=seg.phf,
                phv=seg.phv,
                pf=seg.pf,
                fd=seg.fd,
                fd_mid=seg.fd_mid,
                hor_class=seg.hor_class,
            ))

        highway = TwoLaneHighways(
            segments=py_segments,
            lane_width=data.lane_width,
            shoulder_width=data.shoulder_width,
            apd=data.apd,
            pmhvfl=data.pmhvfl,
            l_de=data.l_de
        )

        # Perform analysis on first segment (index 0)
        seg_idx = 0
        demand_flow_results = highway.determine_demand_flow(seg_idx)
        free_flow_speed = highway.determine_free_flow_speed(seg_idx)
        avg_speed_results = highway.estimate_average_speed(seg_idx)
        percent_followers = highway.estimate_percent_followers(seg_idx)

        return {
            "success": True,
            "results": {
                "facility_summary": highway.summary(),
                "total_length": highway.total_length(),
                "num_segments": highway.num_segments(),
                "segment_analysis": {
                    "demand_flow_analysis": {
                        "demand_flow_i": demand_flow_results[0],
                        "demand_flow_o": demand_flow_results[1],
                        "capacity": demand_flow_results[2],
                    }
                },
                "free_flow_speed": free_flow_speed,
                "average_speed_analysis": {
                    "average_speed": avg_speed_results[0],
                    "horizontal_class": avg_speed_results[1]
                },
                "percent_followers": percent_followers
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

async def query_hcm_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function implementation for HCM query"""
    try:
        model = app.state.embedding_model
        collection = app.state.chroma_collection
        
        question = data.get("question", "")
        top_k = data.get("top_k", 5)

        # Encode query
        query_embedding = model.encode([question])
        
        # Search in Chroma
        search_results = collection.query(
            query_embeddings=[query_embedding[0].tolist()],
            n_results=top_k
        )
        
        # Check if we got results
        if not search_results['documents'][0]:
            return {"success": True, "results": ["No relevant documents found."]}
        
        # Format results
        results = []
        for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
            results.append(f"[{metadata['chapter']}]\n{doc.strip()}")
        
        return {"success": True, "results": results}
        
    except Exception as e:
        # Log the error in production
        print(f"Error querying HCM: {str(e)}")
        return {"success": False, "error": str(e)}

def analyze_highway_segment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single highway segment for detailed performance metrics."""
    try:
        # Convert dict to SegmentInput
        segment_data = data.get("segment", {})

        segment = Segment(
            passing_type=segment_data.get("passing_type", 0),
            length=segment_data.get("length", 1.0),
            grade=segment_data.get("grade", 0.0),
            spl=segment_data.get("spl", 55.0),
            volume=segment_data.get("volume", 400.0)
        )

        highway = TwoLaneHighways(segments=[segment])
        
        # Perform detailed analysis
        vertical_class_range = highway.identify_vertical_class(0)
        vertical_alignment = highway.determine_vertical_alignment(0)
        demand_flow = highway.determine_demand_flow(0)
        free_flow_speed = highway.determine_free_flow_speed(0)
        avg_speed = highway.estimate_average_speed(0)
        percent_followers = highway.estimate_percent_followers(0)
        follower_density = highway.estimate_follower_density(0)
        
        return {
            "success": True,
            "segment_details": {
                "basic_info": {
                    "length": segment.get_length(),
                    "grade": segment.get_grade(),
                    "speed_limit": segment.get_spl(),
                    "volume": segment.get_volume(),
                },
                "analysis_results": {
                    "vertical_class_range": {
                        "min": vertical_class_range[0],
                        "max": vertical_class_range[1]
                    },
                    "vertical_alignment": vertical_alignment,
                    "demand_flow": {
                        "inbound": demand_flow[0],
                        "outbound": demand_flow[1],
                        "capacity": demand_flow[2]
                    },
                    "free_flow_speed": free_flow_speed,
                    "average_speed": {
                        "speed": avg_speed[0],
                        "horizontal_class": avg_speed[1]
                    },
                    "percent_followers": percent_followers,
                    "follower_density": {
                        "fd": follower_density[0],
                        "fd_mid": follower_density[1]
                    }
                }
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# Set function implementations
AVAILABLE_FUNCTIONS["calc_twolanehighway"]["function"] = calc_twolanehighway_function
AVAILABLE_FUNCTIONS["query_hcm"]["function"] = query_hcm_function
AVAILABLE_FUNCTIONS["analyze_highway_segment"]["function"] = analyze_highway_segment_function

@app.post("/tools/list", tags=["tools"], operation_id="post_list_tools")
async def list_tools(request: ListToolsRequest = None):
    """List all available tools with optional category filtering."""
    if request is None:
        request = ListToolsRequest()

    tools = []
    for name, info in AVAILABLE_FUNCTIONS.items():
        if request.category is None or info.get("category") == request.category:
            tools.append({
                "name": name,
                "description": info["description"],
                "category": info.get("category", "general"),
                "parameters": info["parameters"]
            })
    
    return {
        "tools": tools,
        "total_count": len(tools),
        "categories": list(set(info.get("category", "general") for info in AVAILABLE_FUNCTIONS.values()))
    }

@app.post("/tools/call", tags=["tools"], operation_id="call_tool")
async def call_tool(request: ToolCallRequest):
    """Execute a registered function by name."""
    function_name = request.function.name

    if function_name not in AVAILABLE_FUNCTIONS:
        return {
            "success": False,
            "error": f"Function '{function_name}' not found.",
            "available_functions": list(AVAILABLE_FUNCTIONS.keys())
        }
    
    function_impl = AVAILABLE_FUNCTIONS[function_name]["function"]
    if function_impl is None:
        return {
            "success": False,
            "error": f"Function '{function_name}' is not implemented."
        }
    
    try:
        if asyncio.iscoroutinefunction(function_impl):
            result = await function_impl(request.function.arguments)
        else:
            result = function_impl(request.function.arguments)

        return {
            "function_name": function_name,
            "execution_time": time.time(),
            **result
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "function_name": function_name
        }

@app.post("/tools/calc/twolanehighway", tags=["tools"], operation_id="post_calc_twolanehighway")
def calc_twolanehighway(data: TwoLaneHighwaysInput):
    """Calculate twolane highway parameters"""
    return calc_twolanehighway_function(data.model_dump())

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
    """Query the HCM content using Chroma and sentence-transformer embeddings."""
    return await query_hcm_function(data.model_dump())

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
                },
                {
                    "name": "calc-twolanehighway",
                    "description": "Calculate two-lane highway parameters",
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
    description="Highway Capacity Manual API with Transportation Analysis",
    describe_all_responses=True,
    describe_full_response_schema=True  
)

mcp.mount()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)