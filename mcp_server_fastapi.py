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

class SegmentAnalysisRequest(BaseModel):
    segment_index: int = Field(description="Index of the segment to analyzed")
    highway_data: TwoLaneHighwaysInput = Field(description="Data for the highway segment to be analyzed")

class SpeedCalculationRequest(BaseModel):
    segment_index: int = Field(description="Segment index")
    length: float = Field(description="Length for calculation")
    vd: float = Field(description="Demand volume")
    phv: float = Field(description="Percent heavy vehicles")
    rad: float = Field(description="Radius of curve")
    sup_ele: float = Field(description="Superelevation rate")
    highway_data: TwoLaneHighwaysInput = Field(description="Highway data")

class FacilityLOSRequest(BaseModel):
    highway_data: TwoLaneHighwaysInput = Field(description="Data for the highway facility")

# TODO: Read from Yaml
# Function calling registry
AVAILABLE_FUNCTIONS = {
    "calc_twolanehighway": {
        "function": None,
        "description": "Calculate comprehensive two-lane highway analysis",
        "parameters": {
            "type": "object",
            "properties": {
                "segments": {"type": "array", "description": "Highway segments"},
                "lane_width": {"type": "number", "description": "Lane width in feet"},
                "shoulder_width": {"type": "number", "description": "Shoulder width in feet"}
            }
        },
        "category": "transportation"
    },
    "identify_vertical_class": {
        "function": None,
        "description": "Identify vertical alignment class range for a segment",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_demand_flow": {
        "function": None,
        "description": "Calculate demand flow rates and capacity",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_vertical_alignment": {
        "function": None,
        "description": "Determine vertical alignment classification",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_free_flow_speed": {
        "function": None,
        "description": "Calculate free flow speed for a segment",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "estimate_average_speed": {
        "function": None,
        "description": "Estimate average travel speed for a segment",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "estimate_percent_followers": {
        "function": None,
        "description": "Estimate percentage of vehicles following in platoons",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "estimate_average_speed_sf": {
        "function": None,
        "description": "Estimate average speed for specific flow conditions",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "length": {"type": "number", "description": "Length"},
                "vd": {"type": "number", "description": "Demand volume"},
                "phv": {"type": "number", "description": "Percent heavy vehicles"},
                "rad": {"type": "number", "description": "Radius"},
                "sup_ele": {"type": "number", "description": "Superelevation"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "estimate_percent_followers_sf": {
        "function": None,
        "description": "Estimate percent followers for specific flow conditions",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "vd": {"type": "number", "description": "Demand volume"},
                "phv": {"type": "number", "description": "Percent heavy vehicles"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_follower_density_pl": {
        "function": None,
        "description": "Calculate follower density for passing lane segments",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_follower_density_pc_pz": {
        "function": None,
        "description": "Calculate follower density for passing constrained/zone segments",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_adjustment_to_follower_density": {
        "function": None,
        "description": "Calculate adjustment to follower density based on upstream passing lanes",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_segment_los": {
        "function": None,
        "description": "Determine Level of Service for a specific segment",
        "parameters": {
            "type": "object",
            "properties": {
                "segment_index": {"type": "integer", "description": "Segment index"},
                "s_pl": {"type": "number", "description": "Average speed"},
                "capacity": {"type": "integer", "description": "Capacity"},
                "highway_data": {"type": "object", "description": "Highway data"}
            }
        },
        "category": "transportation"
    },
    "determine_facility_los": {
        "function": None,
        "description": "Determine overall facility Level of Service",
        "parameters": {
            "type": "object",
            "properties": {
                "highway_data": {"type": "object", "description": "Complete highway facility data"}
            }
        },
        "category": "transportation"
    },
    "complete_highway_analysis": {
        "function": None,
        "description": "Perform complete HCM analysis following the standard procedure",
        "parameters": {
            "type": "object",
            "properties": {
                "highway_data": {"type": "object", "description": "Complete highway facility data"}
            }
        },
        "category": "transportation"
    },
    "query_hcm": {
        "function": None,
        "description": "Query the Highway Capacity Manual database for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "Question to search in HCM database"},
                "top_k": {"type": "integer", "description": "Number of results to return", "default": 5}
            }
        },
        "category": "research"
    }
}

def create_highway_from_input(highway_input: TwoLaneHighwaysInput) -> TwoLaneHighways:
    """Helper function to create highway object from input data."""
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

    return TwoLaneHighways(
        segments=py_segments,
        lane_width=highway_input.lane_width,
        shoulder_width=highway_input.shoulder_width,
        apd=highway_input.apd,
        pmhvfl=highway_input.pmhvfl,
        l_de=highway_input.l_de
    )

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

# Function implementations
def calc_twolanehighway_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Function implementation for comprehensive two-lane highway calculation."""
    try:
        highway_input = TwoLaneHighwaysInput(**data)
        highway = create_highway_from_input(highway_input)

        results = {}
        
        # Analyze each segment
        for seg_idx in range(len(highway_input.segments)):
            seg_results = {}
            
            # Step 1: Identify vertical class
            vertical_class_range = highway.identify_vertical_class(seg_idx)
            seg_results["vertical_class_range"] = {
                "min": vertical_class_range[0],
                "max": vertical_class_range[1]
            }
            
            # Step 2: Determine demand flow
            demand_flow_results = highway.determine_demand_flow(seg_idx)
            seg_results["demand_flow"] = {
                "inbound": demand_flow_results[0],
                "outbound": demand_flow_results[1],
                "capacity": demand_flow_results[2]
            }
            
            # Step 3: Determine vertical alignment
            vertical_alignment = highway.determine_vertical_alignment(seg_idx)
            seg_results["vertical_alignment"] = vertical_alignment
            
            # Step 4: Determine free flow speed
            free_flow_speed = highway.determine_free_flow_speed(seg_idx)
            seg_results["free_flow_speed"] = free_flow_speed
            
            # Step 5: Estimate average speed
            avg_speed_results = highway.estimate_average_speed(seg_idx)
            seg_results["average_speed"] = {
                "speed": avg_speed_results[0],
                "horizontal_class": avg_speed_results[1]
            }
            
            # Step 6: Estimate percent followers
            percent_followers = highway.estimate_percent_followers(seg_idx)
            seg_results["percent_followers"] = percent_followers
            
            # Step 8: Determine follower density
            if highway_input.segments[seg_idx].passing_type == 2:  # Passing lane
                follower_density_results = highway.determine_follower_density_pl(seg_idx)
                seg_results["follower_density"] = {
                    "fd": follower_density_results[0],
                    "fd_mid": follower_density_results[1]
                }
            else:  # Passing constrained or zone
                follower_density = highway.determine_follower_density_pc_pz(seg_idx)
                seg_results["follower_density"] = {
                    "fd": follower_density,
                    "fd_mid": None
                }
            
            # Determine segment LOS
            segment_capacity = int(seg_results["demand_flow"]["capacity"])
            segment_speed = seg_results["average_speed"]["speed"]
            segment_los = highway.determine_segment_los(seg_idx, segment_speed, segment_capacity)
            seg_results["level_of_service"] = segment_los
            
            results[f"segment_{seg_idx}"] = seg_results

        # Calculate facility-level metrics
        total_length = sum(seg.length for seg in highway_input.segments)
        weighted_fd = 0.0
        weighted_speed = 0.0
        
        for seg_idx, seg in enumerate(highway_input.segments):
            if seg.passing_type == 2:  # Passing lane
                fd_value = results[f"segment_{seg_idx}"]["follower_density"]["fd_mid"]
            else:
                fd_value = results[f"segment_{seg_idx}"]["follower_density"]["fd"]
            
            speed_value = results[f"segment_{seg_idx}"]["average_speed"]["speed"]
            
            weighted_fd += fd_value * seg.length
            weighted_speed += speed_value * seg.length
        
        facility_fd = weighted_fd / total_length
        facility_speed = weighted_speed / total_length
        facility_los = highway.determine_facility_los(facility_fd, facility_speed)
        
        return {
            "success": True,
            "facility_summary": {
                "total_length": total_length,
                "num_segments": len(highway_input.segments),
                "facility_follower_density": facility_fd,
                "facility_average_speed": facility_speed,
                "facility_level_of_service": facility_los
            },
            "segment_results": results
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def identify_vertical_class_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Identify vertical class range for a segment."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        vertical_class_range = highway.identify_vertical_class(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "vertical_class_range": {
                "min": vertical_class_range[0],
                "max": vertical_class_range[1]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_demand_flow_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine demand flow rates and capacity."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        demand_flow_results = highway.determine_demand_flow(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "demand_flow_inbound": demand_flow_results[0],
            "demand_flow_outbound": demand_flow_results[1],
            "capacity": demand_flow_results[2]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_vertical_alignment_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine vertical alignment classification."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        vertical_alignment = highway.determine_vertical_alignment(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "vertical_alignment": vertical_alignment
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_free_flow_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate free flow speed."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Need to run prerequisite steps
        highway.determine_demand_flow(segment_index)
        free_flow_speed = highway.determine_free_flow_speed(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "free_flow_speed": free_flow_speed
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def estimate_average_speed_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate average speed."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        avg_speed_results = highway.estimate_average_speed(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "average_speed": avg_speed_results[0],
            "horizontal_class": avg_speed_results[1]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def estimate_percent_followers_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate percent followers."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        percent_followers = highway.estimate_percent_followers(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "percent_followers": percent_followers
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def estimate_average_speed_sf_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate average speed for specific flow conditions."""
    try:
        segment_index = data["segment_index"]
        length = data["length"]
        vd = data["vd"]
        phv = data["phv"]
        rad = data["rad"]
        sup_ele = data["sup_ele"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        speed_results = highway.estimate_average_speed_sf(segment_index, length, vd, phv, rad, sup_ele)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "average_speed": speed_results[0],
            "horizontal_class": speed_results[1]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def estimate_percent_followers_sf_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate percent followers for specific flow conditions."""
    try:
        segment_index = data["segment_index"]
        vd = data["vd"]
        phv = data["phv"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        
        percent_followers = highway.estimate_percent_followers_sf(segment_index, vd, phv)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "percent_followers": percent_followers
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_follower_density_pl_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate follower density for passing lane segments."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        highway.estimate_average_speed(segment_index)
        highway.estimate_percent_followers(segment_index)
        
        follower_density_results = highway.determine_follower_density_pl(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "follower_density": follower_density_results[0],
            "follower_density_mid": follower_density_results[1]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_follower_density_pc_pz_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate follower density for PC/PZ segments."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps
        highway.determine_demand_flow(segment_index)
        highway.determine_free_flow_speed(segment_index)
        highway.estimate_average_speed(segment_index)
        highway.estimate_percent_followers(segment_index)
        
        follower_density = highway.determine_follower_density_pc_pz(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "follower_density": follower_density
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_adjustment_to_follower_density_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate adjustment to follower density."""
    try:
        segment_index = data["segment_index"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run prerequisite steps for all segments
        for seg_idx in range(len(highway_input.segments)):
            highway.determine_demand_flow(seg_idx)
            highway.determine_free_flow_speed(seg_idx)
            highway.estimate_average_speed(seg_idx)
            highway.estimate_percent_followers(seg_idx)
            highway.determine_follower_density_pc_pz(seg_idx)
        
        adjustment = highway.determine_adjustment_to_follower_density(segment_index)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "follower_density_adjustment": adjustment
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_segment_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine segment Level of Service."""
    try:
        segment_index = data["segment_index"]
        s_pl = data["s_pl"]
        capacity = data["capacity"]
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        segment_los = highway.determine_segment_los(segment_index, s_pl, capacity)
        
        return {
            "success": True,
            "segment_index": segment_index,
            "level_of_service": segment_los,
            "average_speed": s_pl,
            "capacity": capacity
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def determine_facility_los_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine facility Level of Service."""
    try:
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        # Run complete analysis for all segments
        total_length = 0.0
        weighted_fd = 0.0
        weighted_speed = 0.0
        
        for seg_idx in range(len(highway_input.segments)):
            # Run all prerequisite steps
            highway.determine_demand_flow(seg_idx)
            highway.determine_free_flow_speed(seg_idx)
            avg_speed_results = highway.estimate_average_speed(seg_idx)
            highway.estimate_percent_followers(seg_idx)
            
            segment_length = highway_input.segments[seg_idx].length
            total_length += segment_length
            
            # Calculate follower density based on segment type
            if highway_input.segments[seg_idx].passing_type == 2:  # Passing lane
                fd_results = highway.determine_follower_density_pl(seg_idx)
                fd_value = fd_results[1]  # Use fd_mid for passing lanes
            else:  # PC or PZ
                fd_value = highway.determine_follower_density_pc_pz(seg_idx)
            
            weighted_fd += fd_value * segment_length
            weighted_speed += avg_speed_results[0] * segment_length
        
        facility_fd = weighted_fd / total_length
        facility_speed = weighted_speed / total_length
        facility_los = highway.determine_facility_los(facility_fd, facility_speed)
        
        return {
            "success": True,
            "facility_follower_density": facility_fd,
            "facility_average_speed": facility_speed,
            "facility_level_of_service": facility_los,
            "total_length": total_length
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def complete_highway_analysis_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Perform complete HCM analysis following standard procedure."""
    try:
        highway_input = TwoLaneHighwaysInput(**data["highway_data"])
        highway = create_highway_from_input(highway_input)
        
        analysis_results = {
            "facility_info": {
                "total_length": sum(seg.length for seg in highway_input.segments),
                "num_segments": len(highway_input.segments),
                "lane_width": highway_input.lane_width,
                "shoulder_width": highway_input.shoulder_width,
                "apd": highway_input.apd
            },
            "segments": {}
        }
        
        # Analyze each segment following HCM procedure
        for seg_idx in range(len(highway_input.segments)):
            segment_data = highway_input.segments[seg_idx]
            seg_results = {
                "basic_info": {
                    "passing_type": segment_data.passing_type,
                    "length": segment_data.length,
                    "grade": segment_data.grade,
                    "speed_limit": segment_data.spl,
                    "volume": segment_data.volume,
                    "volume_opposite": segment_data.volume_op
                }
            }
            
            # Step 1: Identify vertical class
            vertical_class_range = highway.identify_vertical_class(seg_idx)
            seg_results["step_1_vertical_class"] = {
                "min_length": vertical_class_range[0],
                "max_length": vertical_class_range[1]
            }
            
            # Step 2: Determine demand flow
            demand_flow_results = highway.determine_demand_flow(seg_idx)
            seg_results["step_2_demand_flow"] = {
                "inbound_flow_rate": demand_flow_results[0],
                "outbound_flow_rate": demand_flow_results[1],
                "capacity": demand_flow_results[2]
            }
            
            # Step 3: Determine vertical alignment
            vertical_alignment = highway.determine_vertical_alignment(seg_idx)
            seg_results["step_3_vertical_alignment"] = vertical_alignment
            
            # Step 4: Determine free flow speed
            free_flow_speed = highway.determine_free_flow_speed(seg_idx)
            seg_results["step_4_free_flow_speed"] = free_flow_speed
            
            # Step 5: Estimate average speed
            avg_speed_results = highway.estimate_average_speed(seg_idx)
            seg_results["step_5_average_speed"] = {
                "speed": avg_speed_results[0],
                "horizontal_class": avg_speed_results[1]
            }
            
            # Step 6: Estimate percent followers
            percent_followers = highway.estimate_percent_followers(seg_idx)
            seg_results["step_6_percent_followers"] = percent_followers
            
            # Step 8: Determine follower density
            if segment_data.passing_type == 2:  # Passing lane
                fd_results = highway.determine_follower_density_pl(seg_idx)
                seg_results["step_8_follower_density"] = {
                    "type": "passing_lane",
                    "fd": fd_results[0],
                    "fd_mid": fd_results[1]
                }
                fd_for_los = fd_results[1]  # Use fd_mid for LOS
            else:  # PC or PZ
                fd_value = highway.determine_follower_density_pc_pz(seg_idx)
                seg_results["step_8_follower_density"] = {
                    "type": "pc_or_pz",
                    "fd": fd_value
                }
                fd_for_los = fd_value
            
            # Determine segment LOS
            segment_los = highway.determine_segment_los(
                seg_idx, 
                avg_speed_results[0], 
                int(demand_flow_results[2])
            )
            seg_results["segment_los"] = segment_los
            
            analysis_results["segments"][f"segment_{seg_idx}"] = seg_results
        
        # Calculate facility LOS
        facility_los_result = determine_facility_los_function({"highway_data": data["highway_data"]})
        if facility_los_result["success"]:
            analysis_results["facility_los"] = {
                "follower_density": facility_los_result["facility_follower_density"],
                "average_speed": facility_los_result["facility_average_speed"],
                "level_of_service": facility_los_result["facility_level_of_service"]
            }
        
        return {
            "success": True,
            "analysis_type": "complete_hcm_procedure",
            "results": analysis_results
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "error_type": type(e).__name__}

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
AVAILABLE_FUNCTIONS["identify_vertical_class"]["function"] = identify_vertical_class_function
AVAILABLE_FUNCTIONS["determine_demand_flow"]["function"] = determine_demand_flow_function
AVAILABLE_FUNCTIONS["determine_vertical_alignment"]["function"] = determine_vertical_alignment_function
AVAILABLE_FUNCTIONS["determine_free_flow_speed"]["function"] = determine_free_flow_speed_function
AVAILABLE_FUNCTIONS["estimate_average_speed"]["function"] = estimate_average_speed_function
AVAILABLE_FUNCTIONS["estimate_percent_followers"]["function"] = estimate_percent_followers_function
AVAILABLE_FUNCTIONS["estimate_average_speed_sf"]["function"] = estimate_average_speed_sf_function
AVAILABLE_FUNCTIONS["estimate_percent_followers_sf"]["function"] = estimate_percent_followers_sf_function
AVAILABLE_FUNCTIONS["determine_follower_density_pl"]["function"] = determine_follower_density_pl_function
AVAILABLE_FUNCTIONS["determine_follower_density_pc_pz"]["function"] = determine_follower_density_pc_pz_function
AVAILABLE_FUNCTIONS["determine_adjustment_to_follower_density"]["function"] = determine_adjustment_to_follower_density_function
AVAILABLE_FUNCTIONS["determine_segment_los"]["function"] = determine_segment_los_function
AVAILABLE_FUNCTIONS["determine_facility_los"]["function"] = determine_facility_los_function
AVAILABLE_FUNCTIONS["complete_highway_analysis"]["function"] = complete_highway_analysis_function
AVAILABLE_FUNCTIONS["query_hcm"]["function"] = query_hcm_function

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

@app.post("/tools/analysis/complete", tags=["tools"], operation_id="complete_analysis")
def complete_highway_analysis(data: TwoLaneHighwaysInput):
    """Perform complete HCM analysis."""
    return complete_highway_analysis_function({"highway_data": data.dict()})

@app.post("/tools/analysis/segment", tags=["tools"], operation_id="analyze_segment")
def analyze_segment(data: SegmentAnalysisRequest):
    """Analyze a specific segment."""
    highway = create_highway_from_input(data.highway_data)
    seg_idx = data.segment_index
    
    try:
        # Run analysis steps
        vertical_class = highway.identify_vertical_class(seg_idx)
        demand_flow = highway.determine_demand_flow(seg_idx)
        vertical_alignment = highway.determine_vertical_alignment(seg_idx)
        ffs = highway.determine_free_flow_speed(seg_idx)
        avg_speed = highway.estimate_average_speed(seg_idx)
        pf = highway.estimate_percent_followers(seg_idx)
        
        # Determine follower density based on segment type
        if data.highway_data.segments[seg_idx].passing_type == 2:
            fd_results = highway.determine_follower_density_pl(seg_idx)
            fd_info = {"fd": fd_results[0], "fd_mid": fd_results[1]}
        else:
            fd_value = highway.determine_follower_density_pc_pz(seg_idx)
            fd_info = {"fd": fd_value}
        
        return {
            "success": True,
            "segment_index": seg_idx,
            "results": {
                "vertical_class_range": {"min": vertical_class[0], "max": vertical_class[1]},
                "demand_flow": {"inbound": demand_flow[0], "outbound": demand_flow[1], "capacity": demand_flow[2]},
                "vertical_alignment": vertical_alignment,
                "free_flow_speed": ffs,
                "average_speed": {"speed": avg_speed[0], "horizontal_class": avg_speed[1]},
                "percent_followers": pf,
                "follower_density": fd_info
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/tools/analysis/facility-los", tags=["tools"], operation_id="facility_los")
def facility_los_analysis(data: TwoLaneHighwaysInput):
    """Calculate facility Level of Service."""
    return determine_facility_los_function({"highway_data": data.model_dump()})


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
            "tools": [tool["description"] for tool in AVAILABLE_FUNCTIONS.valus()],
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