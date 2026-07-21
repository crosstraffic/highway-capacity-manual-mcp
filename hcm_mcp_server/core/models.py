from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Chat completion models
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


# Function calling models
class FunctionCall(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCallRequest(BaseModel):
    function: FunctionCall


class ListToolsRequest(BaseModel):
    category: Optional[str] = Field(None, description="Filter tools by category")
    chapter: Optional[int] = Field(None, description="Filter by HCM chapter")


# Transportation analysis models
class SubSegmentInput(BaseModel):
    length: float = Field(default=0.0, description="Length of the sub-segment in miles")
    avg_speed: float = Field(default=0.0, description="Average travel speed in mph")
    hor_class: int = Field(default=1, description="Horizontal alignment class (1-5)")
    design_rad: float = Field(default=0.0, description="Design radius in feet")
    central_angle: float = Field(default=0.0, description="Central angle in degrees")
    sup_ele: float = Field(default=0.0, description="Superelevation rate as decimal")


class SegmentInput(BaseModel):
    passing_type: int = Field(description="Passing type (0=PC, 1=PZ, 2=PL)")
    length: float = Field(description="Segment length in miles")
    grade: float = Field(description="Grade percentage")
    spl: float = Field(description="Speed limit in mph")
    is_hc: bool = Field(default=False, description="Is horizontal curve segment")
    volume: float = Field(default=0.0, description="Traffic volume")
    volume_op: float = Field(default=0.0, description="Opposing direction volume")
    flow_rate: float = Field(default=0.0, description="Flow rate")
    flow_rate_o: float = Field(default=0.0, description="Opposing flow rate")
    capacity: int = Field(default=1700, description="Capacity")
    ffs: float = Field(default=0.0, description="Free flow speed")
    avg_speed: float = Field(default=0.0, description="Average speed")
    vertical_class: int = Field(default=1, description="Vertical alignment class")
    subsegments: List[SubSegmentInput] = Field(default_factory=list, description="List of subsegments")
    phf: float = Field(default=0.92, description="Peak hour factor")
    phv: float = Field(default=0.02, description="Percent heavy vehicles")
    pf: float = Field(default=0.0, description="Percent followers")
    fd: float = Field(default=0.0, description="Follower density")
    fd_mid: float = Field(default=0.0, description="Mid-segment follower density")
    hor_class: int = Field(default=1, description="Horizontal class")


class TwoLaneHighwaysInput(BaseModel):
    segments: List[SegmentInput] = Field(description="List of highway segments")
    lane_width: float = Field(default=12.0, description="Lane width in feet")
    shoulder_width: float = Field(default=6.0, description="Shoulder width in feet")
    apd: float = Field(default=5.0, description="Access point density per mile")
    pmhvfl: float = Field(default=0.02, description="Percent heavy vehicles following")
    l_de: float = Field(default=0.0, description="Length of designated passing zones")


class BasicFreewaysInput(BaseModel):
    """A single basic-freeway (HCM Chapter 12) directional segment."""
    bffs: float = Field(default=65.0, description="Base free-flow speed in mph")
    lw: float = Field(default=12.0, description="Lane width in feet")
    lane_count: int = Field(default=2, description="Lanes in the analysis direction")
    lc_r: int = Field(default=6, description="Right-side lateral clearance in feet")
    lc_l: int = Field(default=6, description="Left-side lateral clearance in feet")
    trd: int = Field(default=0, description="Total ramp density (ramps/mi)")
    apd: int = Field(default=0, description="Access-point density (pts/mi, multilane)")
    grade: float = Field(default=0.0, description="Grade in percent")
    terrain_type: Optional[str] = Field(default=None, description="level / rolling / mountainous")
    speed_limit: int = Field(default=65, description="Posted speed limit in mph")
    phf: float = Field(default=0.95, description="Peak hour factor")
    p_t: float = Field(default=0.05, description="Heavy-vehicle proportion (decimal)")
    sut_percentage: int = Field(default=0, description="Single-unit-truck share of the heavy-vehicle mix; 0 = unknown (general-terrain Exhibit 12-25), or 30/50/70 for the specific-upgrade exhibits 12-26/27/28")
    demand_flow_i: float = Field(default=1000.0, description="Directional demand in veh/h")
    length: float = Field(default=0.625, description="Segment length in miles")
    highway_type: str = Field(default="basic", description="'basic' or 'multilane'")
    city_type: Optional[str] = Field(default=None, description="urban / rural")


# HCM analysis request models
class SegmentAnalysisRequest(BaseModel):
    segment_index: int = Field(description="Index of segment to analyze")
    highway_data: TwoLaneHighwaysInput = Field(description="Highway facility data")


class SpeedCalculationRequest(BaseModel):
    segment_index: int = Field(description="Segment index")
    length: float = Field(description="Length for calculation")
    vd: float = Field(description="Demand volume")
    phv: float = Field(description="Percent heavy vehicles")
    rad: float = Field(description="Radius of curve")
    sup_ele: float = Field(description="Superelevation")
    highway_data: TwoLaneHighwaysInput = Field(description="Highway data")


class FacilityLOSRequest(BaseModel):
    highway_data: TwoLaneHighwaysInput = Field(description="Complete highway facility data")


# Research models
class QueryHCMRequest(BaseModel):
    question: str
    top_k: int = 5


class SearchByChapterRequest(BaseModel):
    chapter: str
    query: Optional[str] = ""
    top_k: Optional[int] = 5


class GetSectionRequest(BaseModel):
    chapter: str
    section: Optional[str] = ""


class SummarizeRequest(BaseModel):
    topic: str
    max_length: Optional[int] = 500
    

class BatchQueryRequest(BaseModel):
    queries: List[str]
    top_k: Optional[int] = 5


# Response models
class StandardResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class FunctionListResponse(BaseModel):
    functions: List[Dict[str, Any]]
    total_count: int
    categories: List[str]
    chapters: Optional[List[int]] = None