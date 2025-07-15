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