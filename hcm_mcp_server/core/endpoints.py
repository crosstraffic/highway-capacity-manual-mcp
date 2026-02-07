import asyncio
import time
from typing import Dict, Any
from fastapi import HTTPException, APIRouter, Depends
from hcm_mcp_server.core.dependencies import get_function_registry
from hcm_mcp_server.core.registry import FunctionRegistry

from .models import (
    ToolCallRequest, ListToolsRequest, FunctionListResponse,
    QueryHCMRequest, TwoLaneHighwaysInput, SegmentAnalysisRequest,
    StandardResponse, SummarizeRequest, BatchQueryRequest,
    SearchByChapterRequest, GetSectionRequest
)


router = APIRouter()

    
@router.post("/tools/list", tags=["tools"], operation_id="list_tools")
async def list_tools(
    request: ListToolsRequest = None,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> FunctionListResponse:
    """List all available tools with optional filtering."""
    if request is None:
        request = ListToolsRequest()
    
    # Get functions based on filters
    if request.category and request.chapter:
        functions = {
            name: info for name, info in registry.get_all_functions().items()
            if info.get("category") == request.category and info.get("chapter") == request.chapter
        }
    elif request.category:
        functions = registry.get_functions_by_category(request.category)
    elif request.chapter:
        functions = registry.get_functions_by_chapter(request.chapter)
    else:
        functions = registry.get_all_functions()
    
    # Format response
    tools = []
    for name, info in functions.items():
        tools.append({
            "name": name,
            "description": info["description"],
            "category": info.get("category", "general"),
            "chapter": info.get("chapter"),
            "step": info.get("step"),
            "parameters": info["parameters"]
        })
    
    return FunctionListResponse(
        functions=tools,
        total_count=len(tools),
        categories=registry.list_categories(),
        chapters=registry.list_chapters()
    )

@router.post("/tools/call", tags=["tools"], operation_id="call_tool")
async def call_tool(
    request: ToolCallRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Execute a function call."""
    function_name = request.function.name
    
    # Get function
    function_impl = registry.get_function(function_name)
    if function_impl is None:
        raise HTTPException(
            status_code=404,
            detail=f"Function '{function_name}' not found"
        )
    
    # Validate parameters
    if not registry.validate_function_parameters(function_name, request.function.arguments):
        raise HTTPException(
            status_code=400,
            detail="Invalid function parameters"
        )
    
    try:
        # Execute the function
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
        raise HTTPException(
            status_code=500,
            detail=f"Function execution failed: {str(e)}"
        )

@router.post("/tools/query-hcm", tags=["research"], operation_id="query_hcm")
async def query_hcm(
    request: QueryHCMRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Query HCM documentation."""
    query_function = registry.get_function("query_hcm")
    
    if query_function is None:
        raise HTTPException(status_code=404, detail="HCM query function not available")
    
    try:
        result = await query_function(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Research function endpoints
@router.post("/research/search-by-chapter", tags=["research"], operation_id="search_hcm_by_chapter")
async def search_hcm_by_chapter(
    request: SearchByChapterRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Search HCM content by specific chapter."""
    search_function = registry.get_function("search_hcm_by_chapter")
    
    if search_function is None:
        raise HTTPException(status_code=404, detail="HCM search function not available")
    
    try:
        result = await search_function(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/research/get-section", tags=["research"], operation_id="get_hcm_section")
async def get_hcm_section(
    request: GetSectionRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Get specific HCM section content."""
    section_function = registry.get_function("get_hcm_section")
    
    if section_function is None:
        raise HTTPException(status_code=404, detail="HCM section retrieval function not available")
    
    try:
        result = await section_function(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/research/summarize", tags=["research"], operation_id="summarize_hcm_content")
async def summarize_hcm_content(
    request: SummarizeRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Summarize HCM content for a given topic."""
    summarize_function = registry.get_function("summarize_hcm_content")

    if summarize_function is None:
        raise HTTPException(status_code=404, detail="HCM summarization function not available")

    try:
        result = await summarize_function(request.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Batch research endpoint for multiple queries
@router.post("/research/batch-query", tags=["research"], operation_id="batch_query_hcm")
async def batch_query_hcm(
    request: BatchQueryRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Perform a batch query on HCM content."""
    query_function = registry.get_function("query_hcm")

    if query_function is None:
        raise HTTPException(status_code=404, detail="HCM query function not available")

    try:
        results = []
        for i, query in enumerate(request.queries):
            try:
                result = await query_function({
                    "question": query.question,
                    "top_k": query.top_k
                })
                results.append({
                    "query_index": i,
                    "query": query,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "query_index": i,
                    "query": query,
                    "result": {
                        "success": False,
                        "error": str(e)
                    }
                })
        return {
            "success": True,
            "batch_size": len(request.queries),
            "results": results,
            "completed_queries": len([r for r in results if r["result"].get("success", False)])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/research/analytics", tags=["research"], operation_id="research_analytics")
async def research_analytics() -> Dict[str, Any]:
    """Get analytics about HCM research database."""
    try:
        return {
            "success":  True,
            "database_status": "available",
            "supported_chapters": ["misc", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", 
                        "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38"],
            "search_functions": [
                "query_hcm",
                "search_hcm_by_chapter",
                "get_hcm_section",
                "summarize_hcm_content"
            ],
            "last_updated": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chapter 15 specific endpoints
@router.post("/analysis/chapter15/complete", tags=["chapter15"], operation_id="chapter15_complete")
async def chapter15_complete_analysis(
    request: TwoLaneHighwaysInput,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Complete Chapter 15 two-lane highway analysis."""
    function_impl = registry.get_function("chapter15_complete_analysis")

    if function_impl is None:
        raise HTTPException(status_code=404, detail="Chapter 15 analysis function not available")
    
    try:
        result = function_impl({"highway_data": request.model_dump()})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analysis/chapter15/segment", tags=["chapter15"], operation_id="chapter15_segment")
async def chapter15_segment_analysis(
    request: SegmentAnalysisRequest,
    registry: FunctionRegistry = Depends(get_function_registry),
) -> Dict[str, Any]:
    """Analyze a specific segment using Chapter 15 methodology."""
    
    # Get multiple functions for comprehensive segment analysis
    functions_to_run = [
        "chapter15_identify_vertical_class",
        "chapter15_determine_demand_flow", 
        "chapter15_determine_vertical_alignment",
        "chapter15_determine_free_flow_speed",
        "chapter15_estimate_average_speed",
        "chapter15_estimate_percent_followers",
        "chapter15_determine_follower_density_pc_pz_function",
        "chapter15_determine_follower_density_pl_function",
        "chapter15_determine_adjustment_to_follower_density"
    ]
    
    results = {}
    
    try:
        for func_name in functions_to_run:
            function_impl = registry.get_function(func_name)
            if function_impl:
                step_result = function_impl(request.model_dump())
                if step_result.get("success"):
                    results[func_name] = step_result
                else:
                    return {"success": False, "error": f"Failed at step {func_name}"}
        
        return {
            "success": True,
            "segment_index": request.segment_index,
            "analysis_results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints
@router.get("/health", tags=["utility"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "HCM-LLM API"}

@router.get("/registry/info", tags=["utility"])
async def registry_info(
    registry: FunctionRegistry = Depends(get_function_registry)
) -> Dict[str, Any]:
    """Get information about the function registry."""
    
    return {
        "total_functions": len(registry.get_all_functions()),
        "categories": registry.list_categories(),
        "chapters": registry.list_chapters(),
        "registry_file": str(registry.registry_file)
    }

@router.post("/registry/reload", tags=["utility"])
async def reload_registry(
    registry: FunctionRegistry = Depends(get_function_registry)
) -> StandardResponse:
    """Reload the function registry from file."""
    try:
        registry.reload_registry()
        return StandardResponse(
            success=True,
            message="Registry reloaded successfully",
            data={"function_count": len(registry.get_all_functions())}
        )
    except Exception as e:
        return StandardResponse(
            success=False,
            error=str(e),
            error_type=type(e).__name__
        )