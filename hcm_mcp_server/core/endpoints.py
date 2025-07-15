import asyncio
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException

from .models import (
    ToolCallRequest, ListToolsRequest, FunctionListResponse,
    QueryHCMRequest, TwoLaneHighwaysInput, SegmentAnalysisRequest,
    StandardResponse
)


def create_endpoints(app: FastAPI) -> None:
    """Create all API endpoints dynamically."""
    
    @app.post("/tools/list", tags=["tools"], operation_id="list_tools")
    async def list_tools(request: ListToolsRequest = None) -> FunctionListResponse:
        """List all available tools with optional filtering."""
        if request is None:
            request = ListToolsRequest()
        
        registry = app.state.function_registry
        
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
    
    @app.post("/tools/call", tags=["tools"], operation_id="call_tool")
    async def call_tool(request: ToolCallRequest) -> Dict[str, Any]:
        """Execute a function call."""
        function_name = request.function.name
        registry = app.state.function_registry
        
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
    
    @app.post("/tools/query-hcm", tags=["research"], operation_id="query_hcm")
    async def query_hcm(request: QueryHCMRequest) -> Dict[str, Any]:
        """Query HCM documentation."""
        registry = app.state.function_registry
        query_function = registry.get_function("query_hcm")
        
        if query_function is None:
            raise HTTPException(status_code=404, detail="HCM query function not available")
        
        try:
            result = await query_function(request.dict())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Chapter 15 specific endpoints
    @app.post("/analysis/chapter15/complete", tags=["chapter15"], operation_id="chapter15_complete")
    async def chapter15_complete_analysis(request: TwoLaneHighwaysInput) -> Dict[str, Any]:
        """Complete Chapter 15 two-lane highway analysis."""
        registry = app.state.function_registry
        function_impl = registry.get_function("chapter15_complete_analysis")
        
        if function_impl is None:
            raise HTTPException(status_code=404, detail="Chapter 15 analysis function not available")
        
        try:
            result = function_impl({"highway_data": request.dict()})
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/analysis/chapter15/segment", tags=["chapter15"], operation_id="chapter15_segment")
    async def chapter15_segment_analysis(request: SegmentAnalysisRequest) -> Dict[str, Any]:
        """Analyze a specific segment using Chapter 15 methodology."""
        registry = app.state.function_registry
        
        # Get multiple functions for comprehensive segment analysis
        functions_to_run = [
            "chapter15_identify_vertical_class",
            "chapter15_determine_demand_flow", 
            "chapter15_determine_vertical_alignment",
            "chapter15_determine_free_flow_speed",
            "chapter15_estimate_average_speed",
            "chapter15_estimate_percent_followers"
        ]
        
        results = {}
        
        try:
            for func_name in functions_to_run:
                function_impl = registry.get_function(func_name)
                if function_impl:
                    step_result = function_impl(request.dict())
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
    @app.get("/health", tags=["utility"])
    async def health_check() -> Dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy", "service": "HCM-LLM API"}
    
    @app.get("/registry/info", tags=["utility"])
    async def registry_info() -> Dict[str, Any]:
        """Get information about the function registry."""
        registry = app.state.function_registry
        
        return {
            "total_functions": len(registry.get_all_functions()),
            "categories": registry.list_categories(),
            "chapters": registry.list_chapters(),
            "registry_file": str(registry.registry_file)
        }
    
    @app.post("/registry/reload", tags=["utility"])
    async def reload_registry() -> StandardResponse:
        """Reload the function registry from file."""
        try:
            registry = app.state.function_registry
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