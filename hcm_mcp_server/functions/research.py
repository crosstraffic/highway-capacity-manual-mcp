from typing import Dict, Any

async def query_hcm_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Query the Highway Capacity Manual database for relevant information."""
    try:
        # This would typically be injected via dependency injection
        # For now, we'll access via the app state (not ideal but works)
        
        # Get current app instance (this is a workaround)
        # In production, you'd use proper dependency injection
        question = data.get("question", "")
        top_k = data.get("top_k", 5)
        
        if not question:
            return {"success": False, "error": "Question parameter is required"}
        
        # This is a placeholder - in real implementation you'd access the app state
        # through proper dependency injection
        try:
            # Access embedding model and collection from app state
            # Note: This requires the app context to be available
            model = None  # app.state.embedding_model
            collection = None  # app.state.chroma_collection
            
            if model is None or collection is None:
                return {
                    "success": False,
                    "error": "HCM database not properly initialized"
                }
            
            # Encode query
            query_embedding = model.encode([question])
            
            # Search in Chroma
            search_results = collection.query(
                query_embeddings=[query_embedding[0].tolist()],
                n_results=top_k
            )
            
            # Check if we got results
            if not search_results['documents'][0]:
                return {
                    "success": True, 
                    "results": ["No relevant documents found."],
                    "query": question
                }
            
            # Format results
            results = []
            for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                results.append({
                    "content": doc.strip(),
                    "chapter": metadata.get('chapter', 'Unknown'),
                    "section": metadata.get('section', 'Unknown'),
                    "source": f"[{metadata.get('chapter', 'Unknown')}] {metadata.get('section', '')}"
                })
            
            return {
                "success": True,
                "query": question,
                "results": results,
                "result_count": len(results)
            }
            
        except Exception:
            # Fallback to mock results for development
            mock_results = [
                {
                    "content": f"Mock HCM content related to: {question}",
                    "chapter": "15",
                    "section": "Two-Lane Highways", 
                    "source": "[Chapter 15] Two-Lane Highways"
                }
            ]
            
            return {
                "success": True,
                "query": question,
                "results": mock_results,
                "result_count": len(mock_results),
                "note": "Using mock results - HCM database not available"
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def search_hcm_by_chapter_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Search HCM content by specific chapter."""
    try:
        chapter = data.get("chapter")
        query = data.get("query", "")
        top_k = data.get("top_k", 5)
        
        if not chapter:
            return {"success": False, "error": "Chapter parameter is required"}
        
        # Enhanced query with chapter filter
        enhanced_query = f"Chapter {chapter}: {query}" if query else f"Chapter {chapter}"
        
        # Use the main query function with enhanced query
        result = await query_hcm_function({
            "question": enhanced_query,
            "top_k": top_k
        })
        
        if result["success"]:
            # Filter results by chapter if available
            if "results" in result:
                filtered_results = [
                    r for r in result["results"] 
                    if str(chapter) in r.get("chapter", "")
                ]
                result["results"] = filtered_results
                result["result_count"] = len(filtered_results)
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_hcm_section_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Get specific HCM section content."""
    try:
        chapter = data.get("chapter")
        section = data.get("section", "")
        
        if not chapter:
            return {"success": False, "error": "Chapter parameter is required"}
        
        # This would query for specific section content
        query = f"Chapter {chapter} {section}".strip()
        
        result = await query_hcm_function({
            "question": query,
            "top_k": 10
        })
        
        return result
        
    except Exception as e:
        return {"success": False, "error": str(e)}


async def summarize_hcm_content_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize HCM content for a given topic."""
    try:
        topic = data.get("topic", "")
        max_length = data.get("max_length", 500)
        
        if not topic:
            return {"success": False, "error": "Topic parameter is required"}
        
        # Search for relevant content
        search_result = await query_hcm_function({
            "question": topic,
            "top_k": 5
        })
        
        if not search_result["success"]:
            return search_result
        
        # Combine and summarize content
        combined_content = ""
        sources = []
        
        for result in search_result.get("results", []):
            combined_content += result.get("content", "") + "\n\n"
            sources.append(result.get("source", "Unknown"))
        
        # Simple summarization (truncate if too long)
        if len(combined_content) > max_length:
            summary = combined_content[:max_length] + "..."
        else:
            summary = combined_content
        
        return {
            "success": True,
            "topic": topic,
            "summary": summary.strip(),
            "sources": list(set(sources)),
            "original_length": len(combined_content),
            "summary_length": len(summary)
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}