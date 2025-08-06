from fastapi import Request

async def get_embedding_model(request: Request):
    return request.app.state.embedding_model

async def get_chroma_collection(request: Request):
    return request.app.state.chroma_collection

async def get_function_registry(request: Request):
    return request.app.state.function_registry
