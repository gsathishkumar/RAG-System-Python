from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from api.router import api_router
from app_settings import settings
import uvicorn
from exceptions import value_error_handler, validation_exception_handler
from fastapi_lifespan import lifespan

# Map the exception types to the handler functions
HANDLERS = {
    ValueError: value_error_handler,
    RequestValidationError: validation_exception_handler,
}

# Register lifespan & Handler with the app
app = FastAPI(
        title="RAG System",
        description="Retrieval and Generation System",
        lifespan=lifespan, 
        exception_handlers=HANDLERS
    )
app.include_router(api_router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="127.0.0.1", 
        port=settings.app_port, 
        reload=True
    )