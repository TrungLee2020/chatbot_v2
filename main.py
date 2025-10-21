"""
Main application entry point.
Refactored from app_new.py with clean architecture.
"""
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import sys
import uvicorn

from app.core.config import settings
from app.api.routes import chat, health
from app.utils.logging import setup_logging, log


# Setup logging
logger = setup_logging(
    log_level=settings.general_log_level,
    log_file=settings.general_log_file
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    log("info", "Application startup: Initializing components")

    # Initialize database
    try:
        from database import create_tables
        create_tables()
        log("info", "Database initialized")
    except Exception as e:
        log("error", f"Database initialization failed: {str(e)}")

    # TODO: Add health checks for external services
    # - vLLM server
    # - Qdrant
    # - Xinference

    log("info", "Application startup complete")

    yield

    # Shutdown
    log("info", "Application shutdown: Cleaning up resources")
    # TODO: Close connections if needed


# Create FastAPI app
app = FastAPI(
    title="VNPOST Chatbot API (Refactored)",
    description="Clean architecture implementation",
    version="2.0.0",
    lifespan=lifespan
)


# Exception handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Handle validation errors."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "loc": error["loc"],
            "msg": error["msg"],
            "type": error["type"]
        })
    log("error", "Validation error", error_details=error_details)
    return JSONResponse(status_code=422, content={"detail": error_details})


# Include routers
app.include_router(chat.router, tags=["Chat"])
app.include_router(health.router, tags=["Health"])


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        log_level=settings.log_level.lower(),
        reload=False
    )
