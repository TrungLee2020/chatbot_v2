"""
Health check API routes.
"""
from fastapi import APIRouter
import time

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "message": "Refactored architecture is operational"
    }
