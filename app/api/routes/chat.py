"""
Chat API routes.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.domain.models import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
from app.api.dependencies import get_chat_service, get_db
from app.utils.logging import set_request_context, clear_request_context
from app.utils.timing import timed

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
@timed
async def chat_endpoint(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
    db: Session = Depends(get_db)
):
    """
    Handle chat requests.

    The actual business logic is delegated to ChatService.
    This endpoint is responsible for:
    - Setting request context for logging
    - Delegating to service layer
    - Clearing context after completion
    """
    set_request_context(
        request.transaction_id,
        request.user_id,
        request.session_id
    )

    try:
        response = await chat_service.process_chat(request)
        return response

    finally:
        clear_request_context()
