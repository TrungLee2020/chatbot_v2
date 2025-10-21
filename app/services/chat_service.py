"""
Chat Service - orchestrates the full chat flow.
This is a simplified version - full implementation will be completed in next iteration.
"""
import logging
import time
from typing import List
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.repositories.chat_history_repo import ChatHistoryRepository
from app.domain.models import ChatRequest, ChatResponse, ChatHistoryItem
from app.core.config import settings
from app.utils.timing import timed
from app.utils.logging import log
from app.utils.validators import normalize_topic

logger = logging.getLogger(__name__)


class ChatService:
    """Service for chat operations."""

    def __init__(
        self,
        llm_service: LLMService,
        rag_service: RAGService,
        chat_repo: ChatHistoryRepository
    ):
        """Initialize chat service."""
        self.llm = llm_service
        self.rag = rag_service
        self.chat_repo = chat_repo

    def _get_chat_history_string(
        self,
        messages: List[ChatHistoryItem]
    ) -> str:
        """Extract recent chat history as string."""
        chat_history = []
        recent_messages = messages[-settings.chat_history_length:]

        for msg in recent_messages:
            if msg.human:
                chat_history.append(f"Human: {msg.human}")
            if msg.chatbot:
                chat_history.append(f"Chatbot: {msg.chatbot}")

        return "\n".join(chat_history)

    @timed
    async def process_chat(
        self,
        request: ChatRequest
    ) -> ChatResponse:
        """
        Process a chat request.

        Note: This is a simplified skeleton. Full implementation will be
        completed in the next iteration to avoid message length limits.
        """
        log("info", "Chat request received",
            topic=request.topic, message_length=len(request.user_message))

        # Normalize topic
        topic_filter = normalize_topic(request.topic)

        # TODO: Implement full chat flow from app_new.py:chat_endpoint
        # For now, return placeholder response

        return ChatResponse(
            bot_message="System is being refactored. Please check back soon.",
            structured_references=None,
            doc_id=[],
            show_ref=0,
            timestamp=time.time(),
            err_id=None,
            retrieval_metrics="{}",
            structured_context=None
        )

    # Additional helper methods will be added in next iteration
