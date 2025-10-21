"""
LLM Service - handles all LLM interactions.
"""
from typing import List, Dict
import logging
from app.adapters.vllm_adapter import VLLMAdapter
from app.core.prompts.chat_prompts import (
    CHATBOT_RESPONSE_PROMPT,
    STANDALONE_QUESTION_PROMPT
)
from app.core.prompts.evaluation_prompts import (
    METADATA_EVALUATION_PROMPT,
    EVALUATION_CHAT_RESPONSE_PROMPT
)
from app.core.config import settings
from app.core.constants import SYSTEM_MAINTENANCE_RESPONSE
from app.utils.timing import timed
from app.utils.logging import log
from app.utils.text_processing import get_token_length

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations."""

    def __init__(self, vllm_adapter: VLLMAdapter):
        """Initialize LLM service with vLLM adapter."""
        self.llm = vllm_adapter

    @timed
    async def rewrite_query(
        self,
        chat_history: str,
        user_message: str
    ) -> str:
        """Rewrite user query with chat history context."""
        if not chat_history:
            return user_message

        if not self.llm.is_available():
            log("warning", "vLLM not available for query rewriting")
            return user_message

        prompt = STANDALONE_QUESTION_PROMPT.format(
            chat_history=chat_history,
            user_message=user_message
        )

        log("debug", "Generating standalone question",
            prompt_length=get_token_length(prompt))

        try:
            response = await self.llm.generate(
                messages=[{"role": "user", "content": prompt + "/no_think"}],
                temperature=0.0,
                top_p=1.0,
                max_tokens=256,
                extra_body={"repetition_penalty": 1.0}
            )

            if response:
                log("info", "Query rewritten",
                    original=user_message, rewritten=response)
                return response.strip()
            else:
                return user_message

        except Exception as e:
            log("error", f"Query rewriting failed: {str(e)}")
            return user_message

    @timed
    async def generate_response(
        self,
        context: str,
        user_message: str,
        topic: str
    ) -> str:
        """Generate chatbot response."""
        if not self.llm.is_available():
            log("error", "vLLM client not available")
            return SYSTEM_MAINTENANCE_RESPONSE

        system_message = CHATBOT_RESPONSE_PROMPT.format(
            context=context,
            topic=topic or "Thông tin tổng hợp"
        )

        log("debug", "Generating bot response",
            system_length=get_token_length(system_message),
            user_length=get_token_length(user_message))

        try:
            response = await self.llm.generate(
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=settings.chatbot_temperature,
                top_p=settings.chatbot_top_p,
                max_tokens=settings.chatbot_max_tokens,
                extra_body={
                    "repetition_penalty": settings.chatbot_repetition_penalty,
                    "chat_template_kwargs": {"enable_thinking": False}
                }
            )

            return str(response) if response else ""

        except Exception as e:
            log("error", f"Response generation failed: {str(e)}")
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý."

    async def evaluate_metadata_need(
        self,
        response: str,
        doc_metadata: str
    ) -> bool:
        """Evaluate if metadata should be shown."""
        if not self.llm.is_available():
            return False

        prompt = METADATA_EVALUATION_PROMPT.format(
            response=response,
            doc_metadata=doc_metadata
        )

        try:
            result = await self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.metadata_eval_temperature,
                max_tokens=settings.metadata_eval_max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )

            parsed = eval(result)
            return parsed.get("is_metadata_relevant", False)

        except Exception as e:
            log("error", f"Metadata evaluation failed: {str(e)}")
            return False

    async def evaluate_response_relevance(
        self,
        user_message: str,
        response: str
    ) -> int:
        """Evaluate if response is relevant to user message."""
        if not self.llm.is_available():
            return 1

        prompt = EVALUATION_CHAT_RESPONSE_PROMPT.format(
            user_message=user_message,
            response=response
        )

        try:
            result = await self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=settings.evaluation_chat_response_temperature,
                max_tokens=settings.evaluation_chat_response_max_tokens
            )

            parsed = eval(result)
            return parsed.get("is_response_eval", 1)

        except Exception as e:
            log("error", f"Response evaluation failed: {str(e)}")
            return 1
