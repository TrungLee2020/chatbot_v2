"""
Adapter for vLLM API client.
Wraps OpenAI-compatible async client for vLLM server.
"""
from openai import AsyncOpenAI
from typing import List, Dict, Optional
import logging
from app.core.config import settings
from app.core.exceptions import VLLMConnectionError

logger = logging.getLogger(__name__)


class VLLMAdapter:
    """Adapter for vLLM server using OpenAI-compatible API."""

    def __init__(self):
        """Initialize vLLM client with configuration from settings."""
        self.server_url = settings.vllm_server_url
        self.model_name = settings.vllm_model_name
        self.client = self._setup_client()

    def _setup_client(self) -> Optional[AsyncOpenAI]:
        """Setup vLLM client with proper error handling and validation."""
        try:
            if not self.server_url:
                raise ValueError("VLLM_SERVER_URL is not configured")

            logger.info(f"Setting up vLLM client for server: {self.server_url}")
            logger.info(f"Model name: {self.model_name}")

            client = AsyncOpenAI(
                base_url=f"{self.server_url}/vllm/v1",
                api_key=settings.vllm_api_key,
                timeout=settings.vllm_timeout,
                max_retries=2
            )

            logger.info("vLLM client initialized successfully")
            return client

        except Exception as e:
            logger.error(f"Failed to setup vLLM client: {str(e)}")
            return None

    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        extra_body: Optional[Dict] = None
    ) -> str:
        """
        Generate text using vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            top_p: Top-p sampling
            max_tokens: Maximum tokens to generate
            extra_body: Extra parameters for vLLM

        Returns:
            Generated text content

        Raises:
            VLLMConnectionError: If vLLM server is unavailable or request fails
        """
        if self.client is None:
            raise VLLMConnectionError("vLLM client not available")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                extra_body=extra_body or {}
            )

            if response.choices and response.choices[0].message.content:
                return str(response.choices[0].message.content)
            else:
                logger.warning("Empty response content from vLLM")
                return ""

        except Exception as e:
            logger.error(f"Error in vLLM generation: {str(e)}", exc_info=True)
            raise VLLMConnectionError(f"vLLM generation failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if vLLM client is available."""
        return self.client is not None
