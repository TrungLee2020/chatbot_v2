"""
Adapter for Xinference reranker model.
"""
from xinference.client import Client
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
from app.core.exceptions import RerankerError

logger = logging.getLogger(__name__)


class XinferenceAdapter:
    """Adapter for Xinference reranker service."""

    def __init__(self):
        """Initialize Xinference client."""
        self.client = Client(settings.xinference_endpoint)
        self.model_name = settings.rerank_model_name
        self.rerank_model = None
        self._initialize_model()

    def _initialize_model(self):
        """Connect to the reranker model."""
        try:
            logger.info("Connecting to the reranker model...")
            models = self.client.list_models()

            for model_id, model_info in models.items():
                if (
                    model_info["model_name"] == self.model_name
                    and model_info["model_type"] == "rerank"
                ):
                    self.rerank_model = self.client.get_model(model_id)
                    logger.info(f"Connected to reranker model. Model ID: {model_id}")
                    return

            if not self.rerank_model:
                logger.error("Custom reranker model not found among launched models.")
                logger.error(f"Available models: {models}")
                raise RerankerError(f"Reranker model '{self.model_name}' not found")

        except Exception as e:
            logger.error(f"Error connecting to reranker model: {e}")
            raise RerankerError(f"Failed to initialize reranker: {str(e)}")

    def rerank(self, corpus: List[str], query: str) -> Dict[str, Any]:
        """
        Rerank a corpus of documents based on query relevance.

        Args:
            corpus: List of document texts to rerank
            query: Query text to rank against

        Returns:
            Reranking results with scores

        Raises:
            RerankerError: If reranking fails
        """
        if not self.rerank_model:
            raise RerankerError("Reranker model not initialized")

        if not corpus:
            logger.warning("Empty corpus provided for reranking")
            return {"results": []}

        try:
            logger.debug(f"Reranking {len(corpus)} documents for query: '{query[:50]}...'")
            result = self.rerank_model.rerank(corpus, query)

            if not result or "results" not in result:
                logger.error(f"Invalid reranker response: {result}")
                raise RerankerError("Invalid reranker response format")

            logger.debug(f"Reranker returned {len(result['results'])} scores")
            return result

        except Exception as e:
            logger.error(f"Reranker API call failed: {str(e)}")
            raise RerankerError(f"Reranking failed: {str(e)}")

    def is_available(self) -> bool:
        """Check if reranker model is available."""
        return self.rerank_model is not None
