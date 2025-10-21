"""
Adapter for embedding model.
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingAdapter:
    """Adapter for HuggingFace embedding model."""

    def __init__(self):
        """Initialize embedding model with configuration."""
        self.model = self._load_model()

    def _load_model(self) -> HuggingFaceEmbedding:
        """Load the embedding model."""
        logger.info(f"Loading embedding model from {settings.embed_model_path}")

        model = HuggingFaceEmbedding(
            model_name=settings.embed_model_path,
            embed_batch_size=settings.embed_model_batch_size,
            max_length=settings.embed_model_max_length,
            trust_remote_code=True,
        )

        logger.info("Embedding model loaded successfully")
        return model

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        return self.model.get_text_embedding(text)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        return [self.get_text_embedding(text) for text in texts]
