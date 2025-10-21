"""
Repository for vector store operations.
"""
from typing import List, Optional, Dict, Any
import logging
from app.adapters.qdrant_adapter import QdrantAdapter
from app.adapters.embedding_adapter import EmbeddingAdapter
from app.core.config import settings
from app.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class VectorStoreRepository:
    """Repository for vector store retrieval operations."""

    def __init__(
        self,
        qdrant_adapter: QdrantAdapter,
        embedding_adapter: EmbeddingAdapter
    ):
        """
        Initialize repository with adapters.

        Args:
            qdrant_adapter: Qdrant database adapter
            embedding_adapter: Embedding model adapter
        """
        self.qdrant = qdrant_adapter
        self.embedder = embedding_adapter

    def multi_vector_search(
        self,
        query: str,
        topic_filter: Optional[str] = None
    ):
        """
        Perform multi-vector search with RRF fusion.

        Args:
            query: Query text
            topic_filter: Optional topic filter

        Returns:
            Search results from Qdrant

        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.debug(f"Starting multi-vector search for query: {query[:50]}...")
            if topic_filter:
                logger.debug(f"With topic filter: {topic_filter}")

            # Generate embeddings for different chunk sizes
            embeddings_dict = {}
            for chunk_size in [128, 256, 512]:
                # Note: In production, you might want to handle chunk_size
                # differently. For now, we just use the same embedding.
                embedding = self.embedder.get_text_embedding(query)
                embeddings_dict[f"text-dense-{chunk_size}"] = embedding

            # Query Qdrant with multi-vector fusion
            results = self.qdrant.query_points(
                embeddings_dict=embeddings_dict,
                topic_filter=topic_filter,
                limit=settings.qdrant_dense_top_k,
                score_threshold=0.3
            )

            logger.info(f"Multi-vector search found {len(results.points)} results")
            return results

        except Exception as e:
            logger.error(f"Multi-vector search failed: {str(e)}")
            raise VectorStoreError(f"Vector search failed: {str(e)}")

    def scroll_collection(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10
    ):
        """
        Scroll through collection points.

        Args:
            filters: Optional filter conditions
            limit: Maximum number of points

        Returns:
            Scroll results
        """
        try:
            from qdrant_client.http import models

            scroll_filter = None
            if filters:
                # Build filter from dict - simplified example
                scroll_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key=k,
                            match=models.MatchValue(value=v)
                        ) for k, v in filters.items()
                    ]
                )

            results = self.qdrant.scroll(
                scroll_filter=scroll_filter,
                limit=limit
            )

            return results

        except Exception as e:
            logger.error(f"Scroll operation failed: {str(e)}")
            raise VectorStoreError(f"Scroll failed: {str(e)}")
