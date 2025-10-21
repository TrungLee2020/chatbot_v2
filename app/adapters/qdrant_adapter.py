"""
Adapter for Qdrant vector database.
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Optional, Dict, Any
import logging
from app.core.config import settings
from app.core.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """Adapter for Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client with configuration."""
        self.client = self._connect()
        self.collection_name = settings.qdrant_collection_name

    def _connect(self) -> QdrantClient:
        """Establish connection to Qdrant server."""
        try:
            logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")

            client = QdrantClient(
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                timeout=30.0,
                prefer_grpc=False
            )

            # Verify connection
            collections = client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Available collections: {collections}")

            return client

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise VectorStoreError(f"Qdrant connection failed: {str(e)}")

    def query_points(
        self,
        embeddings_dict: Dict[str, List[float]],
        topic_filter: Optional[str] = None,
        limit: int = 50,
        score_threshold: float = 0.3
    ):
        """
        Query Qdrant with multi-vector RRF fusion.

        Args:
            embeddings_dict: Dictionary of vector name to embedding vector
            topic_filter: Optional topic to filter results
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            Query results from Qdrant
        """
        try:
            # Build prefetch queries for each vector
            prefetches = []
            for vector_name, embedding in embeddings_dict.items():
                prefetches.append(
                    models.Prefetch(
                        query=embedding,
                        using=vector_name,
                        limit=15
                    )
                )

            # Build search parameters
            search_params = {
                "collection_name": self.collection_name,
                "prefetch": prefetches,
                "query": models.FusionQuery(fusion=models.Fusion.RRF),
                "score_threshold": score_threshold,
                "limit": limit
            }

            # Add topic filter if specified
            if topic_filter:
                search_params["query_filter"] = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="topic",
                            match=models.MatchValue(value=topic_filter)
                        )
                    ]
                )
                logger.debug(f"Applied topic filter: {topic_filter}")

            results = self.client.query_points(**search_params)
            logger.debug(f"Query completed. Number of results: {len(results.points)}")

            return results

        except Exception as e:
            logger.error(f"Error in query_points: {str(e)}")
            raise VectorStoreError(f"Qdrant query failed: {str(e)}")

    def scroll(
        self,
        scroll_filter: Optional[models.Filter] = None,
        limit: int = 10
    ):
        """
        Scroll through collection points.

        Args:
            scroll_filter: Filter conditions
            limit: Maximum number of points to return

        Returns:
            Scroll results
        """
        try:
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit
            )
            return results

        except Exception as e:
            logger.error(f"Error in scroll: {str(e)}")
            raise VectorStoreError(f"Qdrant scroll failed: {str(e)}")

    def get_collections(self):
        """Get list of available collections."""
        return self.client.get_collections()

    def close(self):
        """Close Qdrant client connection."""
        if self.client:
            self.client.close()
            logger.info("Qdrant client closed")
