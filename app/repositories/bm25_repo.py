"""
Repository for BM25 retrieval operations.
"""
from llama_index.retrievers.bm25 import BM25Retriever
from typing import Optional, List
import os
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


class BM25Repository:
    """Repository for BM25-based text retrieval."""

    def __init__(self):
        """Initialize BM25 retriever from persisted data."""
        self.retriever = self._load_retriever()

    def _load_retriever(self) -> Optional[BM25Retriever]:
        """
        Load BM25 retriever from disk.

        Returns:
            BM25Retriever instance or None if not found
        """
        unified_bm25_path = os.path.join(settings.bm25_persist_path, "unified")

        if not os.path.exists(unified_bm25_path):
            logger.warning(
                f"Unified BM25 retriever directory not found at {unified_bm25_path}"
            )
            return None

        try:
            retriever = BM25Retriever.from_persist_dir(unified_bm25_path)
            retriever.similarity_top_k = settings.bm25_top_k

            logger.info("Unified BM25 retriever initialized successfully")
            return retriever

        except Exception as e:
            logger.error(f"Failed to load BM25 retriever: {str(e)}")
            return None

    def search(
        self,
        query: str,
        topic_filter: Optional[str] = None
    ) -> List:
        """
        Search using BM25 algorithm.

        Args:
            query: Query text
            topic_filter: Optional topic filter (applied post-retrieval)

        Returns:
            List of search results (NodeWithScore objects)
        """
        if not self.retriever:
            logger.warning("BM25 retriever not available")
            return []

        try:
            # Retrieve results
            results = self.retriever.retrieve(query)
            logger.info(f"BM25 retrieval found {len(results)} results")

            # Apply topic filter if specified
            if topic_filter:
                from app.core.constants import VALID_TOPICS

                if topic_filter in VALID_TOPICS:
                    results = [
                        r for r in results
                        if r.metadata.get('topic') == topic_filter
                    ]
                    logger.info(
                        f"Filtered BM25 results by topic {topic_filter}: "
                        f"{len(results)} results"
                    )

            return results

        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []

    def is_available(self) -> bool:
        """Check if BM25 retriever is available."""
        return self.retriever is not None
