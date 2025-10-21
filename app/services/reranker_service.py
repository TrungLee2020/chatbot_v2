"""
Reranker Service - handles document reranking logic.
Extracted from query_server_new.py XinferenceReranker class.
"""
from typing import List, Tuple
import logging
from llama_index.core.schema import NodeWithScore
from app.adapters.xinference_adapter import XinferenceAdapter
from app.core.config import settings
from app.utils.text_processing import sigmoid, get_token_length
from app.utils.logging import log

logger = logging.getLogger(__name__)


class RerankerService:
    """Service for reranking search results."""

    def __init__(self, xinference_adapter: XinferenceAdapter):
        """Initialize with Xinference adapter."""
        self.reranker = xinference_adapter

    def rerank_results(
        self,
        results: List[NodeWithScore],
        query: str,
        max_context_length: int,
        max_results: int,
        threshold: float
    ) -> List[Tuple[NodeWithScore, float, float]]:
        """
        Rerank results using Xinference reranker.

        Returns:
            List of (result, logit_score, probability) tuples
        """
        if not results:
            log("warning", "Reranking skipped: empty input")
            return []

        # Validate and clean results
        valid_results = [
            r for r in results
            if hasattr(r, 'node') and hasattr(r.node, 'text') and r.node.text.strip()
        ]

        if not valid_results:
            log("warning", "No valid text candidates after cleaning")
            return []

        # Create corpus
        corpus = [r.node.text for r in valid_results]

        try:
            # Call reranker
            rerank_result = self.reranker.rerank(corpus, query)

            if not rerank_result or "results" not in rerank_result:
                return []

            # Process scores
            results_with_scores = []
            for item in rerank_result["results"]:
                idx = item.get("index")
                if idx is None or idx < 0 or idx >= len(valid_results):
                    continue

                score = item.get("relevance_score", 0)
                results_with_scores.append((
                    valid_results[idx],
                    score,
                    sigmoid(score)
                ))

            # Filter by threshold
            filtered = [r for r in results_with_scores if r[1] >= threshold]
            filtered.sort(key=lambda x: x[1], reverse=True)

            # Trim by token limit
            top_results = []
            total_tokens = 0

            for result, score, prob in filtered:
                tokens = get_token_length(result.node.text)
                if total_tokens + tokens > max_context_length or len(top_results) >= max_results:
                    break
                top_results.append((result, score, prob))
                total_tokens += tokens

            log("info", f"Reranked {len(results)} -> {len(top_results)} results")
            return top_results

        except Exception as e:
            log("error", f"Reranking failed: {str(e)}")
            return []
