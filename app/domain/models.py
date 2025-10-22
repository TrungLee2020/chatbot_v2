"""
Pydantic models for request/response validation.
Consolidated from app_new.py and query_server_new.py
"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any


# ============================================================================
# Chat API Models
# ============================================================================

class ChatHistoryItem(BaseModel):
    """A single chat history item with human and chatbot messages."""
    chatbot: str
    human: str


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    user_id: str
    session_id: str
    transaction_id: str
    user_message: str
    chat_history: List[ChatHistoryItem]
    topic: Optional[str] = None


class RerankingScoreItem(BaseModel):
    """Reranking score details for a single document."""
    doc_id: str
    original_score: float
    final_logit: float
    final_probability: float
    source: str
    topic: str
    text_preview: str


class RetrievalMetrics(BaseModel):
    """Detailed metrics from the retrieval and reranking process."""
    reranking_scores: List[RerankingScoreItem]
    stats: Dict[str, Any]
    context: str
    structured_context: List[Dict[str, Any]] = []
    original_dense_texts: List[Dict[str, Any]]
    original_bm25_texts: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    bot_message: str
    structured_references: Optional[List[Dict[str, Any]]] = None
    doc_id: List[str]
    show_ref: int
    timestamp: float
    err_id: Optional[str]
    retrieval_metrics: str  # JSON string
    structured_context: Optional[List[Dict[str, Any]]] = None


# ============================================================================
# RAG API Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for /rag_query endpoint."""
    topic: Optional[str] = None
    chat_history: str
    user_input: str
    user_id: str
    session_id: str
    transaction_id: str


class BM25Details(BaseModel):
    """BM25 scoring details for a document."""
    term_frequencies: Dict[str, int]
    doc_length: int
    avg_doc_length: float
    score: float
    matched_terms: List[str]


class RetrievalDetails(BaseModel):
    """Detailed retrieval information for RAG response."""
    context: str
    structured_context: List[Dict[str, Any]] = []
    reranking_scores: List[Dict[str, Any]]
    stats: Dict[str, Any]
    original_dense_texts: List[Dict[str, Any]]
    original_bm25_texts: List[Dict[str, Any]]


class QueryResponse(BaseModel):
    """Response model for /rag_query endpoint."""
    response: str
    doc_metadata: List[Dict[str, Any]]
    retrieval_metrics: RetrievalDetails
    doc_ids: str
