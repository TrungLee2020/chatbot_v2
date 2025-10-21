"""
FastAPI dependencies for dependency injection.
"""
from functools import lru_cache
from sqlalchemy.orm import Session

# Adapters
from app.adapters.vllm_adapter import VLLMAdapter
from app.adapters.embedding_adapter import EmbeddingAdapter
from app.adapters.qdrant_adapter import QdrantAdapter
from app.adapters.xinference_adapter import XinferenceAdapter

# Repositories
from app.repositories.vector_store_repo import VectorStoreRepository
from app.repositories.bm25_repo import BM25Repository
from app.repositories.chat_history_repo import ChatHistoryRepository

# Services
from app.services.llm_service import LLMService
from app.services.reranker_service import RerankerService
from app.services.rag_service import RAGService
from app.services.chat_service import ChatService

# Database
from database import SessionLocal


# === Database ===

def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# === Adapters (Singleton) ===

@lru_cache()
def get_vllm_adapter() -> VLLMAdapter:
    """Get VLLMAdapter singleton."""
    return VLLMAdapter()


@lru_cache()
def get_embedding_adapter() -> EmbeddingAdapter:
    """Get EmbeddingAdapter singleton."""
    return EmbeddingAdapter()


@lru_cache()
def get_qdrant_adapter() -> QdrantAdapter:
    """Get QdrantAdapter singleton."""
    return QdrantAdapter()


@lru_cache()
def get_xinference_adapter() -> XinferenceAdapter:
    """Get XinferenceAdapter singleton."""
    return XinferenceAdapter()


# === Repositories ===

@lru_cache()
def get_vector_store_repo() -> VectorStoreRepository:
    """Get VectorStoreRepository singleton."""
    return VectorStoreRepository(
        qdrant_adapter=get_qdrant_adapter(),
        embedding_adapter=get_embedding_adapter()
    )


@lru_cache()
def get_bm25_repo() -> BM25Repository:
    """Get BM25Repository singleton."""
    return BM25Repository()


def get_chat_history_repo(db: Session) -> ChatHistoryRepository:
    """Get ChatHistoryRepository (per-request)."""
    return ChatHistoryRepository(db)


# === Services ===

@lru_cache()
def get_llm_service() -> LLMService:
    """Get LLMService singleton."""
    return LLMService(vllm_adapter=get_vllm_adapter())


@lru_cache()
def get_reranker_service() -> RerankerService:
    """Get RerankerService singleton."""
    return RerankerService(xinference_adapter=get_xinference_adapter())


@lru_cache()
def get_rag_service() -> RAGService:
    """Get RAGService singleton."""
    return RAGService(
        vector_store_repo=get_vector_store_repo(),
        bm25_repo=get_bm25_repo(),
        reranker_service=get_reranker_service()
    )


@lru_cache()
def get_chat_service() -> ChatService:
    """Get ChatService singleton."""
    # Note: ChatHistoryRepo is per-request, passed via endpoint
    return ChatService(
        llm_service=get_llm_service(),
        rag_service=get_rag_service(),
        chat_repo=None  # Will be injected in endpoint
    )
