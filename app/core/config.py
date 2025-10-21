"""
Centralized configuration management using Pydantic Settings.
All environment variables are loaded and validated here.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === API Settings ===
    app_host: str = "0.0.0.0"
    app_port: int = 6868
    log_level: str = "info"

    # === vLLM Settings ===
    vllm_server_url: str
    vllm_model_name: str = ""
    vllm_timeout: float = 60.0
    vllm_api_key: str = "dummy-key"

    # === Chatbot LLM Settings ===
    chatbot_temperature: float = 0.2
    chatbot_top_p: float = 0.95
    chatbot_max_tokens: int = 1024
    chatbot_repetition_penalty: float = 1.05

    # === RAG API Settings ===
    rag_api_url: str = "http://127.0.0.1:8338/rag_query"

    # === Chat History Settings ===
    chat_history_length: int = 3

    # === Embedding Settings ===
    embed_model_path: str = "./models/bge_m3"
    embed_model_batch_size: int = 8
    embed_model_max_length: int = 1024
    embed_chunk_size: int = 1024

    # === Qdrant Settings ===
    qdrant_host: str
    qdrant_port: int = 6333
    qdrant_collection_name: str = "unified_documents"
    qdrant_dense_top_k: int = 50

    # === BM25 Settings ===
    bm25_persist_path: str
    bm25_top_k: int = 30

    # === Reranker Settings ===
    rerank_model_name: str
    reranker_threshold: float = 0.3
    xinference_endpoint: str

    # === Context Settings ===
    max_context_length: int = 2048
    parent_text_max_length: int = 1024
    max_dense_results: int = 30
    max_bm25_results: int = 15
    max_final_context_tokens: int = 2048

    # === Tokenizer Settings ===
    tokenizer_model: str

    # === Logging Settings ===
    general_log_level: str = "INFO"
    general_log_file: str = "chatbot.log"

    # === Evaluation Settings ===
    metadata_eval_temperature: float = 0.1
    metadata_eval_max_tokens: int = 50
    evaluation_chat_response_temperature: float = 0.1
    evaluation_chat_response_max_tokens: int = 50

    # === Routing Settings ===
    route_threshold: float = 0.3

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
