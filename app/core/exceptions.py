"""
Custom exceptions for the application.
"""


class ChatbotException(Exception):
    """Base exception for all chatbot errors."""
    pass


class VLLMConnectionError(ChatbotException):
    """Raised when vLLM server is not available."""
    pass


class RAGServiceError(ChatbotException):
    """Raised when RAG service fails."""
    pass


class RerankerError(ChatbotException):
    """Raised when reranker fails."""
    pass


class VectorStoreError(ChatbotException):
    """Raised when vector store operations fail."""
    pass


class InvalidTopicError(ChatbotException):
    """Raised when an invalid topic is provided."""
    pass


class DatabaseError(ChatbotException):
    """Raised when database operations fail."""
    pass
