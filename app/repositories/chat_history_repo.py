"""
Repository for chat history database operations.
"""
from sqlalchemy.orm import Session
from typing import Optional
import logging
from app.core.exceptions import DatabaseError

logger = logging.getLogger(__name__)


class ChatHistoryRepository:
    """Repository for managing chat history in database."""

    def __init__(self, db_session: Session):
        """
        Initialize repository with database session.

        Args:
            db_session: SQLAlchemy database session
        """
        self.db = db_session

    def save_chat(
        self,
        user_id: str,
        session_id: str,
        question: str,
        answer: str
    ) -> Optional[int]:
        """
        Save a chat exchange to the database.

        Args:
            user_id: User identifier
            session_id: Session identifier
            question: User's question
            answer: Bot's answer

        Returns:
            Chat history record ID if successful, None otherwise

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            # Import here to avoid circular dependency
            from database import ChatHistory

            db_chat_history = ChatHistory(
                user_id=user_id,
                session_id=session_id,
                question=question,
                answer=answer
            )

            self.db.add(db_chat_history)
            self.db.commit()
            self.db.refresh(db_chat_history)

            logger.info(f"Saved chat history with id {db_chat_history.id}")
            return db_chat_history.id

        except Exception as e:
            logger.error(f"Failed to save chat history: {str(e)}")
            self.db.rollback()
            raise DatabaseError(f"Chat history save failed: {str(e)}")

    def get_history(
        self,
        session_id: str,
        limit: int = 10
    ) -> list:
        """
        Retrieve chat history for a session.

        Args:
            session_id: Session identifier
            limit: Maximum number of records to retrieve

        Returns:
            List of chat history records

        Raises:
            DatabaseError: If database operation fails
        """
        try:
            from database import ChatHistory

            history = (
                self.db.query(ChatHistory)
                .filter(ChatHistory.session_id == session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(limit)
                .all()
            )

            return list(reversed(history))

        except Exception as e:
            logger.error(f"Failed to retrieve chat history: {str(e)}")
            raise DatabaseError(f"Chat history retrieval failed: {str(e)}")
