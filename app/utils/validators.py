"""
Validation utilities.
"""
from typing import Optional
from app.core.constants import VALID_TOPICS, LEGACY_TOPIC_MAPPING
from app.utils.logging import log


def normalize_topic(topic: str) -> Optional[str]:
    """
    Normalize topic from legacy format to new unified format.

    Args:
        topic: Topic string to normalize

    Returns:
        Normalized topic code or None if invalid
    """
    if not topic:
        return None

    if topic in VALID_TOPICS:
        return topic

    if topic in LEGACY_TOPIC_MAPPING:
        return LEGACY_TOPIC_MAPPING[topic]

    log("warning", f"Unknown topic '{topic}', searching across all topics")
    return None


def is_valid_text(text) -> bool:
    """
    Check if text is valid (not N/A, nan, NaT, or empty).

    Args:
        text: Text to validate

    Returns:
        True if text is valid, False otherwise
    """
    return text not in ["N/A", "nan", "NaT", ""] and isinstance(text, str)
