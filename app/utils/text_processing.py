"""
Text processing utilities.
"""
from transformers import AutoTokenizer
from app.core.config import settings
from typing import Optional

# Global tokenizer instance
_tokenizer: Optional[AutoTokenizer] = None


def get_tokenizer() -> AutoTokenizer:
    """Get or create tokenizer instance."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(settings.tokenizer_model)
    return _tokenizer


def get_token_length(text: str) -> int:
    """
    Calculate token length of text.

    Args:
        text: Input text

    Returns:
        Number of tokens
    """
    tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))


def sigmoid(x: float) -> float:
    """
    Compute sigmoid function for logit to probability conversion.

    Args:
        x: Input logit value

    Returns:
        Probability value between 0 and 1
    """
    import numpy as np
    x = np.clip(x, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-x))
