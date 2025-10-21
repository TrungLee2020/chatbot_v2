"""
Timing decorator for performance monitoring.
"""
import time
import asyncio
from functools import wraps
from app.utils.logging import log


def timed(func):
    """
    Decorator to measure function execution time.
    Works with both sync and async functions.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            log("info", f"{func.__name__} completed", duration=duration)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start
            log("info", f"{func.__name__} completed", duration=duration)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
