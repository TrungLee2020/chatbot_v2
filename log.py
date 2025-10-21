import structlog
import logging
from contextvars import ContextVar
import os
import sys
import time
import asyncio
from functools import wraps
from logging.handlers import TimedRotatingFileHandler
from transformers import AutoTokenizer
from dotenv import load_dotenv
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo

load_dotenv()

TOKENIZER_MODEL = os.environ.get("TOKENIZER_MODEL")

# Context variables for request-specific information
request_id = ContextVar("request_id", default="")
user_id = ContextVar("user_id", default="")
session_id = ContextVar("session_id", default="")

# Define Vietnam timezone
vietnam_tz = ZoneInfo("Asia/Ho_Chi_Minh")

def setup_logging(log_level: str = "INFO", log_file: str = None):
    # Set up standard logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=30,
            utc=False,  # Use local time
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=False),  # Use local time
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set up the formatter for both console and file handlers
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(ensure_ascii=False),
    )
    console_handler.setFormatter(formatter)
    if log_file:
        file_handler.setFormatter(formatter)

    return structlog.get_logger()

# Create a global logger instance
logger = setup_logging(
    log_level=os.environ.get("GENERAL_LOG_LEVEL", "INFO"),
    log_file=os.environ.get("GENERAL_LOG_FILE", "chatbot.log")
)

def set_request_context(req_id: str, u_id: str, sess_id: str):
    request_id.set(req_id)
    user_id.set(u_id)
    session_id.set(sess_id)

def clear_request_context():
    request_id.set("")
    user_id.set("")
    session_id.set("")

def log(level: str, event: str, **kwargs):
    log_method = getattr(logger, level.lower())
    log_method(event, 
               request_id=request_id.get(),
               user_id=user_id.get(),
               session_id=session_id.get(),
               timestamp=datetime.now(vietnam_tz).isoformat(),
               **kwargs)

def timed(func):
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

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
def get_token_length(text: str) -> int:
    return len(tokenizer.encode(text))