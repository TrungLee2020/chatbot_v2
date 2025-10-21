"""
Logging utilities using structlog.
Migrated from log.py with improvements.
"""
import structlog
import logging
from contextvars import ContextVar
import sys
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from zoneinfo import ZoneInfo

# Context variables for request-specific information
request_id = ContextVar("request_id", default="")
user_id = ContextVar("user_id", default="")
session_id = ContextVar("session_id", default="")

# Define Vietnam timezone
vietnam_tz = ZoneInfo("Asia/Ho_Chi_Minh")


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output

    Returns:
        Configured logger instance
    """
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
            utc=False,
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
            structlog.processors.TimeStamper(fmt="iso", utc=False),
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


def set_request_context(req_id: str, u_id: str, sess_id: str):
    """Set request context variables for logging."""
    request_id.set(req_id)
    user_id.set(u_id)
    session_id.set(sess_id)


def clear_request_context():
    """Clear request context variables."""
    request_id.set("")
    user_id.set("")
    session_id.set("")


def log(level: str, event: str, **kwargs):
    """
    Log an event with structured context.

    Args:
        level: Log level (debug, info, warning, error)
        event: Event description
        **kwargs: Additional structured data
    """
    logger = structlog.get_logger()
    log_method = getattr(logger, level.lower())
    log_method(
        event,
        request_id=request_id.get(),
        user_id=user_id.get(),
        session_id=session_id.get(),
        timestamp=datetime.now(vietnam_tz).isoformat(),
        **kwargs
    )
