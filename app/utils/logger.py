"""Logging configuration for the application with RFC 5424 compliance.

RFC 5424 (The Syslog Protocol) defines a structured log format with:
- Timestamp (ISO 8601 / RFC 3339)
- Hostname
- App-name
- ProcID
- MsgID
- Structured Data
- Message

This implementation uses JSON format for machine-parseable logs that comply
with RFC 5424 structured data requirements.
"""

from datetime import UTC, datetime
import logging
import os
import socket
import sys
from typing import Any

from pythonjsonlogger.jsonlogger import JsonFormatter as BaseJsonFormatter

from app.config import get_settings

# ANSI color codes for text mode
COLORS = {
    logging.DEBUG: "\033[36m",  # Cyan
    logging.INFO: "\033[32m",  # Green
    logging.WARNING: "\033[33m",  # Yellow
    logging.ERROR: "\033[31m",  # Red
    logging.CRITICAL: "\033[35m\033[1m",  # Magenta + Bold
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI colors to the level name."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        use_colors: bool = True,
    ):
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if not self.use_colors:
            return super().format(record)

        # Save original levelname
        original_levelname = record.levelname
        color = COLORS.get(record.levelno, RESET)
        record.levelname = f"{color}{original_levelname}{RESET}"

        try:
            return super().format(record)
        finally:
            # Restore original levelname (in case the record is reused)
            record.levelname = original_levelname


class RFC5424JsonFormatter(BaseJsonFormatter):
    """JSON formatter that produces RFC 5424 compliant structured log entries.

    Maps log fields to RFC 5424 syslog message format:
    - TIMESTAMP: ISO 8601 with timezone
    - HOSTNAME: machine hostname
    - APP-NAME: application name from settings
    - PROCID: process ID
    - MSGID: logger name or message ID
    - STRUCTURED-DATA: additional contextual fields as JSON
    - MSG: the actual log message
    """

    def __init__(
        self,
        *args: Any,
        app_name: str = "app",
        hostname: str | None = None,
        procid: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.app_name = app_name
        self.hostname = hostname or socket.gethostname()
        self.procid = procid or os.getpid()

    def add_fields(
        self,
        log_record: dict[str, Any],
        record: logging.LogRecord,
        message_dict: dict[str, Any],
    ) -> None:
        """Add RFC 5424 fields to the log record."""
        super().add_fields(log_record, record, message_dict)

        # RFC 5424 required fields
        # Use RFC 3339 / ISO 8601 timestamp
        log_record["timestamp"] = datetime.fromtimestamp(
            record.created, tz=UTC
        ).isoformat()
        log_record["hostname"] = self.hostname
        log_record["app_name"] = self.app_name
        log_record["procid"] = self.procid
        log_record["msgid"] = record.name  # Logger name as message ID
        log_record["severity"] = record.levelname.lower()
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        log_record["thread"] = record.threadName
        log_record["thread_id"] = record.thread
        log_record["process_name"] = record.processName

        # Remove duplicate/redundant fields
        for field in ("asctime", "created", "relativeCreated", "msecs"):
            log_record.pop(field, None)

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as RFC 5424 JSON."""
        # Handle exceptions properly
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)

        # Add exception text to the message if present
        if record.exc_text:
            record.message = f"{record.getMessage()}\n{record.exc_text}"

        return super().format(record)


def setup_logging() -> None:
    """Configure root logger with RFC 5424 structured logging.

    Supports two modes:
    - JSON mode (default): RFC 5424 compliant structured JSON logs
    - Text mode: Human-readable colored text logs (for development)
    """
    settings = get_settings()

    # Determine log format from settings
    use_json = settings.log_format.lower() == "json"

    # Create handler
    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        # RFC 5424 JSON formatter for production
        formatter = RFC5424JsonFormatter(
            app_name=settings.app_name,
            hostname=socket.gethostname(),
            procid=os.getpid(),
            json_default=str,  # Serialize non-serializable objects
            json_ensure_ascii=False,
            json_indent=None,  # Compact JSON for production
        )
    else:
        # Colored text formatter for development
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
        formatter = ColoredFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S")

    handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # Remove any existing handlers to avoid duplication
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
