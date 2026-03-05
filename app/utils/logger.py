"""Logging configuration for the application."""

import logging
import sys
from typing import Dict, Optional

from app.config import get_settings

# ANSI color codes
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
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
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


def setup_logging() -> None:
    """Configure root logger with consistent formatting and colored level names."""
    settings = get_settings()

    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
    )

    # Create handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(fmt=log_format, datefmt="%Y-%m-%d %H:%M:%S"))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    # Remove any existing handlers to avoid duplication (force=True equivalent)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "qdrant_client", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)
