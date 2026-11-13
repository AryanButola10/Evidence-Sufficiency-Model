"""
config/logging_config.py

Structured logging setup using Loguru.

Loguru replaces Python's standard logging with a much cleaner API.
One call to setup_logging() configures the entire application.

Usage:
    from config import get_logger
    logger = get_logger(__name__)
    logger.info("Evidence sufficiency score: {score}", score=0.73)
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import settings


def setup_logging() -> None:
    """
    Configure Loguru for the application.
    Called once at FastAPI startup.

    - Logs to stderr with colored output (for terminal)
    - Logs to a rotating file (for persistence)
    - Log level controlled by LOG_LEVEL environment variable
    """
    # Remove default Loguru handler (outputs plain text)
    logger.remove()

    # --- Console handler: colored, human-readable ---
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,   # Show full traceback on errors
        diagnose=True,    # Show variable values in tracebacks
    )

    # --- File handler: JSON-structured for analysis ---
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        rotation="10 MB",    # Create a new file when log reaches 10 MB
        retention="7 days",  # Keep logs for 7 days
        compression="zip",   # Compress old log files
        backtrace=True,
        diagnose=False,      # Disable variable values in file logs (privacy)
    )

    logger.info("Logging configured — level={level}, file={file}",
                level=settings.log_level, file=settings.log_file)


def get_logger(name: str):
    """
    Return a contextualized logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Module started")

    Args:
        name: Usually __name__ from the calling module.

    Returns:
        A Loguru logger bound with the module name.
    """
    return logger.bind(name=name)
