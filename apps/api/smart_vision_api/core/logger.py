"""Logging Configuration

Provides centralized logging setup with automatic log rotation.
Features:
- Time-based rotation (daily)
- Size-based rotation (10MB)
- Log backup management
- Console and file outputs
- Consistent formatting
"""

import logging
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

from .config import settings


def get_logger(name: str = "smart_vision") -> logging.Logger:
    """Configure and return a logger instance with rotation support.

    Creates a logger with console and rotating file handlers.
    Implements both time-based and size-based log rotation.

    Args:
        name: Logger identifier, defaults to "smart_vision"

    Returns:
        logging.Logger: Configured logger instance
    """
    # Initialize logger
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)
    # Prevent duplicate emission via root/uvicorn loggers.
    logger.propagate = False

    # Configure formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set up console output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.LOG_LEVEL)

    # Set up rotating file handler (size-based)
    size_handler = RotatingFileHandler(
        settings.log_dir_path / "smart-vision.log",
        maxBytes=10_000_000,  # 10MB
        backupCount=5,  # Keep 5 backup files
    )
    size_handler.setFormatter(formatter)
    size_handler.setLevel(settings.LOG_LEVEL)

    # Set up timed rotating file handler (daily)
    time_handler = TimedRotatingFileHandler(
        settings.log_dir_path / "smart-vision_daily.log",
        when="midnight",  # Rotate at midnight
        interval=1,  # Daily rotation
        backupCount=30,  # Keep 30 days of logs
    )
    time_handler.setFormatter(formatter)
    time_handler.setLevel(settings.LOG_LEVEL)

    # Add handlers (with deduplication)
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(size_handler)
        logger.addHandler(time_handler)

    return logger
