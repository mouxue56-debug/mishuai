"""Logging setup using loguru."""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config_loader import BASE_DIR, get_main_config


_initialized = False


def setup_logger():
    """Initialize the logger based on config settings."""
    global _initialized
    if _initialized:
        return logger

    try:
        config = get_main_config()
        log_config = config.get("logging", {})
    except FileNotFoundError:
        log_config = {}

    level = log_config.get("level", "INFO")
    log_file = log_config.get("file", "logs/secretary.log")
    rotation = log_config.get("rotation", "10 MB")
    retention = log_config.get("retention", "7 days")

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # File handler
    log_path = BASE_DIR / log_file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    _initialized = True
    logger.info("Logger initialized")
    return logger


def get_logger(name: str = None):
    """Get a configured logger instance."""
    setup_logger()
    if name:
        return logger.bind(name=name)
    return logger
