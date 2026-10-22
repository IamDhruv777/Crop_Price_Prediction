"""
Centralised logging configuration.
"""
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(
    name: str = "crop_price",
    level: str = None,
    log_file: str = None,
) -> logging.Logger:
    """Configure and return a named logger.

    Parameters
    ----------
    name : str
        Logger name (typically the module __name__).
    level : str
        Override log level (default: from LOG_LEVEL env variable or INFO).
    log_file : str
        Override log file path.

    Returns
    -------
    logging.Logger
    """
    log_level_str = level or os.getenv("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    log_file_path = log_file or os.getenv("LOG_FILE", "./logs/app.log")
    log_dir = Path(log_file_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        # Already configured
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler
    fh = logging.handlers.RotatingFileHandler(
        log_file_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(log_level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get (or create) a logger for the given module name."""
    return setup_logging(name=name)
