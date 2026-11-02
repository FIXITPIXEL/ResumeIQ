"""
logger.py – Centralized logging for the Resume Analyzer pipeline.
"""
from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config import LOG_FILE, LOG_LEVEL


def get_logger(name: str = "resume_iq") -> logging.Logger:
    """Return a named logger; safe to call multiple times."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Rotating file handler (5 MB × 3 backups)
    try:
        fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass  # read-only fs – skip file handler

    logger.propagate = False
    return logger


log = get_logger()
