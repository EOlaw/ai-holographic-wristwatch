"""
Logger alias module for AI Holographic Wristwatch System.

Re-exports get_logger from logging_utils so that
from src.core.utils.logger import get_logger works across the project.
"""

import logging
from typing import Optional

try:
    from .logging_utils import get_logger  # noqa: F401
except Exception:
    def get_logger(name: str, context=None):
        return logging.getLogger(name)

__all__ = ["get_logger"]
