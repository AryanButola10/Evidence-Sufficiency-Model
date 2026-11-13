"""
config/__init__.py

Convenience re-exports so other modules can do:
    from config import settings, get_logger
"""
from config.settings import settings
from config.logging_config import get_logger

__all__ = ["settings", "get_logger"]
