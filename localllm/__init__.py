"""
LocalLLM - Your Personal AI Assistant

A lightweight, downloadable Large Language Model that runs entirely on your local machine.
"""

__version__ = "1.0.0"
__author__ = "LocalLLM Team"
__email__ = "support@localllm.ai"

from .model import LocalLLM
from .chat import ChatInterface
from .api import APIServer

__all__ = ["LocalLLM", "ChatInterface", "APIServer"]
