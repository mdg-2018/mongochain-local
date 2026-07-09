"""Mongochain - MongoDB Atlas powered agent framework.

Uses Atlas Vector Search Automated Embedding (`autoEmbed`) for vector storage
and query — no client-side embedding management.
"""

from .agent import MongoAgent
from .config import AgentConfig, MemoryConfig
from .llm import LLMClient
from .memory import MemoryStore

__version__ = "0.2.0"

__all__ = [
    "MongoAgent",
    "AgentConfig",
    "MemoryConfig",
    "LLMClient",
    "MemoryStore",
]
