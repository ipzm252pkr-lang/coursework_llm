from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TypedDict


class Message(TypedDict):
    role: str
    content: str
    turn: int
    timestamp: float


class MemoryStats(TypedDict, total=False):
    strategy: str
    total_messages_added: int
    current_size: int
    context_size_tokens: int


class BaseMemory(ABC):
    """Strategy interface for conversation memory backends."""

    @abstractmethod
    def add(self, role: str, content: str) -> None:
        """Store a message. role must be 'user' or 'assistant'."""

    @abstractmethod
    def get_context(self, query: Optional[str] = None) -> list[dict]:
        """Return messages to inject into the LLM prompt.

        Args:
            query: Used by VectorMemory for semantic retrieval. Ignored by others.

        Returns:
            List of {"role": str, "content": str} dicts, ordered chronologically.
        """

    @abstractmethod
    def get_context_size(self) -> int:
        """Estimate context size in tokens (heuristic: 1.3 tokens/word)."""

    @abstractmethod
    def clear(self) -> None:
        """Reset all stored state."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Return a snapshot of internal metrics."""
