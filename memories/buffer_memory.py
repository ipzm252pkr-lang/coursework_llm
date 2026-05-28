from __future__ import annotations

import time
from typing import Optional

from .base import BaseMemory, Message

_ALLOWED_ROLES = {"user", "assistant"}


class BufferMemory(BaseMemory):
    # Sliding-window memory keeps the last N messages, drops older ones

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 2:
            raise ValueError(f"max_messages must be >= 2, got {max_messages}")

        self._history: list[Message] = []
        self._max_messages = max_messages
        self._total_added = 0
        self._overflow_count = 0
        self._turn = 0

    def add(self, role: str, content: str) -> None:
        self._validate(role, content)

        self._history.append({
            "role": role,
            "content": content.strip(),
            "turn": self._turn,
            "timestamp": time.time(),
        })
        self._total_added += 1
        self._turn += 1

        if len(self._history) > self._max_messages:
            excess = len(self._history) - self._max_messages
            self._history = self._history[excess:]
            self._overflow_count += 1

    def get_context(self, query: Optional[str] = None) -> list[dict]:
        # query is ignored — exists only to match the BaseMemory interface
        return [{"role": m["role"], "content": m["content"]} for m in self._history]

    def get_context_size(self) -> int:
        # ~4 chars per token is the standard approximation for Latin text
        total_chars = sum(len(m["content"]) for m in self._history)
        return max(1, total_chars // 4)

    def clear(self) -> None:
        self._history.clear()
        self._total_added = 0
        self._overflow_count = 0
        self._turn = 0

    def get_stats(self) -> dict:
        return {
            "strategy": "BufferMemory",
            "total_added": self._total_added,
            "current_size": len(self._history),
            "max_size": self._max_messages,
            "overflow_count": self._overflow_count,
            "context_tokens": self.get_context_size(),
            "oldest_turn": self._history[0]["turn"] if self._history else 0,
        }

    @staticmethod
    def _validate(role: str, content: str) -> None:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of {_ALLOWED_ROLES}")
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return f"BufferMemory(max={self._max_messages}, size={len(self._history)})"
