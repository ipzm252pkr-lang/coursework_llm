from __future__ import annotations

import logging
import time
from typing import Optional

from google import genai
from google.genai import types

from .base import BaseMemory, Message

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = {"user", "assistant"}

_COMPRESS_PROMPT = """\
Previous summary: {previous_summary}

New messages:
{messages_text}

Write a concise updated summary (max 100 words).
Preserve all names, personal facts (job, city, pets, preferences), and key topics.
Return only the summary text."""


class SummaryMemory(BaseMemory):
    #LLM-compression memory periodically compresses old messages into a running summary, keeping the active buffer small.

    def __init__(
        self,
        max_messages: int = 10,
        llm: Optional[genai.Client] = None,
        summarize_threshold: int = 5,
        model_name: str = "gemini-3.1-flash-lite",
    ) -> None:
        if max_messages < 2:
            raise ValueError(f"max_messages must be >= 2, got {max_messages}")
        if summarize_threshold >= max_messages:
            raise ValueError(
                f"summarize_threshold ({summarize_threshold}) must be "
                f"less than max_messages ({max_messages})"
            )

        self._history: list[Message] = []
        self._summary = ""
        self._llm = llm
        self._model_name = model_name
        self._max_messages = max_messages
        self._summarize_threshold = summarize_threshold
        self._summary_calls = 0
        self._total_added = 0
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

        if len(self._history) >= self._summarize_threshold:
            self._compress()

    def get_context(self, query: Optional[str] = None) -> list[dict]:
        history = [{"role": m["role"], "content": m["content"]} for m in self._history]

        if not self._summary:
            return history

        return [{"role": "user", "content": f"[Conversation so far]: {self._summary}"}] + history

    def get_context_size(self) -> int:
        summary_chars = len(self._summary)
        history_chars = sum(len(m["content"]) for m in self._history)
        return max(1, (summary_chars + history_chars) // 4)

    def clear(self) -> None:
        self._history.clear()
        self._summary = ""
        self._total_added = 0
        self._summary_calls = 0
        self._turn = 0

    def get_stats(self) -> dict:
        return {
            "strategy": "SummaryMemory",
            "total_added": self._total_added,
            "history_size": len(self._history),
            "summary_chars": len(self._summary),
            "summary_calls": self._summary_calls,
            "context_tokens": self.get_context_size(),
        }

    def _compress(self) -> None:
        if self._llm is None:
            logger.warning("SummaryMemory: no LLM client, falling back to truncation")
            excess = len(self._history) - self._max_messages
            if excess > 0:
                self._history = self._history[excess:]
            return

        batch = self._history[:self._summarize_threshold]
        messages_text = "\n".join(f"{m['role']}: {m['content']}" for m in batch)
        prompt = _COMPRESS_PROMPT.format(
            previous_summary=self._summary or "(none)",
            messages_text=messages_text,
        )

        try:
            response = self._llm.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=200),
            )
            new_summary = response.text.strip()
            if new_summary:
                self._summary = new_summary
                self._summary_calls += 1
                logger.debug("Compressed %d messages → %d chars", self._summarize_threshold, len(self._summary))
            else:
                logger.warning("SummaryMemory: LLM returned an empty summary")

        except Exception as exc:
            logger.error("Compression failed (%s): %s", type(exc).__name__, exc)
            excess = len(self._history) - self._max_messages
            if excess > 0:
                self._history = self._history[excess:]
            return

        self._history = self._history[self._summarize_threshold:]

    @staticmethod
    def _validate(role: str, content: str) -> None:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of {_ALLOWED_ROLES}")
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")

    def __len__(self) -> int:
        return len(self._history)

    def __repr__(self) -> str:
        return (
            f"SummaryMemory(max={self._max_messages}, "
            f"buf={len(self._history)}, calls={self._summary_calls})"
        )
