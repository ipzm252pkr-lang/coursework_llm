from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from memories.base import BaseMemory

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a helpful conversational assistant with memory.
Use the provided conversation history to give consistent, contextually aware responses.
If the user has shared personal facts (name, job, preferences), reference them naturally.
Be concise. Answer in the same language the user writes in."""


@dataclass
class TurnLog:
    turn: int
    query: str
    answer: str
    latency_ms: float
    context_size: int
    context_tokens: int
    strategy: str
    memory_stats: dict
    prompt_chars: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "query": self.query,
            "answer": self.answer,
            "latency_ms": round(self.latency_ms, 2),
            "context_size": self.context_size,
            "context_tokens": self.context_tokens,
            "strategy": self.strategy,
            "memory_stats": self.memory_stats,
            "prompt_chars": self.prompt_chars,
            "timestamp": self.timestamp,
        }


class Chatbot:
    
    def __init__(
        self,
        memory: BaseMemory,
        model_name: str = "gemini-3.1-flash-lite",
        temperature: float = 0.1,
    ) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. "
                "Create a .env file with GEMINI_API_KEY=<your_key>."
            )

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature
        self._memory = memory
        self._strategy_name = type(memory).__name__
        self._turn = 0
        self._log: list[TurnLog] = []
        self._api_errors = 0

        logger.info("Chatbot ready: model=%s strategy=%s", model_name, self._strategy_name)

    def chat(self, user_message: str) -> str:
        if not user_message or not user_message.strip():
            raise ValueError("user_message cannot be empty")

        msg = user_message.strip()

        # Context is fetched BEFORE storing the current message so that
        # VectorMemory searches only the previous conversation, not the
        # query itself (which would always score as an exact match).
        context = self._memory.get_context(query=msg)
        self._memory.add("user", msg)

        prompt = self._build_prompt(context, msg)

        t0 = time.perf_counter()
        answer = self._call_llm(prompt)
        latency = (time.perf_counter() - t0) * 1000

        self._memory.add("assistant", answer)
        self._log.append(TurnLog(
            turn=self._turn,
            query=msg,
            answer=answer,
            latency_ms=latency,
            context_size=len(context),
            context_tokens=self._memory.get_context_size(),
            strategy=self._strategy_name,
            memory_stats=self._memory.get_stats(),
            prompt_chars=len(prompt),
        ))
        self._turn += 1
        return answer

    def reset(self) -> None:
        self._memory.clear()
        self._log.clear()
        self._turn = 0
        self._api_errors = 0

    def get_response_log(self) -> list[dict]:
        return [entry.to_dict() for entry in self._log]

    def get_summary_stats(self) -> dict:
        if not self._log:
            return {"total_turns": 0, "strategy": self._strategy_name}

        latencies = [e.latency_ms for e in self._log]
        tokens = [e.context_tokens for e in self._log]

        return {
            "strategy": self._strategy_name,
            "total_turns": self._turn,
            "mean_latency_ms": round(sum(latencies) / len(latencies), 2),
            "max_latency_ms": round(max(latencies), 2),
            "min_latency_ms": round(min(latencies), 2),
            "mean_context_tokens": round(sum(tokens) / len(tokens), 1),
            "api_errors": self._api_errors,
        }

    def _build_prompt(self, context: list[dict], current_query: str) -> str:
        history = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in context
        ) or "(no history)"

        return (
            f"{_SYSTEM_PROMPT}\n\n"
            f"--- Conversation History ---\n{history}\n"
            f"--- End of History ---\n\n"
            f"User: {current_query}\n"
            f"Assistant:"
        )

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self._temperature,
                    top_p=0.9,
                    max_output_tokens=512,
                ),
            )
            return response.text.strip()

        except Exception as exc:
            self._api_errors += 1
            err = str(exc)

            if "ResourceExhausted" in type(exc).__name__ or "429" in err:
                logger.warning("Rate limit hit")
                return "[ERROR: rate limit exceeded — wait 60s and retry]"

            if "ServiceUnavailable" in type(exc).__name__ or "503" in err:
                logger.warning("Service unavailable")
                return "[ERROR: service temporarily unavailable]"

            logger.error("API error (%s): %s", type(exc).__name__, err)
            return f"[ERROR: {type(exc).__name__}]"

    def __repr__(self) -> str:
        return f"Chatbot(model={self._model_name}, strategy={self._strategy_name}, turns={self._turn})"
