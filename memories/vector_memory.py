from __future__ import annotations

import logging
import time
from typing import Optional

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseMemory, Message

logger = logging.getLogger(__name__)

_ALLOWED_ROLES = {"user", "assistant"}


class VectorMemory(BaseMemory):
   
    # Semantic search memory every message is embedded and stored in ChromaDB.

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "chat_memory",
        recent_window: int = 3,
        top_k: int = 4,
    ) -> None:
        if recent_window < 1:
            raise ValueError(f"recent_window must be >= 1, got {recent_window}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        logger.info("Loading embedding model: %s", model_name)
        self._encoder = SentenceTransformer(model_name)
        self._model_name = model_name

        self._chroma = chromadb.Client()
        self._collection_name = collection_name
        self._collection = self._chroma.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._recent_window = recent_window
        self._top_k = top_k

        self._recent: list[Message] = []
        self._msg_count = 0
        self._total_added = 0
        self._similarities: list[float] = []  

    def add(self, role: str, content: str) -> None:
        self._validate(role, content)
        content = content.strip()

        embedding = self._encoder.encode(content).tolist()

        self._collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[{"role": role, "turn": self._msg_count, "timestamp": time.time()}],
            ids=[f"msg_{self._msg_count}"],
        )

        self._recent.append({"role": role, "content": content, "turn": self._msg_count, "timestamp": time.time()})
        if len(self._recent) > self._recent_window:
            self._recent = self._recent[-self._recent_window:]

        self._msg_count += 1
        self._total_added += 1

    def get_context(self, query: Optional[str] = None) -> list[dict]:
        if self._msg_count == 0:
            return []

        if not query:
            return [{"role": m["role"], "content": m["content"]} for m in self._recent]

        query_emb = self._encoder.encode(query).tolist()
        n = min(self._top_k, self._msg_count)

        results = self._collection.query(
            query_embeddings=[query_emb],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        distances = results["distances"][0]
        self._similarities.extend(1.0 - d for d in distances)

        retrieved = [
            {"role": meta["role"], "content": doc, "turn": meta["turn"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]

        recent_with_turn = [
            {"role": m["role"], "content": m["content"], "turn": m["turn"]}
            for m in self._recent
        ]

        seen: set[int] = set()
        merged: list[dict] = []
        for msg in recent_with_turn + retrieved:
            if msg["turn"] not in seen:
                seen.add(msg["turn"])
                merged.append(msg)

        merged.sort(key=lambda m: m["turn"])
        return [{"role": m["role"], "content": m["content"]} for m in merged]

    def get_context_size(self) -> int:
        total_chars = sum(len(m["content"]) for m in self._recent)
        scale = self._top_k / max(self._recent_window, 1)
        return max(1, int(total_chars * scale) // 4)

    def clear(self) -> None:
        self._chroma.delete_collection(self._collection_name)
        self._collection = self._chroma.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._recent.clear()
        self._similarities.clear()
        self._msg_count = 0
        self._total_added = 0

    def get_stats(self) -> dict:
        return {
            "strategy": "VectorMemory",
            "total_added": self._total_added,
            "collection_size": self._collection.count(),
            "recent_size": len(self._recent),
            "context_tokens": self.get_context_size(),
            "retrieval": self._retrieval_stats(),
        }

    def _retrieval_stats(self) -> dict:
        if not self._similarities:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        arr = np.array(self._similarities)
        return {
            "avg": round(float(arr.mean()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "count": len(self._similarities),
        }

    @staticmethod
    def _validate(role: str, content: str) -> None:
        if role not in _ALLOWED_ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of {_ALLOWED_ROLES}")
        if not content or not content.strip():
            raise ValueError("Message content cannot be empty")

    def __len__(self) -> int:
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"VectorMemory(model={self._model_name}, "
            f"size={self._collection.count()}, top_k={self._top_k})"
        )
