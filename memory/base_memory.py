"""Persistent memory abstractions used by the agentic pipeline."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional


class BaseMemory(ABC):
    """A lightweight JSON-backed memory store with append-only semantics."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        self.storage_path = storage_path or Path("memory_store.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[dict[str, Any]] = self._load()

    def _load(self) -> List[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            with self.storage_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _persist(self) -> None:
        with self.storage_path.open("w", encoding="utf-8") as handle:
            json.dump(self._entries, handle, indent=2, default=str)

    def write(self, entry: dict[str, Any]) -> dict[str, Any]:
        record = dict(entry)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self._entries.append(record)
        self._persist()
        return record

    def read_all(self) -> List[dict[str, Any]]:
        return list(self._entries)

    @abstractmethod
    def search(self, query: str) -> List[dict[str, Any]]:
        """Return entries relevant to a query."""


class WorkingMemory(BaseMemory):
    """Short-lived context for the current task or session."""

    def search(self, query: str) -> List[dict[str, Any]]:
        query_lower = query.lower()
        return [e for e in self._entries if query_lower in json.dumps(e).lower()]


class EpisodicMemory(BaseMemory):
    """History of prior agent actions and outcomes, ranked by keyword hits and recency."""

    def search(self, query: str, top_k: int = 10) -> List[dict[str, Any]]:
        if not self._entries:
            return []

        query_terms = set(re.sub(r"[^\w\s]", "", query.lower()).split())
        now = datetime.now(timezone.utc)
        scored: List[tuple[float, dict[str, Any]]] = []

        for entry in self._entries:
            text = json.dumps(entry).lower()
            keyword_hits = sum(1 for term in query_terms if term in text)
            if keyword_hits == 0:
                continue

            try:
                ts_str = entry.get("timestamp", "")
                ts = datetime.fromisoformat(ts_str) if ts_str else now
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                age_hours = (now - ts).total_seconds() / 3600
                recency = 1.0 / (1.0 + age_hours / 24)
            except Exception:
                recency = 0.5

            score = keyword_hits * 0.7 + recency * 0.3
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]


class SemanticMemory(BaseMemory):
    """Persistent insights distilled from prior runs, retrieved by embedding similarity."""

    _shared_model: Any = None

    @classmethod
    def _get_model(cls) -> Any:
        if cls._shared_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                cls._shared_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
            except Exception:
                cls._shared_model = False  # tried and failed — don't retry
        return cls._shared_model if cls._shared_model is not False else None

    def write(self, entry: dict[str, Any]) -> dict[str, Any]:
        record = dict(entry)
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        model = self._get_model()
        if model is not None:
            text = " ".join(str(v) for v in entry.values() if isinstance(v, str))[:512]
            try:
                embedding = model.encode([text], normalize_embeddings=True)[0]
                record["_embedding"] = embedding.tolist()
            except Exception:
                pass

        self._entries.append(record)
        self._persist()
        return record

    def search(self, query: str, top_k: int = 5) -> List[dict[str, Any]]:
        if not self._entries:
            return []

        model = self._get_model()
        embedded = [(i, e) for i, e in enumerate(self._entries) if "_embedding" in e]

        if model is None or not embedded:
            query_terms = set(query.lower().split())
            return [
                e for e in self._entries
                if any(term in json.dumps(e).lower() for term in query_terms)
            ][:top_k]

        try:
            import numpy as np

            query_vec = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
            scored: List[tuple[float, dict[str, Any]]] = []
            for _, entry in embedded:
                emb = np.array(entry["_embedding"], dtype=np.float32)
                similarity = float(np.dot(query_vec, emb))
                scored.append((similarity, entry))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, entry in scored[:top_k]:
                clean = {k: v for k, v in entry.items() if k != "_embedding"}
                clean["_similarity"] = round(score, 4)
                results.append(clean)
            return results
        except Exception:
            q = query.lower()
            return [e for e in self._entries if q in json.dumps(e).lower()][:top_k]
