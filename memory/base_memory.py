"""Persistent memory abstractions used by the agentic pipeline."""
from __future__ import annotations

import json
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
        return [entry for entry in self._entries if query_lower in json.dumps(entry).lower()]


class EpisodicMemory(BaseMemory):
    """A history of prior agent actions and outcomes."""

    def search(self, query: str) -> List[dict[str, Any]]:
        query_lower = query.lower()
        return [entry for entry in self._entries if query_lower in json.dumps(entry).lower()]


class SemanticMemory(BaseMemory):
    """Persistent facts and reusable insights distilled from prior runs."""

    def search(self, query: str) -> List[dict[str, Any]]:
        query_lower = query.lower()
        return [entry for entry in self._entries if query_lower in json.dumps(entry).lower()]
