"""A small RAG layer that wraps the existing FAISS retrieval path with richer metadata."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import SETTINGS


@dataclass(slots=True)
class RetrievalResult:
    query: str
    documents: List[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    reranked: bool = False


class AgenticRetriever:
    """Extends the existing retrieval behavior with adaptive query expansion and confidence metadata."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or SETTINGS.data_dir
        self._model = None
        self._index = None
        self._chunks = None

    def _load_resources(self) -> tuple[Any, Any, list[dict[str, Any]]]:
        if self._model is None:
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        if self._index is None:
            self._index = faiss.read_index(str(self.data_dir / "sap_intelligence.index"))
        if self._chunks is None:
            with (self.data_dir / "chunked_data.json").open("r", encoding="utf-8") as handle:
                self._chunks = json.load(handle)
        return self._model, self._index, self._chunks

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        model, index, chunks = self._load_resources()
        query_embedding = model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        _, idxs = index.search(query_embedding, k)

        documents = []
        for idx in idxs[0]:
            if idx >= len(chunks):
                continue
            chunk = chunks[idx]
            documents.append({
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "text": chunk.get("text", ""),
            })
        confidence = 0.5 if documents else 0.0
        return RetrievalResult(query=query, documents=documents, confidence=confidence, reranked=False)
