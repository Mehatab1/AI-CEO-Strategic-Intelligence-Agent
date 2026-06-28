"""RAG layer: FAISS retrieval with real confidence scoring and adaptive query expansion."""
from __future__ import annotations

import json
import re
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
    """FAISS retrieval with real distance-based confidence and confidence-gated query expansion."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = data_dir or SETTINGS.data_dir
        self._model: Any = None
        self._index: Any = None
        self._chunks: list[dict[str, Any]] | None = None

    def _load_resources(self) -> tuple[Any, Any, list[dict[str, Any]]]:
        if self._model is None:
            self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        if self._index is None:
            self._index = faiss.read_index(str(self.data_dir / "sap_intelligence.index"))
        if self._chunks is None:
            with (self.data_dir / "chunked_data.json").open("r", encoding="utf-8") as h:
                self._chunks = json.load(h)
        return self._model, self._index, self._chunks

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Single-round retrieval.  Confidence = mean cosine similarity of returned chunks."""
        model, index, chunks = self._load_resources()
        q_vec = np.array(model.encode([query], normalize_embeddings=True), dtype=np.float32)
        distances, idxs = index.search(q_vec, k)

        documents: List[dict[str, Any]] = []
        valid_scores: List[float] = []
        for dist, idx in zip(distances[0], idxs[0]):
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            score = float(dist)  # cosine similarity for L2-normalised BAAI/bge vectors
            documents.append({
                "source": chunk.get("source", ""),
                "title": chunk.get("title", ""),
                "text": chunk.get("text", ""),
                "relevance_score": round(score, 4),
            })
            valid_scores.append(score)

        # Confidence is the mean cosine similarity, clamped to [0, 1].
        # A score of ~0.8+ signals semantically close chunks; ~0.4 signals weak alignment.
        confidence = max(0.0, min(1.0, sum(valid_scores) / len(valid_scores))) if valid_scores else 0.0
        return RetrievalResult(query=query, documents=documents, confidence=confidence, reranked=False)

    def adaptive_retrieve(
        self,
        query: str,
        k: int = 5,
        min_confidence: float = 0.35,
        max_rounds: int = 3,
    ) -> RetrievalResult:
        """Retrieve with confidence-gated retry and automatic query expansion.

        If the first round scores below min_confidence, extracts named entities from
        retrieved text to form a richer follow-up query, retries, and merges unique
        documents across rounds.  Stops early once a round hits min_confidence.
        """
        result = self.retrieve(query, k)
        if result.confidence >= min_confidence or not result.documents:
            return result

        all_docs: List[dict[str, Any]] = list(result.documents)
        seen_titles: set[str] = {d["title"] for d in all_docs}

        for round_num in range(1, max_rounds):
            expanded = self._expand_query(query, round_num, result)
            round_result = self.retrieve(expanded, k)

            for doc in round_result.documents:
                if doc["title"] not in seen_titles:
                    all_docs.append(doc)
                    seen_titles.add(doc["title"])

            if round_result.confidence >= min_confidence:
                break
            result = round_result

        scores = [d.get("relevance_score", 0.0) for d in all_docs]
        merged_conf = max(0.0, min(1.0, sum(scores) / len(scores))) if scores else 0.0

        return RetrievalResult(
            query=query,
            documents=all_docs,
            confidence=merged_conf,
            reranked=True,
        )

    def _expand_query(self, original: str, round_num: int, prev: RetrievalResult) -> str:
        """Build a richer query by extracting capitalized entity phrases from prior results."""
        corpus = " ".join(d.get("text", "") for d in prev.documents[:3])
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", corpus)
        top_entities = list(dict.fromkeys(entities))[:3]  # preserve order, deduplicate

        if round_num == 1 and top_entities:
            return f"{original} {' '.join(top_entities)}"
        return f"SAP enterprise software {original} strategic market analysis"
