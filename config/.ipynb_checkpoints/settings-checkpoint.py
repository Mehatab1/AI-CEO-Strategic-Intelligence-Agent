"""Centralized configuration for the agentic pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path(__file__).resolve().parents[1] / "notebook" / "data"
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    max_retrieval_iterations: int = int(os.getenv("MAX_RETRIEVAL_ITERATIONS", "4"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


SETTINGS = Settings()
