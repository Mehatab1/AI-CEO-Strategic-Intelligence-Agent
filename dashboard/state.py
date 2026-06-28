"""Helpers for keeping the Streamlit dashboard aligned with the agentic workflow."""
from __future__ import annotations

from typing import Any


def build_trace_summary(trace: list[dict[str, Any]]) -> dict[str, int]:
    """Create a compact summary that the dashboard can render quickly."""
    return {stage: sum(1 for entry in trace if entry.get("stage") == stage) for stage in {entry.get("stage") for entry in trace}}
