"""Base abstractions for agents in the evolved platform."""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Optional


def call_ollama(messages: list[dict[str, Any]], timeout: int = 60) -> str:
    """Minimal blocking Ollama /api/chat call.  Returns the assistant content string.

    Used by reflect() and validate() so every agent can self-evaluate without
    importing the heavier agent_core module.  Falls back to "" on any error so
    callers can apply their own heuristic fallback.
    """
    import requests

    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    try:
        resp = requests.post(
            f"{host}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "")
    except Exception:
        return ""


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Pull the first {...} JSON object out of a model response."""
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


class BaseAgent(ABC):
    """Minimal agent interface with memory, tools, execution, reflection and validation hooks."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        available_tools: Optional[list[str]] = None,
    ) -> None:
        self.name = name
        self.role = role
        self.goal = goal
        self.available_tools = available_tools or []
        self.state: dict[str, Any] = {"status": "initialized"}
        self.memory: Any = None

    @abstractmethod
    def execute(self, task: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Perform the agent's responsibility for a task."""

    def reflect(self, result: dict[str, Any], context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ask the LLM to evaluate whether this output is complete and grounded.

        Returns a dict with keys: correct, should_retrieve_again,
        should_ask_another_agent, should_revise, reason.
        Falls back to a heuristic check if Ollama is unavailable.
        """
        evidence_preview = json.dumps(result, default=str)[:1500]
        task_hint = json.dumps(context or {}, default=str)[:300]

        raw = call_ollama([
            {
                "role": "system",
                "content": (
                    f"You are {self.name}, a {self.role} agent. "
                    "Critically evaluate your own output. "
                    "Respond with valid JSON only — no prose before or after."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Task context: {task_hint}\n\n"
                    f"My output: {evidence_preview}\n\n"
                    "Is this output complete, evidence-grounded, and logically consistent?\n"
                    "Respond with exactly this schema:\n"
                    '{"correct": true_or_false, '
                    '"should_retrieve_again": true_or_false, '
                    '"should_ask_another_agent": true_or_false, '
                    '"should_revise": true_or_false, '
                    '"reason": "one sentence"}'
                ),
            },
        ])

        parsed = _extract_json(raw)
        if parsed:
            return {"agent": self.name, **parsed, "result": result}

        # Heuristic fallback: flag outputs where every string field is blank
        is_empty = not any(
            bool(v) for v in result.values() if isinstance(v, (str, list, dict))
        )
        return {
            "agent": self.name,
            "correct": not is_empty,
            "should_retrieve_again": is_empty,
            "should_ask_another_agent": False,
            "should_revise": is_empty,
            "reason": "Output is empty or incomplete." if is_empty else "Output appears adequate.",
            "result": result,
        }

    def validate(self, result: dict[str, Any], context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ask the LLM to validate the output against the evidence supplied in context."""
        evidence = (context or {}).get("evidence_text", "")[:1000]
        output_preview = json.dumps(result, default=str)[:1000]

        raw = call_ollama([
            {
                "role": "system",
                "content": (
                    f"You are {self.name}, a {self.role} agent. "
                    "Validate whether this output is supported by the evidence. "
                    "Respond with valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evidence:\n{evidence}\n\n"
                    f"Output to validate:\n{output_preview}\n\n"
                    "Respond with exactly this schema:\n"
                    '{"status": "Supported|Revised|Unsupported", '
                    '"notes": "one sentence"}'
                ),
            },
        ])

        parsed = _extract_json(raw)
        if parsed:
            return {"agent": self.name, **parsed, "result": result}

        return {
            "agent": self.name,
            "status": "Unverified",
            "notes": "Validation did not return structured output.",
            "result": result,
        }
