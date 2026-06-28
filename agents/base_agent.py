"""Base abstractions for agents in the evolved platform."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseAgent(ABC):
    """Minimal agent interface with memory, tools, execution, reflection and validation hooks."""

    def __init__(self, name: str, role: str, goal: str, available_tools: Optional[list[str]] = None) -> None:
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
        """Inspect the outcome and decide whether to revise it."""
        return {
            "agent": self.name,
            "correct": True,
            "should_retrieve_again": False,
            "should_ask_another_agent": False,
            "should_revise": False,
            "result": result,
        }

    def validate(self, result: dict[str, Any], context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Return a validation record for the produced output."""
        return {
            "agent": self.name,
            "status": "validated",
            "result": result,
        }
