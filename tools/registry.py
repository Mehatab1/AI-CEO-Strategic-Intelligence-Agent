"""A lightweight registry for tools that can be invoked by agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    cost: float = 0.0
    latency: float = 0.0
    handler: Optional[Callable[[dict[str, Any]], Any]] = None


class ToolRegistry:
    """Registers tools and exposes a stable dispatch interface.

    Two additions beyond the original:
    - to_ollama_schema() / all_ollama_schemas(): convert registered tools to the
      Ollama /api/chat tool-call format so the LLM can call them by name.
    - select_tools_for_goal(): lightweight keyword heuristic that narrows the tool
      list before handing it to the LLM, reducing context length.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = self.get(name)
        if tool is None or tool.handler is None:
            raise KeyError(f"Tool '{name}' is not registered or has no handler")
        return tool.handler(arguments or {})

    def definitions(self) -> List[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputs": t.inputs,
                "outputs": t.outputs,
                "cost": t.cost,
                "latency": t.latency,
            }
            for t in self._tools.values()
        ]

    def to_ollama_schema(self, name: str) -> Optional[dict[str, Any]]:
        """Convert a single registered tool to Ollama's native tool-call format."""
        tool = self.get(name)
        if tool is None:
            return None
        properties: Dict[str, Any] = {}
        for input_name, input_type in tool.inputs.items():
            properties[input_name] = {"type": input_type, "description": input_name}
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                },
            },
        }

    def all_ollama_schemas(self) -> List[dict[str, Any]]:
        """All registered tools as Ollama-format tool schemas."""
        return [s for name in self._tools if (s := self.to_ollama_schema(name)) is not None]

    def select_tools_for_goal(self, goal: str) -> List[str]:
        """Return tool names whose description overlaps with the goal text.

        This is a lightweight pre-filter; the LLM makes the final selection
        during the retrieval stage when it sees the actual schemas.  If nothing
        matches, returns all registered tools so we never send an empty list.
        """
        goal_tokens = set(goal.lower().split())
        selected = [
            name
            for name, tool in self._tools.items()
            if goal_tokens & set((tool.name + " " + tool.description).lower().split())
        ]
        return selected or list(self._tools.keys())
