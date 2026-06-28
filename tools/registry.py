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
    """Registers tools and exposes a stable dispatch interface."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        tool = self.get(name)
        if tool is None or tool.handler is None:
            raise KeyError(f"Tool '{name}' is not registered")
        return tool.handler(arguments or {})

    def definitions(self) -> List[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "inputs": tool.inputs,
                "outputs": tool.outputs,
                "cost": tool.cost,
                "latency": tool.latency,
            }
            for tool in self._tools.values()
        ]
