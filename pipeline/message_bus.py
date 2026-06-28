"""A simple in-process message bus for inter-agent communication."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass(slots=True)
class Message:
    kind: str
    sender: str
    recipient: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    """Registers handlers by message kind and dispatches messages to them."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[[Message], None]]] = {}

    def subscribe(self, kind: str, handler: Callable[[Message], None]) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def publish(self, message: Message) -> None:
        for handler in self._handlers.get(message.kind, []):
            handler(message)
