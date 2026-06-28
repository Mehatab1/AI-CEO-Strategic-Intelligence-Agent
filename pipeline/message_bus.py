"""In-process message bus for inter-agent communication."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class MessageKind:
    TASK = "TASK"
    RESULT = "RESULT"
    QUESTION = "QUESTION"
    ANSWER = "ANSWER"
    REQUEST = "REQUEST"
    RESPONSE = "RESPONSE"
    CRITIQUE = "CRITIQUE"
    APPROVAL = "APPROVAL"
    MEMORY_READ = "MEMORY_READ"
    MEMORY_WRITE = "MEMORY_WRITE"
    STATUS = "STATUS"


@dataclass(slots=True)
class Message:
    kind: str
    sender: str
    recipient: str
    payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_correlation(self) -> "Message":
        """Attach a correlation ID so request() can match the reply."""
        self.metadata.setdefault("correlation_id", str(uuid.uuid4()))
        return self


class MessageBus:
    """Registers handlers by message kind and dispatches messages to them.

    All published messages are appended to an internal log so the orchestrator
    can expose a full communication trace in the Streamlit UI.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Callable[[Message], Any]]] = {}
        self._log: List[Message] = []

    def subscribe(self, kind: str, handler: Callable[[Message], Any]) -> None:
        self._handlers.setdefault(kind, []).append(handler)

    def publish(self, message: Message) -> None:
        self._log.append(message)
        for handler in self._handlers.get(message.kind, []):
            handler(message)

    def request(self, message: Message) -> Optional[Any]:
        """Publish a REQUEST and collect the first synchronous RESPONSE.

        Because all agents in this system run in-process, the response handler
        fires synchronously inside publish() before request() returns.
        """
        message.with_correlation()
        cid = message.metadata["correlation_id"]
        replies: List[Any] = []

        def _capture(reply: Message) -> None:
            if reply.metadata.get("correlation_id") == cid:
                replies.append(reply.payload)

        self.subscribe(MessageKind.RESPONSE, _capture)
        self.publish(message)
        return replies[0] if replies else None

    def get_log(self) -> List[Message]:
        return list(self._log)

    def clear_log(self) -> None:
        self._log.clear()
