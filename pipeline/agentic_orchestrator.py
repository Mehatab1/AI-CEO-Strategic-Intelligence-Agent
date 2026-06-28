"""An evolutionary orchestrator that wraps the existing CEO pipeline with agentic abstractions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from memory.base_memory import EpisodicMemory, SemanticMemory, WorkingMemory
from tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """Creates a plan and delegates execution to the orchestrator."""

    def __init__(self, llm_core: Any) -> None:
        super().__init__(name="planner", role="planner", goal="Create an execution plan")
        self.llm_core = llm_core

    def execute(self, task: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        persona_context = context.get("persona_context", "") if context else ""
        plan = self.llm_core.run_plan_stage(str(task), persona_context=persona_context, verbose=False)
        return {"plan": plan, "status": "planned"}


class AgenticOrchestrator:
    """Provides the new workflow while preserving the existing API surface."""

    def __init__(
        self,
        llm_core: Any,
        storage_dir: Optional[Path] = None,
        retrieval_tool_schemas: Optional[list[dict[str, Any]]] = None,
        retrieval_tool_handlers: Optional[dict[str, Any]] = None,
        decide_tool_schema: Optional[dict[str, Any]] = None,
        decide_tool_name: str = "finalize_ceo_answer",
        validate_tool_schema: Optional[dict[str, Any]] = None,
        validate_tool_name: str = "submit_validated_ceo_answer",
    ) -> None:
        self.llm_core = llm_core
        self.storage_dir = storage_dir or Path(".")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.working_memory = WorkingMemory(storage_path=self.storage_dir / "working_memory.json")
        self.episodic_memory = EpisodicMemory(storage_path=self.storage_dir / "episodic_memory.json")
        self.semantic_memory = SemanticMemory(storage_path=self.storage_dir / "semantic_memory.json")
        self.tool_registry = ToolRegistry()
        self.planner = PlannerAgent(llm_core)
        self.trace: list[dict[str, Any]] = []
        self.retrieval_tool_schemas = retrieval_tool_schemas or []
        self.retrieval_tool_handlers = retrieval_tool_handlers or {}
        self.decide_tool_schema = decide_tool_schema
        self.decide_tool_name = decide_tool_name
        self.validate_tool_schema = validate_tool_schema
        self.validate_tool_name = validate_tool_name

    def _record_event(self, stage: str, payload: Any) -> None:
        self.trace.append({"stage": stage, "payload": payload})

    def _register_default_tools(self) -> None:
        self.tool_registry.register(
            ToolSpec(
                name="retrieve_news",
                description="Retrieve evidence from FAISS-backed news chunks",
                inputs={"query": "string", "k": "integer"},
                outputs={"documents": "array"},
                cost=0.0,
                latency=0.0,
                handler=lambda args: [{"source": "faiss", "query": args.get("query", "")}],
            )
        )

    def run(self, question: str, history: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        self.trace = []
        self._register_default_tools()

        self.working_memory.write({"type": "question", "content": question})
        if history:
            self.working_memory.write({"type": "history", "content": history})

        planning = self.planner.execute(question, context={"persona_context": ""})
        self._record_event("plan", planning)
        logger.info("Planned agent run", extra={"question": question, "plan": planning})

        retrieval_trace = self.llm_core.run_retrieval_stage(
            planning["plan"],
            self.retrieval_tool_schemas,
            self.retrieval_tool_handlers,
            persona_context="",
            max_iterations=2,
            verbose=False,
        )
        self._record_event("retrieve", retrieval_trace)
        logger.info("Retrieval completed", extra={"trace": retrieval_trace})

        evidence_text = json.dumps(retrieval_trace, indent=2, default=str)
        analysis = self.llm_core.run_analyze_stage(
            planning["plan"],
            evidence_text,
            self.llm_core.make_analyze_tool_schema("recommendations"),
            persona_context="",
            verbose=False,
        )
        self._record_event("analyze", analysis)

        draft = self.llm_core.run_decide_stage(
            planning["plan"],
            evidence_text,
            analysis,
            self.decide_tool_schema or {},
            self.decide_tool_name,
            persona_context="",
            verbose=False,
        ) or {}
        self._record_event("decide_recommend", draft)

        validated = self.llm_core.run_validate_stage(
            draft,
            evidence_text,
            self.validate_tool_schema or {},
            self.validate_tool_name,
            persona_context="",
            verbose=False,
        )
        if validated:
            final = dict(draft)
            for key, value in validated.items():
                if value:
                    final[key] = value
        else:
            final = dict(draft)
        self._record_event("validate", final)

        self.episodic_memory.write({"question": question, "result": final})
        self.semantic_memory.write({"question": question, "summary": final.get("executive_summary", "")})
        self.working_memory.write({"type": "answer", "content": final})

        return {
            "final_answer": self._format_answer(final),
            "trace": self.trace,
            "artifacts": {
                "working_memory": self.working_memory.read_all(),
                "episodic_memory": self.episodic_memory.read_all(),
                "semantic_memory": self.semantic_memory.read_all(),
            },
        }

    def _format_answer(self, data: dict[str, Any]) -> str:
        sections = [
            ("Executive Summary", data.get("executive_summary")),
            ("Supporting Evidence", data.get("supporting_evidence")),
            ("Recommended Actions", data.get("recommended_actions")),
            ("Priority Level", data.get("priority_level")),
            ("Why This Matters", data.get("why_this_matters")),
        ]
        parts = [f"## {title}\n{content}" for title, content in sections if content]
        return "\n\n".join(parts) if parts else "I wasn't able to produce a validated answer."
