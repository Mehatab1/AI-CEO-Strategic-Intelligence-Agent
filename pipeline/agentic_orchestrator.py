"""Agentic orchestrator: task graph, CriticAgent, wired MessageBus + ToolRegistry."""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from agents.base_agent import BaseAgent, call_ollama
from memory.base_memory import EpisodicMemory, SemanticMemory, WorkingMemory
from pipeline.message_bus import Message, MessageBus, MessageKind
from tools.registry import ToolRegistry, ToolSpec

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Task graph
# ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    description: str
    agent: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"          # pending | running | completed | failed | skipped
    result: Optional[dict[str, Any]] = None
    retries: int = 0
    max_retries: int = 2
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    confidence: float = 0.0


@dataclass
class TaskGraph:
    goal: str
    tasks: List[Task] = field(default_factory=list)
    workflow: str = "pipeline"       # pipeline | parallel | review

    def get(self, task_id: str) -> Optional[Task]:
        return next((t for t in self.tasks if t.id == task_id), None)

    def ready_tasks(self) -> List[Task]:
        """Tasks whose dependencies are all completed and which are still pending."""
        completed_ids = {t.id for t in self.tasks if t.status == "completed"}
        return [
            t for t in self.tasks
            if t.status == "pending"
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def is_complete(self) -> bool:
        return all(t.status in {"completed", "failed", "skipped"} for t in self.tasks)

    def summary(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "total": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == "completed"),
            "failed": sum(1 for t in self.tasks if t.status == "failed"),
            "pending": sum(1 for t in self.tasks if t.status == "pending"),
        }


# ─────────────────────────────────────────────────────────────
# Execution monitor
# ─────────────────────────────────────────────────────────────

class ExecutionMonitor:
    """Tracks every task state change, tool call, and agent decision during a run."""

    def __init__(self) -> None:
        self._events: List[dict[str, Any]] = []

    def record(self, event_type: str, **kwargs: Any) -> None:
        self._events.append({"type": event_type, "timestamp": time.time(), **kwargs})

    def snapshot(self) -> dict[str, Any]:
        return {"event_count": len(self._events), "events": self._events[-20:]}

    def clear(self) -> None:
        self._events.clear()


# ─────────────────────────────────────────────────────────────
# PlannerAgent — real task decomposition + workflow selection
# ─────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """Decomposes goals into a dependency-ordered TaskGraph and selects a workflow."""

    def __init__(self, llm_core: Any) -> None:
        super().__init__(name="planner", role="planner", goal="Build and manage the execution plan")
        self.llm_core = llm_core

    def execute(self, task: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        goal = str(task)
        persona = (context or {}).get("persona_context", "")

        # LLM produces a flat plan (goal + planned_steps list)
        plan = self.llm_core.run_plan_stage(goal, persona_context=persona, verbose=False)

        workflow = self._select_workflow(goal)
        task_graph = self._build_task_graph(goal, plan, workflow)

        return {"plan": plan, "task_graph": task_graph, "workflow": workflow, "status": "planned"}

    def _select_workflow(self, goal: str) -> str:
        """Choose pipeline, parallel, or review based on goal text signals.

        review   — strategic decisions, risk/competitor analysis; invokes CriticAgent.
        parallel — broad intelligence requests spanning multiple domains.
        pipeline — default; specific factual questions.
        """
        g = goal.lower()
        review_signals = (
            "should", "recommend", "strategy", "strategically", "decision",
            "risk", "threat", "versus", " vs ", "compare", "competitor",
            "oracle", "microsoft", "workday", "salesforce", "servicenow",
            "infor", "ifs", "epicor", "positioning", "compete",
        )
        if any(w in g for w in review_signals):
            return "review"
        parallel_signals = (
            "overview", "summary", "everything", "all", "comprehensive",
            "today", "latest", "current", "opportunity", "opportunities",
            "landscape", "what is happening", "state of",
        )
        if any(w in g for w in parallel_signals):
            return "parallel"
        return "pipeline"

    def _build_task_graph(self, goal: str, plan: dict[str, Any], workflow: str) -> TaskGraph:
        """Convert the LLM's planned_steps into a dependency-ordered task graph."""
        graph = TaskGraph(goal=goal, workflow=workflow)
        steps = plan.get("planned_steps", [goal])

        if workflow == "parallel":
            # All retrieval steps are independent; synthesis depends on all of them
            specialist_ids = []
            for i, step in enumerate(steps):
                tid = f"t{i}"
                graph.tasks.append(Task(id=tid, description=step, agent="retriever", depends_on=[]))
                specialist_ids.append(tid)
            graph.tasks.append(Task(
                id="synthesize",
                description="Synthesize gathered evidence into a final answer",
                agent="ceo",
                depends_on=specialist_ids,
            ))

        elif workflow == "review":
            # Sequential retrieval → decision → critic → finalize
            for i, step in enumerate(steps):
                graph.tasks.append(Task(
                    id=f"t{i}",
                    description=step,
                    agent="retriever",
                    depends_on=[f"t{i - 1}"] if i > 0 else [],
                ))
            last = f"t{len(steps) - 1}"
            graph.tasks.append(Task(id="decide", description="Decide and recommend", agent="ceo", depends_on=[last]))
            graph.tasks.append(Task(id="critique", description="Critique the recommendation", agent="critic", depends_on=["decide"]))
            graph.tasks.append(Task(id="finalize", description="Finalize after critique", agent="ceo", depends_on=["critique"]))

        else:  # pipeline
            for i, step in enumerate(steps):
                graph.tasks.append(Task(
                    id=f"t{i}",
                    description=step,
                    agent="retriever",
                    depends_on=[f"t{i - 1}"] if i > 0 else [],
                ))
            graph.tasks.append(Task(
                id="finalize",
                description="Synthesize and finalize",
                agent="ceo",
                depends_on=[f"t{len(steps) - 1}"],
            ))

        return graph

    def replan(self, task_graph: TaskGraph, failed_task: Task, critique: str) -> TaskGraph:
        """Reset a failed task (and its dependents) after critic feedback."""
        logger.info("Replanning", extra={"task": failed_task.id, "critique": critique})
        failed_task.status = "pending"
        failed_task.retries += 1
        failed_task.result = None
        for t in task_graph.tasks:
            if failed_task.id in t.depends_on and t.status in {"failed", "completed"}:
                t.status = "pending"
                t.result = None
        return task_graph


# ─────────────────────────────────────────────────────────────
# CriticAgent — evaluates evidence quality + recommendation soundness
# ─────────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """Evaluates draft recommendations for grounding, consistency, and specificity."""

    def __init__(self) -> None:
        super().__init__(name="critic", role="critic", goal="Evaluate recommendation quality")

    def execute(self, task: Any, context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        draft = (context or {}).get("draft", {})
        evidence = (context or {}).get("evidence_text", "")[:2000]
        draft_preview = json.dumps(draft, default=str)[:1500]

        try:
            raw = call_ollama([
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous strategic critic evaluating an AI-generated "
                        "recommendation. Check: (1) evidence grounding — is every claim "
                        "traceable to the evidence? (2) logical consistency — do the "
                        "conclusions follow? (3) hallucination risk — are there invented "
                        "facts? (4) recommendation specificity — are actions concrete? "
                        "Respond with valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Evidence gathered:\n{evidence}\n\n"
                        f"Draft recommendation:\n{draft_preview}\n\n"
                        "Evaluate this. Respond with exactly this schema:\n"
                        '{"verdict": "approved|rejected|needs_revision", '
                        '"evidence_grounding": "strong|weak|absent", '
                        '"hallucination_risk": "low|medium|high", '
                        '"logical_consistency": "sound|questionable|flawed", '
                        '"recommendation_quality": "specific|vague|missing", '
                        '"feedback": "2-3 sentences of specific actionable feedback", '
                        '"confidence_score": 0.0_to_1.0}'
                    ),
                },
            ], timeout=90)
        except Exception as exc:
            # Ollama unavailable (e.g. test environment, no requests module, offline)
            return {
                "agent": self.name,
                "status": "critic_unavailable",
                "verdict": "needs_revision",
                "feedback": f"Critic could not reach Ollama: {exc}",
                "confidence_score": 0.0,
            }

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                critique = json.loads(match.group())
                return {"agent": self.name, "status": "critiqued", **critique}
            except json.JSONDecodeError:
                pass

        return {
            "agent": self.name,
            "status": "critique_failed",
            "verdict": "needs_revision",
            "feedback": "Critic could not parse the draft. Re-run the decision stage.",
            "confidence_score": 0.0,
        }


# ─────────────────────────────────────────────────────────────
# AgenticOrchestrator
# ─────────────────────────────────────────────────────────────

class AgenticOrchestrator:
    """Drives the task graph through MessageBus, dispatches tools via ToolRegistry,
    runs the CriticAgent on review-mode goals, and computes a real confidence score."""

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
        self.message_bus = MessageBus()
        self.planner = PlannerAgent(llm_core)
        self.critic = CriticAgent()
        self.monitor = ExecutionMonitor()

        self.trace: list[dict[str, Any]] = []
        self.retrieval_tool_schemas = retrieval_tool_schemas or []
        self.retrieval_tool_handlers = retrieval_tool_handlers or {}
        self.decide_tool_schema = decide_tool_schema
        self.decide_tool_name = decide_tool_name
        self.validate_tool_schema = validate_tool_schema
        self.validate_tool_name = validate_tool_name

        self._register_tools()
        self._wire_message_bus()

    # ── setup ───────────────────────────────────────────────

    def _register_tools(self) -> None:
        """Register every retrieval tool in ToolRegistry with its real handler."""
        for schema in self.retrieval_tool_schemas:
            fn = schema.get("function", {})
            name = fn.get("name", "")
            handler = self.retrieval_tool_handlers.get(name)
            if not name or handler is None:
                continue
            params = fn.get("parameters", {}).get("properties", {})
            self.tool_registry.register(ToolSpec(
                name=name,
                description=fn.get("description", ""),
                inputs={k: v.get("type", "string") for k, v in params.items()},
                outputs={"documents": "array"},
                handler=handler,
            ))

    def _wire_message_bus(self) -> None:
        self.message_bus.subscribe(MessageKind.STATUS, self._on_status)
        self.message_bus.subscribe(MessageKind.MEMORY_WRITE, self._on_memory_write)

    def _on_status(self, msg: Message) -> None:
        self.monitor.record("status", sender=msg.sender, detail=msg.payload)

    def _on_memory_write(self, msg: Message) -> None:
        mtype = msg.payload.get("memory_type", "working")
        entry = msg.payload.get("entry", {})
        {"semantic": self.semantic_memory, "episodic": self.episodic_memory}.get(
            mtype, self.working_memory
        ).write(entry)

    # ── internal helpers ────────────────────────────────────

    def _publish_status(self, sender: str, stage: str, detail: str) -> None:
        self.message_bus.publish(Message(
            kind=MessageKind.STATUS,
            sender=sender,
            recipient="monitor",
            payload={"stage": stage, "detail": detail},
        ))

    def _record(self, stage: str, payload: Any) -> None:
        self.trace.append({"stage": stage, "payload": payload})

    def _dispatch_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Route a tool call through ToolRegistry; fall back to direct handler."""
        self.monitor.record("tool_call", tool=tool_name, args=str(args)[:120])
        try:
            return self.tool_registry.dispatch(tool_name, args)
        except KeyError:
            handler = self.retrieval_tool_handlers.get(tool_name)
            return handler(args) if handler else {"error": f"Tool '{tool_name}' not found"}

    def _recall_prior_knowledge(self, question: str) -> list[dict[str, Any]]:
        return self.semantic_memory.search(question, top_k=3) + \
               self.episodic_memory.search(question, top_k=3)

    def _retry_after_critique(
        self,
        plan: dict[str, Any],
        evidence_text: str,
        analysis: dict[str, Any],
        draft: dict[str, Any],
        feedback: str,
    ) -> Optional[dict[str, Any]]:
        """Inject critic feedback into evidence context and re-run the decision stage."""
        augmented = evidence_text + f"\n\nCritic feedback on prior draft:\n{feedback}"
        revised = self.llm_core.run_decide_stage(
            plan, augmented, analysis,
            self.decide_tool_schema or {},
            self.decide_tool_name,
            persona_context="",
            verbose=False,
        )
        return revised or None

    # ── confidence ──────────────────────────────────────────

    def _compute_confidence(
        self,
        retrieval_trace: list[dict[str, Any]],
        analysis: dict[str, Any],
        validated: Optional[dict[str, Any]],
        critique: Optional[dict[str, Any]],
    ) -> float:
        """Real confidence from four evidence signals (max 1.0).

        Breakdown:
        - retrieval_coverage : did the agent actually call tools?         (0–0.30)
        - evidence_agreement : how many analysis observations were found?  (0–0.30)
        - validation_status  : Supported / Revised / Unsupported          (0–0.20)
        - critic_approval    : critic verdict + its own confidence score   (0–0.20)
        """
        real_calls = [e for e in retrieval_trace
                      if e.get("tool") != "finished_retrieving" and "result" in e]
        retrieval_coverage = min(0.30, len(real_calls) * 0.10)

        observations = analysis.get("key_observations", [])
        evidence_agreement = min(0.30, len(observations) * 0.06)

        validation_score = 0.0
        if validated:
            status = (validated.get("validation_status") or "").strip()
            validation_score = {"Supported": 0.20, "Revised": 0.12, "Unsupported": 0.0}.get(status, 0.08)

        critic_score = 0.0
        if critique:
            verdict = critique.get("verdict", "")
            raw = float(critique.get("confidence_score") or 0.0)
            if verdict == "approved":
                critic_score = min(0.20, 0.10 + raw * 0.10)
            elif verdict == "needs_revision":
                critic_score = 0.07
        else:
            critic_score = 0.10   # no critic used — give partial credit

        return round(retrieval_coverage + evidence_agreement + validation_score + critic_score, 3)

    # ── main entry point ────────────────────────────────────

    def run(self, question: str, history: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        self.trace = []
        self.monitor.clear()

        self.working_memory.write({"type": "question", "content": question})
        if history:
            self.working_memory.write({"type": "history", "content": history})

        self._publish_status("orchestrator", "start", f"Processing: {question[:80]}")

        # Memory-first: surface relevant prior knowledge before planning
        prior = self._recall_prior_knowledge(question)
        if prior:
            self.monitor.record("memory_hit", count=len(prior))

        # ── PLAN ───────────────────────────────────────────
        self._publish_status("planner", "plan", "Decomposing goal into tasks")
        planning = self.planner.execute(question, context={"persona_context": ""})
        self._record("plan", planning["plan"])
        task_graph: TaskGraph = planning["task_graph"]
        self.monitor.record("plan", workflow=task_graph.workflow, tasks=task_graph.summary())

        # ── RETRIEVE (tools routed through ToolRegistry) ───
        self._publish_status("orchestrator", "retrieve", "Gathering evidence")
        tracked_handlers = {
            name: (lambda args, n=name: self._dispatch_tool(n, args))
            for name in self.retrieval_tool_handlers
        }
        retrieval_trace = self.llm_core.run_retrieval_stage(
            planning["plan"],
            self.retrieval_tool_schemas,
            tracked_handlers,
            persona_context="",
            max_iterations=2,
            verbose=False,
        )
        self._record("retrieve", retrieval_trace)
        self.monitor.record("retrieve", tool_calls=len(retrieval_trace))
        evidence_text = json.dumps(retrieval_trace, indent=2, default=str)

        # ── ANALYZE ────────────────────────────────────────
        self._publish_status("orchestrator", "analyze", "Analysing evidence")
        analysis = self.llm_core.run_analyze_stage(
            planning["plan"],
            evidence_text,
            self.llm_core.make_analyze_tool_schema("recommendations"),
            persona_context="",
            verbose=False,
        )
        self._record("analyze", analysis)

        # ── DECIDE ────────────────────────────────────────
        self._publish_status("orchestrator", "decide", "Deciding and recommending")
        draft = self.llm_core.run_decide_stage(
            planning["plan"],
            evidence_text,
            analysis,
            self.decide_tool_schema or {},
            self.decide_tool_name,
            persona_context="",
            verbose=False,
        ) or {}
        self._record("decide_recommend", draft)

        # ── CRITIC (review workflow only) ──────────────────
        critique_result: Optional[dict[str, Any]] = None
        if task_graph.workflow == "review":
            self._publish_status("critic", "critique", "Evaluating recommendation quality")
            self.message_bus.publish(Message(
                kind=MessageKind.CRITIQUE,
                sender="orchestrator",
                recipient="critic",
                payload={"draft": draft, "evidence_preview": evidence_text[:500]},
            ))
            critique_result = self.critic.execute(
                "evaluate",
                context={"draft": draft, "evidence_text": evidence_text},
            )
            self._record("critique", critique_result)
            self.monitor.record("critique", verdict=critique_result.get("verdict"))

            # One retry if critic rejects
            if critique_result.get("verdict") == "rejected" and draft:
                self._publish_status("orchestrator", "replan", "Critic rejected — retrying decision")
                feedback = critique_result.get("feedback", "")
                revised = self._retry_after_critique(planning["plan"], evidence_text, analysis, draft, feedback)
                if revised:
                    draft = revised
                    self._record("decide_recommend_revised", draft)

        # ── VALIDATE ──────────────────────────────────────
        self._publish_status("orchestrator", "validate", "Validating answer")
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
        self._record("validate", final)

        # ── CONFIDENCE ────────────────────────────────────
        confidence = self._compute_confidence(retrieval_trace, analysis, validated, critique_result)
        final["_confidence"] = confidence

        # ── PERSIST TO MEMORY ─────────────────────────────
        self.message_bus.publish(Message(
            kind=MessageKind.MEMORY_WRITE,
            sender="orchestrator",
            recipient="memory",
            payload={"memory_type": "episodic", "entry": {"question": question, "result": final}},
        ))
        self.message_bus.publish(Message(
            kind=MessageKind.MEMORY_WRITE,
            sender="orchestrator",
            recipient="memory",
            payload={"memory_type": "semantic", "entry": {
                "question": question,
                "summary": final.get("executive_summary", ""),
                "confidence": confidence,
            }},
        ))
        self.working_memory.write({"type": "answer", "content": final})
        self._publish_status("orchestrator", "complete", f"Confidence: {confidence:.3f}")

        return {
            "final_answer": self._format_answer(final),
            "trace": self.trace,
            "confidence": confidence,
            "workflow": task_graph.workflow,
            "monitor": self.monitor.snapshot(),
            "message_log": [
                {"kind": m.kind, "sender": m.sender, "recipient": m.recipient}
                for m in self.message_bus.get_log()
            ],
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
