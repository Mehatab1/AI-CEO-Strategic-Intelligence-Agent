import tempfile
import types
import unittest
from pathlib import Path

from memory import EpisodicMemory, SemanticMemory, WorkingMemory
from pipeline.agentic_orchestrator import AgenticOrchestrator
from tools.registry import ToolRegistry, ToolSpec


class AgenticArchitectureTests(unittest.TestCase):
    def test_working_memory_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = WorkingMemory(storage_path=Path(tmpdir) / "working.json")
            memory.write({"content": "alpha"})
            entries = memory.read_all()

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["content"], "alpha")

    def test_tool_registry_dispatches_registered_handlers(self):
        registry = ToolRegistry()
        registry.register(
            ToolSpec(
                name="echo",
                description="Echo tool",
                inputs={"value": "string"},
                outputs={"result": "string"},
                cost=0.0,
                latency=0.0,
                handler=lambda args: {"result": args["value"]},
            )
        )

        result = registry.dispatch("echo", {"value": "hi"})
        self.assertEqual(result["result"], "hi")

    def test_orchestrator_runs_with_stubbed_core(self):
        def fake_plan(goal, persona_context="", verbose=True):
            return {"goal": goal, "planned_steps": ["inspect evidence"]}

        def fake_retrieve(plan, tool_schemas, tool_handlers, persona_context="", max_iterations=4, verbose=True):
            return [{"tool": "retrieve_news", "arguments": {"query": "sap ai"}}]

        def fake_analyze(plan, evidence_text, schema, persona_context="", verbose=True):
            return {"key_observations": ["evidence indicates growth"]}

        def fake_decide(plan, evidence_text, analysis, schema, tool_name, persona_context="", verbose=True):
            return {
                "executive_summary": "SAP should focus on AI growth.",
                "supporting_evidence": "Evidence supports this.",
                "recommended_actions": "Prioritize AI investments.",
                "priority_level": "High",
                "why_this_matters": "This drives competitiveness.",
            }

        def fake_validate(draft, evidence_text, schema, tool_name, persona_context="", verbose=True):
            return {
                "validation_status": "Supported",
                "validation_notes": "Evidence was sufficient.",
            }

        fake_core = types.SimpleNamespace(
            run_plan_stage=fake_plan,
            run_retrieval_stage=fake_retrieve,
            run_analyze_stage=fake_analyze,
            run_decide_stage=fake_decide,
            run_validate_stage=fake_validate,
            make_analyze_tool_schema=lambda kind: {},
            attach_default_validation=lambda value: value,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = AgenticOrchestrator(llm_core=fake_core, storage_dir=Path(tmpdir))
            result = orchestrator.run("What should SAP do next?")

            self.assertTrue(result["final_answer"].startswith("## Executive Summary"))
            self.assertGreaterEqual(len(result["trace"]), 5)
            self.assertIsInstance(orchestrator.working_memory, WorkingMemory)
            self.assertIsInstance(orchestrator.episodic_memory, EpisodicMemory)
            self.assertIsInstance(orchestrator.semantic_memory, SemanticMemory)


if __name__ == "__main__":
    unittest.main()
