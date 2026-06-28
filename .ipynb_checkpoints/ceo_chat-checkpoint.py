import json
import os
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


def _locate_and_import_agent_core():
    """Find agent_core.py regardless of where the process's working
    directory ends up (Streamlit, a terminal, or an IDE's run button don't
    all agree on cwd). agent_core.py is expected to live in agents/."""
    here = Path.cwd()
    candidates = [here, here / "agents", here.parent / "agents", here.parent]
    for c in candidates:
        if (c / "agent_core.py").exists():
            sys.path.insert(0, str(c))
            import agent_core as agent_core_module
            return agent_core_module
    raise FileNotFoundError(
        "Could not locate agent_core.py. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + f". Current working directory: {here}"
    )


core = _locate_and_import_agent_core()

# --------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "notebook" / "data"

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
MAX_RETRIEVAL_ITERATIONS = 4  # smaller than the batch notebooks, for chat latency

# Injected into every stage's system prompt (Plan, Retrieve, Analyze, Decide,
# Validate) so the agent never loses track of whose perspective it's working
# from. Without this, a question that names a competitor prominently (e.g.
# "How is Oracle positioning against SAP?") can make the model drift into
# advising the competitor instead of SAP - relying on the model to carry
# this forward through its own restated "goal" text alone is not reliable.
PERSONA_CONTEXT = (
    "You are the Strategic Intelligence Agent working FOR SAP. Every "
    "answer you give - risks, opportunities, recommended actions, "
    "executive summary - must be framed from SAP's perspective and serve "
    "SAP's interests. Even when a question is primarily about a "
    "competitor (e.g. \"How is Oracle positioning against SAP?\"), you are "
    "analyzing what that competitor's activity MEANS FOR SAP's strategy - "
    "you are never advising the competitor itself, and 'we'/'our' in any "
    "question always refers to SAP, regardless of what was discussed in "
    "earlier turns."
)

_model = None
_index = None
_chunks = None


def load_resources():
    """Lazy-load the embedding model, FAISS index, and chunks - only needed
    if the agent actually decides to call retrieve_news during Retrieve."""
    global _model, _index, _chunks

    if _model is None:
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    if _index is None:
        _index = faiss.read_index(str(DATA_DIR / "sap_intelligence.index"))
    if _chunks is None:
        with open(DATA_DIR / "chunked_data.json", "r", encoding="utf-8") as f:
            _chunks = json.load(f)

    return _model, _index, _chunks


def _load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Retrieve-stage tools available to the CEO agent.
# --------------------------------------------------------------------------

def tool_retrieve_news(query: str, k: int = 5):
    model, index, chunks = load_resources()

    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    _, idxs = index.search(query_embedding, k)

    docs = []
    for idx in idxs[0]:
        if idx >= len(chunks):
            continue
        chunk = chunks[idx]
        docs.append({
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "text": chunk.get("text", ""),
        })
    return docs


RETRIEVAL_TOOL_HANDLERS = {
    "retrieve_news": lambda args: tool_retrieve_news(
        args.get("query", ""), int(args.get("k", 5) or 5)
    ),
    "get_opportunities": lambda args: _load_json("opportunities.json"),
    "get_risks": lambda args: _load_json("risks.json"),
    "get_trends": lambda args: _load_json("trends.json"),
    "get_competitor_activity": lambda args: _load_json("competitor_activity.json"),
}

RETRIEVAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_news",
            "description": (
                "Semantic search over the indexed SAP news corpus. Use this for "
                "specific events, products, or topics not already covered by "
                "the pre-computed reports below."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "k": {"type": "integer", "description": "Number of chunks to retrieve (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_opportunities",
            "description": "Pre-extracted, validated business opportunities for SAP.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risks",
            "description": "Pre-extracted, validated business risks for SAP.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trends",
            "description": "Pre-extracted, validated emerging trends relevant to SAP.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_competitor_activity",
            "description": (
                "Pre-extracted, validated competitor activity report "
                "(Oracle, Workday, Salesforce, Microsoft, Infor, IFS, "
                "ServiceNow, Epicor)."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# --------------------------------------------------------------------------
# Decide and Validate tool schemas for the CEO agent's free-form answer.
# --------------------------------------------------------------------------

CEO_DECIDE_TOOL_NAME = "finalize_ceo_answer"
CEO_DECIDE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CEO_DECIDE_TOOL_NAME,
        "description": "Submit your final structured answer to the user's question.",
        "parameters": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 sentences directly answering the question, in plain prose.",
                },
                "supporting_evidence": {
                    "type": "string",
                    "description": (
                        "2-4 sentences describing, in your own plain-prose words, what the "
                        "evidence you retrieved actually shows. Do NOT include tool call "
                        "syntax, function names, arguments, or citation-style brackets like "
                        "[[tool|args=...]] - write it as you would explain it to a person."
                    ),
                },
                "opportunities": {
                    "type": "string",
                    "description": "Relevant opportunities for SAP, in plain prose. Omit or leave blank if not relevant to this question.",
                },
                "risks": {
                    "type": "string",
                    "description": "Relevant risks TO SAP (not to a competitor), in plain prose. Omit or leave blank if not relevant to this question.",
                },
                "recommended_actions": {
                    "type": "string",
                    "description": "Concrete, specific actions SAP should take, in plain prose.",
                },
                "priority_level": {"type": "string", "description": "High, Medium, or Low"},
                "why_this_matters": {
                    "type": "string",
                    "description": "1-2 sentences on the business significance, in plain prose.",
                },
            },
            "required": ["executive_summary", "supporting_evidence", "recommended_actions", "priority_level", "why_this_matters"],
        },
    },
}

CEO_VALIDATE_TOOL_NAME = "submit_validated_ceo_answer"
CEO_VALIDATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CEO_VALIDATE_TOOL_NAME,
        "description": (
            "Check the draft answer against the evidence and submit a "
            "validated, possibly revised, final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 sentences directly answering the question, in plain prose.",
                },
                "supporting_evidence": {
                    "type": "string",
                    "description": (
                        "2-4 sentences describing, in your own plain-prose words, what the "
                        "evidence actually shows. Do NOT include tool call syntax, function "
                        "names, arguments, or citation-style brackets like [[tool|args=...]]."
                    ),
                },
                "opportunities": {
                    "type": "string",
                    "description": "Relevant opportunities for SAP, in plain prose. Omit or leave blank if not relevant.",
                },
                "risks": {
                    "type": "string",
                    "description": "Relevant risks TO SAP (not to a competitor), in plain prose. Omit or leave blank if not relevant.",
                },
                "recommended_actions": {
                    "type": "string",
                    "description": "Concrete, specific actions SAP should take, in plain prose.",
                },
                "priority_level": {"type": "string", "description": "High, Medium, or Low"},
                "why_this_matters": {
                    "type": "string",
                    "description": "1-2 sentences on the business significance, in plain prose.",
                },
                "validation_status": {"type": "string", "description": "Supported, Revised, or Unsupported"},
                "validation_notes": {
                    "type": "string",
                    "description": "1 short sentence explaining the validation status. Use an empty string only if truly Supported with nothing to add.",
                },
            },
            "required": ["executive_summary", "supporting_evidence", "recommended_actions", "priority_level", "why_this_matters", "validation_status", "validation_notes"],
        },
    },
}


def _format_answer(d):
    sections = [
        ("Executive Summary", d.get("executive_summary")),
        ("Supporting Evidence", d.get("supporting_evidence")),
        ("Opportunities", d.get("opportunities")),
        ("Risks", d.get("risks")),
        ("Recommended Actions", d.get("recommended_actions")),
        ("Priority Level", d.get("priority_level")),
        ("Why This Matters", d.get("why_this_matters")),
    ]
    parts = [f"## {title}\n{content}" for title, content in sections if content]

    if d.get("validation_status"):
        notes = (d.get("validation_notes") or "").strip()
        line = f"Validation: **{d['validation_status']}**"
        if notes:
            line += f" — {notes}"
        parts.append(f"\n---\n*{line}*")

    return "\n\n".join(parts) if parts else (
        "I wasn't able to produce a validated answer for this question - try rephrasing it."
    )


def ask_ceo(question, history=None, trace=None, on_stage=None):
    """
    Run the full Goal -> Plan -> Retrieve -> Analyze -> Decide -> Recommend
    -> Validate pipeline for one chat turn.

    history   : list[{"role": "user"|"assistant", "content": str}] - prior
                turns, folded into the Plan stage so the agent has memory
                of the conversation rather than treating every question in
                isolation.
    trace     : optional list this function appends to, one entry per
                pipeline stage ({"stage": name, "payload": ...}), so the
                caller (e.g. the Streamlit UI) can show the full reasoning
                trail.
    on_stage  : optional callback(stage_name: str) called as each stage
                starts, so the UI can show live progress (e.g. update a
                spinner caption).
    """
    history_text = ""
    if history:
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    goal_description = (
        (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
        + f"Current question from the user: {question}"
    )

    def _log(stage, payload):
        if trace is not None:
            trace.append({"stage": stage, "payload": payload})

    def _notify(stage):
        if on_stage is not None:
            on_stage(stage)

    try:
        _notify("plan")
        plan = core.run_plan_stage(goal_description, persona_context=PERSONA_CONTEXT, verbose=False)
        _log("plan", plan)

        _notify("retrieve")
        retrieval_trace = core.run_retrieval_stage(
            plan, RETRIEVAL_TOOL_SCHEMAS, RETRIEVAL_TOOL_HANDLERS,
            persona_context=PERSONA_CONTEXT, max_iterations=MAX_RETRIEVAL_ITERATIONS, verbose=False,
        )
        _log("retrieve", retrieval_trace)
        evidence_text = core.build_evidence_text(retrieval_trace)

        _notify("analyze")
        analysis = core.run_analyze_stage(
            plan, evidence_text, core.make_analyze_tool_schema("recommendations"),
            persona_context=PERSONA_CONTEXT, verbose=False,
        )
        _log("analyze", analysis)

        _notify("decide")
        draft = core.run_decide_stage(
            plan, evidence_text, analysis, CEO_DECIDE_TOOL_SCHEMA, CEO_DECIDE_TOOL_NAME,
            persona_context=PERSONA_CONTEXT, verbose=False,
        ) or {}
        _log("decide_recommend", draft)

        _notify("validate")
        validated = core.run_validate_stage(
            draft, evidence_text, CEO_VALIDATE_TOOL_SCHEMA, CEO_VALIDATE_TOOL_NAME,
            persona_context=PERSONA_CONTEXT, verbose=False,
        )
        if validated:
            # The model sometimes drops content fields when validating - e.g.
            # it marks the answer Unsupported and leaves executive_summary,
            # supporting_evidence, etc. blank instead of echoing the draft
            # back unchanged. Start from the full draft and only let the
            # validate stage override fields it actually filled in, so a
            # verdict of "Unsupported" flags the answer instead of erasing it.
            final = dict(draft)
            for key, value in validated.items():
                if value:
                    final[key] = value
        elif draft:
            final = core.attach_default_validation([dict(draft)])[0]
        else:
            final = {}
        _log("validate", final)

        return _format_answer(final)

    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama: {str(e)}"