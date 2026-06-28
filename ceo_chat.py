"""
CEO Chat — Autonomous Goal-Driven Agentic Pipeline

The Planner Agent examines the question and autonomously decides which specialist
agents, live data sources, and knowledge bases to invoke. Available intelligence tools:

  1. retrieve_knowledge_base     — FAISS semantic search over indexed SAP news corpus
  2. fetch_live_news             — Real-time news from Hacker News + SAP News RSS
  3. run_opportunity_agent       — SAP Opportunity Intelligence Agent
  4. run_risk_agent              — SAP Risk Intelligence Agent
  5. run_trend_agent             — SAP Technology Trend Agent
  6. run_competitor_intelligence — Targeted Competitor Intelligence Agent (live + KB)
  7. recall_memory               — Prior analysis recalled from persistent agent memory

Full pipeline per question:
  Memory Recall → Plan → Retrieve (autonomous tool selection) →
  Analyze → Decide → Critic [review mode] → Validate → Answer
"""
import json
import os
import sys
from pathlib import Path

import requests

from memory.base_memory import EpisodicMemory, SemanticMemory
from pipeline.agentic_orchestrator import CriticAgent
from rag.agentic_retrieval import AgenticRetriever


def _locate_and_import_agent_core():
    """Find agent_core.py regardless of cwd (Streamlit, terminal, or IDE)."""
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
        + f". cwd: {here}"
    )


core = _locate_and_import_agent_core()

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "notebook" / "data"
MEMORY_DIR = BASE_DIR / "memory_store"
MEMORY_DIR.mkdir(exist_ok=True)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# More iterations to support 7-tool set; agent signals done via finished_retrieving
MAX_RETRIEVAL_ITERATIONS = 8

# Injected into every pipeline stage so the agent never loses whose perspective it holds.
# Without this, a question naming a competitor prominently can make the model drift into
# advising the competitor instead of SAP.
PERSONA_CONTEXT = (
    "You are the Strategic Intelligence Agent working FOR SAP. Every "
    "answer you give — risks, opportunities, recommended actions, "
    "executive summary — must be framed from SAP's perspective and serve "
    "SAP's interests. Even when a question is primarily about a "
    "competitor (e.g. 'How is Oracle positioning against SAP?'), you are "
    "analyzing what that competitor's activity MEANS FOR SAP's strategy — "
    "you are never advising the competitor itself, and 'we'/'our' in any "
    "question always refers to SAP, regardless of what was discussed in "
    "earlier turns."
)

# ---------------------------------------------------------------------------
# Shared resources (lazy-loaded on first use)
# ---------------------------------------------------------------------------
_retriever = AgenticRetriever(data_dir=DATA_DIR)
_semantic_memory = SemanticMemory(storage_path=MEMORY_DIR / "semantic_memory.json")
_episodic_memory = EpisodicMemory(storage_path=MEMORY_DIR / "episodic_memory.json")
_critic = CriticAgent()

KNOWN_COMPETITORS = [
    "Oracle", "Microsoft", "Workday", "Salesforce",
    "ServiceNow", "Infor", "IFS", "Epicor",
]


def _load_json(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Tool 1: FAISS Knowledge Base
# ---------------------------------------------------------------------------

def tool_retrieve_knowledge_base(query: str, k: int = 5) -> list:
    """Adaptive semantic retrieval from the indexed SAP news corpus (1,400 chunks)."""
    result = _retriever.adaptive_retrieve(query, k=k)
    return result.documents


# ---------------------------------------------------------------------------
# Tool 2: Live News Fetching
# ---------------------------------------------------------------------------

def tool_fetch_live_news(query: str, max_results: int = 8) -> list:
    """Fetch current articles from Hacker News Algolia API and SAP News RSS feed."""
    articles: list = []

    # Hacker News Algolia — no API key required
    try:
        resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": f"SAP {query}", "tags": "story", "hitsPerPage": max_results},
            timeout=10,
        )
        if resp.ok:
            for hit in resp.json().get("hits", []):
                articles.append({
                    "source": "Hacker News (live)",
                    "title": hit.get("title", ""),
                    "text": (hit.get("story_text") or hit.get("title", ""))[:500],
                    "published": hit.get("created_at", ""),
                    "url": hit.get("url", ""),
                })
    except Exception:
        pass

    # SAP News RSS — no API key required
    try:
        import feedparser  # already in notebooks' requirements
        feed = feedparser.parse("https://news.sap.com/feed/")
        query_lower = query.lower()
        for entry in feed.entries[:30]:
            combined = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
            if not query or query_lower in combined or "sap" in combined:
                articles.append({
                    "source": "SAP News (live)",
                    "title": entry.get("title", ""),
                    "text": entry.get("summary", "")[:500],
                    "published": entry.get("published", ""),
                    "url": entry.get("link", ""),
                })
            if len(articles) >= max_results:
                break
    except Exception:
        pass

    if not articles:
        return [{"note": f"No live articles retrieved for '{query}'. Knowledge base will be used as fallback."}]
    return articles[:max_results]


# ---------------------------------------------------------------------------
# Tool 3: Opportunity Intelligence Agent
# ---------------------------------------------------------------------------

def tool_run_opportunity_agent(focus_area: str = "") -> dict:
    """SAP Opportunity Intelligence Agent — validated opportunities + targeted FAISS evidence."""
    base = _load_json("opportunities.json")
    enrichment: list = []
    if focus_area:
        enrichment = _retriever.adaptive_retrieve(
            f"SAP opportunity growth {focus_area}", k=4
        ).documents
    return {
        "agent": "opportunity_intelligence_agent",
        "focus_area": focus_area or "all opportunities",
        "validated_opportunities": base,
        "supporting_evidence": enrichment,
        "source": "SAP Strategic Knowledge Base + FAISS corpus",
    }


# ---------------------------------------------------------------------------
# Tool 4: Risk Intelligence Agent
# ---------------------------------------------------------------------------

def tool_run_risk_agent(focus_area: str = "") -> dict:
    """SAP Risk Intelligence Agent — validated risks + targeted FAISS threat evidence."""
    base = _load_json("risks.json")
    enrichment: list = []
    if focus_area:
        enrichment = _retriever.adaptive_retrieve(
            f"SAP risk threat vulnerability {focus_area}", k=4
        ).documents
    return {
        "agent": "risk_intelligence_agent",
        "focus_area": focus_area or "all risks",
        "validated_risks": base,
        "supporting_evidence": enrichment,
        "source": "SAP Strategic Knowledge Base + FAISS corpus",
    }


# ---------------------------------------------------------------------------
# Tool 5: Trend Intelligence Agent
# ---------------------------------------------------------------------------

def tool_run_trend_agent(technology_area: str = "") -> dict:
    """SAP Trend Intelligence Agent — validated trends + targeted FAISS market evidence."""
    base = _load_json("trends.json")
    enrichment: list = []
    if technology_area:
        enrichment = _retriever.adaptive_retrieve(
            f"SAP technology market trend {technology_area}", k=4
        ).documents
    return {
        "agent": "trend_intelligence_agent",
        "technology_area": technology_area or "all trends",
        "validated_trends": base,
        "supporting_evidence": enrichment,
        "source": "SAP Strategic Knowledge Base + FAISS corpus",
    }


# ---------------------------------------------------------------------------
# Tool 6: Competitor Intelligence Agent
# ---------------------------------------------------------------------------

def tool_run_competitor_intelligence(competitor_name: str) -> dict:
    """Targeted Competitor Intelligence Agent: live news + FAISS KB + pre-analyzed data."""
    all_competitors = _load_json("competitor_activity.json")
    known_entry = next(
        (c for c in all_competitors
         if competitor_name.lower() in c.get("competitor", "").lower()),
        None,
    )

    # Real-time intelligence fetch — makes this agent genuinely live
    live_news = tool_fetch_live_news(
        f"{competitor_name} enterprise software ERP cloud strategy", max_results=6
    )

    # FAISS knowledge base evidence for this specific competitor
    kb_docs = _retriever.adaptive_retrieve(
        f"{competitor_name} SAP competitive positioning market share strategy", k=5
    ).documents

    return {
        "agent": "competitor_intelligence_agent",
        "competitor": competitor_name,
        "pre_analyzed_activity": known_entry,
        "live_news": live_news,
        "knowledge_base_evidence": kb_docs,
        "known_competitors": KNOWN_COMPETITORS,
    }


# ---------------------------------------------------------------------------
# Tool 7: Memory Recall
# ---------------------------------------------------------------------------

def tool_recall_memory(query: str) -> list:
    """Recall prior intelligence and insights on this topic from persistent agent memory."""
    semantic = _semantic_memory.search(query, top_k=3)
    episodic = _episodic_memory.search(query, top_k=3)
    # Strip internal embedding vectors before returning
    return [
        {k: v for k, v in e.items() if k not in ("_embedding",)}
        for e in (semantic + episodic)
        if e
    ]


# ---------------------------------------------------------------------------
# Tool registry — handlers + schemas
# ---------------------------------------------------------------------------

RETRIEVAL_TOOL_HANDLERS = {
    "retrieve_knowledge_base": lambda args: tool_retrieve_knowledge_base(
        args.get("query", ""), int(args.get("k", 5) or 5)
    ),
    "fetch_live_news": lambda args: tool_fetch_live_news(
        args.get("query", ""), int(args.get("max_results", 8) or 8)
    ),
    "run_opportunity_agent": lambda args: tool_run_opportunity_agent(
        args.get("focus_area", "")
    ),
    "run_risk_agent": lambda args: tool_run_risk_agent(
        args.get("focus_area", "")
    ),
    "run_trend_agent": lambda args: tool_run_trend_agent(
        args.get("technology_area", "")
    ),
    "run_competitor_intelligence": lambda args: tool_run_competitor_intelligence(
        args.get("competitor_name", "")
    ),
    "recall_memory": lambda args: tool_recall_memory(args.get("query", "")),
    # Backward-compat aliases so any cached LLM prompt using old names still works
    "retrieve_news": lambda args: tool_retrieve_knowledge_base(
        args.get("query", ""), int(args.get("k", 5) or 5)
    ),
    "get_opportunities": lambda _: tool_run_opportunity_agent(),
    "get_risks": lambda _: tool_run_risk_agent(),
    "get_trends": lambda _: tool_run_trend_agent(),
    "get_competitor_activity": lambda _: _load_json("competitor_activity.json"),
}

RETRIEVAL_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_knowledge_base",
            "description": (
                "Semantic search over the FAISS-indexed SAP news knowledge base "
                "(1,400 chunks: SAP News, GNews, Hacker News). Use for background "
                "context, historical events, product details, or any topic where "
                "the knowledge base is sufficient and real-time data is not needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "k": {"type": "integer", "description": "Chunks to return (default 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_live_news",
            "description": (
                "Fetch CURRENT, REAL-TIME articles from live sources (Hacker News, "
                "SAP News RSS). Use when the question asks about 'today', 'latest', "
                "'current', 'recent', 'now', 'this week', or when the knowledge base "
                "may be stale. Also useful for fast-moving topics like AI or cloud."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic or keywords to search for"},
                    "max_results": {"type": "integer", "description": "Articles to return (default 8)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_opportunity_agent",
            "description": (
                "Launch the SAP Opportunity Intelligence Agent. Returns validated "
                "strategic opportunities with supporting evidence. "
                "Use when the question is about SAP's growth, market opportunities, "
                "new business areas, expansion, or 'what opportunities does SAP have'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_area": {
                        "type": "string",
                        "description": (
                            "Specific opportunity domain to focus on "
                            "(e.g. 'cloud ERP', 'AI', 'SMB market', 'emerging markets'). "
                            "Leave blank for all opportunities."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_risk_agent",
            "description": (
                "Launch the SAP Risk Intelligence Agent. Returns validated business "
                "risks and threats with evidence. Use when the question is about what "
                "SAP should worry about, vulnerabilities, competitive threats, "
                "regulatory risks, or 'what are the biggest risks for SAP'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "focus_area": {
                        "type": "string",
                        "description": (
                            "Specific risk domain "
                            "(e.g. 'cybersecurity', 'competition', 'regulatory', "
                            "'cloud migration'). Leave blank for all risks."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_trend_agent",
            "description": (
                "Launch the SAP Technology Trend Agent. Returns validated emerging "
                "technology and market trends. Use when the question is about future "
                "directions, technology shifts, market developments, or "
                "'what trends should SAP watch'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "technology_area": {
                        "type": "string",
                        "description": (
                            "Specific technology area "
                            "(e.g. 'AI', 'cloud', 'automation', 'sustainability'). "
                            "Leave blank for all trends."
                        ),
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_competitor_intelligence",
            "description": (
                "Launch the Competitor Intelligence Agent for a SPECIFIC competitor. "
                "Fetches LIVE news AND retrieves knowledge base evidence for that "
                "company. Use when asked about: Oracle, Microsoft, Workday, Salesforce, "
                "ServiceNow, Infor, IFS, or Epicor. Call once per competitor. "
                "For 'what are all competitors doing', call for the 2-3 most relevant ones."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "competitor_name": {
                        "type": "string",
                        "description": "Exact name: Oracle, Microsoft, Workday, Salesforce, ServiceNow, Infor, IFS, or Epicor",
                    },
                },
                "required": ["competitor_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": (
                "Recall prior analysis and intelligence from the agent's persistent "
                "memory stores. Use EARLY in retrieval to check whether this topic "
                "was analyzed before, so you can build on prior work rather than "
                "starting from scratch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to recall from memory"},
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# CEO answer schemas (decide + validate)
# ---------------------------------------------------------------------------

CEO_DECIDE_TOOL_NAME = "finalize_ceo_answer"
CEO_DECIDE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CEO_DECIDE_TOOL_NAME,
        "description": "Submit your final structured answer to the user's strategic question.",
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
                        "2-4 sentences describing what the evidence you retrieved actually shows. "
                        "Do NOT include tool call syntax, function names, or citation brackets "
                        "like [[tool|args=...]]. Write as plain prose for an executive."
                    ),
                },
                "opportunities": {
                    "type": "string",
                    "description": "Relevant opportunities for SAP in plain prose. Leave blank if not relevant.",
                },
                "risks": {
                    "type": "string",
                    "description": "Relevant risks TO SAP (not to a competitor) in plain prose. Leave blank if not relevant.",
                },
                "recommended_actions": {
                    "type": "string",
                    "description": "Concrete, specific actions SAP should take, in plain prose.",
                },
                "priority_level": {"type": "string", "description": "High, Medium, or Low"},
                "why_this_matters": {
                    "type": "string",
                    "description": "1-2 sentences on the business significance.",
                },
            },
            "required": [
                "executive_summary", "supporting_evidence",
                "recommended_actions", "priority_level", "why_this_matters",
            ],
        },
    },
}

CEO_VALIDATE_TOOL_NAME = "submit_validated_ceo_answer"
CEO_VALIDATE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": CEO_VALIDATE_TOOL_NAME,
        "description": "Check the draft answer against evidence and submit a validated final answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "string",
                    "description": "2-3 sentences directly answering the question.",
                },
                "supporting_evidence": {
                    "type": "string",
                    "description": "2-4 sentences on what the evidence shows. No tool syntax.",
                },
                "opportunities": {
                    "type": "string",
                    "description": "Relevant opportunities for SAP in plain prose. Leave blank if not relevant.",
                },
                "risks": {
                    "type": "string",
                    "description": "Relevant risks TO SAP in plain prose. Leave blank if not relevant.",
                },
                "recommended_actions": {
                    "type": "string",
                    "description": "Concrete, specific actions SAP should take.",
                },
                "priority_level": {"type": "string", "description": "High, Medium, or Low"},
                "why_this_matters": {
                    "type": "string",
                    "description": "1-2 sentences on business significance.",
                },
                "validation_status": {
                    "type": "string",
                    "description": "Supported, Revised, or Unsupported",
                },
                "validation_notes": {
                    "type": "string",
                    "description": "1 short sentence explaining validation. Empty string only if fully Supported.",
                },
            },
            "required": [
                "executive_summary", "supporting_evidence",
                "recommended_actions", "priority_level", "why_this_matters",
                "validation_status", "validation_notes",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_workflow(question: str) -> str:
    """Select pipeline / review / parallel based on question semantics.

    review   — strategic decisions, competitor analysis, risk assessment;
               triggers the CriticAgent after the decide stage.
    parallel — broad intelligence requests covering multiple domains.
    pipeline — default; factual questions, specific topic lookups.
    """
    q = question.lower()
    review_signals = (
        "should", "recommend", "strategy", "strategically", "decision",
        "risk", "threat", "versus", " vs ", "compare", "competitor",
        "oracle", "microsoft", "workday", "salesforce", "servicenow",
        "infor", "ifs", "epicor", "positioning", "beat", "win against",
    )
    if any(w in q for w in review_signals):
        return "review"
    parallel_signals = (
        "overview", "comprehensive", "everything", "all about", "summary of",
        "today", "latest", "current", "opportunity", "opportunities", "landscape",
        "what is happening", "state of",
    )
    if any(w in q for w in parallel_signals):
        return "parallel"
    return "pipeline"


def _format_answer(d: dict) -> str:
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

    meta: list[str] = []
    if d.get("validation_status"):
        notes = (d.get("validation_notes") or "").strip()
        line = f"Validation: **{d['validation_status']}**"
        if notes:
            line += f" — {notes}"
        meta.append(line)
    if d.get("_workflow"):
        label = {"review": "review (Critic engaged)", "parallel": "parallel", "pipeline": "pipeline"}.get(
            d["_workflow"], d["_workflow"]
        )
        meta.append(f"Workflow: **{label}**")
    if d.get("_tools_called"):
        meta.append(f"Agents/tools called: **{d['_tools_called']}**")

    if meta:
        parts.append("\n---\n*" + " · ".join(meta) + "*")

    return "\n\n".join(parts) if parts else (
        "I wasn't able to produce a validated answer for this question — try rephrasing it."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def ask_ceo(question, history=None, trace=None, on_stage=None):
    """
    Autonomous Agentic Pipeline — one chat turn.

    Stages:
      0. Memory Recall  — check persistent memory for prior insights before planning
      1. Plan           — LLM states goal + concrete steps (no data access yet)
      2. Retrieve       — LLM autonomously selects tools; calls specialist agents,
                          live news, FAISS KB, and/or memory as needed
      3. Analyze        — LLM summarises what the evidence actually shows
      4. Decide         — LLM produces structured recommendation grounded in evidence
      4b. Critic        — [review mode only] CriticAgent evaluates on 4 axes; can
                          trigger one retry of the decide stage with feedback injected
      5. Validate       — Second independent LLM call checks draft against evidence

    history  : list[{"role": str, "content": str}] — prior turns for continuity
    trace    : mutable list this function appends to (one entry per stage)
    on_stage : callback(stage_name: str) for live UI progress updates
    """
    history_text = ""
    if history:
        history_text = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    workflow = _detect_workflow(question)

    def _log(stage: str, payload):
        if trace is not None:
            trace.append({"stage": stage, "payload": payload})

    def _notify(stage: str):
        if on_stage is not None:
            on_stage(stage)

    try:
        # ── STAGE 0: MEMORY RECALL ────────────────────────────────────────
        # Query persistent memory BEFORE planning so the planner can reference
        # prior intelligence and avoid redundant work.
        _notify("memory_recall")
        prior_knowledge = tool_recall_memory(question)
        _log("memory_recall", prior_knowledge)

        memory_context = ""
        if prior_knowledge:
            clean = [
                {k: v for k, v in e.items() if k not in ("_embedding", "_similarity")}
                for e in prior_knowledge[:2]
            ]
            memory_context = (
                "\n\nRelevant prior intelligence retrieved from agent memory:\n"
                + json.dumps(clean, default=str)[:600]
            )

        goal_description = (
            (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
            + f"Current question from the user: {question}"
            + memory_context
        )

        # ── STAGE 1: PLAN ─────────────────────────────────────────────────
        # LLM states goal + concrete steps before accessing any data.
        _notify("plan")
        plan = core.run_plan_stage(
            goal_description, persona_context=PERSONA_CONTEXT, verbose=False
        )
        _log("plan", plan)

        # ── STAGE 2: RETRIEVE ─────────────────────────────────────────────
        # LLM autonomously selects which of the 7 tools to call, with what
        # arguments, and how many rounds; signals done via finished_retrieving.
        _notify("retrieve")
        retrieval_trace = core.run_retrieval_stage(
            plan,
            RETRIEVAL_TOOL_SCHEMAS,
            RETRIEVAL_TOOL_HANDLERS,
            persona_context=PERSONA_CONTEXT,
            max_iterations=MAX_RETRIEVAL_ITERATIONS,
            verbose=False,
        )
        _log("retrieve", retrieval_trace)
        evidence_text = core.build_evidence_text(retrieval_trace, max_chars=8000)

        # Track which agents/tools were actually called (for the answer footer)
        tools_called = [
            e.get("tool") for e in retrieval_trace
            if e.get("tool") and e.get("tool") != "finished_retrieving"
        ]
        unique_tools = list(dict.fromkeys(tools_called))  # preserve order, deduplicate

        # ── STAGE 3: ANALYZE ─────────────────────────────────────────────
        # LLM summarises key observations from evidence — no deciding yet.
        _notify("analyze")
        analysis = core.run_analyze_stage(
            plan,
            evidence_text,
            core.make_analyze_tool_schema("recommendations"),
            persona_context=PERSONA_CONTEXT,
            verbose=False,
        )
        _log("analyze", analysis)

        # ── STAGE 4: DECIDE + RECOMMEND ───────────────────────────────────
        # LLM produces structured recommendation grounded in plan + evidence + analysis.
        _notify("decide")
        draft = core.run_decide_stage(
            plan, evidence_text, analysis,
            CEO_DECIDE_TOOL_SCHEMA, CEO_DECIDE_TOOL_NAME,
            persona_context=PERSONA_CONTEXT, verbose=False,
        ) or {}
        _log("decide_recommend", draft)

        # ── STAGE 4b: CRITIC [review mode only] ──────────────────────────
        # CriticAgent evaluates evidence grounding, hallucination risk,
        # logical consistency, and recommendation specificity.
        # If verdict == "rejected", critic feedback is injected into the
        # evidence context and the decide stage runs once more.
        if workflow == "review" and draft:
            _notify("critique")
            critique = _critic.execute(
                "evaluate",
                context={"draft": draft, "evidence_text": evidence_text},
            )
            _log("critique", critique)

            if critique.get("verdict") == "rejected":
                feedback = critique.get("feedback", "")
                augmented_evidence = (
                    evidence_text
                    + f"\n\nCritic feedback on prior draft (must be addressed):\n{feedback}"
                )
                revised = core.run_decide_stage(
                    plan, augmented_evidence, analysis,
                    CEO_DECIDE_TOOL_SCHEMA, CEO_DECIDE_TOOL_NAME,
                    persona_context=PERSONA_CONTEXT, verbose=False,
                )
                if revised:
                    draft = revised
                    _log("decide_recommend_revised", draft)

        # ── STAGE 5: VALIDATE ─────────────────────────────────────────────
        # Second, independent LLM call: checks draft against evidence.
        # Can mark Supported / Revised / Unsupported without inventing new evidence.
        _notify("validate")
        validated = core.run_validate_stage(
            draft, evidence_text,
            CEO_VALIDATE_TOOL_SCHEMA, CEO_VALIDATE_TOOL_NAME,
            persona_context=PERSONA_CONTEXT, verbose=False,
        )
        if validated:
            # The validate stage sometimes drops content fields when marking Unsupported.
            # Start from draft and only override fields the validator actually filled in.
            final = dict(draft)
            for key, value in validated.items():
                if value:
                    final[key] = value
        elif draft:
            final = core.attach_default_validation([dict(draft)])[0]
        else:
            final = {}
        _log("validate", final)

        # Metadata for answer footer
        final["_workflow"] = workflow
        final["_tools_called"] = ", ".join(unique_tools) if unique_tools else "none"

        # ── MEMORY WRITE ──────────────────────────────────────────────────
        # Persist this interaction so future questions can recall prior intelligence.
        _episodic_memory.write({
            "question": question,
            "workflow": workflow,
            "tools_called": unique_tools,
            "executive_summary": final.get("executive_summary", ""),
            "validation_status": final.get("validation_status", ""),
        })
        _semantic_memory.write({
            "question": question,
            "summary": final.get("executive_summary", ""),
            "recommended_actions": final.get("recommended_actions", ""),
            "workflow": workflow,
        })

        return _format_answer(final)

    except requests.exceptions.RequestException as e:
        return f"Error calling Ollama: {str(e)}"
