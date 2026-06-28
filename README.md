# AI CEO: Strategic Intelligence Agent

A retrieval-augmented, multi-agent strategic intelligence system that monitors public news
about SAP and answers: **"If you were the CEO today, what would you do next, and why?"**

It collects live news from three independent channels, cleans and indexes it, runs four
specialist agents over it (opportunities, risks, trends, competitor activity), then feeds
those outputs into a CEO agent that produces prioritized, evidence-backed recommendations —
all displayed on an executive dashboard with a live agentic chat interface.

Every reasoning step runs on a **local, open-source LLM via Ollama** (llama3.1:8b / phi4-mini
by default). Zero Anthropic, OpenAI, or Gemini API calls anywhere in the pipeline.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA COLLECTION LAYER                     │
│  SAP News RSS   GNews API   Hacker News API                 │
│       ↓               ↓             ↓                       │
│            01_Data_Collection_cleaned/                       │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│               KNOWLEDGE REPOSITORY LAYER                     │
│  Data Cleaning → Chunking → BAAI/bge-small-en-v1.5 Embeds  │
│                      → FAISS IndexFlatIP                     │
│                   (1,400 chunks indexed)                     │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              STRATEGIC INTELLIGENCE ENGINE                   │
│  ┌──────────────┐ ┌────────────┐ ┌──────────┐ ┌────────┐  │
│  │OpportunityAgt│ │  RiskAgent │ │TrendAgent│ │Compet. │  │
│  │(01_opp.ipynb)│ │(02_rsk.ipy)│ │(03_tr.ip)│ │Monitor │  │
│  └──────┬───────┘ └─────┬──────┘ └────┬─────┘ └───┬────┘  │
│         └───────────────┴─────────────┴────────────┘       │
│                          ↓                                   │
│                    AI CEO Agent                              │
│           (05_ceo_agent.ipynb + ceo_chat.py)                │
└─────────────────────────┬───────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│                EXECUTIVE DASHBOARD (app.py)                  │
│  7 sections: Company Overview / Market Intelligence /        │
│  Opportunity Monitor / Risk Monitor / Sentiment Analysis /   │
│  Strategic Recommendations / CEO Briefing                    │
│  + Live agentic chat with reasoning trail                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Agentic Pipeline

The system implements the **Goal → Plan → Retrieve → Analyze → Decide → Recommend → Validate**
workflow required by the professor's clarification:

```
User Question
     │
     ▼
PlannerAgent ──────────────────────────────────────────────────────
│  • Calls run_plan_stage() — LLM states goal + steps (no data yet)
│  • Selects workflow: pipeline / parallel / review
│  • Builds a TaskGraph with dependency-ordered tasks
     │
     ▼
Retrieve Stage (run_retrieval_stage)
│  • LLM decides which tools to call from ToolRegistry
│  • Tools: retrieve_news, get_opportunities, get_risks,
│           get_trends, get_competitor_activity
│  • LLM decides how many rounds; signals done via finished_retrieving
│  • AgenticRetriever.adaptive_retrieve() retries on low-confidence results
     │
     ▼
Analyze Stage (run_analyze_stage)
│  • Forced tool call: model produces key_observations
│    grounded ONLY in the retrieved evidence — no deciding yet
     │
     ▼
Decide + Recommend Stage (run_decide_stage)
│  • Forced tool call: model produces structured recommendation
│    (executive_summary, supporting_evidence, recommended_actions,
│     priority_level, why_this_matters, risks)
│  • Every field must be traceable to retrieved evidence
     │
     ▼ [review workflow only]
CriticAgent
│  • Evaluates draft on 4 axes: evidence grounding, hallucination
│    risk, logical consistency, recommendation specificity
│  • verdict = approved / needs_revision / rejected
│  • If rejected: feedback injected back and decision stage retried
     │
     ▼
Validate Stage (run_validate_stage)
│  • Second, independent LLM call — checks draft against evidence
│  • Returns: Supported / Revised / Unsupported + notes
│  • Confidence score computed from 4 real signals (not hardcoded)
     │
     ▼
Final Answer + Reasoning Trail displayed in Streamlit
```

Every stage is a **separate, logged LLM call** with a forced tool schema — the model cannot
skip stages or merge them. The "Agent reasoning trail" expander in the chat UI shows all
five stages for every answer.

---

## Data Flow

```
SAP News RSS ──────────────────────────────────────────────────────────────────────┐
GNews API ─────────────────────────────────────────────────────────────────────────┤
Hacker News API ────────────────────────────────────────────────────────────────────┤
                                                                                    ↓
                                                                            03_Merging_Data.ipynb
                                                                                    ↓
                                                                            clean_data.json (250 articles)
                                                                                    ↓
                                                                            embedding_v2.ipynb
                                                                            (chunk + BAAI/bge embed)
                                                                                    ↓
                                                                 ┌──────────────────┴──────────────────┐
                                                                 ↓                                     ↓
                                                         chunked_data.json                sap_intelligence.index
                                                         (1,400 chunks)                   (FAISS IndexFlatIP)
                                                                 │                                     │
                            ┌────────────────────────────────────┴─────────────────────────────────────┘
                            ↓
                 ┌──────────┴───────────┬─────────────────┬──────────────────┐
                 ↓                      ↓                  ↓                  ↓
         opportunities.json         risks.json        trends.json   competitor_activity.json
         (3 items, validated)   (3 items, validated) (3 items)     (8 competitors, validated)
                 └──────────┬───────────┘
                            ↓
                    recommendations.json  +  ceo_briefing.json
                            ↓
                        app.py (Streamlit Dashboard)
```

---

## Technology Stack

| Layer | Technology | Reason |
|---|---|---|
| Data collection | `feedparser`, GNews API, HN Algolia API | Three independent source types; no paid API keys needed |
| Data cleaning | `pandas`, `rapidfuzz` | Fuzzy dedup catches rephrased duplicate headlines across sources |
| Embeddings | `BAAI/bge-small-en-v1.5` (sentence-transformers) | Small, fast, CPU-friendly; on the assignment's approved model list |
| Vector store | FAISS `IndexFlatIP` | Exact cosine search; no approximation needed at 1,400 vectors |
| LLM | Ollama `llama3.1:8b` / `phi4-mini` | Open-source, local, no paid API; required by assignment constraints |
| Sentiment | VADER (`vaderSentiment`) | Fast, deterministic, no per-article LLM call; good enough for directional view |
| Dashboard | Streamlit | Fastest path to an interactive executive dashboard in Python |
| Base agent abstraction | `agents/base_agent.py` | LLM-backed reflect() and validate() on every agent |
| Planner | `pipeline/agentic_orchestrator.PlannerAgent` | Builds TaskGraph, selects pipeline/parallel/review workflow |
| Critic | `pipeline/agentic_orchestrator.CriticAgent` | 4-axis evaluation; can trigger decision retry |
| Memory | `memory/base_memory.py` | WorkingMemory, EpisodicMemory (recency-ranked), SemanticMemory (embedding search) |
| Tool registry | `tools/registry.py` | Typed tool specs, Ollama schema generation, goal-aware selection |
| Message bus | `pipeline/message_bus.py` | 11-kind typed messages (TASK, CRITIQUE, MEMORY_WRITE…), audit log |
| Adaptive retrieval | `rag/agentic_retrieval.AgenticRetriever` | Real confidence from FAISS distances; confidence-gated query expansion |

---

## AI Pipeline (RAG + Agentic)

### Data pipeline (run once)
1. **Collect**: three notebooks pull from SAP News RSS, GNews API, Hacker News API → raw JSON
2. **Merge + clean**: dedup via fuzzy matching, SAP relevance filter, `clean_data.json` (250 docs)
3. **Chunk + embed**: sliding-window chunking → BAAI/bge-small-en-v1.5 embeddings → FAISS index (1,400 chunks)
4. **Specialist agents**: each notebook queries FAISS with domain-specific prompts → validated JSON outputs
5. **CEO synthesis**: reads all four specialist outputs → `recommendations.json` + `ceo_briefing.json`

### Chat pipeline (per question, live)
1. **Plan**: LLM states goal + 3–5 concrete steps *before* touching any data
2. **Retrieve**: LLM autonomously calls available tools until it signals `finished_retrieving`
3. **Analyze**: LLM summarizes what the evidence *actually shows* (forced tool call, no deciding yet)
4. **Decide + Recommend**: LLM produces structured recommendation grounded in plan + evidence + analysis
5. **[Optional] Critic**: CriticAgent evaluates evidence grounding, hallucination risk, consistency
6. **Validate**: second independent LLM call checks draft against evidence; can mark Unsupported

### Confidence computation (real, not hardcoded)
`confidence = retrieval_coverage (0–0.30) + evidence_agreement (0–0.30) + validation_status (0–0.20) + critic_verdict (0–0.20)`

---

## Agentic Components

| Component | File | Behaviour |
|---|---|---|
| `PlannerAgent` | `pipeline/agentic_orchestrator.py` | Decomposes goal into TaskGraph; selects workflow strategy |
| `CriticAgent` | `pipeline/agentic_orchestrator.py` | Evaluates recommendations on 4 axes; can trigger retry |
| `AgenticOrchestrator` | `pipeline/agentic_orchestrator.py` | Drives task graph; wires MessageBus + ToolRegistry |
| `TaskGraph` / `Task` | `pipeline/agentic_orchestrator.py` | Dependency-ordered task tracking with retry logic |
| `ExecutionMonitor` | `pipeline/agentic_orchestrator.py` | Records every tool call, state change, planner decision |
| `AgenticRetriever` | `rag/agentic_retrieval.py` | Real confidence from FAISS distances; adaptive query expansion |
| `SemanticMemory` | `memory/base_memory.py` | Embedding-based search over prior insights |
| `EpisodicMemory` | `memory/base_memory.py` | Recency + keyword ranked prior action history |
| `ToolRegistry` | `tools/registry.py` | Typed specs; Ollama schema generation; goal-aware selection |
| `MessageBus` | `pipeline/message_bus.py` | 11-kind typed messages; request-reply; audit log |
| `BaseAgent.reflect()` | `agents/base_agent.py` | LLM evaluates its own output; signals should_revise |
| `BaseAgent.validate()` | `agents/base_agent.py` | LLM checks output against evidence context |

---

## Design Decisions

**Single knowledge repository.** All three sources feed into one `clean_data.json` → one FAISS
index. Competitor analysis uses the same index queried by competitor name, not a separate DB.

**SAP relevance filter.** Early collection queries pulled in articles with zero connection to SAP.
Fixed by dropping rows that don't mention "SAP" as a whole word.

**Separation of pipeline stages.** Each stage (Plan, Retrieve, Analyze, Decide, Validate) is a
distinct LLM call with a forced tool schema. This prevents the model from silently merging
stages and makes the reasoning trail inspectable.

**VADER over LLM for sentiment.** Scoring 250 articles individually through a local CPU model
would be slow and non-deterministic. VADER is instant, consistent, and directionally accurate.

**CriticAgent on review-mode questions only.** Questions containing "recommend", "strategy",
"risk", "should we" trigger `workflow=review` and invoke the CriticAgent. Factual lookups use
the faster `pipeline` mode to reduce latency.

**Real confidence, not hardcoded.** The previous `confidence = 0.5` stub is replaced by a
four-signal formula that captures actual retrieval quality, analysis depth, validation outcome,
and critic verdict.

**Local LLM only.** Every reasoning step calls `http://localhost:11434` — no LangChain wrappers,
no commercial API SDKs. The assignment constraint (no paid APIs) is enforced at every call site.

---

## Running the Project

```bash
# 1. install Python dependencies
pip install -r requirements.txt

# 2. install Ollama and pull the model
curl https://ollama.ai/install.sh | sh
ollama pull llama3.1:8b

# 3. run the full data + agent pipeline
jupyter nbconvert --to notebook --execute --inplace main.ipynb

# 4. launch the dashboard
streamlit run app.py
```

Or open `main.ipynb` and run cell by cell to inspect each stage's output.

---

## Running Tests

```bash
python -m unittest -q tests.test_agentic_architecture
```

Covers: working memory round-trip, tool registry dispatch, orchestrator end-to-end with stubbed core.

---

## Project Structure

```
01_Data_Collection_cleaned/       data collection notebooks
  01_sap_news.ipynb               SAP News RSS feed
  02_gnew.ipynb                   GNews API (financial + industry news)
  03_hackernews.ipynb             Hacker News API (community/tech)

notebook/
  data_cleaning.ipynb             merge + clean 3 raw sources
  embedding_v2.ipynb              chunking, BAAI/bge embeddings, FAISS index
  data/                           all generated JSON + index files

agents/
  agent_core.py                   Plan/Retrieve/Analyze/Decide/Validate primitives
  base_agent.py                   BaseAgent with LLM-backed reflect() + validate()
  01_opportunity_agent.ipynb
  02_risk_agent.ipynb
  03_trend_agent.ipynb
  04_competitor_monitor.ipynb
  05_ceo_agent.ipynb

pipeline/
  agentic_orchestrator.py         PlannerAgent, CriticAgent, AgenticOrchestrator, TaskGraph
  message_bus.py                  MessageBus with 11 message kinds + audit log

memory/
  base_memory.py                  WorkingMemory, EpisodicMemory, SemanticMemory

rag/
  agentic_retrieval.py            AgenticRetriever with real confidence + adaptive retrieval

tools/
  registry.py                     ToolRegistry with Ollama schema generation

config/
  settings.py                     centralized config (OLLAMA_HOST, model, paths)

tests/
  test_agentic_architecture.py    unit tests

ceo_chat.py                       ask_ceo() — connects chat to full pipeline
app.py                            Streamlit dashboard (7 sections + agentic chat)
main.ipynb                        orchestrates full pipeline end-to-end
```
