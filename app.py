import json
from pathlib import Path
from ceo_chat import ask_ceo
import pandas as pd
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="SAP AI CEO Chatbot", layout="wide")

COMPANY_NAME = "SAP"
INDUSTRY = "Enterprise Software / Cloud ERP"

DATA_DIR = Path(__file__).parent / "notebook" / "data"


def load_json(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_clean_data():
    df = pd.read_json(DATA_DIR / "clean_data.json")
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True, format="mixed")
    return df


@st.cache_data
def score_sentiment(df):
    analyzer = SentimentIntensityAnalyzer()
    texts = (df["title"].fillna("") + ". " + df["clean_text"].fillna("").str.slice(0, 500))
    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = pd.cut(
        df["sentiment_score"],
        bins=[-1, -0.05, 0.05, 1],
        labels=["Negative", "Neutral", "Positive"],
    )
    return df


clean_df = load_clean_data()
opportunities = load_json("opportunities.json")
risks = load_json("risks.json")
trends = load_json("trends.json")
competitors = load_json("competitor_activity.json")







recommendations = load_json("recommendations.json")
briefing = load_json("ceo_briefing.json")

st.title(f"{COMPANY_NAME} AI CEO — Strategic Intelligence Chatbot")
st.caption('"If you were the CEO today, what would you do next and why?" — ask the agent below.')

main_tab, data_tab = st.tabs(["💬 Chat with the CEO Agent", "📊 Supporting Intelligence & Data"])

# ============================================================
# MAIN: Chatbot - this is the primary deliverable.
# ============================================================
with main_tab:
    st.caption(
        "Every answer runs a fully autonomous **Memory Recall → Plan → Retrieve → Analyze "
        "→ Decide → Validate** pipeline. The agent autonomously selects which specialist "
        "agents and data sources to call — Opportunity Agent, Risk Agent, Trend Agent, "
        "Competitor Intelligence Agent, live news, FAISS knowledge base, or memory — "
        "based on your question. For strategic questions the **Critic Agent** evaluates "
        "the draft before validation. Expand 'Agent reasoning trail' to inspect every stage."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "_pending_question" not in st.session_state:
        st.session_state._pending_question = None

    col1, col2 = st.columns([5, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    STAGE_LABELS = {
        "memory_recall": "🧠 Consulting memory...",
        "plan": "📋 Planning...",
        "retrieve": "🔍 Retrieving evidence (autonomous tool selection)...",
        "analyze": "📊 Analyzing evidence...",
        "decide": "✅ Deciding & recommending...",
        "critique": "🔬 Critic evaluating recommendation...",
        "validate": "🔎 Validating recommendation...",
    }

    # Agent/tool display names for the reasoning trail
    _TOOL_LABELS = {
        "retrieve_knowledge_base": "📚 Knowledge Base (FAISS)",
        "fetch_live_news": "🌐 Live News Fetch",
        "run_opportunity_agent": "🚀 Opportunity Intelligence Agent",
        "run_risk_agent": "⚠️ Risk Intelligence Agent",
        "run_trend_agent": "📈 Trend Intelligence Agent",
        "run_competitor_intelligence": "🔭 Competitor Intelligence Agent",
        "recall_memory": "🧠 Memory Recall",
        "retrieve_news": "📚 Knowledge Base (FAISS)",
        "get_opportunities": "🚀 Opportunity Agent",
        "get_risks": "⚠️ Risk Agent",
        "get_trends": "📈 Trend Agent",
        "get_competitor_activity": "🔭 Competitor Monitor",
        "finished_retrieving": None,
    }

    def _render_pipeline(trace):
        if not trace:
            return
        with st.expander(
            "🔧 Agent reasoning trail  "
            "(Memory Recall → Plan → Retrieve → Analyze → Decide → Critic → Validate)"
        ):
            for entry in trace:
                stage = entry.get("stage")
                payload = entry.get("payload") or {}

                if stage == "memory_recall":
                    st.markdown("**0. Memory Recall**")
                    hits = payload if isinstance(payload, list) else []
                    if hits:
                        st.write(f"Found {len(hits)} prior insight(s) — surfaced to planner:")
                        for hit in hits[:3]:
                            label = (
                                hit.get("question")
                                or hit.get("summary")
                                or str(hit)[:120]
                            )
                            st.write(f"• {label[:150]}")
                    else:
                        st.write("*(no prior memory on this topic — starting fresh)*")

                elif stage == "plan":
                    st.markdown("**1. Plan**")
                    st.write("**Goal:**", payload.get("goal"))
                    for step in payload.get("planned_steps", []):
                        st.write("•", step)

                elif stage == "retrieve":
                    st.markdown("**2. Retrieve** — autonomous tool selection")
                    for call in payload:
                        tool = call.get("tool")
                        if tool == "finished_retrieving":
                            st.write("• ✅ *(agent signalled it had gathered enough evidence)*")
                        else:
                            label = _TOOL_LABELS.get(tool, f"`{tool}`")
                            args = call.get("arguments") or {}
                            arg_str = ", ".join(
                                f"{k}={repr(v)}" for k, v in args.items() if v
                            )
                            st.write(f"• {label}" + (f" → `{arg_str}`" if arg_str else ""))

                elif stage == "analyze":
                    st.markdown("**3. Analyze**")
                    for obs in payload.get("key_observations", []):
                        st.write("•", obs)
                    if payload.get("candidate_items"):
                        st.write("**Candidate items identified:**")
                        for item in payload["candidate_items"]:
                            st.write(f"  — {item}")

                elif stage == "decide_recommend":
                    st.markdown("**4. Decide & Recommend** *(draft)*")
                    # Show key fields as readable text rather than raw JSON
                    for field in ("executive_summary", "recommended_actions", "priority_level"):
                        val = payload.get(field)
                        if val:
                            st.write(f"**{field.replace('_', ' ').title()}:** {val}")

                elif stage == "critique":
                    verdict = payload.get("verdict", "unknown")
                    icon = {"approved": "🟢", "needs_revision": "🟡", "rejected": "🔴"}.get(verdict, "⚪")
                    st.markdown(f"**4b. Critic Evaluation** — {icon} **{verdict}**")
                    for axis in (
                        "evidence_grounding", "hallucination_risk",
                        "logical_consistency", "recommendation_quality",
                    ):
                        if payload.get(axis):
                            st.write(f"  • {axis.replace('_', ' ').title()}: **{payload[axis]}**")
                    if payload.get("feedback"):
                        st.write(f"Feedback: {payload['feedback']}")

                elif stage == "decide_recommend_revised":
                    st.markdown("**4c. Revised Decision** *(post-critic)*")
                    val = payload.get("executive_summary") or payload.get("recommended_actions")
                    if val:
                        st.write(val[:300])

                elif stage == "validate":
                    st.markdown("**5. Validate** *(final)*")
                    status = payload.get("validation_status")
                    if status:
                        icon = {"Supported": "✅", "Revised": "🔄", "Unsupported": "❌"}.get(status, "•")
                        st.write(f"{icon} **{status}** — {payload.get('validation_notes', '')}")
                    else:
                        st.write("*(validation stage did not return a status)*")

                st.divider()

    # Empty-state: welcome + example questions, so this reads as a chatbot
    # product rather than a chat box bolted onto a dashboard.
    if not st.session_state.messages:
        st.markdown(
            "👋 **Ask me anything about SAP's current strategic position** - "
            "opportunities, risks, market trends, or specific competitors. "
            "Try one of these, or type your own question below:"
        )
        example_questions = [
            "What should SAP be most worried about right now?",
            "Where's SAP's biggest growth opportunity this quarter?",
            "How is Oracle positioning against SAP?",
        ]
        cols = st.columns(len(example_questions))
        for col, q in zip(cols, example_questions):
            if col.button(q, use_container_width=True):
                st.session_state._pending_question = q
                st.rerun()

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_pipeline(message.get("trace"))

    question = st.chat_input("Ask a strategic question...")
    if st.session_state._pending_question:
        question = st.session_state._pending_question
        st.session_state._pending_question = None

    if question:

        # Prior turns, passed back into the agent so it has memory of the
        # conversation rather than treating every question in isolation.
        history_for_agent = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        trace = []
        with st.status("Running agent pipeline...", expanded=False) as status:
            def _on_stage(stage):
                status.update(label=STAGE_LABELS.get(stage, stage))

            answer = ask_ceo(question, history=history_for_agent, trace=trace, on_stage=_on_stage)
            status.update(label="Pipeline complete", state="complete")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "trace": trace,
            }
        )

        with st.chat_message("assistant", avatar="💼"):
            st.markdown(answer)
            _render_pipeline(trace)

# ============================================================
# SUPPORTING DATA: the underlying pipeline's raw outputs - kept as
# reference/evidence behind the chatbot, not the main deliverable.
# ============================================================
with data_tab:
    st.caption(
        "This is the underlying intelligence pipeline (data collection, retrieval, "
        "agentic analysis, and validation) that powers the chatbot's answers above. "
        "Explore the raw reports here."
    )

    # --------------------------------------------------------
    # Section 1: Company Overview
    # --------------------------------------------------------
    st.header("1. Company Overview")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Company", COMPANY_NAME)
    col2.metric("Industry", INDUSTRY)
    col3.metric("Documents Collected", len(clean_df))

    n_sources = clean_df["source"].nunique()
    col4.metric("Data Sources", n_sources)
    col4.caption("16 outlets across 3 collection channels: SAP News RSS, GNews API, Hacker News API")

    last_update = clean_df["published"].max()
    last_update_str = last_update.strftime("%Y-%m-%d") if pd.notna(last_update) else "Unknown"
    col5.metric("Last Update", last_update_str)

    st.divider()

    # --------------------------------------------------------
    # Section 2: Market Intelligence
    # --------------------------------------------------------
    st.header("2. Market Intelligence")

    sub1, sub2, sub3, sub4 = st.tabs(["Recent News", "Competitor Activity", "Emerging Tech", "SAP Announcements"])

    with sub1:
        recent = clean_df.sort_values("published", ascending=False, na_position="last").head(10)
        for _, row in recent.iterrows():
            date_str = row["published"].strftime("%Y-%m-%d") if pd.notna(row["published"]) else "date unknown"
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['source']} · {date_str}")

    with sub2:
        if not competitors:
            st.info("No competitor data yet - run agents/04_competitor_monitor.ipynb")
        else:
    
            competitor_data = competitors
    
            for c in competitor_data:
                if c.get("mention_count", 0) > 0:
                    st.markdown(
                        f"**{c['competitor']}** ({c['mention_count']} mentions)"
                    )
                    st.write(c.get("summary", ""))
                    st.write("")
    
            zero_hit = [
                c["competitor"]
                for c in competitor_data
                if c.get("mention_count", 0) == 0
            ]
    
            if zero_hit:
                st.caption(
                    f"No recent activity detected for: {', '.join(zero_hit)}"
                )
    with sub3:
        if not trends:
            st.info("No trend data yet - run agents/03_trend_agent.ipynb")
        else:
            for t in trends:
                with st.container(border=True):
                    st.markdown(f"**{t.get('title')}** _{t.get('category')}_")
                    st.write(t.get("description", ""))
                    col1, col2 = st.columns(2)
                    col1.write(f"**Evidence:** {t.get('evidence', '')}")
                    if t.get("confidence_score") is not None:
                        col2.write(f"**Confidence Score:** {t.get('confidence_score')}")
                    if t.get("validation_status"):
                        st.caption(f"Validation: {t['validation_status']}")

    with sub4:
        announcements = clean_df[clean_df["source"] == "SAP News"].sort_values(
            "published", ascending=False, na_position="last"
        ).head(5)
        for _, row in announcements.iterrows():
            st.write(f"- {row['title']}")

    st.divider()

    # --------------------------------------------------------
    # Section 3: Opportunity Monitor
    # --------------------------------------------------------
    st.header("3. Opportunity Monitor")

    if not opportunities:
        st.info("No opportunity data yet - run agents/01_opportunity_agent.ipynb")
    else:
        for o in opportunities:
            with st.container(border=True):
                st.subheader(o.get("title"))
                col1, col2 = st.columns(2)
                col1.write(f"**Impact Level:** {o.get('impact_level')}")
                col2.write(f"**Confidence Score:** {o.get('confidence_score')}")
                st.write(f"**Evidence:** {o.get('evidence')}")
                if o.get("validation_status"):
                    st.caption(f"Validation: {o['validation_status']} — {o.get('validation_notes', '')}")

    st.divider()

    # --------------------------------------------------------
    # Section 4: Risk Monitor
    # --------------------------------------------------------
    st.header("4. Risk Monitor")

    if not risks:
        st.info("No risk data yet - run agents/02_risk_agent.ipynb")
    else:
        for r in risks:
            with st.container(border=True):
                st.subheader(r.get("title"))
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Risk Category:** {r.get('risk_category')}")
                col2.write(f"**Severity Level:** {r.get('severity_level')}")
                col3.write(f"**Confidence Score:** {r.get('confidence_score')}")
                st.write(f"**Evidence:** {r.get('evidence')}")
                if r.get("validation_status"):
                    st.caption(f"Validation: {r['validation_status']} — {r.get('validation_notes', '')}")

    st.divider()

    # --------------------------------------------------------
    # Section 5: Sentiment Analysis
    # --------------------------------------------------------
    st.header("5. Sentiment Analysis")
    st.caption(
        "News sentiment = SAP News + press coverage. "
        "Public sentiment = Hacker News community discussion."
    )

    sentiment_df = score_sentiment(clean_df)
    news_df = sentiment_df[sentiment_df["source"] != "Hacker News"]
    public_df = sentiment_df[sentiment_df["source"] == "Hacker News"]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("News Sentiment")
        if len(news_df) > 0:
            st.metric("Average Score", f"{news_df['sentiment_score'].mean():.2f}")
            st.bar_chart(news_df["sentiment_label"].value_counts())
        else:
            st.write("No news articles found.")

    with col2:
        st.subheader("Public Sentiment (Hacker News)")
        if len(public_df) > 0:
            st.metric("Average Score", f"{public_df['sentiment_score'].mean():.2f}")
            st.bar_chart(public_df["sentiment_label"].value_counts())
        else:
            st.write("No Hacker News articles found.")

    st.subheader("Sentiment Trend Over Time")
    trend_df = sentiment_df.dropna(subset=["published"]).copy()
    if len(trend_df) > 0:
        trend_df["week"] = trend_df["published"].dt.tz_localize(None).dt.to_period("W").dt.start_time
        weekly_sentiment = trend_df.groupby("week")["sentiment_score"].mean()
        st.line_chart(weekly_sentiment)
    else:
        st.write("Not enough dated articles to show a trend.")

    st.divider()

    # --------------------------------------------------------
    # Section 6: Strategic Recommendations
    # --------------------------------------------------------
    st.header("6. Strategic Recommendations")

    if not recommendations:
        st.info("No recommendations yet - run agents/05_ceo_agent.ipynb")
    else:
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        sorted_recs = sorted(recommendations, key=lambda r: priority_order.get(r.get("priority"), 3))

        for r in sorted_recs:
            with st.container(border=True):
                st.subheader(r.get("recommendation"))
                col1, col2 = st.columns(2)
                col1.write(f"**Priority:** {r.get('priority')}")
                col2.write(f"**Risk Level:** {r.get('risk_level')}")
                st.write(f"**Expected Impact:** {r.get('expected_impact')}")
                st.write("**Supporting Evidence:**")
                for e in r.get("supporting_evidence", []):
                    st.write(f"- {e}")

    st.divider()

    # --------------------------------------------------------
    # Section 7: CEO Briefing
    # --------------------------------------------------------
    st.header("7. CEO Briefing")

    if not briefing:
        st.info("No CEO briefing yet - run agents/05_ceo_agent.ipynb")
    else:
        st.subheader("What happened?")
        st.write(briefing.get("what_happened", ""))

        st.subheader("Why does it matter?")
        st.write(briefing.get("why_it_matters", ""))

        st.subheader("What should management do next?")
        st.write(briefing.get("what_to_do_next", ""))