"""
app.py – ResumeIQ: AI Resume Analyzer & Job Recommender (Milestone 6 UI)
Run: streamlit run app.py
"""
from __future__ import annotations

import json
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from config import (
    APP_ICON,
    APP_TITLE,
    JSEARCH_API_KEY,
    LLM_PROVIDER,
    SNAPSHOT_DIR,
    TOP_K_JOBS,
)
from evaluator import (
    aggregate_faithfulness,
    compute_extraction_f1,
    compute_ndcg,
    load_snapshots,
    save_snapshot,
)
from extractor import extract_resume
from jsearch_client import (
    SAMPLE_JOBS,
    build_candidate_pool,
    credits_remaining,
    get_last_api_status,
    test_jsearch_connection,
)
from logger import get_logger
from pdf_parser import parse_resume
from ranker import rank_jobs
from rewriter import rewrite_resume, generate_resume_summary, ResumeSummaryResult
from gap_analyzer import analyze_skill_gaps, GapAnalysis
from resume_chat import ResumeRAGChat, ChatResponse

log = get_logger("app")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

:root {
    --bg-primary:     #0a0f1e;
    --bg-secondary:   #111827;
    --bg-card:        #1a2235;
    --accent-blue:    #4f8ef7;
    --accent-purple:  #a78bfa;
    --accent-green:   #34d399;
    --accent-amber:   #fbbf24;
    --accent-rose:    #f87171;
    --text-primary:   #f1f5f9;
    --text-secondary: #94a3b8;
    --border:         rgba(255,255,255,0.08);
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1a2d 50%, #0f172a 100%);
    color: var(--text-primary);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1525 0%, #111827 100%);
    border-right: 1px solid var(--border);
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1a2235, #1e2d4a);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(79,142,247,0.15);
}
.metric-value {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.25rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Score ring */
.score-ring {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    font-weight: 800;
    margin: 0 auto;
}

/* Job cards */
.job-card {
    background: linear-gradient(135deg, rgba(26,34,53,0.9), rgba(30,45,74,0.9));
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
    transition: all 0.25s ease;
    cursor: pointer;
}
.job-card:hover {
    border-color: rgba(79,142,247,0.5);
    box-shadow: 0 4px 24px rgba(79,142,247,0.12);
    transform: translateY(-2px);
}
.job-title { font-size: 1.05rem; font-weight: 700; color: var(--accent-blue); }
.job-company { font-size: 0.88rem; color: var(--text-secondary); margin-top: 0.1rem; }
.match-pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
}
.match-high   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.match-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.match-low    { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }

/* Skill pill */
.skill-pill {
    display: inline-block;
    background: rgba(79,142,247,0.12);
    border: 1px solid rgba(79,142,247,0.25);
    color: #7fb3ff;
    padding: 0.15rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem;
}
.skill-pill.missing {
    background: rgba(248,113,113,0.1);
    border-color: rgba(248,113,113,0.25);
    color: #f87171;
}
.skill-pill.matched {
    background: rgba(52,211,153,0.1);
    border-color: rgba(52,211,153,0.25);
    color: #34d399;
}

/* Bullet comparison */
.bullet-orig   { background: rgba(248,113,113,0.08); border-left: 3px solid #f87171; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.3rem 0; }
.bullet-new    { background: rgba(52,211,153,0.08); border-left: 3px solid #34d399; padding: 0.6rem 1rem; border-radius: 0 8px 8px 0; margin: 0.3rem 0; }

/* Header gradient */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #4f8ef7 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero-subtitle { color: var(--text-secondary); font-size: 1.05rem; margin-top: 0.5rem; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1.2rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(79,142,247,0.15), rgba(167,139,250,0.1));
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue);
}

/* Navigation Tiles Grid */
.nav-grid-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Secondary Button Tiles (Unselected) */
div[data-testid="stColumn"] button[kind="secondary"] {
    background: rgba(26, 34, 53, 0.75) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    color: #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 0.65rem 0.8rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    white-space: pre-line !important;
    line-height: 1.35 !important;
    transition: all 0.2s ease !important;
    min-height: 3.4rem !important;
    margin-bottom: 0.4rem !important;
}
div[data-testid="stColumn"] button[kind="secondary"]:hover {
    background: rgba(30, 45, 74, 0.95) !important;
    border-color: rgba(79, 142, 247, 0.6) !important;
    color: #ffffff !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(79, 142, 247, 0.2) !important;
}

/* Primary Button Tiles (Active Selected Tile) */
div[data-testid="stColumn"] button[kind="primary"] {
    background: linear-gradient(135deg, #4f8ef7, #7c3aed) !important;
    border: 1.5px solid #a78bfa !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    padding: 0.65rem 0.8rem !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    white-space: pre-line !important;
    line-height: 1.35 !important;
    box-shadow: 0 4px 20px rgba(124, 58, 237, 0.45) !important;
    min-height: 3.4rem !important;
    margin-bottom: 0.4rem !important;
}

/* General Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f8ef7, #7c3aed);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    font-size: 0.95rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(79,142,247,0.4);
}

/* Progress bars */
.stProgress > div > div > div { background: linear-gradient(90deg, #4f8ef7, #a78bfa); border-radius: 999px; }

/* Tips */
.tip-warning { 
    background: rgba(251,191,36,0.1); 
    border: 1px solid rgba(251,191,36,0.25);
    border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0;
    color: #fcd34d; font-size: 0.9rem;
}
.tip-good {
    background: rgba(52,211,153,0.1);
    border: 1px solid rgba(52,211,153,0.25);
    border-radius: 10px; padding: 0.7rem 1rem; margin: 0.4rem 0;
    color: #6ee7b7; font-size: 0.9rem;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Input */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(79,142,247,0.3) !important;
    border-radius: 16px !important;
    background: rgba(79,142,247,0.04) !important;
    padding: 1rem !important;
}

/* Section headers */
h2 { color: var(--text-primary); font-weight: 700; }
h3 { color: var(--accent-blue); font-weight: 600; }

/* Divider */
hr { border-color: var(--border); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(79,142,247,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "resume_raw":     None,
        "resume":         None,
        "jobs":           None,
        "ranked":         None,
        "rewrites":       None,
        "snapshot":       None,
        "resume_summary": None,
        "gap_analysis":   None,
        "chat_engine":    None,
        "chat_messages":  [],
        "active_view":    "📊 Resume Analysis",
        "processing":     False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    target_role = st.text_input(
        "🎯 Target Role",
        placeholder="e.g. ML Engineer, Data Scientist",
        help="Optionally specify your target role for better job matching",
    )

    use_live_api = st.toggle(
        "🌐 Live JSearch API",
        value=bool(JSEARCH_API_KEY),
        help="Requires JSEARCH_API_KEY in .env – uses cached sample data otherwise",
    )

    use_llm = st.toggle(
        "🤖 LLM Features",
        value=bool(LLM_PROVIDER),
        help=f"Enables AI explanations & bullet rewriting (provider: {LLM_PROVIDER})",
    )

    top_k = st.slider("Top K Jobs", 5, 20, TOP_K_JOBS)

    st.divider()
    st.markdown("### 📊 System Status")

    if JSEARCH_API_KEY:
        rem = credits_remaining()
        st.metric("JSearch Credits", rem, delta=None)
        st.progress(rem / 50)

    api_status = "🟢 Configured" if JSEARCH_API_KEY else "🔴 No API Key"
    llm_status = "🟢 Ready" if (LLM_PROVIDER and (
        (LLM_PROVIDER == "groq" and __import__("config").GROQ_API_KEY) or
        (LLM_PROVIDER == "openai" and __import__("config").OPENAI_API_KEY)
    )) else "🟡 Demo Mode"

    st.markdown(f"**JSearch API:** {api_status}")
    st.markdown(f"**LLM:** {llm_status} (`{LLM_PROVIDER}`)")

    if JSEARCH_API_KEY:
        if st.button("⚡ Test JSearch Key", key="test_jsearch_btn", use_container_width=True):
            with st.spinner("Pinging RapidAPI JSearch..."):
                ok, msg = test_jsearch_connection()
                st.session_state["jsearch_test_res"] = (ok, msg)

        if "jsearch_test_res" in st.session_state:
            ok, msg = st.session_state["jsearch_test_res"]
            if ok:
                st.success(f"{msg}")
            else:
                st.error(f"{msg}")
                if "not subscribed" in msg.lower():
                    st.markdown("[👉 Subscribe on RapidAPI (Free 200 req/mo)](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch/pricing)")

    st.divider()

    # Eval snapshots
    snaps = load_snapshots()
    if snaps:
        st.markdown(f"### 📸 {len(snaps)} Snapshot(s)")
        for s in snaps[:3]:
            st.markdown(f"- `{s.resume_name}` | nDCG={s.ndcg:.3f}")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1rem 0;">
    <div class="hero-title">🎯 ResumeIQ</div>
    <div class="hero-subtitle">AI-Powered Resume Analyzer & Intelligent Job Recommender</div>
</div>
""", unsafe_allow_html=True)

# ── File upload ────────────────────────────────────────────────────────────────
resume_file = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"],
    key="resume_uploader",
    help="Upload your resume as a PDF file for analysis",
)

# ── Analyze button ─────────────────────────────────────────────────────────────
if resume_file:
    if st.button("🚀 Analyze Resume & Find Jobs", use_container_width=True):
        st.session_state.processing = True

        with st.spinner("📖 Parsing PDF..."):
            raw = parse_resume(BytesIO(resume_file.read()), filename=resume_file.name)
            st.session_state.resume_raw = raw

        with st.spinner("🧠 Extracting structured data..."):
            resume = extract_resume(raw)
            st.session_state.resume = resume

        with st.spinner("🔍 Building job candidate pool..."):
            if use_live_api and JSEARCH_API_KEY:
                jobs = build_candidate_pool(
                    resume.skills,
                    target_role=target_role,
                    max_jobs=40,
                )
                if not jobs:
                    jobs = SAMPLE_JOBS
            else:
                jobs = SAMPLE_JOBS

            st.session_state.jobs = jobs

        with st.spinner("📊 Ranking & scoring jobs..."):
            ranked = rank_jobs(
                resume,
                jobs,
                top_k=top_k,
                use_llm_explanations=use_llm,
            )
            st.session_state.ranked = ranked

        with st.spinner("✍️ Rewriting weak bullets..."):
            job_ctx = ranked[0].title if ranked else target_role
            rewrites = rewrite_resume(resume, job_context=job_ctx)
            st.session_state.rewrites = rewrites

        with st.spinner("🧠 Generating Executive Summary & Elevator Pitch..."):
            resume_sum = generate_resume_summary(resume, target_role=target_role)
            st.session_state.resume_summary = resume_sum

        with st.spinner("🎯 Analyzing Skill Gaps with AI..."):
            top_desc = ranked[0].description_snippet if ranked else ""
            top_title = ranked[0].title if ranked else target_role
            gap_res = analyze_skill_gaps(resume, job_description=top_desc, job_title=top_title)
            st.session_state.gap_analysis = gap_res

        with st.spinner("💬 Initializing Resume Chat RAG engine..."):
            st.session_state.chat_engine = ResumeRAGChat(resume)
            st.session_state.chat_messages = []

        with st.spinner("💾 Saving evaluation snapshot..."):
            snap = save_snapshot(
                resume_name=resume_file.name,
                ranked_jobs=ranked,
                extracted_skills=resume.skills,
                rewrite_results=rewrites,
            )
            st.session_state.snapshot = snap

        st.session_state.processing = False
        st.success("✅ Analysis complete!")
        st.rerun()


# ── Results tabs ───────────────────────────────────────────────────────────────
resume: Optional[object] = st.session_state.resume
ranked = st.session_state.ranked

if resume is None:
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.95rem; margin-top:2rem;">
        ⬆️ Upload your PDF resume above and click <strong>Analyze</strong> to get started
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Dashboard Navigation Tiles Grid (Zero horizontal scrolling) ────────────────
st.markdown("""
<div class="nav-grid-header">
    <span>🧭</span> <strong>Dashboard Modules</strong>
    <span style="font-size:0.8rem; font-weight:400; color:#94a3b8; margin-left:auto;">Click any tile to explore</span>
</div>
""", unsafe_allow_html=True)

gap_data = st.session_state.gap_analysis
rewrites = st.session_state.rewrites
snap = st.session_state.snapshot

views = [
    {"name": "📊 Resume Analysis", "badge": f"Score {resume.score}/100", "desc": "Overall score & section health"},
    {"name": "🧠 AI Summary & Pitch", "badge": "Elevator Pitch", "desc": "Executive summary & LinkedIn bio"},
    {"name": "🎯 AI Skill Gap Analysis", "badge": f"{gap_data.overall_readiness:.0f}% Match" if gap_data else "Readiness", "desc": "Gap breakdown & 4-week roadmap"},
    {"name": "💬 Chat with Resume (RAG)", "badge": "RAG Assistant", "desc": "Interactive semantic Q&A"},
    {"name": "💼 Job Matches", "badge": f"{len(ranked)} Jobs" if ranked else "Ranked Jobs", "desc": "Matched jobs with fit scores"},
    {"name": "✍️ Bullet Rewriter", "badge": f"{len([r for r in rewrites if r.is_weak])} Weak" if rewrites else "Bullet Enhancer", "desc": "Active impact bullet upgrades"},
    {"name": "🔬 Skills Deep-Dive", "badge": f"{len(resume.skills)} Skills", "desc": "Taxonomy pills & coverage heatmap"},
    {"name": "📈 Evaluation Metrics", "badge": f"nDCG {snap.ndcg:.3f}" if snap else "IR Metrics", "desc": "nDCG@10, F1 & faithfulness"},
    {"name": "🗂️ Raw JSON", "badge": "JSON Export", "desc": "Structured resume schema download"},
]

valid_names = [v["name"] for v in views]
if st.session_state.active_view not in valid_names:
    st.session_state.active_view = valid_names[0]

# Render 3x3 Navigation Tiles Grid
for row_start in (0, 3, 6):
    tile_cols = st.columns(3)
    for col, v in zip(tile_cols, views[row_start:row_start + 3]):
        is_active = (st.session_state.active_view == v["name"])
        with col:
            tile_label = f"{'▶ ' if is_active else ''}{v['name']}\n{v['badge']}"
            if st.button(
                tile_label,
                key=f"tile_nav_{v['name']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
                help=v["desc"],
            ):
                st.session_state.active_view = v["name"]
                st.rerun()

st.markdown("<div style='margin-bottom: 0.8rem;'></div>", unsafe_allow_html=True)
active_tab = st.session_state.active_view

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1 – Resume Analysis
# ═══════════════════════════════════════════════════════════════════════════════
if active_tab == "📊 Resume Analysis":
    st.markdown("## 📊 Resume Analysis")

    # Score overview
    score = resume.score
    score_color = "#34d399" if score >= 75 else "#fbbf24" if score >= 50 else "#f87171"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="background: linear-gradient(135deg, {score_color}, #7c3aed);
                 -webkit-background-clip: text; background-clip: text;">{score}</div>
            <div class="metric-label">Overall Score / 100</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(resume.skills)}</div>
            <div class="metric-label">Skills Detected</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{resume.word_count}</div>
            <div class="metric-label">Word Count</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{resume.quantified_count}</div>
            <div class="metric-label">Quantified Bullets</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(score / 100)

    # Contact info
    c = resume.contact
    if c.name or c.email:
        with st.expander("👤 Contact Information", expanded=True):
            cols = st.columns(3)
            if c.name:     cols[0].markdown(f"**Name:** {c.name}")
            if c.email:    cols[1].markdown(f"**Email:** {c.email}")
            if c.phone:    cols[2].markdown(f"**Phone:** {c.phone}")
            cols2 = st.columns(2)
            if c.linkedin: cols2[0].markdown(f"**LinkedIn:** [{c.linkedin[:50]}]({c.linkedin})")
            if c.github:   cols2[1].markdown(f"**GitHub:** [{c.github[:50]}]({c.github})")

    # Section analysis
    with st.expander("📑 Section Breakdown", expanded=True):
        from config import SECTIONS_REQUIRED
        found = {k.lower() for k in resume.raw_sections}
        for sec in SECTIONS_REQUIRED:
            has = any(sec in k for k in found)
            icon = "✅" if has else "❌"
            st.markdown(f"{icon} **{sec.title()}**")

    # Improvement tips
    if resume.tips:
        st.markdown("### 💡 Improvement Recommendations")
        for tip in resume.tips:
            st.markdown(f"<div class='tip-warning'>{tip}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='tip-good'>🎉 Great resume! No major issues found.</div>", unsafe_allow_html=True)

    # Education
    if resume.education:
        with st.expander("🎓 Education", expanded=False):
            for edu in resume.education:
                st.markdown(f"**{edu.degree}** – {edu.institution} {f'({edu.year})' if edu.year else ''}")
                if edu.gpa:
                    st.markdown(f"GPA: {edu.gpa}")

    # Experience
    if resume.experience:
        with st.expander(f"💼 Experience ({len(resume.experience)} roles)", expanded=False):
            for exp in resume.experience:
                st.markdown(f"**{exp.title}** {'at ' + exp.company if exp.company else ''} {f'({exp.duration})' if exp.duration else ''}")
                for b in exp.bullets[:4]:
                    is_weak = b in [wb for wb in exp.weak_bullets]
                    icon = "⚠️" if is_weak else "•"
                    st.markdown(f"{icon} {b}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – AI Summary & Elevator Pitch
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🧠 AI Summary & Pitch":
    st.markdown("## 🧠 AI Professional Summary & Elevator Pitch")
    st.caption("Auto-generated executive pitch and copyable LinkedIn/Resume summary powered by Generative AI.")

    summary_res: Optional[ResumeSummaryResult] = st.session_state.resume_summary
    if not summary_res:
        st.info("No AI summary generated yet. Click 'Analyze Resume & Find Jobs' above.")
    else:
        badge = getattr(summary_res, "method", "AI Engine")
        st.markdown(f"**Generation Engine:** `{badge}`")
        if getattr(summary_res, "error_message", None):
            st.warning(f"⚠️ {summary_res.error_message}. Enhanced with Smart Profile Engine.")

        # Elevator pitch
        st.markdown("### 🎙️ 30-Second Elevator Pitch")
        st.markdown(f"""
        <div style="background: rgba(124, 58, 237, 0.08); border-left: 4px solid #7c3aed; padding: 1.2rem; border-radius: 8px; margin-bottom: 1.5rem; font-size: 1.05rem; line-height: 1.6; color: #e2e8f0;">
            "{summary_res.elevator_pitch}"
        </div>
        """, unsafe_allow_html=True)

        # Executive Bio / Professional Profile
        st.markdown("### 📄 Professional Summary (Ready for Resume / LinkedIn)")
        st.text_area(
            "Copy & paste into your resume summary section:",
            value=summary_res.professional_summary,
            height=130,
            key="prof_summary_copy_area",
        )

        # Key Career Highlights
        st.markdown("### ⭐ Key Career Highlights")
        for i, h in enumerate(summary_res.key_highlights, 1):
            st.markdown(f"**{i}.** {h}")

        st.divider()
        st.markdown("#### 🔄 Custom Re-generation")
        custom_target = st.text_input("Refocus summary towards specific role/industry:", value=target_role, key="custom_sum_role")
        if st.button("✨ Regenerate Summary", key="regen_sum_btn"):
            with st.spinner("Generating tailored summary..."):
                new_sum = generate_resume_summary(resume, target_role=custom_target)
                st.session_state.resume_summary = new_sum
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – AI Skill Gap Analysis
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🎯 AI Skill Gap Analysis":
    st.markdown("## 🎯 AI-Powered Skill Gap Analysis")
    st.caption("LLM-powered readiness score, prioritized missing competencies, and tailored 4-week learning roadmap.")

    gap_data: Optional[GapAnalysis] = st.session_state.gap_analysis
    if not gap_data:
        st.info("No skill gap analysis available yet. Run analysis first.")
    else:
        badge = getattr(gap_data, "method", "AI Engine")
        st.markdown(f"**Analysis Engine:** `{badge}`")
        if getattr(gap_data, "error_message", None):
            st.info(f"ℹ️ {gap_data.error_message}")

        # Top Metric Cards
        col_g1, col_g2, col_g3 = st.columns(3)
        readiness = gap_data.overall_readiness
        col_g1.metric("Target Role Readiness", f"{readiness:.0f}%", delta=f"{len(gap_data.matched_skills)} matched skills")
        col_g2.metric("Missing Critical / High Gaps", len(gap_data.gap_details))
        col_g3.metric("Candidate Strength Skills", len(gap_data.matched_skills))

        st.progress(min(max(readiness / 100.0, 0.0), 1.0))

        # Strategic AI Summary
        if gap_data.summary:
            st.markdown("### 📋 Strategic Gap Assessment")
            st.markdown(f"""
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem; line-height: 1.6;">
                {gap_data.summary}
            </div>
            """, unsafe_allow_html=True)

        # 4-Week Action Plan
        if gap_data.action_plan:
            st.markdown("### 🗓️ Recommended 4-Week Bridge Plan")
            st.markdown(f"""
            <div style="background: rgba(56, 189, 248, 0.08); border-left: 4px solid #38bdf8; padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 1.5rem; line-height: 1.7; font-size: 0.95rem;">
                {gap_data.action_plan.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

        # Detailed Gap Breakdown
        if gap_data.gap_details:
            st.markdown("### 🔍 Skill Gap Priorities & Tailored Learning Paths")
            for gap in gap_data.gap_details:
                with st.expander(f"**{gap.skill.title()}** — Priority: {gap.priority} ({gap.estimated_time})", expanded=(gap.priority == "Critical")):
                    c_l, c_r = st.columns([2, 1])
                    with c_l:
                        st.markdown(f"**Category:** `{gap.category}` | **Estimated Time:** `{gap.estimated_time}`")
                        st.markdown(f"**Learning Pathway:** {gap.recommendation}")
                    with c_r:
                        st.markdown("**Suggested Resources:**")
                        for res in gap.resources[:3]:
                            search_link = f"https://www.google.com/search?q={res.replace(' ', '+')}"
                            st.markdown(f"• [{res}]({search_link})")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 – Chat with Resume (RAG)
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "💬 Chat with Resume (RAG)":
    st.markdown("## 💬 Chat with Your Resume (RAG Q&A)")
    st.caption("Ask questions about your skills, achievements, experience, and career trajectory using semantic retrieval over your resume.")

    chat_engine: Optional[ResumeRAGChat] = st.session_state.chat_engine
    if not chat_engine:
        if resume:
            st.session_state.chat_engine = ResumeRAGChat(resume)
            chat_engine = st.session_state.chat_engine
        else:
            st.info("Please upload and analyze a resume first.")
            st.stop()

    # Pre-set prompt chips
    st.markdown("**Quick Prompts:**")
    qc1, qc2, qc3, qc4 = st.columns(4)
    q_chosen = None
    if qc1.button("⭐ Top AI Strengths", use_container_width=True):
        q_chosen = "What are my top machine learning and AI strengths based on my resume?"
    if qc2.button("🛠️ Gaps for Senior Role", use_container_width=True):
        q_chosen = "What should I add to my experience to qualify for a Senior AI Engineer position?"
    if qc3.button("📝 Summarize Roles", use_container_width=True):
        q_chosen = "Summarize my key work experience roles and achievements."
    if qc4.button("💡 Project Highlights", use_container_width=True):
        q_chosen = "What are the most technically impressive projects or achievements on my resume?"

    # Display chat message history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📚 Retrieved Resume Context Chunks", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f"• `{src}`")

    # Input form
    user_query = st.chat_input("Ask a question about your resume background...")
    active_query = q_chosen or user_query

    if active_query:
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving resume context & generating response..."):
                resp: ChatResponse = chat_engine.answer_query(active_query)
                st.markdown(resp.answer)
                if resp.error_message:
                    st.caption(f"⚡ Note: {resp.error_message}")
                if resp.sources:
                    with st.expander("📚 Retrieved Resume Context Chunks", expanded=False):
                        for src in resp.sources:
                            st.markdown(f"• `{src}`")

        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": resp.answer,
            "sources": resp.sources,
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 – Job Matches
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "💼 Job Matches":
    st.markdown("## 💼 Job Recommendations")

    if not ranked:
        st.info("No jobs ranked yet. Please run analysis first.")
    else:
        # Check if JSearch API failed due to subscription or connectivity
        jsearch_stat = get_last_api_status()
        if jsearch_stat.get("error_type") == "NOT_SUBSCRIBED":
            st.warning(
                "⚠️ **Live JSearch Notice**: RapidAPI returned `403 Forbidden` (*\"You are not subscribed to this API\"*).\n\n"
                "Currently displaying **8 curated fallback jobs**. To enable live job queries from LinkedIn/Indeed:\n"
                "1. Visit the [RapidAPI JSearch Pricing Page](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch/pricing)\n"
                "2. Click **Subscribe to Test** on the **Basic (Free)** plan ($0/mo, 200 requests)\n"
                "3. Once subscribed, click **⚡ Test JSearch Key** in the sidebar to verify."
            )
        elif jsearch_stat.get("error_type") == "INVALID_KEY":
            st.warning("⚠️ **Live JSearch Notice**: RapidAPI returned `401 Unauthorized` (Invalid API Key). Check `JSEARCH_API_KEY` in `.env`.")
        # Summary metrics
        scores = [j.match_score for j in ranked]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top Match", f"{scores[0]:.1f}%")
        c2.metric("Avg Match", f"{sum(scores)/len(scores):.1f}%")
        c3.metric("Jobs Ranked", len(ranked))

        st.divider()

        # Job cards
        for i, job in enumerate(ranked):
            match_cls = "match-high" if job.match_score >= 70 else "match-medium" if job.match_score >= 45 else "match-low"
            emp_type = job.employment_type.replace("_", " ").title() if job.employment_type else ""

            with st.expander(
                f"{'🥇' if i==0 else '🥈' if i==1 else '🥉' if i==2 else f'{i+1}.'} "
                f"{job.title} — {job.company} | {job.match_score:.0f}% Match",
                expanded=(i < 3),
            ):
                col_main, col_scores = st.columns([2, 1])

                with col_main:
                    st.markdown(f"**🏢 {job.company}** | 📍 {job.location} | {emp_type}")
                    st.markdown(f"💰 {job.salary_range}")
                    if job.explanation:
                        st.markdown(f"> {job.explanation}")

                    # Skills
                    st.markdown("**✅ Matched Skills:**")
                    if job.matched_skills:
                        pills = " ".join(f'<span class="skill-pill matched">{s}</span>' for s in job.matched_skills)
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("No direct skill matches")

                    st.markdown("**❌ Missing Skills:**")
                    if job.missing_skills:
                        pills = " ".join(f'<span class="skill-pill missing">{s}</span>' for s in job.missing_skills[:8])
                        st.markdown(pills, unsafe_allow_html=True)
                    else:
                        st.caption("None – great fit!")

                    if job.apply_link and not job.apply_link.startswith("https://example"):
                        st.markdown(f"[🔗 Apply Now]({job.apply_link})")

                with col_scores:
                    st.markdown(f"""
                    <div class="metric-card" style="margin-bottom:0.5rem">
                        <div class="metric-value" style="font-size:1.8rem">{job.match_score:.0f}%</div>
                        <div class="metric-label">Overall Match</div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown(f"**Semantic:** {job.semantic_score:.1f}%")
                    st.progress(job.semantic_score / 100)
                    st.markdown(f"**Skill Overlap:** {job.skill_overlap:.1f}%")
                    st.progress(job.skill_overlap / 100)
                    st.markdown(f"**Seniority Fit:** {job.seniority_fit:.1f}%")
                    st.progress(job.seniority_fit / 100)

        # Download ranked table
        df_ranked = pd.DataFrame([{
            "Job": j.title,
            "Company": j.company,
            "Location": j.location,
            "Match %": j.match_score,
            "Salary": j.salary_range,
            "Matched Skills": ", ".join(j.matched_skills),
            "Missing Skills": ", ".join(j.missing_skills),
        } for j in ranked])

        st.download_button(
            "⬇️ Download Rankings CSV",
            df_ranked.to_csv(index=False),
            file_name="job_rankings.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 – Bullet Rewriter
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "✍️ Bullet Rewriter":
    st.markdown("## ✍️ AI Bullet Rewriter")

    rewrites = st.session_state.rewrites
    if not rewrites:
        st.info("No rewrites available. Run analysis first.")
    else:
        weak = [r for r in rewrites if r.is_weak]
        strong = [r for r in rewrites if not r.is_weak]

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Bullets", len(rewrites))
        c2.metric("Weak Bullets", len(weak), delta=f"-{len(weak)}" if weak else None, delta_color="inverse")
        avg_faith = sum(r.faithfulness_score for r in rewrites)/len(rewrites) if rewrites else 1.0
        c3.metric("Avg Faithfulness", f"{avg_faith:.2f}")

        # Check if any LLM quota or connection issue occurred
        llm_errs = [getattr(r, "error_message", None) for r in rewrites if getattr(r, "error_message", None)]
        if llm_errs:
            st.warning(
                f"⚠️ **LLM Notice**: `{llm_errs[0]}`\n\n"
                "The rewriter automatically engaged the **Smart Action Engine** (offline rule-based upgrade) so your bullets are enhanced without interruption. "
                "To use cloud LLMs (GPT-4o or LLaMA-3), please replenish OpenAI API credits or switch to Groq (`LLM_PROVIDER=groq`) in `.env`."
            )

        st.divider()

        if weak:
            st.markdown("### ⚠️ Rewritten Bullets (Weak → Strong)")
            for i, r in enumerate(weak, 1):
                engine_badge = getattr(r, "method", "AI Engine")
                st.markdown(f"**Bullet #{i}** | Engine: `{engine_badge}` | Faithfulness: **{r.faithfulness_score:.2f}**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Before (Weak):**")
                    st.markdown(f"<div class='bullet-orig'>❌ {r.original}</div>", unsafe_allow_html=True)
                with col_b:
                    st.markdown("**After (Upgraded):**")
                    st.markdown(f"<div class='bullet-new'>✅ {r.rewritten}</div>", unsafe_allow_html=True)
                st.caption(r.improvement_note)
                st.markdown("---")
        else:
            st.success("🎉 No weak bullets detected! Your experience bullets are strong.")

        if strong:
            with st.expander(f"✅ Strong Bullets ({len(strong)}) – No Changes Needed"):
                for r in strong:
                    st.markdown(f"• {r.original}")

        # Manual rewrite tool
        st.divider()
        st.markdown("### 🛠️ Manual Bullet Rewriter")
        custom_bullet = st.text_area("Enter a bullet to rewrite:", height=80, key="manual_bullet")
        custom_job_ctx = st.text_input("Job context (optional):", key="manual_job_ctx")
        if st.button("✨ Rewrite This Bullet", key="rewrite_btn"):
            if custom_bullet.strip():
                from rewriter import rewrite_bullet
                with st.spinner("Rewriting..."):
                    result = rewrite_bullet(custom_bullet, custom_job_ctx, resume.skills)
                if getattr(result, "error_message", None):
                    st.warning(f"⚠️ Note: {result.error_message}. Enhanced with Smart Action Engine.")
                st.markdown(f"**Engine:** `{getattr(result, 'method', 'AI Engine')}` | **Faithfulness:** `{result.faithfulness_score:.2f}`")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown("**Before:**")
                    st.markdown(f"<div class='bullet-orig'>❌ {result.original}</div>", unsafe_allow_html=True)
                with col_m2:
                    st.markdown("**After:**")
                    st.markdown(f"<div class='bullet-new'>✅ {result.rewritten}</div>", unsafe_allow_html=True)
                st.caption(result.improvement_note)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 – Skills Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🔬 Skills Deep-Dive":
    st.markdown("## 🔬 Skills Deep-Dive")

    if resume.skills:
        # All detected skills
        pills = " ".join(f'<span class="skill-pill">{s}</span>' for s in resume.skills)
        st.markdown(f"**Detected Skills ({len(resume.skills)}):**")
        st.markdown(pills, unsafe_allow_html=True)

        st.divider()

        if ranked:
            # Skills gap analysis across all jobs
            st.markdown("### 📉 Skill Gap Analysis")
            all_missing: dict[str, int] = {}
            for job in ranked:
                for ms in job.missing_skills:
                    all_missing[ms] = all_missing.get(ms, 0) + 1

            if all_missing:
                gap_df = pd.DataFrame(
                    sorted(all_missing.items(), key=lambda x: x[1], reverse=True),
                    columns=["Missing Skill", "Demanded by # Jobs"],
                )
                gap_df["Priority"] = gap_df["Demanded by # Jobs"].apply(
                    lambda x: "🔴 High" if x >= 3 else "🟡 Medium" if x >= 2 else "🟢 Low"
                )
                st.dataframe(gap_df, use_container_width=True, hide_index=True)

                # Top missing skills suggestion
                top_missing = gap_df.head(5)["Missing Skill"].tolist()
                st.markdown("#### 📚 Learning Recommendations")
                for skill in top_missing:
                    search_url = f"https://www.google.com/search?q=learn+{skill.replace(' ', '+')}+course"
                    st.markdown(f"- **{skill}** → [Find courses 🔗]({search_url})")

            # Skill coverage heatmap by job
            st.divider()
            st.markdown("### 🔥 Skill Coverage by Job")
            heatmap_data = {}
            for job in ranked[:6]:
                for s in job.matched_skills:
                    if s not in heatmap_data:
                        heatmap_data[s] = {}
                    heatmap_data[s][job.title[:20]] = 1

            if heatmap_data:
                hm_df = pd.DataFrame(heatmap_data).T.fillna(0)
                st.dataframe(hm_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    else:
        st.warning("No skills extracted yet.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 – Evaluation Metrics
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "📈 Evaluation Metrics":
    st.markdown("## 📈 Evaluation Metrics")

    snap = st.session_state.snapshot

    if snap:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("nDCG@10", f"{snap.ndcg:.4f}", help="Ranking quality vs objective qualification ground-truth (0.85-0.98 is high quality)")
        c2.metric("Avg Faithfulness", f"{snap.avg_faithfulness:.4f}", help="Bullet rewrite faithfulness")
        c3.metric("Jobs Ranked", len(snap.ranking_scores))
        c4.metric("Skills Extracted", len(snap.extraction_skills))

        st.caption("ℹ️ **nDCG@10 Methodology**: Evaluates the hybrid ranking order against independent ground-truth qualification relevance (skill coverage + domain alignment + seniority fit).")
        st.divider()

        # Score distribution
        st.markdown("### 📊 Match Score Distribution")
        score_df = pd.DataFrame({
            "Job Rank": list(range(1, len(snap.ranking_scores)+1)),
            "Match Score": snap.ranking_scores,
        })
        st.bar_chart(score_df.set_index("Job Rank"))

        # Ground truth F1 evaluation
        st.markdown("### 🎯 Extraction F1 (Ground Truth Comparison)")
        gt_input = st.text_area(
            "Enter ground-truth skills (comma-separated) to compute F1:",
            placeholder="python, pytorch, docker, kubernetes, ...",
            height=80,
            key="gt_skills",
        )
        if gt_input.strip():
            gt_skills = [s.strip().lower() for s in gt_input.split(",") if s.strip()]
            f1_result = compute_extraction_f1(snap.extraction_skills, gt_skills)
            col1, col2, col3 = st.columns(3)
            col1.metric("Precision", f"{f1_result['precision']:.3f}")
            col2.metric("Recall",    f"{f1_result['recall']:.3f}")
            col3.metric("F1 Score",  f"{f1_result['f1']:.3f}")
            st.markdown(f"TP={f1_result['tp']} | FP={f1_result['fp']} | FN={f1_result['fn']}")

        # Snapshot history
        st.divider()
        st.markdown("### 📸 Snapshot History")
        all_snaps = load_snapshots()
        if all_snaps:
            snap_df = pd.DataFrame([{
                "Resume": s.resume_name,
                "nDCG": s.ndcg,
                "Avg Faithfulness": s.avg_faithfulness,
                "Skills Extracted": len(s.extraction_skills),
                "Jobs Ranked": len(s.ranking_scores),
                "Timestamp": time.strftime("%Y-%m-%d %H:%M", time.localtime(s.timestamp)),
            } for s in all_snaps])
            st.dataframe(snap_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run analysis to generate evaluation metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9 – Raw JSON
# ═══════════════════════════════════════════════════════════════════════════════
elif active_tab == "🗂️ Raw JSON":
    st.markdown("## 🗂️ Structured Resume JSON")

    if resume:
        json_str = resume.to_json()
        st.download_button(
            "⬇️ Download Resume JSON",
            json_str,
            file_name="resume_structured.json",
            mime="application/json",
        )
        st.json(json.loads(json_str))
