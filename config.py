"""
config.py – Central configuration for the Resume Analyzer & Job Recommender.
All environment variables, model names, thresholds, and constants live here.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ── API keys ───────────────────────────────────────────────────────────────────
JSEARCH_API_KEY: str = os.getenv("JSEARCH_API_KEY", "")
OPENAI_API_KEY: str  = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY: str    = os.getenv("GROQ_API_KEY", "")

# ── LLM settings ──────────────────────────────────────────────────────────────
LLM_PROVIDER: str      = os.getenv("LLM_PROVIDER", "groq")   # "openai" | "groq"
LLM_MODEL: str         = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b").split("#")[0].strip()
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS: int    = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Embedding model ────────────────────────────────────────────────────────────
EMBED_MODEL: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

# ── JSearch API ────────────────────────────────────────────────────────────────
# ── JSearch API (Supports OpenWebNinja direct & RapidAPI) ─────────────────────
JSEARCH_OPENWEBNINJA_URL: str = "https://api.openwebninja.com/jsearch/search"
JSEARCH_RAPIDAPI_URL: str     = "https://jsearch.p.rapidapi.com/search"
JSEARCH_PROVIDER: str         = os.getenv(
    "JSEARCH_PROVIDER",
    "openwebninja" if JSEARCH_API_KEY.startswith("ak_") else "rapidapi"
).lower()
JSEARCH_BASE_URL: str         = (
    JSEARCH_OPENWEBNINJA_URL if JSEARCH_PROVIDER == "openwebninja" else JSEARCH_RAPIDAPI_URL
)
JSEARCH_HOST: str             = "jsearch.p.rapidapi.com"
JSEARCH_CREDIT_CAP: int       = int(os.getenv("JSEARCH_CREDIT_CAP", "50"))   # max requests/session
JSEARCH_CACHE_TTL: int        = int(os.getenv("JSEARCH_CACHE_TTL", "3600"))  # seconds

# ── Paths ──────────────────────────────────────────────────────────────────────
CACHE_DIR: Path       = BASE_DIR / ".cache"
LOG_DIR: Path         = BASE_DIR / "logs"
RAG_DATA_DIR: Path    = BASE_DIR / "rag_data"
TEST_RESUMES_DIR: Path = BASE_DIR / "test_resumes"
SNAPSHOT_DIR: Path    = BASE_DIR / "snapshots"

for _d in (CACHE_DIR, LOG_DIR, RAG_DATA_DIR, TEST_RESUMES_DIR, SNAPSHOT_DIR):
    _d.mkdir(exist_ok=True)

# ── Scoring weights ────────────────────────────────────────────────────────────
SEMANTIC_WEIGHT: float   = float(os.getenv("SEMANTIC_WEIGHT", "0.55"))
SKILL_WEIGHT: float      = float(os.getenv("SKILL_WEIGHT", "0.35"))
SENIORITY_WEIGHT: float  = float(os.getenv("SENIORITY_WEIGHT", "0.10"))

# ── Resume scoring ─────────────────────────────────────────────────────────────
MIN_WORDS: int = 200
MAX_WORDS: int = 900
MIN_SKILLS: int = 5
MIN_BULLETS_WITH_NUMS: int = 3

SECTIONS_REQUIRED = [
    "education", "experience", "skills", "projects"
]
SECTIONS_OPTIONAL = [
    "certifications", "summary", "objective", "awards", "publications"
]

# ── Skill taxonomy ─────────────────────────────────────────────────────────────
SKILL_ALIASES: dict[str, str] = {
    "ml":                "machine learning",
    "dl":                "deep learning",
    "cv":                "computer vision",
    "nlp":               "natural language processing",
    "natural language processing": "nlp",
    "tf":                "tensorflow",
    "sk-learn":          "scikit-learn",
    "sklearn":           "scikit-learn",
    "scikit learn":      "scikit-learn",
    "torch":             "pytorch",
    "hf":                "huggingface",
    "hugging face":      "huggingface",
    "gpt":               "openai api",
    "generative ai":     "genai",
    "gen ai":            "genai",
    "large language model": "llm",
    "large language models": "llm",
    "retrieval augmented generation": "rag",
    "k8s":               "kubernetes",
    "gke":               "kubernetes",
    "git hub":           "git",
    "ms sql":            "sql",
    "mysql":             "sql",
    "postgresql":        "sql",
    "postgres":          "sql",
    "pbi":               "power bi",
    "powerbi":           "power bi",
    "js":                "javascript",
    "ts":                "typescript",
    "node":              "node.js",
    "nodejs":            "node.js",
    "react.js":          "react",
    "reactjs":           "react",
    "vue.js":            "vue",
    "vuejs":             "vue",
}

CANONICAL_SKILLS: list[str] = [
    # Core programming
    "python", "r", "java", "c++", "c#", "javascript", "typescript",
    "golang", "rust", "scala", "julia", "bash",
    # Data
    "sql", "pandas", "numpy", "spark", "hadoop", "dbt", "airflow",
    "kafka", "elasticsearch",
    # ML / AI
    "machine learning", "deep learning", "scikit-learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost",
    "computer vision", "nlp", "transformers", "huggingface",
    "llm", "rag", "langchain", "openai api", "genai",
    "prompt engineering", "fine-tuning", "federated learning",
    "reinforcement learning",
    # MLOps / DevOps
    "mlops", "docker", "kubernetes", "git", "github actions", "ci/cd",
    "aws", "azure", "gcp", "terraform", "fastapi", "flask",
    "streamlit", "gradio",
    # BI / Analytics
    "power bi", "tableau", "excel", "looker", "dax",
    # Soft/Other
    "statistics", "linear algebra", "data analysis", "data engineering",
    "node.js", "react", "vue", "mongodb", "redis", "postgresql",
    "opencv", "scipy", "matplotlib", "seaborn", "plotly",
]

# ── Weak-bullet patterns ───────────────────────────────────────────────────────
WEAK_BULLET_PATTERNS: list[str] = [
    r"^(worked on|helped with|assisted in|responsible for|involved in)",
    r"^(did|made|used|handled|managed|performed)\b",
    r"^(various|multiple|different|several)\b",
]

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL: str   = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: Path   = LOG_DIR / "app.log"

# ── UI ─────────────────────────────────────────────────────────────────────────
APP_TITLE   = "ResumeIQ – AI Resume Analyzer & Job Recommender"
APP_ICON    = "🎯"
TOP_K_JOBS  = 10
