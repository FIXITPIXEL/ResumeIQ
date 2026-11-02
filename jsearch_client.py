"""
jsearch_client.py – Milestone 3: JSearch API client with disk caching and credit cap.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import requests

from config import (
    CACHE_DIR,
    JSEARCH_API_KEY,
    JSEARCH_BASE_URL,
    JSEARCH_CREDIT_CAP,
    JSEARCH_HOST,
    JSEARCH_PROVIDER,
)
from logger import get_logger
from llm_client import llm_complete

log = get_logger("jsearch_client")

_CREDIT_COUNTER_FILE = CACHE_DIR / "credit_counter.json"


def _get_api_url_and_headers() -> tuple[str, dict, str]:
    """Determine URL, headers, and provider name (OpenWebNinja direct vs RapidAPI)."""
    key = JSEARCH_API_KEY.strip()
    is_own = key.startswith("ak_") or JSEARCH_PROVIDER == "openwebninja"
    if is_own:
        url = "https://api.openwebninja.com/jsearch/search"
        headers = {"x-api-key": key}
        return url, headers, "OpenWeb Ninja"
    else:
        url = "https://jsearch.p.rapidapi.com/search"
        headers = {
            "X-RapidAPI-Key": key,
            "X-RapidAPI-Host": JSEARCH_HOST,
        }
        return url, headers, "RapidAPI"

_CREDIT_COUNTER_FILE = CACHE_DIR / "credit_counter.json"


# ── Credit tracking & status ───────────────────────────────────────────────────

class CreditCapExceeded(Exception):
    pass


_last_api_status: dict = {
    "checked": False,
    "success": False,
    "status_code": None,
    "message": "",
    "error_type": "",
}


def get_last_api_status() -> dict:
    return _last_api_status


def _load_credits() -> dict:
    if _CREDIT_COUNTER_FILE.exists():
        try:
            return json.loads(_CREDIT_COUNTER_FILE.read_text())
        except Exception:
            pass
    return {"used": 0, "reset_ts": time.time() + 86400}


def _save_credits(state: dict) -> None:
    _CREDIT_COUNTER_FILE.write_text(json.dumps(state))


def _can_consume_credit() -> bool:
    state = _load_credits()
    if time.time() > state["reset_ts"]:
        return True
    return state["used"] < JSEARCH_CREDIT_CAP


def _record_consumed_credit() -> None:
    state = _load_credits()
    if time.time() > state["reset_ts"]:
        state = {"used": 0, "reset_ts": time.time() + 86400}
    state["used"] += 1
    _save_credits(state)
    log.debug("JSearch credits used: %d/%d", state["used"], JSEARCH_CREDIT_CAP)


def credits_remaining() -> int:
    state = _load_credits()
    if time.time() > state["reset_ts"]:
        return JSEARCH_CREDIT_CAP
    return max(0, JSEARCH_CREDIT_CAP - state["used"])


def test_jsearch_connection() -> tuple[bool, str]:
    """Test JSearch connectivity with a live test query."""
    if not JSEARCH_API_KEY:
        return False, "JSEARCH_API_KEY is not set in .env"

    url, headers, provider_name = _get_api_url_and_headers()
    params = {
        "query": "software engineer",
        "page": "1",
        "num_pages": "1",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data.get("data", [])
            _last_api_status.update({
                "checked": True,
                "success": True,
                "status_code": 200,
                "message": f"Connected to {provider_name}! Returned {len(jobs)} live jobs.",
                "error_type": "",
            })
            return True, f"Connected to {provider_name} successfully! ({len(jobs)} live jobs received)"
        elif resp.status_code == 403:
            if provider_name == "OpenWeb Ninja":
                msg = "OpenWeb Ninja 403: Invalid or unpaid API key. Check dashboard at https://app.openwebninja.com/"
            else:
                msg = "RapidAPI 403: You are not subscribed to JSearch. Go to https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch/pricing and click 'Subscribe to Test' (Free tier)."
            _last_api_status.update({
                "checked": True,
                "success": False,
                "status_code": 403,
                "message": msg,
                "error_type": "NOT_SUBSCRIBED" if provider_name != "OpenWeb Ninja" else "INVALID_KEY",
            })
            return False, msg
        elif resp.status_code == 401:
            msg = f"{provider_name} 401: Invalid API Key. Please verify JSEARCH_API_KEY in .env."
            _last_api_status.update({
                "checked": True,
                "success": False,
                "status_code": 401,
                "message": msg,
                "error_type": "INVALID_KEY",
            })
            return False, msg
        else:
            msg = f"{provider_name} HTTP {resp.status_code}: {resp.text[:120]}"
            _last_api_status.update({
                "checked": True,
                "success": False,
                "status_code": resp.status_code,
                "message": msg,
                "error_type": "HTTP_ERROR",
            })
            return False, msg
    except Exception as e:
        msg = f"{provider_name} connection error: {e}"
        _last_api_status.update({
            "checked": True,
            "success": False,
            "status_code": None,
            "message": msg,
            "error_type": "CONNECTION_ERROR",
        })
        return False, msg


# ── Disk cache ─────────────────────────────────────────────────────────────────

def _cache_key(query: str, page: int) -> Path:
    h = hashlib.md5(f"{query}|{page}".encode()).hexdigest()
    return CACHE_DIR / f"jsearch_{h}.json"


def _read_cache(path: Path, ttl: int) -> Optional[list[dict]]:
    if path.exists() and (time.time() - path.stat().st_mtime) < ttl:
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return None


def _write_cache(path: Path, data: list[dict]) -> None:
    path.write_text(json.dumps(data, indent=2))


# ── LLM query generation ───────────────────────────────────────────────────────

def generate_search_queries(skills: list[str], target_role: str = "") -> list[str]:
    """
    Use the LLM to generate diverse JSearch query strings from resume skills.
    Returns 3–5 queries.
    """
    if not skills:
        return [target_role or "software engineer intern"]

    prompt = f"""You are a job search query optimizer.
Given these skills extracted from a resume: {', '.join(skills[:20])}
{"And target role: " + target_role if target_role else ""}

Generate 4 concise job search queries optimized for the JSearch (Indeed/LinkedIn) API.
Rules:
- Each query should be 3-7 words
- Mix broad and specific queries
- Focus on roles that match the skill set
- Output ONLY a JSON array of strings, nothing else

Example output: ["machine learning engineer", "python data scientist", "mlops engineer", "AI engineer pytorch"]"""

    try:
        raw = llm_complete(prompt, system="You output only valid JSON arrays.")
        # Extract JSON array
        import re
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        if m:
            queries = json.loads(m.group())
            queries = [q for q in queries if isinstance(q, str) and len(q) > 2]
            log.info("LLM generated %d search queries", len(queries))
            return queries[:5]
    except Exception as e:
        log.warning("LLM query generation failed: %s", e)

    # Fallback: simple heuristic
    top_skills = skills[:3]
    return [f"{' '.join(top_skills)} developer", f"{top_skills[0]} engineer"]


# ── API call ───────────────────────────────────────────────────────────────────

def _fetch_from_api(query: str, page: int = 1, num_pages: int = 1) -> list[dict]:
    if not JSEARCH_API_KEY:
        log.warning("JSEARCH_API_KEY not set – returning empty results")
        return []

    if not _can_consume_credit():
        log.warning("JSearch credit cap (%d) reached", JSEARCH_CREDIT_CAP)
        return []

    url, headers, provider_name = _get_api_url_and_headers()
    params = {
        "query": query,
        "page": str(page),
        "num_pages": str(num_pages),
        "date_posted": "month",
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=25)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data.get("data", [])
            _record_consumed_credit()
            _last_api_status.update({
                "checked": True,
                "success": True,
                "status_code": 200,
                "message": f"Returned {len(jobs)} jobs from {provider_name} for '{query}'",
                "error_type": "",
            })
            log.info("%s returned %d jobs for query '%s'", provider_name, len(jobs), query)
            return jobs
        elif resp.status_code == 403:
            msg = f"{provider_name} 403: Access forbidden / subscription required."
            _last_api_status.update({
                "checked": True,
                "success": False,
                "status_code": 403,
                "message": msg,
                "error_type": "NOT_SUBSCRIBED",
            })
            log.error("JSearch 403 (%s): %s", provider_name, msg)
            return []
        else:
            resp.raise_for_status()
            return []
    except Exception as e:
        log.error("JSearch API error (%s): %s", provider_name, e)
        _last_api_status.update({
            "checked": True,
            "success": False,
            "status_code": getattr(getattr(e, "response", None), "status_code", None),
            "message": str(e),
            "error_type": "ERROR",
        })
        return []


def search_jobs(
    query: str,
    page: int = 1,
    ttl: int = 3600,
    use_cache: bool = True,
) -> list[dict]:
    """Search jobs with caching. Returns list of raw JSearch job dicts."""
    cache_path = _cache_key(query, page)
    if use_cache:
        cached = _read_cache(cache_path, ttl)
        if cached is not None:
            log.debug("Cache hit for query '%s'", query)
            return cached

    jobs = _fetch_from_api(query, page)
    if jobs and use_cache:
        _write_cache(cache_path, jobs)
    return jobs


# ── Candidate pool builder ─────────────────────────────────────────────────────

def build_candidate_pool(
    skills: list[str],
    target_role: str = "",
    max_jobs: int = 50,
    use_cache: bool = True,
) -> list[dict]:
    """
    Build a de-duplicated candidate job pool from multiple LLM-generated queries.
    """
    queries = generate_search_queries(skills, target_role)
    seen_ids: set[str] = set()
    pool: list[dict] = []

    for query in queries:
        if len(pool) >= max_jobs:
            break
        try:
            jobs = search_jobs(query, use_cache=use_cache)
            for job in jobs:
                jid = job.get("job_id", "")
                if jid and jid not in seen_ids:
                    seen_ids.add(jid)
                    pool.append(job)
        except CreditCapExceeded as e:
            log.warning(str(e))
            break

    log.info("Candidate pool: %d unique jobs from %d queries", len(pool), len(queries))
    return pool[:max_jobs]


# ── Sample jobs (offline fallback) ────────────────────────────────────────────

SAMPLE_JOBS: list[dict] = [
    {
        "job_id": "sample_001",
        "job_title": "Machine Learning Engineer",
        "employer_name": "TechCorp AI",
        "job_city": "San Francisco",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """We are seeking an ML Engineer to build and scale machine learning systems.
Requirements:
- Proficiency in Python, pytorch, tensorflow, scikit-learn
- Experience with mlops, docker, kubernetes, CI/CD pipelines
- Deep learning model training and deployment with fastapi
- Familiarity with LLMs, transformers, and huggingface ecosystem
- SQL databases and data pipelines with airflow or kafka
- Strong fundamentals in statistics and machine learning""",
        "job_min_salary": 120000,
        "job_max_salary": 160000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/ml-engineer",
    },
    {
        "job_id": "sample_002",
        "job_title": "Data Scientist",
        "employer_name": "DataDriven Inc",
        "job_city": "New York",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Join our data science team to derive insights from large datasets.
Requirements:
- Python, pandas, numpy, scikit-learn, and statistics
- SQL for data querying and manipulation
- Data analysis, visualization with tableau or power bi
- Machine learning model development and evaluation
- A/B testing and experimentation frameworks
- Communication skills to present findings""",
        "job_min_salary": 100000,
        "job_max_salary": 140000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/data-scientist",
    },
    {
        "job_id": "sample_003",
        "job_title": "GenAI / LLM Engineer",
        "employer_name": "AI Frontier",
        "job_city": "Remote",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Build cutting-edge generative AI applications and pipelines.
Requirements:
- Python, langchain, rag, prompt engineering, openai api
- huggingface transformers, fine-tuning LLMs
- fastapi, streamlit, gradio for serving AI apps
- Vector databases (Pinecone, Chroma, Weaviate)
- genai, llm architecture understanding
- docker, git for deployment""",
        "job_min_salary": 130000,
        "job_max_salary": 180000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/genai-engineer",
    },
    {
        "job_id": "sample_004",
        "job_title": "Computer Vision Engineer",
        "employer_name": "VisionAI Labs",
        "job_city": "Seattle",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Develop computer vision models for real-time applications.
Requirements:
- python, opencv, pytorch, tensorflow, keras
- deep learning, computer vision pipeline development
- Object detection, segmentation, tracking algorithms
- ONNX, TensorRT model optimization
- docker, kubernetes for deployment
- Experience with video analytics""",
        "job_min_salary": 115000,
        "job_max_salary": 155000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/cv-engineer",
    },
    {
        "job_id": "sample_005",
        "job_title": "MLOps Engineer",
        "employer_name": "CloudScale",
        "job_city": "Austin",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Own the ML infrastructure and deployment pipelines.
Requirements:
- mlops, docker, kubernetes, terraform
- aws or azure or gcp cloud platforms
- CI/CD, github actions, Jenkins
- python scripting and automation
- airflow, kafka, spark for data pipelines
- Monitoring with Prometheus, Grafana
- ML model serving with fastapi, mlflow""",
        "job_min_salary": 125000,
        "job_max_salary": 165000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/mlops-engineer",
    },
    {
        "job_id": "sample_006",
        "job_title": "Data Engineer",
        "employer_name": "DataPipeline Co",
        "job_city": "Chicago",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Design and maintain robust data infrastructure.
Requirements:
- python, sql, spark, hadoop, kafka
- airflow for orchestration and scheduling
- AWS Redshift, Glue, or Azure Data Factory
- dbt for data transformation
- data engineering best practices and data modeling
- Strong SQL and database design skills
- elasticsearch for search indexing""",
        "job_min_salary": 110000,
        "job_max_salary": 150000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/data-engineer",
    },
    {
        "job_id": "sample_007",
        "job_title": "NLP Research Scientist",
        "employer_name": "LanguageAI",
        "job_city": "Boston",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Advance the state of NLP research and applications.
Requirements:
- nlp, transformers, pytorch, huggingface, fine-tuning
- python, research paper implementation
- rag, llm, prompt engineering experience
- Information retrieval, text classification, NER
- Strong statistics and linear algebra background
- Publication record or equivalent industry work""",
        "job_min_salary": 140000,
        "job_max_salary": 190000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/nlp-scientist",
    },
    {
        "job_id": "sample_008",
        "job_title": "AI Product Manager",
        "employer_name": "ProductAI",
        "job_city": "San Jose",
        "job_country": "US",
        "job_employment_type": "FULLTIME",
        "job_description": """Drive AI product strategy and roadmap.
Requirements:
- Understanding of machine learning, llm, genai concepts
- data analysis, sql, excel for product metrics
- Experience with agile, product roadmap, stakeholder management
- python scripting for data exploration a plus
- Ability to communicate ML concepts to business stakeholders""",
        "job_min_salary": 130000,
        "job_max_salary": 170000,
        "job_salary_currency": "USD",
        "job_apply_link": "https://example.com/jobs/ai-pm",
    },
]
