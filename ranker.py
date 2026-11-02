"""
ranker.py – Milestone 4: JD skill extraction, hybrid scoring, per-job explanations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import SEMANTIC_WEIGHT, SKILL_WEIGHT, SENIORITY_WEIGHT
from extractor import ResumeStructured, normalize_skills
from logger import get_logger
from llm_client import llm_complete

log = get_logger("ranker")

# Lazy-loaded sentence transformer
_model = None


def _get_embed_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBED_MODEL
        log.info("Loading embedding model: %s", EMBED_MODEL)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


# Salary parsing
_SALARY_RE = re.compile(r"\$?([\d,]+)\s*[–\-to]+\s*\$?([\d,]+)", re.IGNORECASE)
_SENIORITY_LEVELS = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3, "lead": 4, "principal": 5, "staff": 5,
}


@dataclass
class JobScore:
    job_id: str
    title: str
    company: str
    location: str
    employment_type: str
    match_score: float          # 0–100
    semantic_score: float
    skill_overlap: float
    seniority_fit: float
    matched_skills: list[str]
    missing_skills: list[str]
    job_skills: list[str]
    explanation: str = ""
    salary_range: str = ""
    apply_link: str = ""
    description_snippet: str = ""


def _parse_salary(job: dict) -> str:
    mn = job.get("job_min_salary")
    mx = job.get("job_max_salary")
    curr = job.get("job_salary_currency", "USD")
    if mn and mx:
        return f"${int(mn):,} – ${int(mx):,} {curr}/yr"
    desc = job.get("job_description", "")
    m = _SALARY_RE.search(desc)
    if m:
        return f"${m.group(1)} – ${m.group(2)}"
    return "Not disclosed"


def _seniority_score(title: str, resume: ResumeStructured) -> float:
    """Score 0–1 based on seniority fit."""
    title_lower = title.lower()
    exp_years = len(resume.experience)  # proxy: number of roles
    job_level = 2  # default mid
    for level, val in _SENIORITY_LEVELS.items():
        if level in title_lower:
            job_level = val
            break
    resume_level = min(exp_years, 4)  # 0..4
    diff = abs(job_level - resume_level)
    return max(0.0, 1.0 - diff * 0.25)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def _generate_explanation(
    resume: ResumeStructured,
    job: dict,
    score_obj: JobScore,
    use_llm: bool = True,
) -> str:
    """Generate a concise natural-language explanation for the match."""
    if use_llm:
        prompt = f"""You are a career counsellor. Explain in 2-3 short sentences why this resume is a {score_obj.match_score:.0f}% match for the job.

Resume skills: {', '.join(resume.skills[:15])}
Job title: {score_obj.title} at {score_obj.company}
Matched skills: {', '.join(score_obj.matched_skills[:8])}
Missing skills: {', '.join(score_obj.missing_skills[:5])}

Be direct and specific. Start with the match score context."""
        try:
            return llm_complete(prompt, max_tokens=200)
        except Exception as e:
            log.debug("LLM explanation failed: %s", e)

    # Rule-based fallback
    parts = []
    if score_obj.matched_skills:
        parts.append(f"Strong match on {', '.join(score_obj.matched_skills[:3])}.")
    if score_obj.missing_skills:
        parts.append(f"Consider adding: {', '.join(score_obj.missing_skills[:3])}.")
    if score_obj.match_score >= 75:
        parts.append("Excellent overall alignment.")
    elif score_obj.match_score >= 50:
        parts.append("Good fit with some skill gaps.")
    else:
        parts.append("Partial match – focus on bridging skill gaps.")
    return " ".join(parts)


def rank_jobs(
    resume: ResumeStructured,
    jobs: list[dict],
    top_k: int = 10,
    use_llm_explanations: bool = True,
) -> list[JobScore]:
    """
    Score and rank jobs against the resume using hybrid scoring:
    - Semantic similarity (sentence-transformers)
    - Skill overlap (Jaccard-like)
    - Seniority fit heuristic

    Returns top_k ranked JobScore objects.
    """
    if not jobs:
        return []

    model = _get_embed_model()

    # Encode resume
    resume_text = " ".join(resume.skills) + " " + " ".join(
        bullet for exp in resume.experience for bullet in exp.bullets
    )
    resume_emb = model.encode(resume_text, convert_to_numpy=True)

    # Encode all JDs
    descriptions = [j.get("job_description", "") for j in jobs]
    job_embs = model.encode(descriptions, convert_to_numpy=True, batch_size=32, show_progress_bar=False)

    resume_skill_set = set(resume.skills)
    scored: list[JobScore] = []

    for i, job in enumerate(jobs):
        jd = job.get("job_description", "")
        job_skills = normalize_skills(jd)
        job_skill_set = set(job_skills)

        # Skill overlap
        if job_skill_set:
            matched = sorted(resume_skill_set & job_skill_set)
            missing = sorted(job_skill_set - resume_skill_set)
            skill_overlap = len(matched) / len(job_skill_set)
        else:
            matched, missing, skill_overlap = [], [], 0.0

        # Semantic similarity
        sem_sim = _cosine_sim(resume_emb, job_embs[i])

        # Seniority fit
        title = job.get("job_title", "")
        senior_fit = _seniority_score(title, resume)

        # Hybrid score
        score = (
            SEMANTIC_WEIGHT  * sem_sim +
            SKILL_WEIGHT     * skill_overlap +
            SENIORITY_WEIGHT * senior_fit
        ) * 100

        score_obj = JobScore(
            job_id=job.get("job_id", str(i)),
            title=title,
            company=job.get("employer_name", "Unknown"),
            location=f"{job.get('job_city', '')} {job.get('job_country', '')}".strip(),
            employment_type=job.get("job_employment_type", ""),
            match_score=round(score, 1),
            semantic_score=round(sem_sim * 100, 1),
            skill_overlap=round(skill_overlap * 100, 1),
            seniority_fit=round(senior_fit * 100, 1),
            matched_skills=matched,
            missing_skills=missing,
            job_skills=job_skills,
            salary_range=_parse_salary(job),
            apply_link=job.get("job_apply_link", ""),
            description_snippet=jd[:300] + "..." if len(jd) > 300 else jd,
        )
        scored.append(score_obj)

    # Sort descending
    scored.sort(key=lambda x: x.match_score, reverse=True)
    top = scored[:top_k]

    # Generate explanations (only top jobs to save LLM calls)
    for js in top:
        job_dict = next((j for j in jobs if j.get("job_id") == js.job_id), {})
        js.explanation = _generate_explanation(resume, job_dict, js, use_llm=use_llm_explanations)

    log.info("Ranked %d jobs; top score=%.1f%%", len(top), top[0].match_score if top else 0)
    return top
