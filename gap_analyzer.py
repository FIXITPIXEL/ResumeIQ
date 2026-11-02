"""
gap_analyzer.py – AI-Powered Skill Gap Analysis.
Compares resume skills against target job requirements and generates
actionable learning recommendations using LLM reasoning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from extractor import ResumeStructured, normalize_skills
from llm_client import llm_complete
from logger import get_logger

log = get_logger("gap_analyzer")


@dataclass
class SkillGap:
    """Represents a single skill gap with learning recommendations."""
    skill: str
    priority: str           # "Critical", "Important", "Nice-to-Have"
    category: str            # "Technical", "Tool", "Framework", "Soft Skill"
    recommendation: str      # LLM-generated learning path
    estimated_time: str      # e.g., "2-4 weeks"
    resources: list[str] = field(default_factory=list)


@dataclass
class GapAnalysis:
    """Complete gap analysis result."""
    matched_skills: list[str]
    missing_skills: list[str]
    gap_details: list[SkillGap]
    overall_readiness: float      # 0–100
    summary: str                  # LLM-generated summary
    action_plan: str              # LLM-generated action plan
    method: str = "AI (LLM)"
    error_message: Optional[str] = None


def _categorize_skill(skill: str) -> str:
    """Categorize a skill into a broad category."""
    frameworks = {"react", "vue", "angular", "flask", "fastapi", "django", "streamlit",
                  "langchain", "pytorch", "tensorflow", "keras", "scikit-learn",
                  "xgboost", "lightgbm", "huggingface", "gradio"}
    tools = {"docker", "kubernetes", "git", "aws", "azure", "gcp", "terraform",
             "airflow", "kafka", "spark", "hadoop", "tableau", "power bi",
             "mlops", "ci/cd", "github actions", "elasticsearch"}
    languages = {"python", "r", "java", "c++", "c#", "javascript", "typescript",
                 "golang", "rust", "scala", "julia", "bash", "sql"}

    sl = skill.lower()
    if sl in frameworks:
        return "Framework"
    elif sl in tools:
        return "Tool / Platform"
    elif sl in languages:
        return "Programming Language"
    else:
        return "Technical Skill"


def _prioritize_skill(skill: str, job_title: str) -> str:
    """Heuristically prioritize a missing skill."""
    sl = skill.lower()
    title_lower = job_title.lower()

    # Core skills matching job title keywords
    critical_map = {
        "ml": ["machine learning", "deep learning", "pytorch", "tensorflow", "scikit-learn"],
        "data": ["python", "sql", "pandas", "spark", "data analysis"],
        "ai": ["llm", "nlp", "genai", "langchain", "rag", "transformers"],
        "backend": ["python", "java", "fastapi", "flask", "docker", "kubernetes"],
        "frontend": ["javascript", "typescript", "react", "vue"],
    }

    for keyword, critical_skills in critical_map.items():
        if keyword in title_lower and sl in critical_skills:
            return "Critical"

    # General importance heuristic
    high_demand = {"python", "sql", "docker", "aws", "machine learning", "deep learning",
                   "llm", "genai", "pytorch", "tensorflow"}
    if sl in high_demand:
        return "Important"

    return "Nice-to-Have"


def _estimate_learning_time(skill: str) -> str:
    """Estimate time to learn a skill."""
    advanced = {"machine learning", "deep learning", "nlp", "computer vision",
                "reinforcement learning", "federated learning", "mlops"}
    moderate = {"pytorch", "tensorflow", "kubernetes", "spark", "langchain",
                "rag", "llm", "genai", "aws", "azure", "gcp"}

    sl = skill.lower()
    if sl in advanced:
        return "4-8 weeks"
    elif sl in moderate:
        return "2-4 weeks"
    else:
        return "1-2 weeks"


def _heuristic_gap_recommendation(skill: str) -> str:
    """Generate a rule-based learning recommendation when LLM is unavailable."""
    category = _categorize_skill(skill)
    time_est = _estimate_learning_time(skill)

    templates = {
        "Framework": f"Learn {skill} through official documentation tutorials and build a small project. Practice with real-world datasets and deploy a demo. Estimated time: {time_est}.",
        "Tool / Platform": f"Get hands-on with {skill} through guided labs and certification prep. Focus on practical deployment scenarios. Estimated time: {time_est}.",
        "Programming Language": f"Complete an interactive course on {skill}, then solve coding challenges on LeetCode/HackerRank. Build a portfolio project. Estimated time: {time_est}.",
        "Technical Skill": f"Study {skill} fundamentals through online courses (Coursera/Udemy), then apply concepts in a hands-on project. Estimated time: {time_est}.",
    }
    return templates.get(category, templates["Technical Skill"])


def _heuristic_resources(skill: str) -> list[str]:
    """Generate default learning resource suggestions."""
    return [
        f"Official {skill} documentation & tutorials",
        f"Coursera / Udemy courses on {skill}",
        f"YouTube: '{skill} crash course' or '{skill} project tutorial'",
        f"GitHub: Search for '{skill} examples' / '{skill} projects'",
    ]


def analyze_skill_gaps(
    resume: ResumeStructured,
    job_description: str = "",
    job_title: str = "",
) -> GapAnalysis:
    """
    Perform AI-powered skill gap analysis comparing resume skills
    against job requirements. Uses LLM for detailed recommendations.
    """
    # Extract skills from job description
    if job_description:
        job_skills = normalize_skills(job_description)
    else:
        job_skills = []

    resume_skill_set = set(s.lower() for s in resume.skills)
    job_skill_set = set(s.lower() for s in job_skills)

    matched = sorted(resume_skill_set & job_skill_set)
    missing = sorted(job_skill_set - resume_skill_set)

    # Calculate readiness
    if job_skill_set:
        readiness = (len(matched) / len(job_skill_set)) * 100
    else:
        readiness = 100.0

    # Generate gap details
    gap_details: list[SkillGap] = []
    method = "AI (LLM)"
    error_message: Optional[str] = None

    for skill in missing:
        priority = _prioritize_skill(skill, job_title)
        category = _categorize_skill(skill)
        time_est = _estimate_learning_time(skill)

        # Try LLM for recommendation
        try:
            prompt = f"""You are a career development advisor. Generate a concise, actionable learning recommendation for someone who needs to learn "{skill}".

Context:
- Their current skills: {', '.join(resume.skills[:15])}
- Target role: {job_title or 'Data Science / ML Engineer'}
- Skill category: {category}
- Priority: {priority}

Provide in 2-3 sentences:
1. What to learn first (most impactful aspect)
2. Best resource type (course, project, certification)
3. How to demonstrate this skill on a resume

Respond with ONLY the recommendation text, no headers or bullet points."""

            recommendation = llm_complete(prompt, max_tokens=150)
            if not recommendation or recommendation.startswith("[LLM unavailable"):
                raise ValueError("LLM unavailable")
            resources = _heuristic_resources(skill)

        except Exception as e:
            recommendation = _heuristic_gap_recommendation(skill)
            resources = _heuristic_resources(skill)
            if not error_message:
                error_message = f"LLM unavailable: {str(e)[:60]}"
                method = "Hybrid (Heuristic + AI)"

        gap_details.append(SkillGap(
            skill=skill,
            priority=priority,
            category=category,
            recommendation=recommendation,
            estimated_time=time_est,
            resources=resources,
        ))

    # Sort by priority
    priority_order = {"Critical": 0, "Important": 1, "Nice-to-Have": 2}
    gap_details.sort(key=lambda g: priority_order.get(g.priority, 3))

    # Generate overall summary and action plan with LLM
    summary = ""
    action_plan = ""

    try:
        summary_prompt = f"""You are a career advisor. Provide a 3-4 sentence skill gap summary.

Candidate skills: {', '.join(resume.skills[:20])}
Target role: {job_title or 'Data Science / ML Engineer'}
Matching skills: {', '.join(matched[:10])}
Missing skills: {', '.join(missing[:10])}
Readiness: {readiness:.0f}%

Be encouraging but honest. Mention strongest areas and top gaps. Respond with ONLY the summary."""

        summary = llm_complete(summary_prompt, max_tokens=200)
        if summary.startswith("[LLM unavailable"):
            raise ValueError("LLM unavailable")

        plan_prompt = f"""Create a concise 4-week action plan for bridging these skill gaps:
Missing skills (by priority): {', '.join(s.skill for s in gap_details[:6])}
Current strengths: {', '.join(matched[:8])}

Format as:
Week 1: [focus area]
Week 2: [focus area]
Week 3: [focus area]
Week 4: [focus area]

Keep each week to ONE sentence. Respond with ONLY the plan."""

        action_plan = llm_complete(plan_prompt, max_tokens=200)
        if action_plan.startswith("[LLM unavailable"):
            raise ValueError("LLM unavailable")

    except Exception:
        if not summary:
            summary = _heuristic_summary(matched, missing, readiness, job_title)
        if not action_plan:
            action_plan = _heuristic_action_plan(gap_details)
        method = "Hybrid (Heuristic + AI)" if method != "AI (LLM)" else method

    return GapAnalysis(
        matched_skills=matched,
        missing_skills=missing,
        gap_details=gap_details,
        overall_readiness=round(readiness, 1),
        summary=summary,
        action_plan=action_plan,
        method=method,
        error_message=error_message,
    )


def _heuristic_summary(matched: list[str], missing: list[str], readiness: float, job_title: str) -> str:
    """Generate a rule-based summary when LLM is unavailable."""
    role = job_title or "the target role"
    parts = []
    if readiness >= 80:
        parts.append(f"You are highly prepared for {role} with {readiness:.0f}% skill alignment.")
    elif readiness >= 50:
        parts.append(f"You have a solid foundation for {role} with {readiness:.0f}% skill match.")
    else:
        parts.append(f"There is room for growth toward {role} with {readiness:.0f}% current alignment.")

    if matched:
        parts.append(f"Your strengths in {', '.join(matched[:3])} provide a strong technical base.")
    if missing:
        parts.append(f"Focus on acquiring {', '.join(missing[:3])} to strengthen your candidacy.")
    return " ".join(parts)


def _heuristic_action_plan(gaps: list[SkillGap]) -> str:
    """Generate a rule-based 4-week plan when LLM is unavailable."""
    critical = [g for g in gaps if g.priority == "Critical"]
    important = [g for g in gaps if g.priority == "Important"]
    nice = [g for g in gaps if g.priority == "Nice-to-Have"]

    lines = []
    if critical:
        lines.append(f"Week 1: Deep-dive into {critical[0].skill} — the most critical gap for your target role.")
        if len(critical) > 1:
            lines.append(f"Week 2: Build proficiency in {critical[1].skill} through hands-on projects.")
        elif important:
            lines.append(f"Week 2: Strengthen {important[0].skill} to complement your existing skill set.")
    elif important:
        lines.append(f"Week 1: Start with {important[0].skill} — high-impact for your career goals.")
        if len(important) > 1:
            lines.append(f"Week 2: Level up in {important[1].skill} through practical application.")

    if len(lines) < 2:
        lines.append("Week 2: Practice by building a portfolio project combining your new and existing skills.")

    lines.append("Week 3: Build a portfolio project that demonstrates your newly acquired skills in a real-world context.")
    lines.append("Week 4: Refine your resume, update your GitHub portfolio, and prepare for technical interviews.")

    return "\n".join(lines)
