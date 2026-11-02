"""
extractor.py – Milestone 2: NER, skill normalization, and structured resume JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

from config import (
    CANONICAL_SKILLS,
    SKILL_ALIASES,
    WEAK_BULLET_PATTERNS,
    MIN_WORDS,
    MAX_WORDS,
    MIN_SKILLS,
    MIN_BULLETS_WITH_NUMS,
    SECTIONS_REQUIRED,
)
from logger import get_logger
from pdf_parser import ResumeRaw

log = get_logger("extractor")

# Pre-compiled skill matchers
_SKILL_PATTERNS = {
    skill: re.compile(
        r"(?<![a-z\-])" + re.escape(skill) + r"(?![a-z\-])",
        re.IGNORECASE,
    )
    for skill in CANONICAL_SKILLS
}

# Pre-compiled alias matchers
_ALIAS_PATTERNS = {
    alias: re.compile(
        r"(?<![a-z\-])" + re.escape(alias) + r"(?![a-z\-])",
        re.IGNORECASE,
    )
    for alias in SKILL_ALIASES
}

_WEAK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in WEAK_BULLET_PATTERNS]

# Regex helpers
_EMAIL_RE   = re.compile(r"[\w.\-+]+@[\w\-]+\.[a-z]{2,}", re.IGNORECASE)
_PHONE_RE   = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_URL_RE     = re.compile(r"https?://[^\s]+|linkedin\.com/[^\s]+|github\.com/[^\s]+", re.IGNORECASE)
_YEAR_RE    = re.compile(r"\b(19|20)\d{2}\b")
_NUM_RE     = re.compile(r"\d+%|\d+\+|\b\d{2,}\b")
_BULLET_RE  = re.compile(r"^[\u2022\u2023\u25e6\u25cf\-\*\+•]\s*", re.MULTILINE)
_BULLET_START = re.compile(r"^[\u2022\u2023\u25e6\u25cf\-\*\+•]\s*")

# Degree keywords
_DEGREE_MAP = {
    "b.tech": "B.Tech", "b.e": "B.E", "b.sc": "B.Sc", "b.s": "B.S",
    "m.tech": "M.Tech", "m.s": "M.S", "m.sc": "M.Sc",
    "ph.d": "Ph.D", "phd": "Ph.D", "mba": "MBA",
    "bachelor": "Bachelor's", "master": "Master's", "doctoral": "Ph.D",
}


@dataclass
class ContactInfo:
    name: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    github: str = ""
    location: str = ""


@dataclass
class Education:
    degree: str = ""
    institution: str = ""
    year: str = ""
    gpa: str = ""


@dataclass
class Experience:
    title: str = ""
    company: str = ""
    duration: str = ""
    bullets: list[str] = field(default_factory=list)
    weak_bullets: list[str] = field(default_factory=list)


@dataclass
class ResumeStructured:
    contact: ContactInfo = field(default_factory=ContactInfo)
    education: list[Education] = field(default_factory=list)
    experience: list[Experience] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    certifications: list[str] = field(default_factory=list)
    raw_sections: dict[str, str] = field(default_factory=dict)
    score: int = 0
    tips: list[str] = field(default_factory=list)
    word_count: int = 0
    quantified_count: int = 0

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent)


# ── Skill normalization ────────────────────────────────────────────────────────

def normalize_skills(text: str) -> list[str]:
    """
    Extract and normalize skills from free text.
    1. Resolve aliases → canonical forms
    2. Match canonical skills with word-boundary patterns
    Returns a sorted, deduplicated list.
    """
    # 1. Alias substitution on a working copy
    working = text
    for alias, canonical in SKILL_ALIASES.items():
        pat = _ALIAS_PATTERNS.get(alias)
        if pat and pat.search(working):
            working = pat.sub(canonical, working)

    # 2. Match canonicals
    found: set[str] = set()
    for skill, pat in _SKILL_PATTERNS.items():
        if pat.search(working):
            found.add(skill)

    result = sorted(found)
    log.debug("Normalized skills (%d): %s", len(result), result)
    return result


# ── Contact extraction ─────────────────────────────────────────────────────────

def _extract_contact(header_text: str) -> ContactInfo:
    c = ContactInfo()

    emails = _EMAIL_RE.findall(header_text)
    c.email = emails[0] if emails else ""

    phones = _PHONE_RE.findall(header_text)
    c.phone = phones[0].strip() if phones else ""

    urls = _URL_RE.findall(header_text)
    for u in urls:
        ul = u.lower()
        if "linkedin" in ul:
            c.linkedin = u
        elif "github" in ul:
            c.github = u

    # Name heuristic: first non-empty line of header that isn't email/phone/url
    for line in header_text.splitlines():
        stripped = line.strip()
        if stripped and not _EMAIL_RE.search(stripped) and not _PHONE_RE.search(stripped) and not _URL_RE.search(stripped):
            if len(stripped.split()) <= 6:  # names are short
                c.name = stripped
                break

    return c


# ── Education extraction ───────────────────────────────────────────────────────

def _extract_education(edu_text: str) -> list[Education]:
    results: list[Education] = []
    lines = [l.strip() for l in edu_text.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        line = lines[i].lower()
        degree_key = next((k for k in _DEGREE_MAP if k in line), None)
        if degree_key:
            edu = Education()
            edu.degree = _DEGREE_MAP[degree_key]
            edu.institution = lines[i]  # use full line as institution placeholder
            years = _YEAR_RE.findall(lines[i])
            if not years and i + 1 < len(lines):
                years = _YEAR_RE.findall(lines[i + 1])
            edu.year = " – ".join(years[-2:]) if len(years) >= 2 else (years[0] if years else "")
            gpa_m = re.search(r"(gpa|cgpa|grade)[:\s]+([0-9.]+)", lines[i], re.IGNORECASE)
            if gpa_m:
                edu.gpa = gpa_m.group(2)
            results.append(edu)
        i += 1
    return results or [Education(institution="See full text")]


# ── Experience extraction ──────────────────────────────────────────────────────

def _is_weak_bullet(bullet: str) -> bool:
    return any(p.search(bullet.strip()) for p in _WEAK_PATTERNS)


def _extract_experience(exp_text: str) -> list[Experience]:
    if not exp_text.strip():
        return []

    # ── Step 1: join wrapped bullet lines ─────────────────────────────────────
    # PDFs often wrap long bullets onto the next line without a bullet marker.
    # We join a continuation line (doesn't start with a bullet/year/caps-word)
    # back to its parent bullet.
    raw_lines = exp_text.splitlines()
    joined: list[str] = []
    for raw_line in raw_lines:
        stripped = raw_line.strip()
        if not stripped:
            joined.append("")  # preserve blank separators
            continue
        if _BULLET_START.match(stripped):
            joined.append(stripped)  # new bullet
        elif joined and joined[-1] and not _BULLET_START.match(joined[-1]) and not _YEAR_RE.match(joined[-1]):
            # Non-bullet, non-year line after a non-bullet → new header or continuation
            # Heuristic: if previous joined line was a bullet, this is continuation
            if _BULLET_START.match(joined[-1].lstrip()):
                joined[-1] = joined[-1].rstrip() + " " + stripped
            else:
                joined.append(stripped)
        elif joined and joined[-1] and _BULLET_START.match(joined[-1].lstrip()):
            # Previous line IS a bullet → this is a wrapped continuation
            joined[-1] = joined[-1].rstrip() + " " + stripped
        else:
            joined.append(stripped)

    rejoined_text = "\n".join(joined)

    # ── Step 2: split into role blocks ────────────────────────────────────────
    blocks = re.split(r"\n{2,}", rejoined_text.strip())
    experiences: list[Experience] = []

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue
        exp = Experience()
        # First line → title / company
        header = lines[0]
        if "," in header and len(header.split(",")) >= 2:
            # e.g. "AI Model Trainer / Evaluator (Freelance), Outlier 2023 - Present"
            parts = header.split(",", 1)
            exp.title   = parts[0].strip()
            exp.company = parts[1].strip()
        elif " at " in header.lower():
            parts = re.split(r"\s+at\s+", header, flags=re.IGNORECASE)
            exp.title, exp.company = parts[0], parts[1]
        elif " | " in header:
            parts = header.split(" | ")
            exp.title, exp.company = parts[0], parts[1] if len(parts) > 1 else ""
        elif " – " in header or " - " in header:
            parts = re.split(r"\s[–\-]\s", header, maxsplit=1)
            exp.title, exp.company = parts[0], parts[1] if len(parts) > 1 else ""
        else:
            exp.title = header

        # Duration
        years = _YEAR_RE.findall(block)
        exp.duration = " – ".join(years[-2:]) if len(years) >= 2 else (years[0] if years else "")

        # Bullets — match any line starting with a bullet char or dash
        for line in lines[1:]:
            if _BULLET_START.match(line) or line.startswith("-"):
                bullet = _BULLET_START.sub("", line).strip()
                if bullet:
                    exp.bullets.append(bullet)
                    if _is_weak_bullet(bullet):
                        exp.weak_bullets.append(bullet)
        experiences.append(exp)

    return experiences


# ── Resume scoring ─────────────────────────────────────────────────────────────

def _score_resume(
    resume: ResumeStructured,
    full_text: str,
) -> tuple[int, list[str]]:
    tips: list[str] = []
    score = 0

    # 1. Sections (30 pts)
    found_sections = {k.lower() for k in resume.raw_sections}
    sections_found = sum(1 for s in SECTIONS_REQUIRED if any(s in k for k in found_sections))
    score += int(30 * sections_found / len(SECTIONS_REQUIRED))
    for s in SECTIONS_REQUIRED:
        if not any(s in k for k in found_sections):
            tips.append(f"❌ Missing required section: **{s.title()}**")

    # 2. Skills (25 pts)
    skill_score = min(len(resume.skills), 15) / 15 * 25
    score += int(skill_score)
    if len(resume.skills) < MIN_SKILLS:
        tips.append(f"⚠️ Only {len(resume.skills)} skills detected. Aim for at least {MIN_SKILLS}.")

    # 3. Quantified impact (20 pts)
    nums = len(_NUM_RE.findall(full_text))
    score += int(min(nums, 8) / 8 * 20)
    if nums < MIN_BULLETS_WITH_NUMS:
        tips.append("📊 Add quantified achievements (e.g., 'improved accuracy by 15%', 'reduced latency by 200ms').")

    # 4. Length (15 pts)
    words = len(full_text.split())
    if MIN_WORDS <= words <= MAX_WORDS:
        score += 15
    elif words < MIN_WORDS:
        score += 5
        tips.append(f"📝 Resume is too short ({words} words). Add more details to projects/experience.")
    else:
        score += 8
        tips.append(f"✂️ Resume is long ({words} words). Keep it concise (ideally ≤{MAX_WORDS} words).")

    # 5. Weak bullets (10 pts)
    all_weak = [b for exp in resume.experience for b in exp.weak_bullets]
    if not all_weak:
        score += 10
    elif len(all_weak) <= 2:
        score += 6
        tips.append(f"💪 {len(all_weak)} weak bullet(s) found. Use strong action verbs (Built, Developed, Achieved).")
    else:
        score += 2
        tips.append(f"💪 {len(all_weak)} weak bullets. Rewrite using the **STAR** format with strong verbs.")

    # Contact completeness tip
    c = resume.contact
    if not c.linkedin:
        tips.append("🔗 Add your **LinkedIn** profile URL.")
    if not c.github:
        tips.append("💻 Add your **GitHub** profile URL.")

    resume.word_count = words
    resume.quantified_count = nums
    return min(score, 100), tips


# ── Main extractor ─────────────────────────────────────────────────────────────

def extract_resume(raw: ResumeRaw) -> ResumeStructured:
    """Convert a ResumeRaw into a fully structured ResumeStructured."""
    result = ResumeStructured(raw_sections=raw.sections)

    # Contact
    header_text = raw.sections.get("header", raw.full_text[:500])
    result.contact = _extract_contact(header_text)

    # Education
    edu_text = raw.sections.get("education", "")
    result.education = _extract_education(edu_text)

    # Experience (try multiple keys)
    exp_keys = ["experience", "professional experience", "work experience", "internships"]
    exp_text = next((raw.sections[k] for k in exp_keys if k in raw.sections), "")
    result.experience = _extract_experience(exp_text)

    # Skills (from skills section + full text)
    skills_text = raw.sections.get("skills", "") + "\n" + raw.sections.get("technical skills", "") + "\n" + raw.sections.get("core competencies", "")
    result.skills = normalize_skills(raw.full_text)  # use full text for recall

    # Projects
    proj_text = raw.sections.get("projects", raw.sections.get("academic projects", raw.sections.get("personal projects", "")))
    result.projects = [l.strip() for l in proj_text.splitlines() if l.strip()][:10]

    # Certifications
    cert_text = raw.sections.get("certifications", "")
    result.certifications = [l.strip() for l in cert_text.splitlines() if l.strip()][:10]

    # Score
    result.score, result.tips = _score_resume(result, raw.full_text)

    log.info(
        "Extraction complete | name=%s | skills=%d | score=%d",
        result.contact.name, len(result.skills), result.score,
    )
    return result
