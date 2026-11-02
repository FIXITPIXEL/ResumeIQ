"""
pdf_parser.py – Milestone 1: PDF parsing and section splitting.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Union

import pdfplumber

from config import SECTIONS_OPTIONAL, SECTIONS_REQUIRED
from logger import get_logger

log = get_logger("pdf_parser")

# All recognized section headers (canonical lower-case)
ALL_SECTIONS = SECTIONS_REQUIRED + SECTIONS_OPTIONAL + [
    "professional experience", "work experience", "internships",
    "academic projects", "personal projects", "technical skills",
    "core competencies", "achievements", "honors", "extracurricular",
    "volunteer", "languages",
]

# Pattern to detect a section header line
_SECTION_RE = re.compile(
    r"^(?P<header>" + "|".join(re.escape(s) for s in sorted(ALL_SECTIONS, key=len, reverse=True)) + r")\s*[:\-–—]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class ResumeRaw:
    """Raw parsed resume before structured extraction."""
    full_text: str
    sections: dict[str, str] = field(default_factory=dict)
    pages: int = 0
    filename: str = ""


def extract_text_from_pdf(source: Union[str, Path, IO]) -> tuple[str, int]:
    """Extract raw text from a PDF file or file-like object."""
    pages_text: list[str] = []
    try:
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages_text.append(text)
        log.info("Extracted %d pages from PDF", len(pages_text))
    except Exception as exc:
        log.error("PDF extraction failed: %s", exc)
        raise

    full = "\n".join(pages_text)
    return full, len(pages_text)


def split_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections.
    Returns a dict mapping normalized section name → content.
    """
    lines = text.splitlines()
    sections: dict[str, str] = {}
    current_section = "header"
    buffer: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = _SECTION_RE.match(stripped)
        if m:
            # Save previous section
            sections[current_section] = "\n".join(buffer).strip()
            current_section = m.group("header").lower().strip()
            buffer = []
        else:
            buffer.append(line)

    # flush last section
    sections[current_section] = "\n".join(buffer).strip()
    log.debug("Found sections: %s", list(sections.keys()))
    return sections


def parse_resume(source: Union[str, Path, IO], filename: str = "") -> ResumeRaw:
    """Full pipeline: PDF → raw text → sections."""
    full_text, page_count = extract_text_from_pdf(source)

    # Light cleaning
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r"[ \t]{2,}", " ", full_text)

    sections = split_sections(full_text)

    return ResumeRaw(
        full_text=full_text,
        sections=sections,
        pages=page_count,
        filename=filename,
    )
