"""
resume_chat.py – Conversational Resume Chat using RAG.
Indexes resume sections and bullet points into a mini vector index,
retrieves top relevant context, and answers candidate questions with an LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from extractor import ResumeStructured
from llm_client import llm_complete
from logger import get_logger
from ranker import _get_embed_model, _cosine_sim

log = get_logger("resume_chat")


@dataclass
class ChatResponse:
    answer: str
    sources: list[str]
    method: str = "AI (RAG + LLM)"
    error_message: Optional[str] = None


class ResumeRAGChat:
    def __init__(self, resume: ResumeStructured):
        self.resume = resume
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self._build_index()

    def _build_index(self):
        chunks = []
        # Contact / Overview
        if self.resume.contact.name:
            chunks.append(f"Candidate Name: {self.resume.contact.name}")
        if self.resume.skills:
            chunks.append(f"Key Technical & Professional Skills: {', '.join(self.resume.skills)}")

        # Education
        for edu in self.resume.education:
            chunks.append(f"Education: {edu.degree} at {edu.institution} ({edu.year}) {f'GPA: {edu.gpa}' if edu.gpa else ''}".strip())

        # Experience
        for exp in self.resume.experience:
            role_header = f"Role: {exp.title} at {exp.company} ({exp.duration})"
            chunks.append(role_header)
            for bullet in exp.bullets:
                chunks.append(f"Experience Achievement ({exp.company}): {bullet}")

        # Projects
        for proj in self.resume.projects:
            chunks.append(f"Project: {proj}")

        # Certifications
        for cert in self.resume.certifications:
            chunks.append(f"Certification: {cert}")

        if not chunks:
            chunks.append("Resume contains standard background in software and machine learning.")

        self.chunks = chunks
        model = _get_embed_model()
        self.embeddings = model.encode(self.chunks, convert_to_numpy=True)
        log.info("Built Resume RAG index with %d chunks", len(self.chunks))

    def retrieve(self, query: str, top_k: int = 4) -> list[str]:
        if not self.chunks or self.embeddings is None:
            return []
        model = _get_embed_model()
        q_emb = model.encode(query, convert_to_numpy=True)
        sims = [_cosine_sim(q_emb, ce) for ce in self.embeddings]
        top_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_indices]

    def answer_query(self, query: str, history: Optional[list[dict]] = None) -> ChatResponse:
        relevant_chunks = self.retrieve(query, top_k=4)
        context_str = "\n".join(f"- {c}" for c in relevant_chunks)

        prompt = f"""You are an insightful AI Career Assistant and Resume Coach.
Answer the user's question accurately and constructively based on their resume context.

RESUME CONTEXT (Retrieved from user's parsed resume):
{context_str}

USER QUESTION:
{query}

GUIDELINES:
1. Provide a direct, specific answer referencing their actual experience, skills, and projects.
2. If suggesting improvements, give concrete examples based on their background.
3. Be encouraging, professional, and concise (max 3-4 paragraphs or crisp bullet points).
4. If the resume context doesn't contain the requested information, acknowledge it honestly.
"""

        try:
            raw_res = llm_complete(
                prompt=prompt,
                system="You are an expert resume coach providing precise, tailored career advice.",
                max_tokens=400,
            )
            if not raw_res or raw_res.startswith("[LLM unavailable"):
                raise ValueError("LLM is currently unavailable or quota exceeded")

            return ChatResponse(
                answer=raw_res,
                sources=relevant_chunks,
                method="AI (RAG + LLM)",
            )
        except Exception as e:
            err_msg = str(e)
            log.warning("LLM Chat failed (%s). Generating fallback heuristic response.", err_msg)
            fallback_answer = self._generate_fallback(query, relevant_chunks)
            return ChatResponse(
                answer=fallback_answer,
                sources=relevant_chunks,
                method="Semantic RAG Search (Offline Fallback)",
                error_message=f"LLM offline or quota exhausted: {err_msg[:60]}",
            )

    def _generate_fallback(self, query: str, relevant_chunks: list[str]) -> str:
        q_low = query.lower()
        if "skill" in q_low or "strength" in q_low:
            return (
                f"**Top Extracted Skills:**\n{', '.join(self.resume.skills[:15])}\n\n"
                f"**Most Relevant Background:**\n" +
                "\n".join(f"• {c}" for c in relevant_chunks[:3])
            )
        elif "experience" in q_low or "role" in q_low or "work" in q_low:
            roles = [f"• {exp.title} at {exp.company} ({exp.duration})" for exp in self.resume.experience]
            return "**Experience Summary:**\n" + ("\n".join(roles) if roles else "No experience roles parsed.")
        else:
            return (
                f"I found the following most relevant details in your resume for your query:\n\n" +
                "\n".join(f"• {c}" for c in relevant_chunks[:3]) +
                "\n\n*(Note: For generative multi-turn reasoning, configure active LLM API credentials in `.env`)*"
            )
