"""
test_genai_features.py – Unit tests for the new GenAI modules:
1. rewriter.generate_resume_summary
2. gap_analyzer.analyze_skill_gaps
3. resume_chat.ResumeRAGChat
"""
import pytest
from extractor import ResumeStructured, ContactInfo, Experience, Education
from rewriter import generate_resume_summary, ResumeSummaryResult
from gap_analyzer import analyze_skill_gaps, GapAnalysis
from resume_chat import ResumeRAGChat, ChatResponse


@pytest.fixture
def sample_resume():
    return ResumeStructured(
        contact=ContactInfo(name="Sripal Jena", email="sripal@example.com"),
        skills=["python", "pytorch", "machine learning", "deep learning", "docker", "aws"],
        education=[Education(degree="B.Tech Computer Science", institution="National Institute of Technology", year="2020-2024")],
        experience=[
            Experience(
                title="AI Engineer",
                company="Tech Corp",
                duration="2024 - Present",
                bullets=[
                    "Spearheaded real-time fraud detection pipeline using PyTorch, reducing latency by 35%.",
                    "Assisted with basic model evaluation.",
                ],
                weak_bullets=["Assisted with basic model evaluation."],
            )
        ],
        projects=["Built RAG knowledge assistant with FAISS and LangChain."],
    )


class TestResumeSummary:
    def test_summary_generation(self, sample_resume):
        res = generate_resume_summary(sample_resume, target_role="Senior AI Engineer")
        assert isinstance(res, ResumeSummaryResult)
        assert len(res.elevator_pitch) > 10
        assert len(res.professional_summary) > 20
        assert len(res.key_highlights) >= 2


class TestGapAnalyzer:
    def test_skill_gap_analysis(self, sample_resume):
        job_desc = "Seeking Machine Learning Engineer with Python, PyTorch, Kubernetes, Kafka, and Snowflake."
        gap = analyze_skill_gaps(sample_resume, job_description=job_desc, job_title="MLOps Engineer")
        assert isinstance(gap, GapAnalysis)
        assert "python" in gap.matched_skills or "pytorch" in gap.matched_skills
        assert "kubernetes" in gap.missing_skills or "kafka" in gap.missing_skills
        assert len(gap.gap_details) > 0
        assert gap.overall_readiness > 0.0


class TestResumeRAGChat:
    def test_chat_indexing_and_answer(self, sample_resume):
        chat = ResumeRAGChat(sample_resume)
        assert len(chat.chunks) > 0
        assert chat.embeddings is not None

        # Test retrieval
        retrieved = chat.retrieve("What is the fraud detection project?", top_k=2)
        assert len(retrieved) > 0
        assert any("fraud" in r.lower() for r in retrieved)

        # Test answering
        resp = chat.answer_query("Summarize my experience with PyTorch")
        assert isinstance(resp, ChatResponse)
        assert len(resp.answer) > 10
        assert len(resp.sources) > 0
