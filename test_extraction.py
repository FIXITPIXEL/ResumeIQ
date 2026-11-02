"""
test_extraction.py – Milestone 1 & 2: Messy-resume test set and extraction F1 checks.
Run: python -m pytest test_extraction.py -v
"""
from __future__ import annotations

import pytest
from io import StringIO
from extractor import normalize_skills, _is_weak_bullet
from evaluator import compute_extraction_f1, compute_ndcg

# ── Messy resume samples (text) ────────────────────────────────────────────────

MESSY_RESUME_1 = """
John doe  
johndoe @gmail.com  | +1(555) 123 4567 | linkedin.com/in/johndoe  

EDUCATION
B.tech  Computer Science – IIT Bombay (2020–2024)  CGPA: 8.5

SKILLS
Python ,  pytorch, TensorFlow  , SK-learn, SQL, Docker, AWS  
NLP , Transformers,LangChain,  HuggingFace

EXPERIENCE
ML Intern at FinTech Corp | Jun 2023 – Aug 2023
• Worked on building fraud detection models
• Helped with data preprocessing using pandas
• Responsible for model evaluation

Machine Learning Engineer | StartupXYZ | 2022–2023
• Developed a recommendation system increasing CTR by 22 percent
• Built RAG pipeline with Langchain and Pinecone reducing search time by 40%
"""

MESSY_RESUME_2 = """
PRIYA SHARMA
priya.sharma@outlook.com | GitHub: github.com/priyasharma | Bangalore, India

OBJECTIVE
Seeking a challenging role in data science and AI

TECHNICAL SKILLS
Languages: Python, R, SQL, JavaScript
Frameworks: Scikit learn, Keras, OpenCV, FastAPI, Streamlit
Cloud: Azure, GCP
Visualization: Tableau, Power BI, Matplotlib, Plotly

PROJECTS
1. Sentiment Analysis System
   Used python and pytorch to classify 100K+ tweets with 91% accuracy
2. Real-time Object Detection
   Implemented YOLO using opencv and tensorflow deployed via docker

EDUCATION
Master of Science in Data Science – Stanford University (2021–2023)
"""

# ── Skill extraction tests ─────────────────────────────────────────────────────

class TestSkillNormalization:
    def test_alias_resolution(self):
        skills = normalize_skills("I use sklearn and torch daily")
        assert "scikit-learn" in skills, "sklearn alias not resolved"
        assert "pytorch" in skills, "torch alias not resolved"

    def test_case_insensitive(self):
        skills = normalize_skills("PYTHON and TensorFlow")
        assert "python" in skills
        assert "tensorflow" in skills

    def test_comma_separated(self):
        skills = normalize_skills("Python, SQL, Docker, AWS, MLOps")
        assert "python" in skills
        assert "sql" in skills
        assert "docker" in skills

    def test_noisy_resume_1(self):
        skills = normalize_skills(MESSY_RESUME_1)
        assert "python" in skills
        assert "pytorch" in skills
        assert "nlp" in skills or "transformers" in skills

    def test_noisy_resume_2(self):
        skills = normalize_skills(MESSY_RESUME_2)
        assert "scikit-learn" in skills
        assert "opencv" in skills
        assert "docker" in skills

    def test_no_false_positives(self):
        skills = normalize_skills("I enjoy cooking and traveling")
        assert len(skills) == 0, f"Unexpected skills: {skills}"


# ── Weak bullet detection ──────────────────────────────────────────────────────

class TestWeakBulletDetection:
    @pytest.mark.parametrize("bullet,expected", [
        ("Worked on building fraud detection models", True),
        ("Helped with data preprocessing", True),
        ("Responsible for model evaluation", True),
        ("Did various ML tasks", True),
        ("Developed a recommendation system increasing CTR by 22%", False),
        ("Built and deployed a real-time fraud detection pipeline", False),
        ("Achieved 94.2 F1 on NER task", False),
        ("Reduced inference latency by 3× using TensorRT", False),
    ])
    def test_bullet_weakness(self, bullet, expected):
        assert _is_weak_bullet(bullet) == expected, f"Failed for: {bullet!r}"


# ── nDCG computation ───────────────────────────────────────────────────────────

class TestNDCG:
    def test_perfect_ordering(self):
        scores = [90, 80, 70, 60]  # already sorted descending
        ndcg = compute_ndcg(scores)
        assert ndcg == 1.0, f"Expected 1.0, got {ndcg}"

    def test_reversed_ordering(self):
        scores = [60, 70, 80, 90]  # worst order
        ndcg = compute_ndcg(scores)
        assert ndcg < 1.0

    def test_empty(self):
        assert compute_ndcg([]) == 0.0

    def test_single_item(self):
        assert compute_ndcg([75]) == 1.0


# ── Extraction F1 ──────────────────────────────────────────────────────────────

class TestExtractionF1:
    def test_perfect_match(self):
        result = compute_extraction_f1(["python", "pytorch"], ["python", "pytorch"])
        assert result["f1"] == 1.0

    def test_partial_match(self):
        result = compute_extraction_f1(["python", "pytorch", "docker"], ["python", "pytorch"])
        assert result["precision"] < 1.0
        assert result["recall"] == 1.0

    def test_no_match(self):
        result = compute_extraction_f1(["java", "rust"], ["python", "pytorch"])
        assert result["f1"] == 0.0

    def test_empty_prediction(self):
        result = compute_extraction_f1([], ["python"])
        assert result["f1"] == 0.0
