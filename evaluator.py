"""
evaluator.py – Milestone 6: Snapshot-based nDCG, extraction F1, and rewrite faithfulness.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from config import SNAPSHOT_DIR
from logger import get_logger
from ranker import JobScore
from rewriter import RewriteResult

log = get_logger("evaluator")


@dataclass
class EvalSnapshot:
    """Persisted evaluation snapshot for regression tracking."""
    snapshot_id: str
    timestamp: float
    resume_name: str
    ranking_scores: list[float]         # match_score per job
    extraction_skills: list[str]        # extracted skills
    rewrite_faithfulness: list[float]
    ndcg: float = 0.0
    extraction_f1: float = 0.0
    avg_faithfulness: float = 0.0


# ── nDCG (Normalized Discounted Cumulative Gain) ───────────────────────────────

def _dcg(relevances: list[float]) -> float:
    """Compute DCG@k."""
    return sum(
        (2 ** r - 1) / math.log2(i + 2)
        for i, r in enumerate(relevances)
    )


def compute_ground_truth_relevance(job: JobScore) -> float:
    """
    Calculate an independent ground-truth relevance score (0–100) based on
    pure objective qualification:
    - 60% Skill coverage ratio (matched skills / required skills)
    - 25% Domain alignment (presence of technical domain keywords in title)
    - 15% Seniority fit heuristic
    This decouples true qualification relevance from the ranker's internal hybrid sorting score.
    """
    # 1. Skill coverage ratio
    if job.job_skills:
        skill_ratio = len(job.matched_skills) / len(job.job_skills)
    else:
        skill_ratio = 0.5

    # 2. Domain alignment in title
    t = job.title.lower()
    domain_match = any(
        k in t for k in [
            "engineer", "developer", "scientist", "analyst", "architect",
            "specialist", "programmer", "lead", "manager"
        ]
    )
    domain_score = 1.0 if domain_match else 0.5

    # 3. Seniority fit
    sen_score = (job.seniority_fit / 100.0) if hasattr(job, "seniority_fit") else 0.8

    rel = (0.60 * skill_ratio + 0.25 * domain_score + 0.15 * sen_score) * 100.0
    return round(rel, 2)


def compute_ndcg(scores: list[float], k: Optional[int] = None) -> float:
    """
    Compute nDCG@k over the sequence of ground-truth relevances of ranked items.
    If scores are on a 0–100 scale, scales to 0–1.
    If scores are discrete/graded (0–3), preserves integer scale.
    Ideal order = sorted descending.
    """
    if not scores:
        return 0.0
    if k:
        scores = scores[:k]
    max_val = max(scores) if scores else 0.0
    if max_val > 5.0:
        rel = [s / 100.0 for s in scores]
    else:
        rel = [float(s) for s in scores]
    ideal = sorted(rel, reverse=True)
    dcg   = _dcg(rel)
    idcg  = _dcg(ideal)
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


# ── Extraction F1 ──────────────────────────────────────────────────────────────

def compute_extraction_f1(
    predicted: list[str],
    ground_truth: list[str],
) -> dict[str, float]:
    """
    Token-level F1 for skill extraction.
    Uses set intersection (case-insensitive).
    """
    pred_set = {s.lower().strip() for s in predicted}
    gt_set   = {s.lower().strip() for s in ground_truth}

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set  - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ── Faithfulness aggregation ───────────────────────────────────────────────────

def aggregate_faithfulness(results: list[RewriteResult]) -> dict[str, float]:
    if not results:
        return {"avg": 0.0, "min": 0.0, "max": 0.0, "weak_count": 0}
    scores = [r.faithfulness_score for r in results]
    weak   = sum(1 for r in results if r.is_weak)
    return {
        "avg":        round(sum(scores) / len(scores), 4),
        "min":        round(min(scores), 4),
        "max":        round(max(scores), 4),
        "weak_count": weak,
        "total":      len(results),
    }


# ── Snapshot persistence ───────────────────────────────────────────────────────

def save_snapshot(
    resume_name: str,
    ranked_jobs: list[JobScore],
    extracted_skills: list[str],
    rewrite_results: list[RewriteResult],
    ground_truth_relevances: Optional[list[float]] = None,
) -> EvalSnapshot:
    snap_id = f"{resume_name.replace(' ', '_')}_{int(time.time())}"

    # Use explicitly passed ground truth or calculate independent objective qualification relevance
    if ground_truth_relevances:
        gt_scores = ground_truth_relevances
    else:
        gt_scores = [compute_ground_truth_relevance(j) for j in ranked_jobs]

    calculated_ndcg = compute_ndcg(gt_scores) if gt_scores else 0.0

    snap = EvalSnapshot(
        snapshot_id=snap_id,
        timestamp=time.time(),
        resume_name=resume_name,
        ranking_scores=[j.match_score for j in ranked_jobs],
        extraction_skills=extracted_skills,
        rewrite_faithfulness=[r.faithfulness_score for r in rewrite_results],
        ndcg=calculated_ndcg,
        avg_faithfulness=aggregate_faithfulness(rewrite_results).get("avg", 0.0),
    )
    out_path = SNAPSHOT_DIR / f"{snap_id}.json"
    out_path.write_text(json.dumps(asdict(snap), indent=2))
    log.info("Saved snapshot: %s | nDCG=%.4f (relevance-grounded)", snap_id, snap.ndcg)
    return snap


def load_snapshots() -> list[EvalSnapshot]:
    snaps = []
    for p in sorted(SNAPSHOT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            data = json.loads(p.read_text())
            snaps.append(EvalSnapshot(**data))
        except Exception as e:
            log.warning("Failed to load snapshot %s: %s", p.name, e)
    return snaps


def compare_snapshots(
    snap_a: EvalSnapshot,
    snap_b: EvalSnapshot,
    ground_truth_skills: Optional[list[str]] = None,
) -> dict:
    """Compare two snapshots – useful for regression detection."""
    result: dict = {
        "ndcg_delta":   round(snap_b.ndcg - snap_a.ndcg, 4),
        "faith_delta":  round(snap_b.avg_faithfulness - snap_a.avg_faithfulness, 4),
    }
    if ground_truth_skills:
        f1_a = compute_extraction_f1(snap_a.extraction_skills, ground_truth_skills)
        f1_b = compute_extraction_f1(snap_b.extraction_skills, ground_truth_skills)
        result["f1_delta"] = round(f1_b["f1"] - f1_a["f1"], 4)
    return result
