"""
Skill Recency Scorer

Signal: Are the candidate's AI/ML skills current, deep, and endorsed?

Methodology:
  - Filter skills to those matching the JD's target skill set
  - Weight each skill by: proficiency × recency decay × endorsement bonus
  - Top-5 skills dominate — depth beats breadth
"""
from __future__ import annotations

import math
from typing import Any, Optional

from app.models.candidate import Candidate, Skill, SkillProficiency
from app.services.signal_scorers.base import BaseScorer, ScorerResult

_TARGET_SKILLS: set[str] = {
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "llm", "large language model", "fine-tuning", "fine-tuning llms",
    "rag", "retrieval augmented generation", "prompt engineering",
    "langchain", "langgraph", "llamaindex",
    "vector database", "vector search", "embedding", "embeddings",
    "semantic search", "hybrid search", "bm25", "faiss", "pinecone",
    "weaviate", "qdrant", "pgvector", "elasticsearch",
    "reranking", "learning to rank", "ltr", "recommendation system",
    "ndcg", "evaluation framework", "model evaluation", "a/b testing",
    "mlflow", "weights & biases", "wandb", "kubeflow", "airflow",
    "feature store", "model serving", "triton", "bentoml",
    "kubernetes", "docker", "gcp", "aws", "azure", "spark", "kafka",
    "python", "pytorch", "tensorflow", "jax", "hugging face", "transformers",
}

_PROFICIENCY_SCORE: dict[SkillProficiency, float] = {
    SkillProficiency.BEGINNER:     0.25,
    SkillProficiency.INTERMEDIATE: 0.50,
    SkillProficiency.ADVANCED:     0.75,
    SkillProficiency.EXPERT:       1.00,
}


def _recency_weight(duration_months: int) -> float:
    """Exponential decay — full weight for recent skills, floor of 0.2 for old ones."""
    return max(0.2, math.exp(-0.35 * (duration_months / 12)))


def _matches_target(skill_name: str) -> bool:
    name = skill_name.lower()
    return any(t in name or name in t for t in _TARGET_SKILLS)


class SkillRecencyScorer(BaseScorer):
    name = "skill_recency"
    weight = 2.0
    description = "Scores depth and recency of AI/ML skills relevant to the JD."

    def score(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]] = None,
    ) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: Candidate) -> ScorerResult:
        if not candidate.skills:
            return self._missing("no skills listed")

        matched: list[Skill] = [s for s in candidate.skills if _matches_target(s.name)]
        if not matched:
            return self._result(0.1, "no AI/ML skills matching JD requirements found")

        weighted_scores: list[float] = []
        for skill in matched:
            prof_score  = _PROFICIENCY_SCORE[skill.proficiency]
            recency_w   = _recency_weight(skill.duration_months)
            # Log-scaled endorsement bonus, capped at 0.2
            endorse_bonus = min(0.2, math.log1p(skill.endorsements) / 25)
            weighted_scores.append(prof_score * recency_w + endorse_bonus)

        # Top-5 skills dominate; long tail matters less
        top_scores = sorted(weighted_scores, reverse=True)[:5]
        raw = sum(top_scores) / len(top_scores)

        top_names = [
            s.name for s in sorted(matched, key=lambda x: x.duration_months, reverse=True)[:3]
        ]
        rationale = (
            f"{len(matched)} relevant skills; "
            f"top: {', '.join(top_names)}; "
            f"avg weighted score: {raw:.2f}"
        )
        return self._result(self._clamp(raw), rationale)
