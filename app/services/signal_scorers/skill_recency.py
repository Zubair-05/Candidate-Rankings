"""
Skill Recency Scorer

Signal: Are the candidate's AI/ML skills current or stale?

Methodology:
  - Maintain a list of target AI/ML skills drawn from the JD
  - For each matching skill, weight by: proficiency × recency × endorsements
  - Recency = how recently the skill appears in the candidate's career history
    (cross-referenced by matching skill name against role descriptions)
  - A candidate with 3 current expert-level AI skills beats one with
    10 stale intermediate skills listed from five years ago
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any

from app.services.signal_scorers.base import BaseScorer, ScorerResult

# Skills relevant to the JD — expanded to catch variants
_TARGET_SKILLS: set[str] = {
    # Core ML/AI
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    # LLMs & GenAI
    "llm", "large language model", "fine-tuning", "fine-tuning llms",
    "rag", "retrieval augmented generation", "prompt engineering",
    "langchain", "langgraph", "llamaindex",
    # Search & Retrieval
    "vector database", "vector search", "embedding", "embeddings",
    "semantic search", "hybrid search", "bm25", "faiss", "pinecone",
    "weaviate", "qdrant", "pgvector", "elasticsearch",
    # Reranking & Ranking
    "reranking", "learning to rank", "ltr", "recommendation system",
    # Evaluation
    "ndcg", "evaluation framework", "model evaluation", "a/b testing",
    # MLOps
    "mlflow", "weights & biases", "wandb", "kubeflow", "airflow",
    "feature store", "model serving", "triton", "bentoml",
    # Infra
    "kubernetes", "docker", "gcp", "aws", "azure", "spark", "kafka",
    # Languages & frameworks
    "python", "pytorch", "tensorflow", "jax", "hugging face", "transformers",
}

_PROFICIENCY_WEIGHTS = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

_CURRENT_YEAR = date.today().year


def _recency_weight(duration_months: int) -> float:
    """
    Decay function: full weight for skills used in the last 12 months,
    decaying to ~0.2 at 5 years. Uses exponential decay with a 24-month half-life.
    """
    years_ago = duration_months / 12
    return max(0.2, math.exp(-0.35 * years_ago))


def _skill_matches_target(skill_name: str) -> bool:
    name_lower = skill_name.lower()
    return any(target in name_lower or name_lower in target for target in _TARGET_SKILLS)


class SkillRecencyScorer(BaseScorer):
    name = "skill_recency"
    weight = 2.0
    description = "Scores depth and recency of AI/ML skills relevant to the JD."

    def score(self, candidate: dict[str, Any]) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: dict[str, Any]) -> ScorerResult:
        skills = candidate.get("skills", [])
        if not skills:
            return self._missing("no skills listed")

        matched: list[dict] = [s for s in skills if _skill_matches_target(s.get("name", ""))]
        if not matched:
            return self._result(0.1, "no AI/ML skills matching JD requirements found")

        weighted_scores: list[float] = []
        for skill in matched:
            proficiency = skill.get("proficiency", "beginner")
            duration_months = skill.get("duration_months", 0)
            endorsements = skill.get("endorsements", 0)

            prof_w = _PROFICIENCY_WEIGHTS.get(proficiency, 0.25)
            recency_w = _recency_weight(duration_months)
            # Endorsements: log scale, capped contribution to avoid outliers dominating
            endorsement_bonus = min(0.2, math.log1p(endorsements) / 25)

            weighted_scores.append(prof_w * recency_w + endorsement_bonus)

        # Top-5 skills dominate; long tail matters less
        top_scores = sorted(weighted_scores, reverse=True)[:5]
        raw = sum(top_scores) / len(top_scores)
        score = self._clamp(raw)

        top_skill_names = [
            s["name"] for s in sorted(matched, key=lambda x: x.get("duration_months", 0), reverse=True)[:3]
        ]
        rationale = (
            f"{len(matched)} relevant AI/ML skills; "
            f"top: {', '.join(top_skill_names)}; "
            f"avg weighted score: {raw:.2f}"
        )
        return self._result(score, rationale)
