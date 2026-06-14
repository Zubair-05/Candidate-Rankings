"""
Domain Relevance Scorer

Signal: Has this person worked in AI/ML/tech product domains, or in unrelated fields?

Methodology:
  - Score each career role's industry against a domain relevance tier
  - Weight by duration (longer tenures count more)
  - Recent roles count more than old ones (same recency decay as skill scorer)
  - Product company experience > consulting/services for this JD
"""
from __future__ import annotations

import math
from typing import Any

from app.services.signal_scorers.base import BaseScorer, ScorerResult

# Industry → domain relevance score (0.0–1.0)
# Unmapped industries default to 0.3
_INDUSTRY_SCORES: dict[str, float] = {
    # Direct AI/ML product work
    "artificial intelligence": 1.0,
    "machine learning": 1.0,
    "ai": 1.0,
    "data science": 0.95,
    # Core tech product
    "software": 0.9,
    "saas": 0.9,
    "cloud computing": 0.9,
    "developer tools": 0.9,
    "internet": 0.85,
    "e-commerce": 0.85,
    "fintech": 0.85,
    "edtech": 0.80,
    "healthtech": 0.80,
    "adtech": 0.80,
    "search": 0.90,
    "information technology": 0.75,
    # Data-heavy but not pure ML
    "analytics": 0.75,
    "big data": 0.75,
    "telecommunications": 0.65,
    "media": 0.60,
    "gaming": 0.65,
    # Adjacent — some ML exposure but not primary
    "banking": 0.55,
    "finance": 0.55,
    "insurance": 0.50,
    "retail": 0.45,
    "healthcare": 0.50,
    "logistics": 0.45,
    # Consulting/services — penalised per JD ("not consulting")
    "it services": 0.30,
    "consulting": 0.25,
    "staffing": 0.20,
    "outsourcing": 0.20,
    # Unrelated
    "manufacturing": 0.20,
    "construction": 0.15,
    "paper products": 0.10,
    "real estate": 0.15,
    "education": 0.35,
}


def _industry_score(industry: str) -> float:
    key = industry.lower().strip()
    # Exact match first
    if key in _INDUSTRY_SCORES:
        return _INDUSTRY_SCORES[key]
    # Partial match
    for mapped_key, score in _INDUSTRY_SCORES.items():
        if mapped_key in key or key in mapped_key:
            return score
    return 0.30  # unknown industry — neutral-low


class DomainRelevanceScorer(BaseScorer):
    name = "domain_relevance"
    weight = 2.5
    description = "Scores how domain-relevant the candidate's industry history is for an AI product role."

    def score(self, candidate: dict[str, Any]) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: dict[str, Any]) -> ScorerResult:
        history = candidate.get("career_history", [])
        if not history:
            return self._missing("no career history")

        # Sort newest first — more recent roles carry more weight
        sorted_roles = sorted(history, key=lambda r: r.get("start_date", ""), reverse=True)

        weighted_sum = 0.0
        total_weight = 0.0
        role_notes: list[str] = []

        for i, role in enumerate(sorted_roles):
            industry = role.get("industry", "")
            duration = role.get("duration_months", 0)
            ind_score = _industry_score(industry)

            # Recency decay: most recent role has full weight, older roles decay
            recency_w = math.exp(-0.15 * i)
            # Duration weight: longer tenures count more (log scale to avoid 10yr dominance)
            duration_w = math.log1p(max(duration, 1))

            role_weight = recency_w * duration_w
            weighted_sum += ind_score * role_weight
            total_weight += role_weight

            if i < 3:
                role_notes.append(f"{industry}({ind_score:.1f})")

        score = self._clamp(weighted_sum / total_weight if total_weight > 0 else 0.3)

        rationale = (
            f"{len(history)} roles scored; "
            f"recent industries: {', '.join(role_notes)}; "
            f"weighted domain fit: {score:.2f}"
        )
        return self._result(score, rationale)
