"""
Domain Relevance Scorer

Signal: Has this person worked in domains relevant to *this specific JD*?

Methodology:
  - Embed each unique industry from the candidate's career history
  - Compute cosine similarity against the JD embedding (passed via context)
  - Weight each role by recency and duration — recent, longer tenures matter most
  - Falls back gracefully if no JD embedding is available in context

JD-agnostic by design: no hardcoded domain scores. A Healthcare AI JD
rewards HealthTech candidates; a FinTech JD rewards Finance candidates.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from app.models.candidate import Candidate
from app.services.signal_scorers.base import BaseScorer, ScorerResult

# Process-level cache — avoids re-encoding the same industry string
# across 2000 candidates in a single ranking run.
_industry_embedding_cache: dict[str, np.ndarray] = {}


def _get_industry_embedding(industry: str, model: Any) -> np.ndarray:
    key = industry.lower().strip()
    if key not in _industry_embedding_cache:
        emb = model.encode(key, normalize_embeddings=True, convert_to_numpy=True)
        _industry_embedding_cache[key] = emb.astype(np.float32)
    return _industry_embedding_cache[key]


class DomainRelevanceScorer(BaseScorer):
    name = "domain_relevance"
    weight = 2.5
    description = (
        "Scores domain fit by computing cosine similarity between the candidate's "
        "industry history and the JD embedding. Adapts to any JD automatically — "
        "no hardcoded domain scores."
    )

    def score(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]] = None,
    ) -> ScorerResult:
        try:
            return self._compute(candidate, context)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]],
    ) -> ScorerResult:
        jd_embedding: Optional[np.ndarray] = context.get("jd_embedding") if context else None
        model: Any = context.get("embed_model") if context else None

        if jd_embedding is None or model is None:
            return self._missing(
                "JD embedding not in context — "
                "ensure retrieval.build_index() ran before scoring"
            )

        # Sort newest first — recent domain experience is more relevant
        sorted_roles = sorted(candidate.career_history, key=lambda r: r.start_date, reverse=True)

        weighted_sum = 0.0
        total_weight = 0.0
        role_notes: list[str] = []

        for i, role in enumerate(sorted_roles):
            if not role.industry:
                continue

            industry_emb = _get_industry_embedding(role.industry, model)

            # Cosine similarity (both L2-normalised) mapped from [-1,1] → [0,1]
            similarity    = float(np.dot(industry_emb, jd_embedding))
            domain_score  = (similarity + 1.0) / 2.0

            recency_w  = math.exp(-0.15 * i)
            duration_w = math.log1p(max(role.duration_months, 1))
            role_weight = recency_w * duration_w

            weighted_sum  += domain_score * role_weight
            total_weight  += role_weight

            if i < 3:
                role_notes.append(f"{role.industry}({domain_score:.2f})")

        if total_weight == 0:
            return self._missing("no scoreable industries in career history")

        score = self._clamp(weighted_sum / total_weight)
        rationale = (
            f"{len(sorted_roles)} roles; "
            f"recent: {', '.join(role_notes)}; "
            f"weighted fit: {score:.3f}"
        )
        return self._result(score, rationale)
