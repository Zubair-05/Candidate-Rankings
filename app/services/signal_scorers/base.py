"""
Abstract base class for all signal scorers.

Every scorer must:
  - declare a unique `name`, a `weight`, and a human-readable `description`
  - implement `score(candidate, context)` returning a ScorerResult
  - never raise — catch all exceptions internally and return a neutral result

The `context` dict is computed once per ranking run by the engine and
passed to every scorer. It carries JD-derived signals (embeddings, parsed
requirements) so scorers can adapt to the JD rather than hardcode assumptions.

Current context keys:
  jd_query (str)                    — raw JD text
  jd_embedding (np.ndarray | None)  — L2-normalised embedding of the JD query
  embed_model (SentenceTransformer) — model instance for encoding new strings
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.models.candidate import Candidate


class ScorerResult(BaseModel):
    scorer: str
    score: float = Field(ge=0.0, le=1.0)
    rationale: str
    data_available: bool = True  # False when required fields were missing


class BaseScorer(ABC):
    name: str
    weight: float
    description: str

    @abstractmethod
    def score(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]] = None,
    ) -> ScorerResult:
        ...

    # ------------------------------------------------------------------
    # Shared helpers — available to all subclasses
    # ------------------------------------------------------------------

    def _missing(self, reason: str) -> ScorerResult:
        """Return a neutral mid-point score when required data is absent."""
        return ScorerResult(
            scorer=self.name,
            score=0.5,
            rationale=f"[data unavailable] {reason}",
            data_available=False,
        )

    def _result(self, score: float, rationale: str) -> ScorerResult:
        return ScorerResult(scorer=self.name, score=round(score, 4), rationale=rationale)

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))
