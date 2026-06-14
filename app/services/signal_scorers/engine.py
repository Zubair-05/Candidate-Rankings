"""
Signal scoring engine.

Runs every scorer in REGISTERED_SCORERS against a candidate,
normalises weights at runtime, and returns a SignalScoringResult.

The engine is intentionally scorer-agnostic — it knows nothing about
individual dimensions. Adding a scorer requires zero changes here.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel

from app.services.signal_scorers import REGISTERED_SCORERS
from app.services.signal_scorers.base import BaseScorer, ScorerResult

logger = logging.getLogger(__name__)


class SignalScoringResult(BaseModel):
    candidate_id: str
    composite_score: float          # 0.0–1.0, weighted average of all scorers
    scorer_breakdown: list[ScorerResult]
    scorers_run: int
    scorers_missing_data: int       # how many fell back to neutral 0.5


def score_candidate(
    candidate: dict[str, Any],
    scorers: list[BaseScorer] | None = None,
) -> SignalScoringResult:
    """
    Run all registered scorers against a single candidate dict.

    Args:
        candidate: Raw candidate dict from candidates.jsonl
        scorers:   Override scorer list (defaults to REGISTERED_SCORERS).
                   Useful for testing individual scorers in isolation.
    """
    active_scorers = scorers if scorers is not None else REGISTERED_SCORERS
    candidate_id = candidate.get("candidate_id", "unknown")

    results: list[ScorerResult] = []
    total_weight = sum(s.weight for s in active_scorers)

    for scorer in active_scorers:
        result = scorer.score(candidate)
        results.append(result)

    # Weighted composite — weights are normalised so their sum equals 1
    composite = sum(
        r.score * (s.weight / total_weight)
        for s, r in zip(active_scorers, results)
    )

    missing_count = sum(1 for r in results if not r.data_available)

    return SignalScoringResult(
        candidate_id=candidate_id,
        composite_score=round(composite, 4),
        scorer_breakdown=results,
        scorers_run=len(results),
        scorers_missing_data=missing_count,
    )


def score_candidates_bulk(
    candidates: list[dict[str, Any]],
    top_k: int = 50,
) -> list[SignalScoringResult]:
    """
    Score a list of candidates and return the top_k by composite score.

    This is the Layer 2 → Layer 3 handoff point.
    """
    t0 = time.perf_counter()

    scored = [score_candidate(c) for c in candidates]
    scored.sort(key=lambda r: r.composite_score, reverse=True)
    top = scored[:top_k]

    elapsed = time.perf_counter() - t0
    logger.info(
        "signal scoring: %d candidates → top %d in %.2fs "
        "(composite range %.3f–%.3f)",
        len(candidates),
        len(top),
        elapsed,
        top[-1].composite_score if top else 0,
        top[0].composite_score if top else 0,
    )
    return top
