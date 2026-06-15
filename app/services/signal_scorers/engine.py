"""
Signal scoring engine.

Parses raw candidate dicts into typed Candidate models at the boundary,
then runs every registered scorer. Weights are normalised at runtime —
adding a scorer requires zero changes here.
"""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, ValidationError

from app.logger import get_logger, log_latency
from app.models.candidate import Candidate
from app.services.signal_scorers import REGISTERED_SCORERS
from app.services.signal_scorers.base import BaseScorer, ScorerResult

logger = get_logger(__name__)


class SignalScoringResult(BaseModel):
    candidate_id: str
    composite_score: float
    scorer_breakdown: list[ScorerResult]
    scorers_run: int
    scorers_missing_data: int


def build_scoring_context(jd_query: str) -> dict[str, Any]:
    """
    Build the context dict passed to every scorer for a ranking run.
    Called once per /rank request — not once per candidate.
    Reuses the embedding model already loaded by the retrieval layer.
    """
    from app.services.retrieval import _embed_model

    context: dict[str, Any] = {"jd_query": jd_query}

    if _embed_model is not None:
        t0 = time.perf_counter()
        jd_embedding = _embed_model.encode(
            jd_query, normalize_embeddings=True, convert_to_numpy=True,
        ).astype(np.float32)
        logger.debug(
            "JD embedding encoded",
            extra={"latency_ms": round((time.perf_counter() - t0) * 1000, 1), "dims": jd_embedding.shape[0]},
        )
        context["jd_embedding"] = jd_embedding
        context["embed_model"]  = _embed_model
    else:
        logger.warning(
            "Retrieval model not loaded — domain_relevance scorer will fall back to neutral score. "
            "Ensure retrieval.build_index() is called at startup."
        )

    return context


def score_candidate(
    candidate: Candidate,
    context: Optional[dict[str, Any]] = None,
    scorers: Optional[list[BaseScorer]] = None,
) -> SignalScoringResult:
    """Run all registered scorers against a typed Candidate object."""
    active_scorers = scorers if scorers is not None else REGISTERED_SCORERS
    total_weight   = sum(s.weight for s in active_scorers)

    results: list[ScorerResult] = [
        scorer.score(candidate, context=context)
        for scorer in active_scorers
    ]

    composite = sum(
        r.score * (s.weight / total_weight)
        for s, r in zip(active_scorers, results)
    )

    return SignalScoringResult(
        candidate_id=candidate.candidate_id,
        composite_score=round(composite, 4),
        scorer_breakdown=results,
        scorers_run=len(results),
        scorers_missing_data=sum(1 for r in results if not r.data_available),
    )


def score_candidates_bulk(
    raw_candidates: list[dict[str, Any]],
    jd_query: str,
    top_k: int = 50,
) -> list[SignalScoringResult]:
    """
    Parse raw dicts → Candidate models at the boundary, score all,
    return top_k sorted by composite score descending.
    """
    t0           = time.perf_counter()
    parse_errors = 0
    scored: list[SignalScoringResult] = []

    with log_latency(logger, "build_scoring_context"):
        context = build_scoring_context(jd_query)

    t_score = time.perf_counter()
    for raw in raw_candidates:
        try:
            candidate = Candidate.model_validate(raw)
        except ValidationError as exc:
            parse_errors += 1
            logger.debug(
                "Candidate skipped — validation error",
                extra={"candidate_id": raw.get("candidate_id"), "error": str(exc)},
            )
            continue
        scored.append(score_candidate(candidate, context=context))

    if parse_errors:
        logger.warning(
            "Candidates skipped due to schema validation errors",
            extra={"skipped": parse_errors, "total": len(raw_candidates)},
        )

    scored.sort(key=lambda r: r.composite_score, reverse=True)
    top = scored[:top_k]

    score_ms  = round((time.perf_counter() - t_score) * 1000, 1)
    total_ms  = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "layer2_scoring complete",
        extra={
            "total_ms":    total_ms,
            "scoring_ms":  score_ms,
            "input":       len(raw_candidates),
            "valid":       len(scored),
            "parse_errors": parse_errors,
            "output":      len(top),
            "score_min":   top[-1].composite_score if top else 0,
            "score_max":   top[0].composite_score  if top else 0,
        },
    )
    return top
