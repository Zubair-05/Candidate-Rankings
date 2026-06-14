import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalResult
from app.services import retrieval
from app.services.signal_scorers.engine import SignalScoringResult, score_candidates_bulk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ranking"])

_CANDIDATES_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "candidates.jsonl"

# Module-level cache — candidates.jsonl loaded once on first request
_candidates_by_id: dict[str, dict] | None = None


def _load_candidates() -> dict[str, dict]:
    global _candidates_by_id
    if _candidates_by_id is not None:
        return _candidates_by_id

    logger.info("Loading candidates.jsonl into memory…")
    result: dict[str, dict] = {}
    with open(_CANDIDATES_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                result[c["candidate_id"]] = c
            except (json.JSONDecodeError, KeyError):
                pass

    _candidates_by_id = result
    logger.info("Loaded %d candidates", len(result))
    return result


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = Field(default=2000, ge=1, le=10000)
    dense_pool: int = Field(default=3000, ge=100)
    sparse_pool: int = Field(default=3000, ge=100)


class RankRequest(BaseModel):
    query: str
    retrieval_top_k: int = Field(default=2000, description="Candidates Layer 1 returns")
    signal_top_k: int = Field(default=50, description="Candidates Layer 2 returns for LLM")


class RankedCandidate(BaseModel):
    rank: int
    candidate_id: str
    composite_score: float
    scorer_breakdown: list[dict]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float


class RankResponse(BaseModel):
    query: str
    total_retrieved: int
    total_scored: int
    top_k: int
    candidates: list[RankedCandidate]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/retrieve",
    response_model=RetrievalResult,
    summary="Layer 1 — hybrid retrieval only",
)
def retrieve_candidates(req: RetrieveRequest) -> RetrievalResult:
    """Run hybrid retrieval (dense + BM25 + RRF). Returns ranked candidate IDs."""
    return retrieval.retrieve(
        query=req.query,
        top_k=req.top_k,
        dense_pool=req.dense_pool,
        sparse_pool=req.sparse_pool,
    )


@router.post(
    "/rank",
    response_model=RankResponse,
    summary="Layer 1 + 2 — retrieval then signal scoring",
)
def rank_candidates(req: RankRequest) -> RankResponse:
    """
    Full Layer 1→2 pipeline:
      1. Hybrid retrieval → top retrieval_top_k candidates
      2. Signal scoring across all registered scorers → top signal_top_k
    """
    # Layer 1
    retrieval_result = retrieval.retrieve(query=req.query, top_k=req.retrieval_top_k)
    retrieved_ids = {c.candidate_id: c for c in retrieval_result.candidates}

    # Hydrate with full candidate data for signal scoring
    candidates_store = _load_candidates()
    hydrated: list[dict] = []
    for cid in retrieved_ids:
        if cid in candidates_store:
            hydrated.append(candidates_store[cid])

    if not hydrated:
        raise HTTPException(status_code=500, detail="No candidates hydrated — check candidates.jsonl path")

    # Layer 2
    scored = score_candidates_bulk(hydrated, top_k=req.signal_top_k)

    ranked: list[RankedCandidate] = []
    for rank, result in enumerate(scored, start=1):
        retrieval_meta = retrieved_ids.get(result.candidate_id)
        ranked.append(RankedCandidate(
            rank=rank,
            candidate_id=result.candidate_id,
            composite_score=result.composite_score,
            scorer_breakdown=[r.model_dump() for r in result.scorer_breakdown],
            dense_rank=retrieval_meta.dense_rank if retrieval_meta else None,
            sparse_rank=retrieval_meta.sparse_rank if retrieval_meta else None,
            rrf_score=retrieval_meta.rrf_score if retrieval_meta else 0.0,
        ))

    return RankResponse(
        query=req.query,
        total_retrieved=len(retrieval_result.candidates),
        total_scored=len(hydrated),
        top_k=len(ranked),
        candidates=ranked,
    )
