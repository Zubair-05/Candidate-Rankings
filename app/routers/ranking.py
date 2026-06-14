import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import fitz  # pymupdf
import docx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalResult
from app.services import retrieval
from app.services.signal_scorers.engine import SignalScoringResult, score_candidates_bulk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ranking"])

_CANDIDATES_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "candidates.jsonl"

# Module-level cache — candidates.jsonl loaded once on first request
_candidates_by_id: Optional[Dict[str, dict]] = None


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
    scorer_breakdown: List[dict]
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float


class RankResponse(BaseModel):
    query: str
    total_retrieved: int
    total_scored: int
    top_k: int
    candidates: List[RankedCandidate]


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
    hydrated: List[dict] = []
    for cid in retrieved_ids:
        if cid in candidates_store:
            hydrated.append(candidates_store[cid])

    if not hydrated:
        raise HTTPException(status_code=500, detail="No candidates hydrated — check candidates.jsonl path")

    # Layer 2
    scored = score_candidates_bulk(hydrated, jd_query=req.query, top_k=req.signal_top_k)

    ranked: List[RankedCandidate] = []
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


def _extract_text_from_pdf(data: bytes) -> str:
    doc = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)


def _extract_text_from_docx(data: bytes) -> str:
    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


@router.post(
    "/rank/upload",
    response_model=RankResponse,
    summary="Layer 1 + 2 — accepts JD as text, PDF, or DOCX",
)
async def rank_candidates_upload(
    jd_text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    retrieval_top_k: int = Form(default=2000),
    signal_top_k: int = Form(default=50),
) -> RankResponse:
    """
    Accepts a JD either as plain text (jd_text form field) or as an uploaded
    PDF / DOCX file. Runs the full Layer 1 → 2 pipeline and returns ranked candidates.
    """
    if file is not None:
        raw = await file.read()
        ct = file.content_type or ""
        filename = file.filename or ""
        if "pdf" in ct or filename.lower().endswith(".pdf"):
            query = _extract_text_from_pdf(raw)
        elif "word" in ct or "docx" in ct or filename.lower().endswith(".docx"):
            query = _extract_text_from_docx(raw)
        elif filename.lower().endswith(".doc"):
            raise HTTPException(status_code=400, detail="Legacy .doc format not supported — please upload .docx or PDF")
        else:
            query = raw.decode("utf-8", errors="replace")
    elif jd_text:
        query = jd_text.strip()
    else:
        raise HTTPException(status_code=422, detail="Provide either jd_text or a file upload")

    if len(query) < 20:
        raise HTTPException(status_code=422, detail="Job description too short — needs at least 20 characters")

    return rank_candidates(RankRequest(
        query=query,
        retrieval_top_k=retrieval_top_k,
        signal_top_k=signal_top_k,
    ))


@router.get(
    "/candidates/{candidate_id}",
    summary="Get full candidate profile by ID",
)
def get_candidate(candidate_id: str) -> dict:
    candidates_store = _load_candidates()
    candidate = candidates_store.get(candidate_id)
    if candidate is None:
        raise HTTPException(status_code=404, detail=f"Candidate {candidate_id!r} not found")
    return candidate
