from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.models.retrieval import RetrievalResult
from app.services import retrieval

router = APIRouter(prefix="/api", tags=["ranking"])


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 2000
    dense_pool: int = 3000
    sparse_pool: int = 3000


@router.post("/retrieve", response_model=RetrievalResult, summary="Layer 1 — hybrid retrieval")
def retrieve_candidates(req: RetrieveRequest) -> RetrievalResult:
    """
    Run hybrid retrieval (dense + BM25 + RRF) and return the top-k candidates.
    This is the input to Layer 2 signal scorers.
    """
    return retrieval.retrieve(
        query=req.query,
        top_k=req.top_k,
        dense_pool=req.dense_pool,
        sparse_pool=req.sparse_pool,
    )
