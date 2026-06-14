from typing import Optional

from pydantic import BaseModel, Field


class RetrievalCandidate(BaseModel):
    candidate_id: str
    dense_rank: Optional[int] = None       # rank in dense (embedding) results, None if absent
    sparse_rank: Optional[int] = None      # rank in BM25 results, None if absent
    rrf_score: float = 0.0                 # fused score — higher is better
    dense_score: Optional[float] = None    # raw cosine similarity
    text_blob: str = ""                    # kept for Layer 2 signal scorers


class RetrievalResult(BaseModel):
    query: str
    top_k: int
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    dense_count: int = 0                # how many candidates dense returned
    sparse_count: int = 0               # how many candidates BM25 returned
    overlap_count: int = 0              # how many appeared in both
