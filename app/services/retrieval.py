"""
Layer 1 — Hybrid retrieval: dense (embeddings) + sparse (BM25) fused via RRF.

Entry point: build_index() on startup, then retrieve() per query.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.models.retrieval import RetrievalCandidate, RetrievalResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
_EMBEDDINGS_PATH = _DATA_DIR / "embeddings.npy"
_IDS_PATH = _DATA_DIR / "candidate_ids.json"
_BLOBS_PATH = _DATA_DIR / "text_blobs.json"
_METADATA_PATH = _DATA_DIR / "embeddings_metadata.json"

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at startup
# ---------------------------------------------------------------------------
_embeddings: Optional[np.ndarray] = None       # (N, D) float32, L2-normalised
_candidate_ids: Optional[list[str]] = None
_text_blobs: Optional[list[str]] = None
_bm25_index: Optional[BM25Okapi] = None
_embed_model: Optional[SentenceTransformer] = None
_embed_model_name: str = "all-MiniLM-L6-v2"

# RRF constant — standard value; larger = less penalty for low ranks
_RRF_K = 60


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """Load embeddings + build BM25 index. Call once at application startup."""
    global _embeddings, _candidate_ids, _text_blobs, _bm25_index, _embed_model, _embed_model_name

    t0 = time.perf_counter()
    logger.info("Loading candidate index from %s", _DATA_DIR)

    with open(_IDS_PATH) as f:
        _candidate_ids = json.load(f)

    with open(_BLOBS_PATH) as f:
        _text_blobs = json.load(f)

    _embeddings = np.load(str(_EMBEDDINGS_PATH)).astype(np.float32)

    with open(_METADATA_PATH) as f:
        meta = json.load(f)
    _embed_model_name = meta.get("model", _embed_model_name)

    logger.info(
        "Embeddings loaded: shape=%s model=%s", _embeddings.shape, _embed_model_name
    )

    # BM25 — tokenise on whitespace (fast, sufficient for English tech text)
    logger.info("Building BM25 index over %d documents…", len(_text_blobs))
    tokenised = [blob.lower().split() for blob in _text_blobs]
    _bm25_index = BM25Okapi(tokenised)

    # Load the same embedding model for query encoding
    logger.info("Loading embedding model: %s", _embed_model_name)
    _embed_model = SentenceTransformer(_embed_model_name)

    elapsed = time.perf_counter() - t0
    logger.info("Index ready in %.1fs — %d candidates", elapsed, len(_candidate_ids))


def retrieve(
    query: str,
    top_k: int = 2000,
    dense_pool: int = 3000,
    sparse_pool: int = 3000,
) -> RetrievalResult:
    """
    Hybrid retrieval: dense cosine search + BM25, fused with RRF.

    Args:
        query:       The job description text (or a summary of it).
        top_k:       Final number of candidates to return after fusion.
        dense_pool:  How many top candidates dense search nominates before fusion.
        sparse_pool: How many top candidates BM25 nominates before fusion.

    Returns:
        RetrievalResult sorted by rrf_score descending.
    """
    _require_index()

    t0 = time.perf_counter()

    # --- Dense retrieval ---
    query_emb = _encode_query(query)
    cosine_scores = _embeddings @ query_emb          # dot product == cosine (normalised)
    dense_top_idx = np.argpartition(cosine_scores, -dense_pool)[-dense_pool:]
    dense_top_idx = dense_top_idx[np.argsort(cosine_scores[dense_top_idx])[::-1]]

    dense_ranks: dict[str, int] = {}
    dense_scores: dict[str, float] = {}
    for rank, idx in enumerate(dense_top_idx, start=1):
        cid = _candidate_ids[idx]
        dense_ranks[cid] = rank
        dense_scores[cid] = float(cosine_scores[idx])

    # --- Sparse retrieval (BM25) ---
    tokens = query.lower().split()
    bm25_scores = _bm25_index.get_scores(tokens)
    sparse_top_idx = np.argpartition(bm25_scores, -sparse_pool)[-sparse_pool:]
    sparse_top_idx = sparse_top_idx[np.argsort(bm25_scores[sparse_top_idx])[::-1]]

    sparse_ranks: dict[str, int] = {}
    for rank, idx in enumerate(sparse_top_idx, start=1):
        sparse_ranks[_candidate_ids[idx]] = rank

    # --- Reciprocal Rank Fusion ---
    all_cids = set(dense_ranks) | set(sparse_ranks)
    rrf_scores: dict[str, float] = {}
    for cid in all_cids:
        score = 0.0
        if cid in dense_ranks:
            score += 1.0 / (_RRF_K + dense_ranks[cid])
        if cid in sparse_ranks:
            score += 1.0 / (_RRF_K + sparse_ranks[cid])
        rrf_scores[cid] = score

    sorted_cids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:top_k]

    # --- Build id→blob lookup (needed by Layer 2) ---
    id_to_blob = {cid: _text_blobs[i] for i, cid in enumerate(_candidate_ids)}

    candidates = [
        RetrievalCandidate(
            candidate_id=cid,
            dense_rank=dense_ranks.get(cid),
            sparse_rank=sparse_ranks.get(cid),
            rrf_score=rrf_scores[cid],
            dense_score=dense_scores.get(cid),
            text_blob=id_to_blob.get(cid, ""),
        )
        for cid in sorted_cids
    ]

    overlap = len(set(dense_ranks) & set(sparse_ranks))
    elapsed = time.perf_counter() - t0
    logger.info(
        "retrieve() → %d candidates (dense=%d sparse=%d overlap=%d) in %.2fs",
        len(candidates), len(dense_ranks), len(sparse_ranks), overlap, elapsed,
    )

    return RetrievalResult(
        query=query,
        top_k=top_k,
        candidates=candidates,
        dense_count=len(dense_ranks),
        sparse_count=len(sparse_ranks),
        overlap_count=overlap,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_query(query: str) -> np.ndarray:
    # bge models expect a prefix for asymmetric search
    if "bge" in _embed_model_name.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    emb = _embed_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def _require_index() -> None:
    if _embeddings is None or _bm25_index is None:
        raise RuntimeError(
            "Retrieval index not built. Call retrieval.build_index() at startup."
        )
