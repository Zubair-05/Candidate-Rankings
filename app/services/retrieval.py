"""
Layer 1 — Hybrid retrieval: dense (embeddings) + sparse (BM25) fused via RRF.

Entry point: build_index() on startup, then retrieve() per query.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.logger import get_logger, log_latency
from app.models.retrieval import RetrievalCandidate, RetrievalResult

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR        = Path(__file__).resolve().parents[2] / "data" / "processed"
_EMBEDDINGS_PATH = _DATA_DIR / "embeddings.npy"
_IDS_PATH        = _DATA_DIR / "candidate_ids.json"
_BLOBS_PATH      = _DATA_DIR / "text_blobs.json"
_METADATA_PATH   = _DATA_DIR / "embeddings_metadata.json"

# ---------------------------------------------------------------------------
# Module-level singletons — loaded once at startup
# ---------------------------------------------------------------------------
_embeddings:      Optional[np.ndarray]         = None   # (N, D) float32, L2-normalised
_candidate_ids:   Optional[list[str]]           = None
_text_blobs:      Optional[list[str]]           = None
_bm25_index:      Optional[BM25Okapi]           = None
_embed_model:     Optional[SentenceTransformer] = None
_embed_model_name: str = "all-MiniLM-L6-v2"

_RRF_K = 60  # RRF constant — larger = less penalty for low ranks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """Load embeddings + build BM25 index. Called once at application startup."""
    global _embeddings, _candidate_ids, _text_blobs, _bm25_index, _embed_model, _embed_model_name

    logger.info("Loading candidate index", extra={"data_dir": str(_DATA_DIR)})

    with log_latency(logger, "load_embeddings_and_ids"):
        with open(_IDS_PATH) as f:
            _candidate_ids = json.load(f)
        with open(_BLOBS_PATH) as f:
            _text_blobs = json.load(f)
        _embeddings = np.load(str(_EMBEDDINGS_PATH)).astype(np.float32)
        with open(_METADATA_PATH) as f:
            meta = json.load(f)
        _embed_model_name = meta.get("model", _embed_model_name)

    logger.info(
        "Embeddings loaded",
        extra={"shape": str(_embeddings.shape), "model": _embed_model_name},
    )

    with log_latency(logger, "bm25_index_build", extra={"corpus_size": len(_text_blobs)}):
        tokenised  = [blob.lower().split() for blob in _text_blobs]
        _bm25_index = BM25Okapi(tokenised)

    with log_latency(logger, "embedding_model_load", extra={"model": _embed_model_name}):
        _embed_model = SentenceTransformer(_embed_model_name)

    logger.info(
        "Index ready",
        extra={"candidates": len(_candidate_ids), "model": _embed_model_name},
    )


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
        dense_pool:  How many candidates dense search nominates before fusion.
        sparse_pool: How many candidates BM25 nominates before fusion.
    """
    _require_index()
    t0 = time.perf_counter()

    # --- Dense retrieval ---
    t_dense = time.perf_counter()
    query_emb    = _encode_query(query)
    cosine_scores = _embeddings @ query_emb
    dense_top_idx = np.argpartition(cosine_scores, -dense_pool)[-dense_pool:]
    dense_top_idx = dense_top_idx[np.argsort(cosine_scores[dense_top_idx])[::-1]]

    dense_ranks:  dict[str, int]   = {}
    dense_scores: dict[str, float] = {}
    for rank, idx in enumerate(dense_top_idx, start=1):
        cid = _candidate_ids[idx]
        dense_ranks[cid]  = rank
        dense_scores[cid] = float(cosine_scores[idx])
    dense_ms = round((time.perf_counter() - t_dense) * 1000, 1)

    # --- Sparse retrieval (BM25) ---
    t_sparse    = time.perf_counter()
    tokens      = query.lower().split()
    bm25_scores = _bm25_index.get_scores(tokens)
    sparse_top_idx = np.argpartition(bm25_scores, -sparse_pool)[-sparse_pool:]
    sparse_top_idx = sparse_top_idx[np.argsort(bm25_scores[sparse_top_idx])[::-1]]

    sparse_ranks: dict[str, int] = {}
    for rank, idx in enumerate(sparse_top_idx, start=1):
        sparse_ranks[_candidate_ids[idx]] = rank
    sparse_ms = round((time.perf_counter() - t_sparse) * 1000, 1)

    # --- Reciprocal Rank Fusion ---
    t_rrf    = time.perf_counter()
    all_cids = set(dense_ranks) | set(sparse_ranks)
    rrf_scores: dict[str, float] = {
        cid: (
            (1.0 / (_RRF_K + dense_ranks[cid])  if cid in dense_ranks  else 0.0) +
            (1.0 / (_RRF_K + sparse_ranks[cid]) if cid in sparse_ranks else 0.0)
        )
        for cid in all_cids
    }
    sorted_cids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)[:top_k]
    rrf_ms = round((time.perf_counter() - t_rrf) * 1000, 1)

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

    overlap     = len(set(dense_ranks) & set(sparse_ranks))
    total_ms    = round((time.perf_counter() - t0) * 1000, 1)

    logger.info(
        "layer1_retrieve complete",
        extra={
            "total_ms":   total_ms,
            "dense_ms":   dense_ms,
            "sparse_ms":  sparse_ms,
            "rrf_ms":     rrf_ms,
            "output":     len(candidates),
            "dense_pool": len(dense_ranks),
            "sparse_pool": len(sparse_ranks),
            "overlap":    overlap,
            "top_k":      top_k,
        },
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
    if "bge" in _embed_model_name.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    emb = _embed_model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
    return emb.astype(np.float32)


def _require_index() -> None:
    if _embeddings is None or _bm25_index is None:
        raise RuntimeError(
            "Retrieval index not built. Call retrieval.build_index() at startup."
        )
