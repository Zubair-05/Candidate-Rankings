"""
Layer 1 — Hybrid retrieval: dense (embeddings) + sparse (TF-IDF) fused via RRF.

Qdrant handles both vector types and runs RRF fusion natively.
The TF-IDF vectorizer (fitted at ingest time) is loaded once at startup
and used to encode queries into the same sparse space as the index.
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from app.config import settings
from app.logger import get_logger, log_latency
from app.models.retrieval import RetrievalCandidate, RetrievalResult
from qdrant_client import QdrantClient
from qdrant_client.http.models import Prefetch, SparseVector
from qdrant_client.http.models import Fusion

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR         = Path(__file__).resolve().parents[2] / "data" / "processed"
_METADATA_PATH    = _DATA_DIR / "embeddings_metadata.json"
_VECTORIZER_PATH  = _DATA_DIR / "tfidf_vectorizer.pkl"

# ---------------------------------------------------------------------------
# Module-level singletons — initialised once at startup via build_index()
# ---------------------------------------------------------------------------
_qdrant:      Optional[QdrantClient]      = None
_embed_model: Optional[SentenceTransformer] = None
_tfidf:       Optional[TfidfVectorizer]   = None
_embed_model_name: str = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index() -> None:
    """
    Connect to Qdrant, load the TF-IDF vectorizer, and load the embedding model.
    Called once at application startup — never per request.
    """
    global _qdrant, _embed_model, _tfidf, _embed_model_name

    import json
    if _METADATA_PATH.exists():
        with open(_METADATA_PATH) as f:
            meta = json.load(f)
        _embed_model_name = meta.get("model", _embed_model_name)

    with log_latency(logger, "qdrant_connect", extra={"url": settings.qdrant_url}):
        _qdrant = QdrantClient(url=settings.qdrant_url)
        count = _qdrant.count(collection_name=settings.qdrant_collection).count
        logger.info(
            "Qdrant connected",
            extra={"collection": settings.qdrant_collection, "points": count},
        )

    with log_latency(logger, "tfidf_vectorizer_load"):
        with open(_VECTORIZER_PATH, "rb") as f:
            _tfidf = pickle.load(f)
        logger.info(
            "TF-IDF vectorizer loaded",
            extra={"vocab_size": len(_tfidf.vocabulary_)},
        )

    with log_latency(logger, "embedding_model_load", extra={"model": _embed_model_name}):
        _embed_model = SentenceTransformer(_embed_model_name)

    logger.info("Index ready", extra={"model": _embed_model_name})


def retrieve(
    query: str,
    top_k: int = 2000,
    dense_pool: int = 3000,
    sparse_pool: int = 3000,
) -> RetrievalResult:
    """
    Hybrid retrieval via Qdrant:
      1. Dense prefetch  — cosine similarity over 384-dim embeddings → top dense_pool
      2. Sparse prefetch — TF-IDF dot product → top sparse_pool
      3. RRF fusion      — Qdrant merges both ranked lists natively → top_k

    Args:
        query:       Job description text.
        top_k:       Final candidates to return after fusion.
        dense_pool:  How many dense results feed into RRF.
        sparse_pool: How many sparse results feed into RRF.
    """
    _require_index()
    t0 = time.perf_counter()

    # Encode query — dense and sparse
    t_enc = time.perf_counter()
    query_dense  = _encode_dense(query)
    query_sparse = _encode_sparse(query)
    enc_ms = round((time.perf_counter() - t_enc) * 1000, 1)

    # Qdrant hybrid query with native RRF fusion
    t_search = time.perf_counter()
    results = _qdrant.query_points(
        collection_name=settings.qdrant_collection,
        prefetch=[
            Prefetch(query=query_dense.tolist(), using="dense",  limit=dense_pool),
            Prefetch(query=query_sparse,         using="sparse", limit=sparse_pool),
        ],
        query=Fusion.RRF,
        limit=top_k,
        with_payload=True,
    ).points
    search_ms = round((time.perf_counter() - t_search) * 1000, 1)

    candidates = [
        RetrievalCandidate(
            candidate_id=pt.payload["candidate_id"],
            dense_rank=None,    # Qdrant RRF doesn't expose per-source ranks
            sparse_rank=None,
            rrf_score=pt.score,
            dense_score=None,
            text_blob="",
        )
        for pt in results
    ]

    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    logger.info(
        "layer1_retrieve complete",
        extra={
            "total_ms":   total_ms,
            "encode_ms":  enc_ms,
            "search_ms":  search_ms,
            "dense_pool": dense_pool,
            "sparse_pool": sparse_pool,
            "output":     len(candidates),
            "top_k":      top_k,
        },
    )

    return RetrievalResult(
        query=query,
        top_k=top_k,
        candidates=candidates,
        dense_count=dense_pool,
        sparse_count=sparse_pool,
        overlap_count=0,  # not available from Qdrant RRF
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_dense(query: str) -> np.ndarray:
    if "bge" in _embed_model_name.lower():
        query = f"Represent this sentence for searching relevant passages: {query}"
    return _embed_model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)


def _encode_sparse(query: str) -> SparseVector:
    matrix = _tfidf.transform([query])
    row = matrix[0]
    return SparseVector(
        indices=row.indices.tolist(),
        values=row.data.tolist(),
    )


def _require_index() -> None:
    if _qdrant is None or _tfidf is None or _embed_model is None:
        raise RuntimeError(
            "Retrieval index not initialised. Call retrieval.build_index() at startup."
        )
