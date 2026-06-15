"""
Ingest processed candidate data into Qdrant.

Reads:
  data/processed/embeddings.npy       — dense vectors (100k × 384, float32)
  data/processed/candidate_ids.json   — ordered list of candidate IDs
  data/processed/text_blobs.json      — text representation per candidate
  data/processed/embeddings_metadata.json

Builds:
  TF-IDF sparse vectors (BM25 approximation) over the full corpus
  Saves fitted vectorizer to data/processed/tfidf_vectorizer.pkl

Upserts to Qdrant:
  Collection: $QDRANT_COLLECTION (default: candidates)
  Vectors:    dense  — 384-dim cosine
              sparse — TF-IDF indices + values

Usage:
  python scripts/ingest_to_qdrant.py [--batch-size 512] [--recreate]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from app.config import settings

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
_VECTORIZER_PATH = _DATA_DIR / "tfidf_vectorizer.pkl"

DENSE_DIM = 384


def _load_processed() -> tuple[np.ndarray, list[str], list[str]]:
    print("Loading processed files…")
    embeddings = np.load(str(_DATA_DIR / "embeddings.npy")).astype(np.float32)

    with open(_DATA_DIR / "candidate_ids.json") as f:
        candidate_ids: list[str] = json.load(f)

    with open(_DATA_DIR / "text_blobs.json") as f:
        text_blobs: list[str] = json.load(f)

    assert len(embeddings) == len(candidate_ids) == len(text_blobs), (
        f"Length mismatch: embeddings={len(embeddings)}, "
        f"ids={len(candidate_ids)}, blobs={len(text_blobs)}"
    )
    print(f"  {len(candidate_ids):,} candidates, embedding dim={embeddings.shape[1]}")
    return embeddings, candidate_ids, text_blobs


def _build_tfidf(text_blobs: list[str]) -> TfidfVectorizer:
    print("Fitting TF-IDF vectorizer (BM25 approximation)…")
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,      # log(1+tf) — closer to BM25 term frequency saturation
        min_df=2,               # ignore terms that appear in < 2 docs
        max_df=0.95,            # ignore terms that appear in > 95% of docs
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=65_536,    # sparse vector max index
    )
    vectorizer.fit(text_blobs)
    elapsed = round((time.perf_counter() - t0) * 1000)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"  vocab={vocab_size:,} terms, fitted in {elapsed}ms")

    with open(_VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  Saved vectorizer → {_VECTORIZER_PATH}")
    return vectorizer


def _setup_collection(client: QdrantClient, recreate: bool) -> None:
    collection = settings.qdrant_collection
    exists = any(c.name == collection for c in client.get_collections().collections)

    if exists and recreate:
        print(f"Deleting existing collection '{collection}'…")
        client.delete_collection(collection)
        exists = False

    if not exists:
        print(f"Creating collection '{collection}'…")
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams()),
            },
        )
        print("  Collection created.")
    else:
        count = client.count(collection_name=collection).count
        print(f"Collection '{collection}' already exists ({count:,} points). "
              f"Pass --recreate to wipe and reload.")


def _upsert_batch(
    client: QdrantClient,
    batch_ids: list[int],
    batch_cids: list[str],
    batch_dense: np.ndarray,
    batch_sparse_matrix,
) -> None:
    points = []
    for local_idx, (point_id, cid) in enumerate(zip(batch_ids, batch_cids)):
        sparse_row = batch_sparse_matrix[local_idx]
        indices = sparse_row.indices.tolist()
        values  = sparse_row.data.tolist()

        points.append(PointStruct(
            id=point_id,
            vector={
                "dense":  batch_dense[local_idx].tolist(),
                "sparse": SparseVector(indices=indices, values=values),
            },
            payload={"candidate_id": cid},
        ))

    client.upsert(collection_name=settings.qdrant_collection, points=points)


def main(batch_size: int, recreate: bool) -> None:
    embeddings, candidate_ids, text_blobs = _load_processed()
    vectorizer = _build_tfidf(text_blobs)

    print(f"Connecting to Qdrant at {settings.qdrant_url}…")
    client = QdrantClient(url=settings.qdrant_url)
    _setup_collection(client, recreate=recreate)

    n = len(candidate_ids)
    print(f"Upserting {n:,} candidates in batches of {batch_size}…")

    t0 = time.perf_counter()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        batch_cids   = candidate_ids[start:end]
        batch_ids    = list(range(start, end))
        batch_dense  = embeddings[start:end]
        batch_blobs  = text_blobs[start:end]
        batch_sparse = vectorizer.transform(batch_blobs)

        _upsert_batch(client, batch_ids, batch_cids, batch_dense, batch_sparse)

        pct = round((end / n) * 100)
        elapsed = round(time.perf_counter() - t0)
        print(f"  {end:>7,} / {n:,}  ({pct}%)  {elapsed}s elapsed", end="\r", flush=True)

    elapsed = round(time.perf_counter() - t0)
    print(f"\nDone. Upserted {n:,} candidates in {elapsed}s.")

    count = client.count(collection_name=settings.qdrant_collection).count
    print(f"Qdrant collection '{settings.qdrant_collection}' now has {count:,} points.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest candidates into Qdrant")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the collection before ingesting")
    args = parser.parse_args()
    main(batch_size=args.batch_size, recreate=args.recreate)
