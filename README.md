# Candidate Ranking System

A production-grade AI pipeline that ranks 100,000 candidates against a job description the way a great recruiter would — not by keyword matching, but by understanding career trajectory, skill recency, domain fit, and behavioral signals.

Built for the **Redrob India Runs Hackathon — Track 1: Data & AI Challenge**.

---

## How it works

The system runs candidates through a three-layer funnel:

```
100,000 candidates
      ↓  Layer 1 — Hybrid Retrieval (dense + BM25 + RRF)
    2,000 candidates
      ↓  Layer 2 — Signal Scoring (5 structured dimensions)
      50 candidates
      ↓  Layer 3 — LLM Judge (Claude)
      20 candidates  →  final ranked CSV (top 100)
```

### Layer 1 — Hybrid Retrieval
Combines two fundamentally different search methods and fuses them:

- **Dense search**: cosine similarity over 384-dim sentence embeddings (`all-MiniLM-L6-v2`) — finds semantically relevant candidates even when they don't share exact keywords with the JD
- **BM25**: keyword frequency search over candidate text blobs — finds exact technical term matches
- **RRF (Reciprocal Rank Fusion)**: merges both ranked lists so candidates appearing in both get boosted

### Layer 2 — Signal Scoring
Five structured scorers run on the top 2,000. Each scorer returns a `0–1` score with a human-readable rationale. Scores are weighted and combined into a composite.

| Scorer | What it measures |
|---|---|
| Career Trajectory | Title-level progression over time. Fast-tracker patterns rewarded. Company size excluded — startup growth ≠ inferior. |
| Skill Recency | AI/ML skill depth weighted by proficiency × recency decay × endorsements. Expert skills from 6 months ago beat intermediate skills from 3 years ago. |
| Domain Relevance | Cosine similarity between candidate industry history and the JD embedding. Adapts to any JD automatically — no hardcoded domain scores. |
| Activity Signals | Redrob platform engagement: open-to-work status, recruiter response rate, GitHub activity, profile completeness, notice period. |
| Seniority Calibration | Bell-curve YOE fit + title level + AI skill depth. Fast-tracker override: a 3-year candidate with a Senior title and expert AI skills is not penalised by YOE alone. |

The scorer registry is **plugin-based** — adding a new dimension requires creating one file and registering it. The engine normalises weights at runtime.

### Layer 3 — LLM Judge *(in progress)*
Top 50 candidates from Layer 2 are passed to Claude (`claude-sonnet-4-6`). Claude evaluates each candidate's actual work descriptions, identifies fit reasons and risks, and produces a structured JSON output validated by Pydantic.

---

## Architecture decisions

**Why hybrid search?**
Dense search finds semantically similar candidates but misses exact technical terms. BM25 finds exact matches but has no semantic understanding. A candidate who "built production vector search infrastructure" and one who "designed intelligent retrieval systems" mean the same thing — dense search finds both, BM25 finds only the first.

**Why not just use the LLM for all 100k?**
Cost and latency. An LLM call per candidate across 100k would cost ~$200 and take hours. Layers 1 and 2 are fast, deterministic filters that give Claude only the candidates worth reasoning about deeply.

**Why is domain relevance JD-aware?**
Hardcoding "AI = 1.0, Healthcare = 0.5" bakes in assumptions about one JD. If tomorrow's role is "Senior ML Engineer in Healthcare", those scores are backwards. Instead, we embed each candidate's industry and compute cosine similarity against the JD embedding — the scorer adapts to any JD automatically.

**Why typed Pydantic models for candidates?**
`dict.get("field", default)` is fragile — silent failures, no type safety, no IDE support. The `Candidate` model mirrors the schema exactly. Raw dicts are parsed once at the ingestion boundary; everything downstream works with typed objects.

---

## Project structure

```
candidate-ranking/
├── app/
│   ├── main.py                          # FastAPI app, startup index loading
│   ├── models/
│   │   ├── candidate.py                 # Typed Pydantic model for full candidate schema
│   │   └── retrieval.py                 # RetrievalCandidate, RetrievalResult
│   ├── services/
│   │   ├── retrieval.py                 # Layer 1: build_index(), retrieve()
│   │   └── signal_scorers/
│   │       ├── __init__.py              # REGISTERED_SCORERS — add new scorers here
│   │       ├── base.py                  # BaseScorer abstract class, ScorerResult
│   │       ├── engine.py                # score_candidates_bulk() — Layer 2 entry point
│   │       ├── career_trajectory.py
│   │       ├── skill_recency.py
│   │       ├── domain_relevance.py
│   │       ├── activity_signals.py
│   │       └── seniority.py
│   └── routers/
│       ├── health.py                    # GET /health
│       └── ranking.py                   # POST /api/retrieve, POST /api/rank
├── notebooks/
│   └── generate_embeddings.ipynb        # Run on Google Colab (T4 GPU) to generate embeddings
├── data/
│   ├── raw/                             # gitignored — candidates.jsonl
│   └── processed/                       # gitignored — embeddings.npy, candidate_ids.json, text_blobs.json
├── evals/                               # Precision@K, NDCG — coming soon
├── requirements.txt
└── .env.example
```

---

## Getting started

### Prerequisites
- Python 3.9+
- `data/raw/candidates.jsonl` — download from Redrob Google Drive
- `data/processed/` files — generate using the Colab notebook

### Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
```

### Generate embeddings (one-time)
Open `notebooks/generate_embeddings.ipynb` in Google Colab, set runtime to T4 GPU, mount your Google Drive with `candidates.jsonl`, and run all cells. Download the output files to `data/processed/`.

### Run the API

```bash
venv/bin/uvicorn app.main:app --reload
```

The server loads the embedding index and builds the BM25 index on startup (~30 seconds for 100k candidates).

### Test the pipeline

```bash
# Layer 1 — hybrid retrieval
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Senior AI Engineer with vector search, LLMs, and production ML experience",
    "top_k": 10
  }' | python3 -m json.tool

# Layer 1 + 2 — retrieval + signal scoring
curl -s -X POST http://localhost:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Senior AI Engineer with vector search, LLMs, and production ML experience",
    "signal_top_k": 20
  }' | python3 -m json.tool
```

---

## Adding a new scorer

1. Create `app/services/signal_scorers/my_scorer.py` implementing `BaseScorer`
2. Add it to `REGISTERED_SCORERS` in `app/services/signal_scorers/__init__.py`

The engine picks it up automatically. Weights are normalised at runtime — no other changes needed.

```python
class MyScorer(BaseScorer):
    name = "my_scorer"
    weight = 1.0
    description = "What this scorer measures."

    def score(self, candidate: Candidate, context: dict | None = None) -> ScorerResult:
        try:
            # ... your logic using candidate.profile, candidate.skills, etc.
            return self._result(score=0.8, rationale="reason here")
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")
```

---

## Output format

```csv
candidate_id,rank,score,reasoning
CAND_0061257,1,0.88,"Staff ML Engineer, 8yrs in AI/ML. Strong vector search background. High recruiter responsiveness."
...
```

Top 100 candidates. Ranks 1–20 are Claude-judged. Ranks 21–100 use Layer 2 composite score order.

---

## Submission
**Deadline: June 28**
1. This GitHub repo
2. PDF deck — architecture, explainability output, eval results
3. Ranked CSV in the above format
