# CLAUDE.md — Candidate Ranking System

## What this is
A production-grade AI candidate ranking system built for the Redrob India Runs hackathon (Track 1 — Data & AI Challenge). It ranks candidates against a job description the way a great recruiter would — not by keyword matching, but by understanding career trajectory, skill recency, domain fit, and behavioral signals.

## Audience
This codebase will be reviewed by the Redrob AI hiring team as a Senior AI/ML Engineer application. Code quality, architectural decisions, and documentation are evaluated as much as correctness. Write as a senior engineer would — no MVP shortcuts, no dict.get() chains, no hardcoded assumptions.

## Problem being solved
Recruiters miss good candidates because keyword filters can't interpret context. This system uses a multi-signal scoring pipeline with LLM-as-judge to produce a ranked shortlist with explainable reasoning per candidate.

## Tech stack

* API: FastAPI + Pydantic v2 (async, typed throughout)
* Embeddings: sentence-transformers (`all-MiniLM-L6-v2` for dev, upgrade to `BAAI/bge-large-en-v1.5` before final submission)
* Keyword search: BM25 via rank_bm25 library
* Fusion: Reciprocal Rank Fusion (RRF) merging dense + BM25 results
* LLM judge: Claude (`claude-sonnet-4-6`) via Anthropic SDK — structured Pydantic output
* Database: PostgreSQL with pgvector extension (future: store candidates, embeddings, RCA results)
* Cache: Redis (hot query caching — not yet wired)
* Deployment: GCP Cloud Run + Docker
* CI/CD: GitHub Actions

---

## Architecture — three layers + funnel

```
100,000 candidates
      ↓  Layer 1 — hybrid retrieval (dense + BM25 + RRF)
    2,000 candidates  ← semantically + keyword relevant
      ↓  Layer 2 — signal scorers (5 dimensions, plugin registry)
      50 candidates   ← structurally strong, not just good writers
      ↓  Layer 3 — Claude LLM judge
      20 candidates   ← deeply evaluated, ranked with reasoning
      ↓  Final output
     CSV top 100 (ranks 21–100 use Layer 2 composite order)
```

---

### Layer 1: Hybrid retrieval (COMPLETE)

**Files:** `app/services/retrieval.py`, `app/models/retrieval.py`, `app/routers/ranking.py`

- Loads `data/processed/embeddings.npy` + `candidate_ids.json` + `text_blobs.json` at startup via `build_index()`
- Dense search: cosine similarity over numpy matrix (L2-normalised dot product) → top 3000
- BM25 search: keyword search over text blobs → top 3000
- RRF fusion (k=60): merges both ranked lists → top 2000
- Endpoint: `POST /api/retrieve`

**Embeddings:**
- Generated in `notebooks/generate_embeddings.ipynb` on Google Colab (T4 GPU)
- Model: `all-MiniLM-L6-v2` — 384 dims, normalised float32
- 100,000 candidates encoded, saved to `data/processed/`
- Files in `data/processed/` are gitignored — regenerate from notebook if missing

---

### Layer 2: Signal scoring engine (COMPLETE)

**Files:** `app/services/signal_scorers/` (plugin package), `app/models/candidate.py`

#### Plugin architecture
- `BaseScorer` abstract class: every scorer declares `name`, `weight`, `description`; implements `score(candidate: Candidate, context: dict) -> ScorerResult`
- `REGISTERED_SCORERS` list in `__init__.py` — adding a scorer = create file + append to list, zero engine changes
- Engine normalises weights at runtime, builds context once per ranking run
- `context` dict carries `jd_embedding` + `embed_model` so scorers can adapt to any JD

#### Candidate model
- `app/models/candidate.py`: full Pydantic model mirroring `candidate_schema.json`
- All scorers receive typed `Candidate` objects — no `dict.get()` anywhere
- Raw dicts parsed via `Candidate.model_validate()` at the boundary in the engine
- Validation errors logged + skipped, never silently corrupt scoring

#### Five scorers

| Scorer | Weight | Signal |
|---|---|---|
| `CareerTrajectoryScorer` | 2.5 | Title level progression + tenure quality. Company size excluded intentionally — startup growth ≠ inferior. Fast-tracker promotion patterns rewarded. |
| `SkillRecencyScorer` | 2.0 | AI/ML skill depth weighted by proficiency × recency decay × endorsements. Top-5 skills count, not the long tail. |
| `DomainRelevanceScorer` | 2.5 | Cosine similarity between candidate industry history and JD embedding. No hardcoded domain scores — adapts to any JD automatically. |
| `ActivitySignalsScorer` | 1.5 | Redrob platform signals: open_to_work, recruiter response rate, GitHub activity, profile completeness, notice period. |
| `SeniorityScorer` | 1.5 | YOE bell curve (weight 0.25) + title level fit (0.45) + AI skill depth (0.30). Fast-tracker override: Senior+ title with expert AI skills floors YOE penalty. |

#### Key design decisions made
- **Company size removed from CareerTrajectoryScorer**: a startup engineer with full ownership grows faster than a FAANG engineer with narrow scope — company size context goes to Layer 3 (Claude reads descriptions)
- **DomainRelevanceScorer is JD-agnostic**: embeds each candidate industry string, computes cosine similarity against JD embedding. A Healthcare JD rewards HealthTech candidates; an E-commerce JD rewards retail candidates — zero config
- **SeniorityScorer uses soft signals, not hard cutoffs**: fast-tracker override ensures a 3-year candidate with Senior title + expert AI skills scores ~0.93, not penalised by YOE alone
- **CareerSwitchScorer is planned but not built yet**: will handle candidates transitioning from SDE → AI Engineer (title-skill gap, recent AI certifications, transferable seniority)

#### Endpoint
`POST /api/rank` — runs full Layer 1→2 pipeline, returns top-k with scorer breakdown per candidate

---

### Layer 3: LLM judge (NOT YET BUILT)

- Top 20 from Layer 2 passed to Claude (`claude-sonnet-4-6`)
- Claude returns structured JSON: overall score, top 3 fit reasons, top 2 risks, confidence
- Pydantic model validates every LLM response — never trust raw strings
- Prompts stored in DB with version field — never hardcoded

---

## Current project structure (actual)

```
candidate-ranking/
├── app/
│   ├── main.py                          # FastAPI app + lifespan (calls build_index on startup)
│   ├── models/
│   │   ├── candidate.py                 # Typed Pydantic model for full candidate schema
│   │   └── retrieval.py                 # RetrievalCandidate, RetrievalResult
│   ├── services/
│   │   ├── retrieval.py                 # build_index(), retrieve() — Layer 1
│   │   └── signal_scorers/
│   │       ├── __init__.py              # REGISTERED_SCORERS list
│   │       ├── base.py                  # BaseScorer, ScorerResult
│   │       ├── engine.py                # score_candidate(), score_candidates_bulk()
│   │       ├── career_trajectory.py
│   │       ├── skill_recency.py
│   │       ├── domain_relevance.py
│   │       ├── activity_signals.py
│   │       └── seniority.py
│   ├── routers/
│   │   ├── health.py                    # GET /health
│   │   └── ranking.py                   # POST /api/retrieve, POST /api/rank
│   ├── db/                              # Not yet implemented
│   └── observability/                   # Not yet implemented
├── notebooks/
│   └── generate_embeddings.ipynb        # Colab notebook — generates embeddings.npy
├── data/
│   ├── raw/                             # gitignored — candidates.jsonl, schema, sample CSV
│   └── processed/                       # gitignored — embeddings.npy, candidate_ids.json, text_blobs.json
├── evals/                               # Not yet built
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## What's next (Layer 3 + final pipeline)

1. **Layer 3 — LLM judge** (`app/services/llm_judge.py`)
   - Build Claude call with structured Pydantic output
   - Takes top 50 from Layer 2, returns top 20 reordered with reasoning
   - `POST /api/rank` should chain Layer 1 → 2 → 3

2. **Final CSV output**
   - Ranks 1–20: Claude-judged
   - Ranks 21–100: Layer 2 composite score order
   - Format: `candidate_id, rank, score, reasoning`

3. **Eval framework** (`evals/eval_runner.py`)
   - Precision@K (K=5, K=10)
   - NDCG
   - Latency p50/p95 per pipeline stage
   - Vector-only vs BM25-only vs hybrid comparison

4. **Future scorer to add**
   - `CareerSwitchScorer` — detects SDE→AI transitions via title-skill gap, recent AI certifications, transferable seniority

5. **Upgrade embedding model**
   - Swap `all-MiniLM-L6-v2` → `BAAI/bge-large-en-v1.5` in notebook
   - Re-run eval comparison before final submission

---

## Key implementation rules

* All LLM responses validated through Pydantic — never trust raw strings
* Scorers receive typed `Candidate` objects, never raw dicts — parsing happens once at the boundary
* Domain scorer adapts to the JD via embedding similarity — no hardcoded industry scores ever
* Every pipeline stage logs: input size, output size, latency
* No scorer ever raises — all exceptions caught internally, neutral 0.5 returned with `data_available=False`
* Weights are normalised at runtime — adding a scorer never requires recalculating others

## What NOT to do

* Do not use `dict.get()` inside scorers — parse to `Candidate` at the boundary
* Do not hardcode domain relevance scores — use JD embedding similarity
* Do not hardcode prompts — version them in DB
* Do not call Claude for all candidates — LLM pass runs only on top 20
* Do not hard-filter on YOE — use bell curve + fast-tracker override
* Do not reward larger company size as a proxy for growth

## Submission requirements (deadline: June 28)

1. GitHub repo — clean, working, documented
2. PDF deck — problem framing, architecture, explainability output, eval results
3. Ranked output CSV: `candidate_id, rank, score, reasoning` — top 100

## Running locally

```bash
# Create venv and install deps
python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Start the API (embeddings load on startup — takes ~30s)
venv/bin/uvicorn app.main:app --reload

# Test Layer 1
curl -s -X POST http://localhost:8000/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior AI Engineer with vector search and LLM experience", "top_k": 10}' \
  | python3 -m json.tool

# Test Layer 1+2
curl -s -X POST http://localhost:8000/api/rank \
  -H "Content-Type: application/json" \
  -d '{"query": "Senior AI Engineer with vector search and LLM experience", "signal_top_k": 20}' \
  | python3 -m json.tool
```

## Dataset
`data/raw/candidates.jsonl` — 100,000 candidate profiles, gitignored. Download from Redrob Google Drive and place in `data/raw/`. Run the Colab notebook to regenerate `data/processed/` files if needed.
