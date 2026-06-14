# CLAUDE.md — Candidate Ranking System

## What this is
A production-grade AI candidate ranking system built for the Redrob India Runs hackathon (Track 1 — Data & AI Challenge). It ranks candidates against a job description the way a great recruiter would — not by keyword matching, but by understanding career trajectory, skill recency, domain fit, and behavioral signals.

## Problem being solved
Recruiters miss good candidates because keyword filters can't interpret context. This system uses a multi-signal scoring pipeline with LLM-as-judge to produce a ranked shortlist with explainable reasoning per candidate.

## Tech stack

* API: FastAPI + Pydantic v2 (async, typed throughout)
* Database: PostgreSQL with pgvector extension (candidates, JDs, embeddings, results)
* Cache: Redis (hot query caching)
* Embeddings: sentence-transformers (local) or Claude embeddings via Anthropic SDK
* Keyword search: BM25 via rank_bm25 library
* Fusion: Reciprocal Rank Fusion (RRF) merging vector + BM25 results
* LLM judge: Claude (claude-sonnet-4-6) via Anthropic SDK — structured Pydantic output
* Async jobs: FastAPI BackgroundTasks (Kafka/Inngest noted as production scaling path)
* Deployment: GCP Cloud Run + Docker
* CI/CD: GitHub Actions

## Architecture — three layers

### Layer 1: Semantic fit
* Embed JD and candidate profiles using sentence-transformers
* Store embeddings in pgvector
* Cosine similarity search returns top-50 candidates

### Layer 2: Structured signal scoring (main differentiator)
Five explicit scorers run on the top-50:

1. Career trajectory score — promotion patterns, growth direction over time
2. Skill recency score — skills weighted by how recently they were applied
3. Domain relevance score — industry/domain match between candidate history and JD
4. Activity/behavioral score — platform activity signals from dataset
5. Seniority calibration — right-sized vs overqualified vs underqualified

RRF merges vector search results with BM25 keyword results before signal scoring.

### Layer 3: LLM synthesis
* Top 20 candidates from Layer 1+2 passed to Claude
* Claude returns structured JSON: overall score, top 3 fit reasons, top 2 risks, confidence
* Pydantic model validates every LLM response — no raw string output
* Prompt versions stored in DB, not hardcoded

## Output format
Each ranked candidate gets a structured card:

```json
{
  "rank": 1,
  "candidate_id": "...",
  "overall_score": 8.7,
  "fit_reasons": ["...", "...", "..."],
  "risks": ["...", "..."],
  "confidence": "high",
  "signal_breakdown": {
    "semantic_fit": 0.82,
    "career_trajectory": 9,
    "skill_recency": 8,
    "domain_relevance": 9,
    "activity_signals": 6,
    "seniority_calibration": 8
  }
}
```

## Project structure

```
candidate-ranking/
├── app/
│   ├── main.py                  # FastAPI app, routes
│   ├── models/
│   │   ├── candidate.py         # Pydantic models for candidate
│   │   ├── job_description.py   # Pydantic models for JD
│   │   └── ranking.py           # Pydantic models for output
│   ├── services/
│   │   ├── embeddings.py        # Embedding generation
│   │   ├── vector_search.py     # pgvector search
│   │   ├── bm25_search.py       # BM25 keyword search
│   │   ├── rrf.py               # Reciprocal Rank Fusion
│   │   ├── signal_scorers.py    # Five structured signal scorers
│   │   └── llm_judge.py         # Claude scoring with structured output
│   ├── db/
│   │   ├── postgres.py          # PostgreSQL connection + queries
│   │   └── redis.py             # Redis cache
│   └── observability/
│       └── logger.py            # Structured logging, latency, cost tracking
├── evals/
│   ├── eval_runner.py           # Precision@K, NDCG, latency benchmarks
│   └── test_cases/              # Ground truth samples from dataset
├── data/
│   └── ingest.py                # Dataset ingestion script (Redrob dataset)
├── Dockerfile
├── docker-compose.yml           # PostgreSQL + Redis + app
├── .github/workflows/ci.yml     # GitHub Actions
└── README.md
```

## Key implementation rules

* All LLM responses validated through Pydantic — never trust raw strings
* Every pipeline stage logs: input size, output size, latency, cost
* Prompts stored in PostgreSQL `prompts` table with version field — never hardcoded
* Redis caches rankings for identical JD+candidate_set pairs (TTL: 1 hour)
* BackgroundTasks for async ingestion — POST /rank returns job_id immediately, GET /rank/{job_id} polls result
* All errors caught and logged with stage context — no silent failures

## Eval metrics

* Precision@K (K=5, K=10)
* NDCG (Normalized Discounted Cumulative Gain)
* Latency p50/p95 per pipeline stage
* Cost per ranking run (Claude token usage tracked)
* Retrieval quality: vector-only vs BM25-only vs hybrid comparison

## What NOT to do

* Do not frame this as a RAG system — it is a multi-signal ranking pipeline
* Do not hardcode prompts — always version them in DB
* Do not call Claude for all candidates — LLM pass runs only on top-20 from earlier stages
* Do not block on embedding generation — always async
* Do not skip evals — they are a submission requirement and a differentiator

## Submission requirements (deadline: June 28)

1. GitHub repo — clean, working, documented
2. PDF deck — problem framing, architecture, explainability output, eval results
3. Ranked output CSV in Redrob's specified format

## Dataset
Located in `/data/raw/` after download from Redrob Google Drive link. Run `python data/ingest.py` to parse and load into PostgreSQL. First task before any pipeline work: understand every field in the dataset.
