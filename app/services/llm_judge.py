"""
Layer 3 — LLM Judge

Takes the top N candidates from Layer 2, evaluates each one against the JD
using a language model, and returns a final reranked shortlist with structured
reasoning per candidate.

Design decisions:
- Provider-agnostic: swap LLM_PROVIDER in .env — zero code changes
- One structured call per candidate, not a batched mega-prompt
- Parallel execution via asyncio.gather with a semaphore concurrency cap
- .with_structured_output() handles JSON parsing + Pydantic validation per provider
- Two-strike retry: if structured output fails, retry once with an explicit JSON
  instruction; if it fails again, fall back gracefully with llm_judged=False
- Final score = 0.4 × layer2_composite + 0.6 × llm_score so Claude's reading
  has majority weight but can't fully override strong structural signals
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema — every LLM response is validated against this
# ---------------------------------------------------------------------------

class LLMJudgement(BaseModel):
    candidate_id: str = Field(description="Candidate ID exactly as provided")
    llm_score: float = Field(ge=0.0, le=1.0, description="Overall fit score 0–1")
    fit_reasons: list[str] = Field(
        min_length=1, max_length=3,
        description="Top 1–3 specific reasons from their actual work experience",
    )
    risks: list[str] = Field(
        max_length=2,
        description="0–2 genuine risks or gaps for this role",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="How confident you are in this assessment")
    summary: str = Field(
        max_length=200,
        description="One recruiter-friendly sentence summarising fit",
    )


class JudgedCandidate(BaseModel):
    candidate_id: str
    rank: int
    final_score: float
    layer2_score: float
    llm_score: Optional[float] = None
    llm_judged: bool
    fit_reasons: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: Optional[float] = None
    summary: Optional[str] = None
    scorer_breakdown: list[dict] = Field(default_factory=list)
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float = 0.0


# ---------------------------------------------------------------------------
# LLM factory — swap provider via .env
# ---------------------------------------------------------------------------

def _build_llm() -> BaseChatModel:
    provider = settings.llm_provider
    model    = settings.llm_model
    temp     = settings.llm_temperature

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temp,
            api_key=settings.anthropic_api_key or None,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=settings.openai_api_key or None,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=settings.groq_api_key or None,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}. Choose anthropic, openai, or groq.")


# Module-level LLM instance — built lazily so startup doesn't fail if no key set
_llm: Optional[BaseChatModel] = None
_structured_llm: Any = None


def _get_structured_llm() -> Any:
    global _llm, _structured_llm
    if _structured_llm is None:
        _llm = _build_llm()
        _structured_llm = _llm.with_structured_output(LLMJudgement)
    return _structured_llm


# ---------------------------------------------------------------------------
# Prompt builder — focused summary, not raw JSON
# ---------------------------------------------------------------------------

def _build_prompt(jd: str, candidate: dict, layer2_score: float, scorer_breakdown: list[dict]) -> str:
    profile = candidate.get("profile", {})
    name    = profile.get("anonymized_name", candidate.get("candidate_id"))
    title   = profile.get("current_title", "")
    yoe     = profile.get("years_of_experience", "?")
    summary = profile.get("summary", "")

    career_lines = []
    for role in sorted(
        candidate.get("career_history", []),
        key=lambda r: r.get("start_date", ""),
        reverse=True,
    )[:5]:  # most recent 5 roles only — keep prompt tight
        desc = role.get("description", "")[:300]
        career_lines.append(
            f"  • {role.get('company')} | {role.get('title')} | "
            f"{role.get('duration_months', '?')} months\n    {desc}"
        )

    skills = ", ".join(
        f"{s['name']} ({s.get('proficiency','')})"
        for s in sorted(
            candidate.get("skills", []),
            key=lambda s: s.get("endorsements", 0),
            reverse=True,
        )[:10]
    )

    breakdown_lines = " | ".join(
        f"{r['scorer'].replace('_',' ')}: {round(r['score']*100)}%"
        for r in scorer_breakdown
    )

    career_block = "\n".join(career_lines) if career_lines else "  (no career history)"

    return f"""You are a senior technical recruiter evaluating candidates for the following role.

JOB DESCRIPTION:
{jd.strip()}

---

CANDIDATE: {name}
Current title: {title} | {yoe} years experience
Candidate ID: {candidate.get('candidate_id')}

Career (most recent first):
{career_block}

Top skills: {skills or '(none listed)'}

{'Professional summary: ' + summary[:400] if summary else ''}

Signal scoring (automated, for context only):
Layer 2 composite: {round(layer2_score * 100)}% | {breakdown_lines}

---

Evaluate this candidate's FIT for the job description above.
Base your reasoning on their ACTUAL work descriptions — not just job titles or skill names.
Be specific: cite what they built, shipped, or owned. Do not hallucinate details not present above.
Return candidate_id exactly as: {candidate.get('candidate_id')}"""


# ---------------------------------------------------------------------------
# Single candidate judge — with retry + graceful fallback
# ---------------------------------------------------------------------------

async def _judge_one(
    semaphore: asyncio.Semaphore,
    jd: str,
    candidate: dict,
    layer2_score: float,
    scorer_breakdown: list[dict],
    retrieval_meta: dict,
) -> JudgedCandidate:
    cid = candidate.get("candidate_id", "unknown")

    async with semaphore:
        structured_llm = _get_structured_llm()

        for attempt in range(2):
            try:
                prompt = _build_prompt(jd, candidate, layer2_score, scorer_breakdown)
                judgement: LLMJudgement = await structured_llm.ainvoke(prompt)

                final_score = round(
                    0.4 * layer2_score + 0.6 * judgement.llm_score, 4
                )
                return JudgedCandidate(
                    candidate_id=cid,
                    rank=0,  # assigned after sorting
                    final_score=final_score,
                    layer2_score=layer2_score,
                    llm_score=judgement.llm_score,
                    llm_judged=True,
                    fit_reasons=judgement.fit_reasons,
                    risks=judgement.risks,
                    confidence=judgement.confidence,
                    summary=judgement.summary,
                    scorer_breakdown=scorer_breakdown,
                    dense_rank=retrieval_meta.get("dense_rank"),
                    sparse_rank=retrieval_meta.get("sparse_rank"),
                    rrf_score=retrieval_meta.get("rrf_score", 0.0),
                )
            except Exception as exc:
                if attempt == 0:
                    logger.warning("LLM judge attempt 1 failed for %s: %s — retrying", cid, exc)
                else:
                    logger.error("LLM judge failed for %s after 2 attempts: %s — using Layer 2 score", cid, exc)

    # Graceful fallback: use Layer 2 score, mark as not LLM-judged
    return JudgedCandidate(
        candidate_id=cid,
        rank=0,
        final_score=round(layer2_score, 4),
        layer2_score=layer2_score,
        llm_judged=False,
        scorer_breakdown=scorer_breakdown,
        dense_rank=retrieval_meta.get("dense_rank"),
        sparse_rank=retrieval_meta.get("sparse_rank"),
        rrf_score=retrieval_meta.get("rrf_score", 0.0),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def judge_candidates(
    layer2_results: list[dict],
    candidates_store: dict[str, dict],
    jd: str,
) -> list[JudgedCandidate]:
    """
    Run LLM judgement in parallel over the top N Layer 2 candidates.

    layer2_results: list of RankedCandidate dicts (from ranking router)
    candidates_store: {candidate_id: raw_candidate_dict}
    jd: job description text

    Returns final top-k candidates sorted by final_score, with ranks assigned.
    """
    pool_size  = settings.layer3_candidate_pool
    final_topk = settings.layer3_final_top_k
    pool = layer2_results[:pool_size]

    logger.info(
        "Layer 3: judging %d candidates (pool=%d, final_top_k=%d, provider=%s model=%s)",
        len(pool), pool_size, final_topk, settings.llm_provider, settings.llm_model,
    )

    t0 = time.perf_counter()
    semaphore = asyncio.Semaphore(settings.layer3_max_concurrency)

    tasks = []
    for item in pool:
        cid = item["candidate_id"]
        raw = candidates_store.get(cid)
        if raw is None:
            logger.warning("Candidate %s not in store — skipping Layer 3", cid)
            continue
        tasks.append(_judge_one(
            semaphore=semaphore,
            jd=jd,
            candidate=raw,
            layer2_score=item["composite_score"],
            scorer_breakdown=item.get("scorer_breakdown", []),
            retrieval_meta={
                "dense_rank":  item.get("dense_rank"),
                "sparse_rank": item.get("sparse_rank"),
                "rrf_score":   item.get("rrf_score", 0.0),
            },
        ))

    judged = await asyncio.gather(*tasks)

    # Sort by final_score, assign ranks, return top-k
    judged_sorted = sorted(judged, key=lambda c: c.final_score, reverse=True)
    for i, c in enumerate(judged_sorted, start=1):
        c.rank = i

    elapsed = time.perf_counter() - t0
    top = judged_sorted[:final_topk]

    llm_judged_count = sum(1 for c in top if c.llm_judged)
    logger.info(
        "Layer 3 complete: %d/%d LLM-judged in %.1fs (score range %.3f–%.3f)",
        llm_judged_count, len(top), elapsed,
        top[-1].final_score if top else 0,
        top[0].final_score if top else 0,
    )
    return top
