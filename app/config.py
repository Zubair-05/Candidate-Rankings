"""
Application settings loaded from environment variables.

All runtime configuration lives here — never scattered across files.
Change behaviour by editing .env, no code changes needed.
"""
from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM provider ──────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "openai", "groq"] = "anthropic"
    llm_model: str = "claude-sonnet-4-6"
    llm_temperature: float = 0.1

    # ── Layer 3 pipeline knobs ────────────────────────────────────────────────
    layer3_candidate_pool: int = 50   # how many from Layer 2 enter the LLM judge
    layer3_final_top_k: int = 20      # how many ranked candidates come out
    layer3_max_concurrency: int = 5   # parallel LLM calls cap

    # ── Provider API keys ─────────────────────────────────────────────────────
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    groq_api_key: str = ""

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "candidates"

    model_config = {"env_file": ".env", "extra": "ignore"}


# Module-level singleton — import this everywhere
settings = Settings()
