"""
Signal scorer registry.

To add a new scorer:
  1. Create a module in this package implementing BaseScorer.
  2. Import it here and append an instance to REGISTERED_SCORERS.

The engine (engine.py) reads REGISTERED_SCORERS at runtime — no other
changes required.
"""
from app.services.signal_scorers.base import BaseScorer, ScorerResult
from app.services.signal_scorers.career_trajectory import CareerTrajectoryScorer
from app.services.signal_scorers.skill_recency import SkillRecencyScorer
from app.services.signal_scorers.domain_relevance import DomainRelevanceScorer
from app.services.signal_scorers.activity_signals import ActivitySignalsScorer
from app.services.signal_scorers.seniority import SeniorityScorer

REGISTERED_SCORERS: list[BaseScorer] = [
    CareerTrajectoryScorer(),
    SkillRecencyScorer(),
    DomainRelevanceScorer(),
    ActivitySignalsScorer(),
    SeniorityScorer(),
]

__all__ = ["BaseScorer", "ScorerResult", "REGISTERED_SCORERS"]
