"""
Activity Signals Scorer

Signal: Is this candidate active, hireable, and likely to engage?

Methodology:
  - Uses Redrob platform signals exclusively (redrob_signals block)
  - Combines availability signals, engagement signals, and verification signals
  - A brilliant candidate who ignores recruiters is a wasted top-20 slot
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from app.services.signal_scorers.base import BaseScorer, ScorerResult

_TODAY = date.today()
_STALE_THRESHOLD_DAYS = 180  # 6 months inactive = low signal


def _days_since(date_str: str) -> int:
    try:
        last = date.fromisoformat(date_str)
        return (_TODAY - last).days
    except (ValueError, TypeError):
        return 999


class ActivitySignalsScorer(BaseScorer):
    name = "activity_signals"
    weight = 1.5
    description = "Scores platform engagement, availability, and recruiter responsiveness."

    def score(self, candidate: dict[str, Any]) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: dict[str, Any]) -> ScorerResult:
        signals = candidate.get("redrob_signals", {})
        if not signals:
            return self._missing("redrob_signals block missing")

        notes: list[str] = []
        component_scores: list[tuple[float, float]] = []  # (score, weight)

        # --- 1. Availability (weight 2.0) ---
        open_to_work = signals.get("open_to_work_flag", False)
        days_inactive = _days_since(signals.get("last_active_date", ""))
        recency_score = self._clamp(1.0 - (days_inactive / _STALE_THRESHOLD_DAYS))
        availability = (0.6 if open_to_work else 0.2) + 0.4 * recency_score
        component_scores.append((self._clamp(availability), 2.0))
        notes.append(f"open={open_to_work}, inactive={days_inactive}d")

        # --- 2. Recruiter responsiveness (weight 2.5) ---
        response_rate = signals.get("recruiter_response_rate", -1)
        interview_rate = signals.get("interview_completion_rate", -1)
        if response_rate >= 0 and interview_rate >= 0:
            responsiveness = 0.6 * response_rate + 0.4 * interview_rate
        elif response_rate >= 0:
            responsiveness = response_rate
        else:
            responsiveness = 0.5  # unknown → neutral
        component_scores.append((self._clamp(responsiveness), 2.5))
        notes.append(f"response_rate={response_rate:.2f}")

        # --- 3. GitHub activity (weight 1.5) ---
        github_score = signals.get("github_activity_score", -1)
        if github_score >= 0:
            # Normalise 0–100 → 0–1
            gh_normalised = github_score / 100
            component_scores.append((gh_normalised, 1.5))
            notes.append(f"github={github_score:.0f}/100")
        # No penalty for missing GitHub — not everyone links it

        # --- 4. Profile quality (weight 1.0) ---
        completeness = signals.get("profile_completeness_score", 0) / 100
        verified = sum([
            signals.get("verified_email", False),
            signals.get("verified_phone", False),
            signals.get("linkedin_connected", False),
        ]) / 3
        profile_quality = 0.5 * completeness + 0.5 * verified
        component_scores.append((self._clamp(profile_quality), 1.0))
        notes.append(f"completeness={completeness:.0%}, verified={verified:.0%}")

        # --- 5. Notice period (weight 0.5 — light signal) ---
        notice_days = signals.get("notice_period_days", 90)
        # Sweet spot 15–60 days; long notice is a mild negative
        notice_score = self._clamp(1.0 - max(0, notice_days - 60) / 120)
        component_scores.append((notice_score, 0.5))

        # Weighted average
        total_weight = sum(w for _, w in component_scores)
        composite = sum(s * w for s, w in component_scores) / total_weight

        return self._result(
            self._clamp(composite),
            "; ".join(notes),
        )
