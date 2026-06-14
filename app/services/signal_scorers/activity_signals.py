"""
Activity Signals Scorer

Signal: Is this candidate active, hireable, and likely to engage?

Uses Redrob platform signals exclusively. A brilliant candidate who ignores
recruiters is a wasted top-20 slot — this scorer surfaces that risk early.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from app.models.candidate import Candidate, RedrobSignals
from app.services.signal_scorers.base import BaseScorer, ScorerResult

_TODAY = date.today()
_STALE_DAYS = 180  # 6 months inactive = low engagement signal


class ActivitySignalsScorer(BaseScorer):
    name = "activity_signals"
    weight = 1.5
    description = "Scores platform engagement, availability, and recruiter responsiveness."

    def score(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]] = None,
    ) -> ScorerResult:
        try:
            return self._compute(candidate.redrob_signals)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, signals: RedrobSignals) -> ScorerResult:
        notes: list[str] = []
        components: list[tuple[float, float]] = []  # (score, weight)

        # --- 1. Availability (weight 2.0) ---
        days_inactive   = (_TODAY - signals.last_active_date).days
        recency_score   = self._clamp(1.0 - days_inactive / _STALE_DAYS)
        availability    = (0.6 if signals.open_to_work_flag else 0.2) + 0.4 * recency_score
        components.append((self._clamp(availability), 2.0))
        notes.append(f"open={signals.open_to_work_flag}, inactive={days_inactive}d")

        # --- 2. Recruiter responsiveness (weight 2.5) ---
        responsiveness = (
            0.6 * signals.recruiter_response_rate
            + 0.4 * signals.interview_completion_rate
        )
        components.append((self._clamp(responsiveness), 2.5))
        notes.append(f"response_rate={signals.recruiter_response_rate:.2f}")

        # --- 3. GitHub activity (weight 1.5) ---
        if signals.github_activity_score >= 0:
            components.append((signals.github_activity_score / 100, 1.5))
            notes.append(f"github={signals.github_activity_score:.0f}/100")

        # --- 4. Profile quality (weight 1.0) ---
        completeness    = signals.profile_completeness_score / 100
        verified_count  = sum([signals.verified_email, signals.verified_phone, signals.linkedin_connected])
        profile_quality = 0.5 * completeness + 0.5 * (verified_count / 3)
        components.append((self._clamp(profile_quality), 1.0))
        notes.append(f"completeness={completeness:.0%}, verified={verified_count}/3")

        # --- 5. Notice period (weight 0.5) ---
        # Sweet spot ≤60 days; linearly decays up to 180
        notice_score = self._clamp(1.0 - max(0, signals.notice_period_days - 60) / 120)
        components.append((notice_score, 0.5))

        total_weight = sum(w for _, w in components)
        composite    = sum(s * w for s, w in components) / total_weight

        return self._result(self._clamp(composite), "; ".join(notes))
