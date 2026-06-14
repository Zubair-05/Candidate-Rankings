"""
Career Trajectory Scorer

Signal: Did this person grow, stagnate, or regress?

Methodology:
  - Map each role title to a seniority level (1–6)
  - Reward upward title progression over time
  - Reward healthy tenure per role (deep enough to ship, not job-hopping)
  - Penalise lateral moves that consume years without advancement

Deliberately excludes company size: a startup engineer with full ownership
grows faster than a large-company engineer with narrow scope. Company size
context belongs in Layer 3 where Claude reads role descriptions directly.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.models.candidate import Candidate, CareerRole
from app.services.signal_scorers.base import BaseScorer, ScorerResult

_TITLE_LEVELS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"vp|vice president|chief|cto|ceo|coo", re.I), 6),
    (re.compile(r"director|head of", re.I), 5),
    (re.compile(r"principal|distinguished|fellow", re.I), 5),
    (re.compile(r"staff|architect", re.I), 4),
    (re.compile(r'tech lead|engineering manager|ml manager|ai manager|(?:^|\s)lead(?:\s|$)', re.I), 4),
    (re.compile(r"senior|sr\.?", re.I), 3),
    (re.compile(r"junior|jr\.?|associate|intern|trainee", re.I), 1),
]
_DEFAULT_LEVEL = 2

# Sweet spot: 18–42 months per role — long enough to ship, short enough to grow.
_TENURE_SWEET_MIN = 18
_TENURE_SWEET_MAX = 42


def _title_level(title: str) -> int:
    for pattern, level in _TITLE_LEVELS:
        if pattern.search(title):
            return level
    return _DEFAULT_LEVEL


def _tenure_score(duration_months: int, promoted_after: bool) -> float:
    """
    Score a single role's tenure. A short tenure followed by a promotion
    is a fast-tracker signal, not a job-hop.
    """
    if promoted_after and duration_months < _TENURE_SWEET_MIN:
        return 0.9
    if duration_months < 6:
        return 0.2
    if duration_months < 12:
        return 0.5
    if _TENURE_SWEET_MIN <= duration_months <= _TENURE_SWEET_MAX:
        return 1.0
    if duration_months <= 60:
        return 0.85
    return 0.6  # > 5 years without promotion — possible stagnation


class CareerTrajectoryScorer(BaseScorer):
    name = "career_trajectory"
    weight = 2.5
    description = (
        "Measures upward title progression and healthy tenure patterns. "
        "Company size is intentionally excluded — startup and large-company "
        "growth are equally valued; Claude evaluates scope in Layer 3."
    )

    def score(
        self,
        candidate: Candidate,
        context: Optional[dict[str, Any]] = None,
    ) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: Candidate) -> ScorerResult:
        history = candidate.career_history
        if not history:
            return self._missing("no career history")

        sorted_roles: list[CareerRole] = sorted(history, key=lambda r: r.start_date)
        levels = [_title_level(r.title) for r in sorted_roles]

        score = 0.5

        # --- 1. Title progression ---
        if len(levels) >= 2:
            promotions  = sum(1 for a, b in zip(levels, levels[1:]) if b > a)
            regressions = sum(1 for a, b in zip(levels, levels[1:]) if b < a)
            progression_ratio = (promotions - regressions) / max(len(levels) - 1, 1)
            score += 0.25 * progression_ratio

        # --- 2. Peak level reached ---
        peak_level = max(levels)
        peak_bonus = self._clamp((peak_level - 2) / 4)
        score += 0.15 * peak_bonus

        # --- 3. Tenure quality ---
        tenure_scores = [
            _tenure_score(
                sorted_roles[i].duration_months,
                promoted_after=(i + 1 < len(levels) and levels[i + 1] > levels[i]),
            )
            for i in range(len(sorted_roles))
        ]
        avg_tenure = sum(tenure_scores) / len(tenure_scores)
        score += 0.10 * (avg_tenure - 0.5) * 2

        score = self._clamp(score)
        rationale = (
            f"Peak level {peak_level}/6 ({candidate.profile.current_title}); "
            f"{len(history)} roles; "
            f"trajectory: {' → '.join(str(l) for l in levels)}; "
            f"avg tenure score: {avg_tenure:.2f}"
        )
        return self._result(score, rationale)
