"""
Career Trajectory Scorer

Signal: Did this person grow, stagnate, or regress?

Methodology:
  - Map each role title to a seniority level (1–6)
  - Reward upward progression over time
  - Reward increasing company size (more scope, more impact)
  - Penalise lateral moves that consume years without advancement
"""
from __future__ import annotations

import re
from typing import Any

from app.services.signal_scorers.base import BaseScorer, ScorerResult

# Title → numeric seniority level.
# Checked in order — first match wins.
_TITLE_LEVELS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"vp|vice president|chief|cto|ceo|coo", re.I), 6),
    (re.compile(r"director|head of", re.I), 5),
    (re.compile(r"principal|distinguished|fellow", re.I), 5),
    (re.compile(r"staff|architect", re.I), 4),
    (re.compile(r"lead|manager|tech lead", re.I), 4),
    (re.compile(r"senior|sr\.?", re.I), 3),
    (re.compile(r"junior|jr\.?|associate|intern|trainee", re.I), 1),
]
_DEFAULT_LEVEL = 2  # plain "Engineer / Analyst / Scientist"

_COMPANY_SIZE_ORDER = [
    "1-10", "11-50", "51-200", "201-500",
    "501-1000", "1001-5000", "5001-10000", "10001+",
]


def _title_level(title: str) -> int:
    for pattern, level in _TITLE_LEVELS:
        if pattern.search(title):
            return level
    return _DEFAULT_LEVEL


def _company_size_rank(size: str) -> int:
    try:
        return _COMPANY_SIZE_ORDER.index(size)
    except ValueError:
        return 0


class CareerTrajectoryScorer(BaseScorer):
    name = "career_trajectory"
    weight = 2.5
    description = "Measures upward progression in title level and company scope over time."

    def score(self, candidate: dict[str, Any]) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: dict[str, Any]) -> ScorerResult:
        history = candidate.get("career_history", [])
        if not history:
            return self._missing("no career history")

        # Sort oldest → newest by start_date
        sorted_roles = sorted(history, key=lambda r: r.get("start_date", ""))

        levels = [_title_level(r.get("title", "")) for r in sorted_roles]
        sizes = [_company_size_rank(r.get("company_size", "")) for r in sorted_roles]

        score = 0.5  # neutral baseline

        # --- Title progression ---
        if len(levels) >= 2:
            promotions = sum(1 for a, b in zip(levels, levels[1:]) if b > a)
            regressions = sum(1 for a, b in zip(levels, levels[1:]) if b < a)
            progression_ratio = (promotions - regressions) / max(len(levels) - 1, 1)
            score += 0.25 * progression_ratio  # ±0.25

        # --- Absolute peak level reached ---
        peak_level = max(levels)
        # Level 3 = Senior (meets JD), level 4+ = above target
        peak_bonus = self._clamp((peak_level - 2) / 4)  # 0 at level 2, 1.0 at level 6
        score += 0.15 * peak_bonus

        # --- Company size growth ---
        if len(sizes) >= 2:
            size_growth = (sizes[-1] - sizes[0]) / max(len(_COMPANY_SIZE_ORDER) - 1, 1)
            score += 0.10 * self._clamp(size_growth, -1.0, 1.0)

        score = self._clamp(score)

        current_title = candidate.get("profile", {}).get("current_title", "unknown")
        rationale = (
            f"Peak level {peak_level}/6 ({current_title}); "
            f"{len(history)} roles; "
            f"title trajectory: {' → '.join(str(l) for l in levels)}"
        )
        return self._result(score, rationale)
