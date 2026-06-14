"""
Seniority Calibration Scorer

Signal: Is this candidate right-sized for the role?

Methodology — three inputs as discussed:
  1. Years of experience fit   (bell curve, not hard cutoff)
  2. Title level fit           (target = Senior/Staff/Lead = levels 3–4)
  3. Skill depth               (avg proficiency of AI/ML skills)

A 3-year fast-tracker with Senior title and expert-level skills scores well.
A 12-year candidate stuck at mid-level with intermediate skills scores lower.
Neither is hard-filtered — Layer 3 makes the final call.
"""
from __future__ import annotations

import math
import re
from typing import Any

from app.services.signal_scorers.base import BaseScorer, ScorerResult

# JD target band
_YOE_MIN = 5.0
_YOE_MAX = 9.0
_YOE_IDEAL = 7.0   # centre of the bell curve

# Title → level (same mapping as career_trajectory.py)
_TITLE_LEVELS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"vp|vice president|chief|cto|ceo|coo", re.I), 6),
    (re.compile(r"director|head of", re.I), 5),
    (re.compile(r"principal|distinguished|fellow", re.I), 5),
    (re.compile(r"staff|architect", re.I), 4),
    (re.compile(r"lead|manager|tech lead", re.I), 4),
    (re.compile(r"senior|sr\.?", re.I), 3),
    (re.compile(r"junior|jr\.?|associate|intern|trainee", re.I), 1),
]
_DEFAULT_LEVEL = 2

_PROFICIENCY_WEIGHTS = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.0}

_AI_SKILL_KEYWORDS = {
    "machine learning", "deep learning", "nlp", "llm", "rag", "embedding",
    "vector", "pytorch", "tensorflow", "python", "transformers", "mlops",
}


def _title_level(title: str) -> int:
    for pattern, level in _TITLE_LEVELS:
        if pattern.search(title):
            return level
    return _DEFAULT_LEVEL


def _yoe_fit(yoe: float) -> float:
    """
    Bell curve centred on _YOE_IDEAL. Full score within [_YOE_MIN, _YOE_MAX].
    Falls off smoothly outside the band — never hard zero.
    """
    if _YOE_MIN <= yoe <= _YOE_MAX:
        return 1.0
    # Gaussian decay outside the band
    distance = min(abs(yoe - _YOE_MIN), abs(yoe - _YOE_MAX))
    return max(0.2, math.exp(-0.08 * distance ** 2))


class SeniorityScorer(BaseScorer):
    name = "seniority_calibration"
    weight = 1.5
    description = (
        "Measures right-sizing: years fit (bell curve), title level, and AI skill depth. "
        "Fast-trackers score well on title+skill even if YOE is below band."
    )

    def score(self, candidate: dict[str, Any]) -> ScorerResult:
        try:
            return self._compute(candidate)
        except Exception as exc:
            return self._missing(f"unexpected error: {exc}")

    def _compute(self, candidate: dict[str, Any]) -> ScorerResult:
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        history = candidate.get("career_history", [])

        # --- 1. Years of experience (weight 0.4) ---
        yoe = profile.get("years_of_experience")
        if yoe is None:
            yoe_score = 0.5
            yoe_note = "yoe=unknown"
        else:
            yoe_score = _yoe_fit(float(yoe))
            yoe_note = f"yoe={yoe:.1f}"

        # --- 2. Title level fit (weight 0.4) ---
        current_title = profile.get("current_title", "")
        level = _title_level(current_title)
        # Ideal = 3–4; below = under, above = slightly over-leveled for the role
        if level in (3, 4):
            title_score = 1.0
        elif level == 2:
            title_score = 0.6
        elif level == 1:
            title_score = 0.3
        elif level == 5:
            title_score = 0.75   # principal/director — strong but possibly over-leveled
        else:  # level 6 — VP/CXO
            title_score = 0.5

        # Boost if history shows they've held Senior+ titles even if current is lower
        if history:
            peak_level = max(_title_level(r.get("title", "")) for r in history)
            if peak_level > level:
                title_score = max(title_score, 0.75)

        # --- 3. AI skill depth (weight 0.2) ---
        ai_skills = [
            s for s in skills
            if any(kw in s.get("name", "").lower() for kw in _AI_SKILL_KEYWORDS)
        ]
        if ai_skills:
            avg_prof = sum(
                _PROFICIENCY_WEIGHTS.get(s.get("proficiency", "beginner"), 0.25)
                for s in ai_skills
            ) / len(ai_skills)
            skill_depth_score = self._clamp(avg_prof)
        else:
            skill_depth_score = 0.2

        composite = 0.4 * yoe_score + 0.4 * title_score + 0.2 * skill_depth_score

        rationale = (
            f"{yoe_note}, title='{current_title}'(level {level}), "
            f"ai_skills={len(ai_skills)}, "
            f"yoe_fit={yoe_score:.2f}, title_fit={title_score:.2f}, "
            f"skill_depth={skill_depth_score:.2f}"
        )
        return self._result(self._clamp(composite), rationale)
