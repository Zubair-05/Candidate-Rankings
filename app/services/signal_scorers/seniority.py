"""
Seniority Calibration Scorer

Signal: Is this candidate right-sized for the role?

Three inputs — none are hard cutoffs:
  1. Years of experience fit  (bell curve, weight 0.25 — least important)
  2. Title level fit          (weight 0.45 — primary signal)
  3. AI skill depth           (weight 0.30 — what they can actually do)

Weight rationale: YOE is the weakest signal because people gain depth at
different rates. A 3-year candidate who has a Staff title and expert-level
AI skills has clearly earned that level — penalising them heavily for YOE
would be wrong. Title and skill depth together tell a more honest story.

Fast-tracker override: if title ≥ Senior AND skill depth ≥ 0.75, the YOE
component is floored so it cannot drag the composite below 0.70. These
candidates go to Layer 3 where Claude reads their actual work.

Layer 3 (Claude) makes the final call on all borderline cases.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

from app.models.candidate import Candidate, SkillProficiency
from app.services.signal_scorers.base import BaseScorer, ScorerResult

_YOE_MIN  = 5.0
_YOE_MAX  = 9.0

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

_PROFICIENCY_SCORE: dict[SkillProficiency, float] = {
    SkillProficiency.BEGINNER:     0.25,
    SkillProficiency.INTERMEDIATE: 0.50,
    SkillProficiency.ADVANCED:     0.75,
    SkillProficiency.EXPERT:       1.00,
}

_AI_KEYWORDS = {
    "machine learning", "deep learning", "nlp", "llm", "rag",
    "embedding", "vector", "pytorch", "tensorflow", "python",
    "transformers", "mlops",
}


def _title_level(title: str) -> int:
    for pattern, level in _TITLE_LEVELS:
        if pattern.search(title):
            return level
    return _DEFAULT_LEVEL


def _yoe_fit(yoe: float) -> float:
    if _YOE_MIN <= yoe <= _YOE_MAX:
        return 1.0
    distance = min(abs(yoe - _YOE_MIN), abs(yoe - _YOE_MAX))
    return max(0.2, math.exp(-0.08 * distance ** 2))


class SeniorityScorer(BaseScorer):
    name = "seniority_calibration"
    weight = 1.5
    description = (
        "Bell-curve YOE fit + title level + AI skill depth. "
        "No hard cutoffs — fast-trackers with strong titles score well below the YOE band."
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
        # --- 1. YOE fit (weight 0.4) ---
        yoe       = candidate.profile.years_of_experience
        yoe_score = _yoe_fit(yoe)

        # --- 2. Title level fit (weight 0.4) ---
        current_level = _title_level(candidate.profile.current_title)
        if current_level in (3, 4):
            title_score = 1.0
        elif current_level == 2:
            title_score = 0.6
        elif current_level == 1:
            title_score = 0.3
        elif current_level == 5:
            title_score = 0.75
        else:  # VP/CXO — over-leveled
            title_score = 0.5

        # Rescue: candidate may have stepped down for a startup move
        peak_level = max((_title_level(r.title) for r in candidate.career_history), default=current_level)
        if peak_level > current_level:
            title_score = max(title_score, 0.75)

        # --- 3. AI skill depth (weight 0.30) ---
        ai_skills = [
            s for s in candidate.skills
            if any(kw in s.name.lower() for kw in _AI_KEYWORDS)
        ]
        if ai_skills:
            avg_prof    = sum(_PROFICIENCY_SCORE[s.proficiency] for s in ai_skills) / len(ai_skills)
            skill_score = self._clamp(avg_prof)
        else:
            skill_score = 0.2

        # Fast-tracker override: strong title + strong skills = YOE is not the
        # bottleneck. Floor the composite so YOE can't drag a clearly-capable
        # candidate below the threshold for Layer 3.
        fast_tracker = current_level >= 3 and skill_score >= 0.75
        if fast_tracker:
            yoe_score = max(yoe_score, 0.65)

        composite = 0.25 * yoe_score + 0.45 * title_score + 0.30 * skill_score

        fast_tracker_note = " [fast-tracker override applied]" if fast_tracker else ""
        rationale = (
            f"yoe={yoe:.1f}(fit={yoe_score:.2f}), "
            f"title='{candidate.profile.current_title}'(level {current_level}, fit={title_score:.2f}), "
            f"ai_skills={len(ai_skills)}(depth={skill_score:.2f})"
            f"{fast_tracker_note}"
        )
        return self._result(self._clamp(composite), rationale)
