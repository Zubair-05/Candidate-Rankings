"""
Pydantic models for a candidate profile.

Mirrors candidate_schema.json exactly. All fields match the JSONL structure
produced by the Redrob dataset. Parse at the ingestion boundary — scorers
and downstream code receive typed objects, never raw dicts.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums — match the schema's enum constraints exactly
# ---------------------------------------------------------------------------

class CompanySize(str, Enum):
    XS   = "1-10"
    S    = "11-50"
    M    = "51-200"
    L    = "201-500"
    XL   = "501-1000"
    XXL  = "1001-5000"
    XXXL = "5001-10000"
    ENTERPRISE = "10001+"


class SkillProficiency(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"
    EXPERT       = "expert"


class LanguageProficiency(str, Enum):
    BASIC          = "basic"
    CONVERSATIONAL = "conversational"
    PROFESSIONAL   = "professional"
    NATIVE         = "native"


class InstitutionTier(str, Enum):
    TIER_1  = "tier_1"
    TIER_2  = "tier_2"
    TIER_3  = "tier_3"
    TIER_4  = "tier_4"
    UNKNOWN = "unknown"


class WorkMode(str, Enum):
    REMOTE   = "remote"
    HYBRID   = "hybrid"
    ONSITE   = "onsite"
    FLEXIBLE = "flexible"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Profile(BaseModel):
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience: float = Field(ge=0, le=50)
    current_title: str
    current_company: str
    current_company_size: CompanySize
    current_industry: str


class CareerRole(BaseModel):
    company: str
    title: str
    start_date: date
    end_date: Optional[date] = None
    duration_months: int = Field(ge=0)
    is_current: bool
    industry: str
    company_size: CompanySize
    description: str


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str] = None
    tier: InstitutionTier = InstitutionTier.UNKNOWN


class Skill(BaseModel):
    name: str
    proficiency: SkillProficiency
    endorsements: int = Field(ge=0)
    duration_months: int = Field(ge=0, default=0)


class Certification(BaseModel):
    name: str
    issuer: str
    year: int


class Language(BaseModel):
    language: str
    proficiency: LanguageProficiency


class SalaryRange(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)


class RedrobSignals(BaseModel):
    profile_completeness_score: float = Field(ge=0, le=100)
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int = Field(ge=0)
    applications_submitted_30d: int = Field(ge=0)
    recruiter_response_rate: float = Field(ge=0, le=1)
    avg_response_time_hours: float = Field(ge=0)
    skill_assessment_scores: dict[str, float] = Field(default_factory=dict)
    connection_count: int = Field(ge=0)
    endorsements_received: int = Field(ge=0)
    notice_period_days: int = Field(ge=0, le=180)
    expected_salary_range_inr_lpa: SalaryRange
    preferred_work_mode: WorkMode
    willing_to_relocate: bool
    github_activity_score: float = Field(ge=-1, le=100)
    search_appearance_30d: int = Field(ge=0)
    saved_by_recruiters_30d: int = Field(ge=0)
    interview_completion_rate: float = Field(ge=0, le=1)
    offer_acceptance_rate: float = Field(ge=-1, le=1)
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


# ---------------------------------------------------------------------------
# Root model
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    candidate_id: str = Field(pattern=r"^CAND_[0-9]{7}$")
    profile: Profile
    career_history: list[CareerRole] = Field(min_length=1)
    education: list[Education] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)
    redrob_signals: RedrobSignals
