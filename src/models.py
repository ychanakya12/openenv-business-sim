"""
models.py — All Pydantic typed models for the Business Sim OpenEnv environment.
Satisfies OpenEnv spec: typed models for action_space and observation_space.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class TechStack(str, Enum):
    cheap    = "cheap"
    standard = "standard"
    premium  = "premium"


class MarketPhase(str, Enum):
    boom      = "boom"
    stable    = "stable"
    recession = "recession"


class Domain(str, Enum):
    ai     = "ai"
    web    = "web"
    mobile = "mobile"
    data   = "data"
    devops = "devops"


class Project(BaseModel):
    """A client project the CEO can accept."""
    id:                 str
    name:               str
    domain:             Domain

    # Factor 1 — Profit Potential
    base_profit:        float
    profit_variance:    float          # 0.0–1.0

    # Factor 2 — Risk Level
    base_risk:          float          # 0.0–1.0
    hidden_risk:        float          # 0.0–1.0  (uncertainty / unknown scope)

    # Factor 3 — Team Capability
    skill_required:     float          # 0.0–1.0

    # Factor 4 — Deadline Pressure (encoded as duration + tightness)
    duration_quarters:  int
    deadline_tight:     bool

    # Factor 5 — Market Demand sensitivity
    demand_sensitivity: float

    # Factor 6 — Reputation Impact
    reputation_gain:    float
    reputation_loss:    float

    # Resource cost (tooling, infra, licenses)
    resource_cost:      float


class TeamState(BaseModel):
    """Snapshot of the company's engineering team."""
    size:         int
    skill:        float   # 0.0–1.0  average skill level
    burnout:      float   # 0.0–1.0  employee load / fatigue
    domain_focus: Domain  # strongest domain


# ── Action Space (what the CEO decides each quarter) ──────────────────────────

class CEOAction(BaseModel):
    """
    ACTION SPACE — OpenEnv spec requires a typed action model.
    All fields have safe defaults so a no-op action is always valid.
    """
    accept_project_id: Optional[str]  = Field(
        default=None,
        description="8-char project ID from available_projects, or null to skip"
    )
    hire_count:        int            = Field(
        default=0, ge=0, le=5,
        description="Number of developers to hire this quarter"
    )
    fire_count:        int            = Field(
        default=0, ge=0, le=3,
        description="Number of developers to let go"
    )
    training_budget:   float          = Field(
        default=0.0, ge=0.0, le=50_000.0,
        description="Budget allocated to team skill training"
    )
    tech_stack:        TechStack      = Field(
        default=TechStack.standard,
        description="Technology stack choice affects quality, speed, cost"
    )
    reduce_workload:   bool           = Field(
        default=False,
        description="Pay $5k to rest team and reduce burnout by 0.25"
    )


# ── Observation Space (what the CEO sees each quarter) ────────────────────────

class CompanyObservation(BaseModel):
    """
    OBSERVATION SPACE — OpenEnv spec requires a typed observation model.
    Exposes all 6 key factors so the agent can make informed decisions.
    """
    # Identity
    session_id:          str
    quarter:             int
    max_quarters:        int
    goal:                str

    # Factor 1 — Finances
    budget:              float
    resource_pool:       float         # man-hours proxy

    # Factor 3 + Resource — Team state
    team:                TeamState

    # Factor 6 — Reputation
    reputation:          float         # 0.0–1.0

    # Factor 5 — Market
    market_phase:        MarketPhase
    domain_demand:       dict          # {"ai": 1.3, "web": 0.9, ...}

    # Factor 2 — Available projects (all 6 factors visible per project)
    available_projects:  list[Project]
    active_risks:        list[str]

    # Step feedback (mirrors BrowserGym pattern)
    last_action_result:  Optional[str] = None
    last_action_error:   Optional[str] = None


# ── Step / State return types ─────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: CompanyObservation
    reward:      float
    done:        bool
    info:        dict


class FullState(BaseModel):
    """Returned by GET /state — includes internal state for counterfactual analysis."""
    observation:          CompanyObservation
    internal:             dict
    episode_history:      list[dict]
    counterfactual_hint:  Optional[str] = None


class TaskInfo(BaseModel):
    id:           str
    difficulty:   str
    max_quarters: int
    description:  str
