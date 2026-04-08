"""
company_env.py — Core business simulation environment.

Integrates all 6 key factors:
  1. Profit Potential      — base_profit × market demand × tech multiplier
  2. Risk Level            — base_risk + skill_gap + burnout + hidden_risk
  3. Team Capability       — skill match determines success probability
  4. Deadline Pressure     — tight deadlines increase burnout on acceptance
  5. Market Demand         — domain_demand dict shifts per MarketPhase
  6. Reputation Impact     — per-project gain/loss, affects future pipeline
  + Resource / Burnout     — overloaded team degrades skill and increases errors
"""
import random
import uuid

from src.models import (
    CEOAction, CompanyObservation, StepResult, FullState,
    Project, TechStack, TeamState, Domain,
)
from src.environment.market_agent import MarketAgent
from src.environment.adversarial  import AdversarialAgent


# ── Synthetic project pool ────────────────────────────────────────────────────
# Numbers calibrated to realistic SMB software project ranges.
# hidden_risk simulates scope creep, unreliable clients, changing requirements.

PROJECTS_POOL: list[dict] = [
    dict(
        name="AI Recommendation Engine", domain=Domain.ai,
        base_profit=90_000, profit_variance=0.40,
        base_risk=0.65,     hidden_risk=0.25,
        skill_required=0.75, duration_quarters=2, deadline_tight=True,
        demand_sensitivity=1.4, reputation_gain=0.10, reputation_loss=0.20,
        resource_cost=8_000,
    ),
    dict(
        name="E-commerce Platform", domain=Domain.web,
        base_profit=35_000, profit_variance=0.15,
        base_risk=0.20,     hidden_risk=0.10,
        skill_required=0.30, duration_quarters=1, deadline_tight=False,
        demand_sensitivity=0.9, reputation_gain=0.05, reputation_loss=0.08,
        resource_cost=2_000,
    ),
    dict(
        name="Mobile Fitness App", domain=Domain.mobile,
        base_profit=55_000, profit_variance=0.30,
        base_risk=0.40,     hidden_risk=0.15,
        skill_required=0.50, duration_quarters=2, deadline_tight=False,
        demand_sensitivity=1.1, reputation_gain=0.07, reputation_loss=0.12,
        resource_cost=3_500,
    ),
    dict(
        name="Real-time Data Pipeline", domain=Domain.data,
        base_profit=65_000, profit_variance=0.25,
        base_risk=0.50,     hidden_risk=0.20,
        skill_required=0.60, duration_quarters=1, deadline_tight=True,
        demand_sensitivity=1.2, reputation_gain=0.08, reputation_loss=0.15,
        resource_cost=5_000,
    ),
    dict(
        name="Legacy System Migration", domain=Domain.web,
        base_profit=28_000, profit_variance=0.20,
        base_risk=0.35,     hidden_risk=0.30,
        skill_required=0.40, duration_quarters=2, deadline_tight=False,
        demand_sensitivity=0.8, reputation_gain=0.04, reputation_loss=0.10,
        resource_cost=1_500,
    ),
    dict(
        name="Cloud DevOps Pipeline", domain=Domain.devops,
        base_profit=42_000, profit_variance=0.15,
        base_risk=0.25,     hidden_risk=0.10,
        skill_required=0.45, duration_quarters=1, deadline_tight=True,
        demand_sensitivity=1.0, reputation_gain=0.06, reputation_loss=0.09,
        resource_cost=4_000,
    ),
    dict(
        name="ML Fraud Detection API", domain=Domain.ai,
        base_profit=78_000, profit_variance=0.35,
        base_risk=0.60,     hidden_risk=0.20,
        skill_required=0.70, duration_quarters=2, deadline_tight=True,
        demand_sensitivity=1.3, reputation_gain=0.09, reputation_loss=0.18,
        resource_cost=7_000,
    ),
    dict(
        name="Inventory SaaS Platform", domain=Domain.web,
        base_profit=32_000, profit_variance=0.10,
        base_risk=0.18,     hidden_risk=0.08,
        skill_required=0.28, duration_quarters=1, deadline_tight=False,
        demand_sensitivity=0.85, reputation_gain=0.04, reputation_loss=0.07,
        resource_cost=1_000,
    ),
    dict(
        name="IoT Sensor Dashboard", domain=Domain.data,
        base_profit=48_000, profit_variance=0.28,
        base_risk=0.45,     hidden_risk=0.18,
        skill_required=0.55, duration_quarters=2, deadline_tight=False,
        demand_sensitivity=1.1, reputation_gain=0.07, reputation_loss=0.13,
        resource_cost=4_500,
    ),
    dict(
        name="AR Product Visualizer", domain=Domain.mobile,
        base_profit=60_000, profit_variance=0.38,
        base_risk=0.55,     hidden_risk=0.22,
        skill_required=0.65, duration_quarters=2, deadline_tight=True,
        demand_sensitivity=1.15, reputation_gain=0.08, reputation_loss=0.16,
        resource_cost=6_000,
    ),
]

TASK_GOALS: dict[str, str] = {
    "single_quarter_survival": (
        "End the quarter with positive cash flow. "
        "Balance risk vs profit — your team skill is 0.4, budget is $100k."
    ),
    "four_quarter_growth": (
        "Grow company revenue by 30% over 4 quarters. "
        "Invest in team capability, adapt to market phase shifts."
    ),
    "adversarial_resilience": (
        "Survive 8 quarters with reputation >= 0.6 despite adversarial shocks. "
        "Manage burnout, hidden risks, and budget simultaneously."
    ),
}

SALARY_PER_DEV    = 8_000
HIRE_COST         = 15_000
BURNOUT_RECOVERY  = 5_000


class CompanyEnv:
    """
    The core OpenEnv environment.
    Exposes reset() / step() / get_full_state() matching the OpenEnv spec.
    """

    def __init__(
        self,
        task_id:      str = "single_quarter_survival",
        max_quarters: int = 1,
        difficulty:   str = "easy",
    ):
        self.task_id      = task_id
        self.max_quarters = max_quarters
        self.difficulty   = difficulty
        self.market       = MarketAgent()
        self.adversarial  = AdversarialAgent(difficulty)
        self._session_id  = str(uuid.uuid4())
        self.history:     list[dict] = []
        self._reset_state()

    # ── OpenEnv API ───────────────────────────────────────────────────────────

    def reset(self) -> "_ResetResult":
        self._session_id = str(uuid.uuid4())
        self._reset_state()
        return _ResetResult(self._build_obs())

    def step(self, action: CEOAction) -> StepResult:
        if self.done:
            return StepResult(
                observation=self._build_obs(error="Episode already done. Call reset()."),
                reward=0.0, done=True, info={},
            )

        messages:  list[str] = []
        reward                = 0.0
        error_msg: str | None = None

        try:
            # ── 1. Fixed quarterly salary cost ───────────────
            salary       = self.team.size * SALARY_PER_DEV
            self.budget -= salary
            messages.append(f"Salaries paid: −${salary:,.0f}")

            # ── 2. Resource pool refresh ─────────────────────
            self.resource_pool = float(self.team.size * 100)

            # ── 3. Hiring ────────────────────────────────────
            if action.hire_count > 0:
                cost = action.hire_count * HIRE_COST
                if self.budget < cost:
                    error_msg = (
                        f"Insufficient budget to hire {action.hire_count} devs "
                        f"(need ${cost:,.0f}, have ${self.budget:,.0f})."
                    )
                    action = action.model_copy(update={"hire_count": 0})
                else:
                    self.budget           -= cost
                    self.team.size        += action.hire_count
                    self.resource_pool    += action.hire_count * 100
                    messages.append(
                        f"Hired {action.hire_count} devs (−${cost:,.0f})"
                    )

            # ── 4. Firing ─────────────────────────────────────
            if action.fire_count > 0:
                fired             = min(action.fire_count, self.team.size - 1)
                self.team.size    = max(1, self.team.size - fired)
                messages.append(f"Released {fired} devs")

            # ── 5. Training (Factor 3 improvement) ───────────
            if action.training_budget > 0:
                self.budget      -= action.training_budget
                gain              = min(0.15, action.training_budget / 50_000)
                self.team.skill   = min(1.0, self.team.skill + gain)
                self.team.burnout = max(0.0, self.team.burnout - 0.05)
                messages.append(
                    f"Training: skill → {self.team.skill:.2f} "
                    f"(−${action.training_budget:,.0f})"
                )

            # ── 6. Burnout recovery ───────────────────────────
            if action.reduce_workload:
                self.budget       -= BURNOUT_RECOVERY
                self.team.burnout  = max(0.0, self.team.burnout - 0.25)
                messages.append(
                    f"Team rested — burnout → {self.team.burnout:.2f} "
                    f"(−${BURNOUT_RECOVERY:,.0f})"
                )

            # ── 7. Project execution (all 6 factors) ─────────
            if action.accept_project_id:
                project = self._find_project(action.accept_project_id)
                if project is None:
                    error_msg = (
                        f"Project ID '{action.accept_project_id}' not found "
                        f"in available_projects."
                    )
                else:
                    ok, net, msg, rep_delta = self._execute_project(
                        project, action.tech_stack
                    )
                    self.budget     += net
                    self.reputation  = max(0.0, min(1.0,
                                         self.reputation + rep_delta))
                    reward          += net / 100_000

                    # Factor 4: deadline pressure raises burnout
                    if project.deadline_tight:
                        burn_gain         = 0.10 + self.team.burnout * 0.05
                        self.team.burnout = min(1.0,
                                           self.team.burnout + burn_gain)
                        messages.append(
                            f"Tight deadline: burnout → {self.team.burnout:.2f}"
                        )
                    messages.append(msg)

            # ── 8. Burnout penalty (Resource factor) ─────────
            if self.team.burnout > 0.6:
                skill_drain       = self.team.burnout * 0.03
                self.team.skill   = max(0.05, self.team.skill - skill_drain)
                reward           -= 0.15
                messages.append(
                    f"⚠ High burnout ({self.team.burnout:.2f}) "
                    f"draining team skill → {self.team.skill:.2f}"
                )

            # ── 9. Tech-debt bleed ────────────────────────────
            if "tech_debt" in self.active_risks and random.random() < 0.40:
                self.reputation  = max(0.0, self.reputation - 0.04)
                reward          -= 0.05
                messages.append("Tech debt causing client complaints (−rep)")

            # ── 10. Adversarial shocks (Factor 10) ───────────
            self.reputation, self.budget, shocks = self.adversarial.apply(
                self.reputation, self.budget
            )
            if shocks:
                messages.append(f"External shocks: {', '.join(shocks)}")

            # ── 11. Market advances ───────────────────────────
            self.market.step()

            # ── 12. Shaped reward across all factors ─────────
            reward += self.reputation * 0.20          # Factor 6
            reward += min(0.30, self.budget / 500_000) # Factor 1
            reward -= self.team.burnout * 0.10         # Resource
            if self.budget < 0:
                reward -= 1.0

        except Exception as exc:
            error_msg = str(exc)

        # ── Advance quarter ───────────────────────────────────────────────────
        self.quarter += 1
        self.done     = (
            self.quarter > self.max_quarters
            or self.budget    < -50_000
            or self.reputation <= 0.0
        )

        self.history.append({
            "quarter":    self.quarter - 1,
            "action":     action.model_dump(),
            "reward":     round(reward, 4),
            "budget":     round(self.budget, 2),
            "reputation": round(self.reputation, 3),
            "burnout":    round(self.team.burnout, 3),
            "team_skill": round(self.team.skill, 3),
            "messages":   messages,
        })

        return StepResult(
            observation=self._build_obs(
                last_result="; ".join(messages) if messages else None,
                error=error_msg,
            ),
            reward=round(reward, 4),
            done=self.done,
            info={
                "quarter":    self.quarter,
                "budget":     round(self.budget, 2),
                "reputation": round(self.reputation, 3),
            },
        )

    def get_full_state(self) -> FullState:
        """GET /state — full internal state for counterfactual analysis."""
        hint: str | None = None
        if self.history:
            last = self.history[-1]
            if last["reward"] < 0:
                if last["burnout"] > 0.6:
                    hint = (
                        "Counterfactual: Using reduce_workload before that project "
                        "would have prevented burnout-driven skill loss and the reward penalty."
                    )
                elif last["action"].get("accept_project_id"):
                    hint = (
                        "Counterfactual: Allocating $20k to training first would have "
                        "raised team skill and improved project success probability by ~30%."
                    )
                elif last["budget"] < 20_000:
                    hint = (
                        "Counterfactual: Accepting a smaller safe project to rebuild budget "
                        "would have been more sustainable this quarter."
                    )

        return FullState(
            observation=self._build_obs(),
            internal={
                "market_phase":    self.market.phase,
                "domain_demand":   self.market.domain_demand(),
                "active_risks":    self.active_risks,
                "difficulty":      self.difficulty,
                "resource_pool":   self.resource_pool,
            },
            episode_history=self.history,
            counterfactual_hint=hint,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    # ── Private helpers ───────────────────────────────────────────────────────

    def _reset_state(self):
        self.quarter       = 1
        self.budget        = 100_000.0
        self.resource_pool = 500.0
        self.team          = TeamState(
            size=5, skill=0.40, burnout=0.0, domain_focus=Domain.web
        )
        self.reputation    = 0.70
        self.active_risks: list[str] = []
        self.done          = False
        self.market.reset()
        self.history       = []
        self._cached_projects: list[Project] = self._sample_projects()

    def _execute_project(
        self,
        project: Project,
        tech:    TechStack,
    ) -> tuple[bool, float, str, float]:
        """
        Simulate project outcome incorporating all risk factors.
        Returns (success, net_money_change, message, reputation_delta).
        """
        # Factor 3: skill gap drives failure probability up
        skill_gap       = max(0.0, project.skill_required - self.team.skill)

        # Resource / burnout penalty
        burnout_penalty = self.team.burnout * 0.15

        # Factor 5: domain demand from market
        demand          = self.market.domain_demand().get(project.domain.value, 1.0)

        # Factor 10: hidden risk is revealed partially at random
        revealed_hidden = project.hidden_risk * random.random()

        # Factor 2: total failure probability
        total_risk = min(
            0.95,
            project.base_risk
            + skill_gap        * 0.50
            + burnout_penalty
            + revealed_hidden,
        )

        # Tech stack modifies risk and profit (Factor 3 / Factor 1)
        profit_mult = demand * project.demand_sensitivity
        if tech == TechStack.cheap:
            total_risk  = min(0.95, total_risk + 0.15)
            profit_mult *= 0.75
            if "tech_debt" not in self.active_risks:
                self.active_risks.append("tech_debt")
        elif tech == TechStack.premium:
            total_risk  = max(0.0, total_risk - 0.10)
            profit_mult *= 1.20

        success       = random.random() > total_risk
        variance_roll = 1.0 + random.uniform(
            -project.profit_variance, project.profit_variance
        )

        if success:
            earned    = (
                project.base_profit * profit_mult * variance_roll
                - project.resource_cost
            )
            rep_delta = project.reputation_gain
            msg       = (
                f"✓ '{project.name}' succeeded "
                f"+${earned:,.0f} (market×{demand:.1f}, risk={total_risk:.2f})"
            )
            return True, earned, msg, rep_delta
        else:
            penalty   = project.base_profit * 0.30 + project.resource_cost
            rep_delta = -project.reputation_loss
            msg       = (
                f"✗ '{project.name}' FAILED "
                f"−${penalty:,.0f} (risk={total_risk:.2f}, "
                f"skill_gap={skill_gap:.2f})"
            )
            return False, -penalty, msg, rep_delta

    def _find_project(self, pid: str) -> Project | None:
        for p in self._cached_projects:
            if p.id == pid:
                return p
        return None

    def _sample_projects(self) -> list[Project]:
        n    = self.market.project_count()
        pool = random.sample(PROJECTS_POOL, min(n, len(PROJECTS_POOL)))
        return [Project(id=str(uuid.uuid4())[:8], **p) for p in pool]

    def _build_obs(
        self,
        last_result: str | None = None,
        error:       str | None = None,
    ) -> CompanyObservation:
        self._cached_projects = self._sample_projects()
        return CompanyObservation(
            session_id         = self._session_id,
            quarter            = self.quarter,
            max_quarters       = self.max_quarters,
            goal               = TASK_GOALS.get(self.task_id, ""),
            budget             = round(self.budget, 2),
            resource_pool      = round(self.resource_pool, 1),
            team               = self.team.model_copy(),
            reputation         = round(self.reputation, 3),
            market_phase       = self.market.phase,
            domain_demand      = self.market.domain_demand(),
            available_projects = self._cached_projects,
            active_risks       = self.active_risks.copy(),
            last_action_result = last_result,
            last_action_error  = error,
        )


class _ResetResult:
    """Mirrors the BrowserGymEnv reset() return pattern."""
    def __init__(self, observation: CompanyObservation):
        self.observation = observation
        self.done        = False
        self.reward      = 0.0
