"""
market_agent.py — Dynamic market environment (Factor 5: Market Demand).

Simulates real-world business cycles: boom → stable → recession.
Each phase affects domain demand, project availability, and profit multipliers.
"""
import random
from src.models import MarketPhase

# Per-phase domain demand multipliers (calibrated to real IT market cycles)
DOMAIN_DEMAND_BY_PHASE: dict[MarketPhase, dict[str, float]] = {
    MarketPhase.boom: {
        "ai":     1.4,
        "web":    1.1,
        "mobile": 1.2,
        "data":   1.3,
        "devops": 1.0,
    },
    MarketPhase.stable: {
        "ai":     1.1,
        "web":    1.0,
        "mobile": 1.0,
        "data":   1.0,
        "devops": 1.0,
    },
    MarketPhase.recession: {
        "ai":     0.8,
        "web":    0.7,
        "mobile": 0.6,
        "data":   0.9,
        "devops": 1.1,   # infrastructure stays essential in downturns
    },
}


class MarketAgent:
    """
    Markov chain market simulator.
    Transition probabilities modelled on typical 6-month business cycles.
    """

    PHASES = [MarketPhase.boom, MarketPhase.stable, MarketPhase.recession]

    # Transition matrix: from_phase → [p(boom), p(stable), p(recession)]
    TRANSITION: dict[MarketPhase, list[float]] = {
        MarketPhase.boom:      [0.30, 0.50, 0.20],
        MarketPhase.stable:    [0.20, 0.50, 0.30],
        MarketPhase.recession: [0.10, 0.40, 0.50],
    }

    def __init__(self):
        self.phase = MarketPhase.stable

    def reset(self):
        self.phase = MarketPhase.stable

    def step(self) -> MarketPhase:
        """Advance market one quarter using Markov transition."""
        weights    = self.TRANSITION[self.phase]
        self.phase = random.choices(self.PHASES, weights=weights, k=1)[0]
        return self.phase

    def domain_demand(self) -> dict[str, float]:
        return DOMAIN_DEMAND_BY_PHASE[self.phase].copy()

    def project_count(self) -> int:
        """Fewer projects available in recession, more in boom."""
        return {
            MarketPhase.boom:      4,
            MarketPhase.stable:    3,
            MarketPhase.recession: 2,
        }[self.phase]

    def profit_multiplier(self) -> float:
        return {
            MarketPhase.boom:      1.25,
            MarketPhase.stable:    1.00,
            MarketPhase.recession: 0.75,
        }[self.phase]
