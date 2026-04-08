"""
adversarial.py — Adversarial Critic Agent (Factor 10: Uncertainty / Hidden Risk).

Injects realistic business shocks to prevent the Decision Agent from learning
brittle strategies. Shock intensity scales with task difficulty.
"""
import random


# (shock_name, base_probability, reputation_delta, budget_delta)
SHOCK_CATALOG: list[tuple[str, float, float, float]] = [
    ("client_dispute",      0.12, -0.08,  -5_000),
    ("key_developer_quit",  0.10, -0.05, -10_000),
    ("budget_audit",        0.08,  0.00, -15_000),
    ("market_rumour",       0.10, -0.10,      0),
    ("competitor_launch",   0.10, -0.05,      0),
    ("surprise_tax_bill",   0.06,  0.00,  -8_000),
    ("data_breach_scare",   0.05, -0.12,  -6_000),
]

DIFFICULTY_MULTIPLIER: dict[str, float] = {
    "easy":   0.40,
    "medium": 1.00,
    "hard":   1.80,
}


class AdversarialAgent:
    """
    Introduces hidden complications to expose weaknesses in the CEO agent's
    strategy and push it toward resilience.
    """

    def __init__(self, difficulty: str = "easy"):
        self.multiplier = DIFFICULTY_MULTIPLIER.get(difficulty, 1.0)

    def apply(
        self, reputation: float, budget: float
    ) -> tuple[float, float, list[str]]:
        """
        Apply random shocks for this quarter.
        Returns updated (reputation, budget, list_of_triggered_shock_names).
        """
        triggered: list[str] = []
        for name, prob, rep_delta, budget_delta in SHOCK_CATALOG:
            if random.random() < prob * self.multiplier:
                reputation = max(0.0, min(1.0, reputation + rep_delta))
                budget    += budget_delta
                triggered.append(name)
        return reputation, budget, triggered
