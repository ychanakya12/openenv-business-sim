"""task_hard.py — Adversarial Resilience grader."""


def grade(env) -> float:
    """
    Score 0.0–1.0 weighted across all 6 factors:
      - Survival (budget > 0)
      - Profit (budget vs $200k benchmark)
      - Reputation >= 0.6 (Factor 6)
      - Low burnout (Resource factor)
      - Reputation bonus for hitting 0.6 target
    """
    survived          = 1.0 if env.budget > 0              else 0.0
    profit_score      = min(1.0, max(0.0, env.budget / 200_000.0))
    reputation_score  = env.reputation
    low_burnout_score = 1.0 - env.team.burnout
    rep_target_bonus  = 0.15 if env.reputation >= 0.6       else 0.0

    raw = (
        survived          * 0.25
        + profit_score    * 0.25
        + reputation_score* 0.20
        + low_burnout_score* 0.15
        + rep_target_bonus
    )
    return round(min(1.0, max(0.0, raw)), 4)
