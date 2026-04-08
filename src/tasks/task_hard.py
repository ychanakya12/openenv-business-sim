"""task_hard.py — Adversarial Resilience grader."""


def grade(env) -> float:
    """
    Score strictly in (0.0, 1.0) weighted across all 6 factors.
    OpenEnv requirement: score must never equal exactly 0.0 or 1.0.
    """
    survived          = 1.0 if env.budget > 0              else 0.0
    profit_score      = min(0.99, max(0.01, env.budget / 200_000.0))
    reputation_score  = min(0.99, max(0.01, env.reputation))
    low_burnout_score = min(0.99, max(0.01, 1.0 - env.team.burnout))
    rep_target_bonus  = 0.15 if env.reputation >= 0.6       else 0.0

    raw = (
        survived          * 0.25
        + profit_score    * 0.25
        + reputation_score* 0.20
        + low_burnout_score* 0.15
        + rep_target_bonus
    )
    # Clamp first, then round, then clamp again as safety net
    score = round(min(max(raw, 0.01), 0.99), 3)
    return min(max(score, 0.01), 0.99)
