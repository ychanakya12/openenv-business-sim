"""task_medium.py — Four Quarter Growth grader."""


def grade(env) -> float:
    """
    Score strictly in (0.0, 1.0) based on 30% revenue growth over 4 quarters.
    Also rewards reputation maintenance (Factor 6).
    OpenEnv requirement: score must never equal exactly 0.0 or 1.0.
    """
    if not env.history:
        return 0.01

    # Factor 1: profit growth
    growth        = (env.budget - 100_000.0) / 100_000.0
    growth_score  = min(0.99, max(0.01, growth / 0.30))

    # Factor 6: reputation maintained (clamp to avoid 0.0 or 1.0)
    rep_score     = min(0.99, max(0.01, env.reputation))

    # Weighted combination
    raw = growth_score * 0.70 + rep_score * 0.30
    # Clamp first, then round, then clamp again as safety net
    score = round(min(max(raw, 0.01), 0.99), 3)
    return min(max(score, 0.01), 0.99)
