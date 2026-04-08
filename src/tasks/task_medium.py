"""task_medium.py — Four Quarter Growth grader."""


def grade(env) -> float:
    """
    Score 0.0–1.0 based on 30% revenue growth over 4 quarters.
    Also rewards reputation maintenance (Factor 6).
    """
    if not env.history:
        return 0.0

    # Factor 1: profit growth
    growth        = (env.budget - 100_000.0) / 100_000.0
    growth_score  = min(1.0, max(0.0, growth / 0.30))

    # Factor 6: reputation maintained
    rep_score     = env.reputation

    # Weighted combination
    raw = growth_score * 0.70 + rep_score * 0.30
    return round(min(1.0, max(0.0, raw)), 4)
