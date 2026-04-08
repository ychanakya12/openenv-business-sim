"""task_easy.py — Single Quarter Survival grader."""


def grade(env) -> float:
    """
    Score strictly in (0.0, 1.0) based on ending budget after 1 quarter.
    OpenEnv requirement: score must never equal exactly 0.0 or 1.0.
    """
    if env.budget <= 0:
        val = 1.0 + env.budget / 50_000.0
    else:
        profit = env.budget - 100_000.0
        val = profit / 50_000.0
    # Clamp first, then round — guarantees strictly in (0.01, 0.99)
    clamped = min(max(val, 0.01), 0.99)
    score = round(clamped, 3)
    # Final safety net: rounding could still produce exactly 0.0 or 1.0
    return min(max(score, 0.01), 0.99)
