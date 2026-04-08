"""task_easy.py — Single Quarter Survival grader."""


def grade(env) -> float:
    """
    Score 0.0–1.0 based on ending budget after 1 quarter.
    Partial credit: even breaking even is better than loss.
    """
    if env.budget <= 0:
        # Partial credit for how close they came
        val = 1.0 + env.budget / 50_000
        return min(0.999, max(0.001, round(val, 4)))
    profit = env.budget - 100_000.0
    val = profit / 50_000.0
    return min(0.999, max(0.001, round(val, 4)))
