"""task_hard.py - Adversarial Resilience grader."""

def grade(env) -> float:
    """
    Score weighted across survival, profit, reputation, and burnout.
    Guaranteed strictly in (0.01, 0.99).
    """
    # Survival Factor (0.0 to 0.3)
    # Map (0, 8) quarters to (0.0, 0.3)
    survival_score = (len(env.history) / 8.0) * 0.3

    # Budget Health (0.0 to 0.25)
    if env.budget <= 0:
        budget_score = 0.0
    else:
        budget_score = min(0.25, (env.budget / 200000.0) * 0.25)

    # Reputation Factor (0.0 to 0.25)
    rep_score = env.reputation * 0.25

    # Team Health (0.0 to 0.19)
    burnout_factor = max(0.0, 1.0 - env.team.burnout)
    team_score = burnout_factor * 0.19

    raw_score = 0.01 + survival_score + budget_score + rep_score + team_score
    
    return round(float(min(0.99, max(0.01, raw_score))), 4)
