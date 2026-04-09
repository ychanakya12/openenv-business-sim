"""task_medium.py - Four Quarter Growth grader."""

def grade(env) -> float:
    """
    Score based on 30% revenue growth over 4 quarters.
    Guaranteed strictly in (0.01, 0.99).
    """
    # 1. Survival & Budget Factor (0.01 to 0.6)
    if env.budget <= -50000:
        budget_score = 0.0
    elif env.budget <= 100000:
        # Bankrupt or lost money: Map (-50k, 100k) to (0.0, 0.3)
        budget_score = ((env.budget + 50000) / 150000) * 0.3
    else:
        # Map (100k, 130k+) to (0.3, 0.6)
        growth = (env.budget - 100000) / 30000 # 1.0 at 30% growth
        budget_score = 0.3 + min(0.3, growth * 0.3)

    # 2. Reputation Factor (0.0 to 0.3)
    rep_score = env.reputation * 0.3

    # 3. Persistence Factor (0.0 to 0.09)
    # Reward sticking it out for more quarters
    persistence = (len(env.history) / 4.0) * 0.09

    raw_score = 0.01 + budget_score + rep_score + persistence
    
    return round(float(min(0.99, max(0.01, raw_score))), 4)
