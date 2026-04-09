"""task_easy.py - Single Quarter Survival grader."""

def grade(env) -> float:
    """
    Score based on ending budget after 1 quarter.
    Guaranteed strictly in (0.01, 0.99).
    Very granular to show partial progress.
    """
    # Start with a base score
    if env.budget <= -50000:
        raw_score = 0.01
    elif env.budget <= 0:
        # Bankrupt but survived part of the quarter / didn't lose everything
        # Map (-50k, 0) to (0.01, 0.2)
        raw_score = 0.01 + ((env.budget + 50000) / 50000) * 0.19
    elif env.budget < 100000:
        # Survived but lost money
        # Map (0, 100k) to (0.2, 0.5)
        raw_score = 0.2 + (env.budget / 100000) * 0.3
    else:
        # Profit!
        # Map (100k, 200k) to (0.5, 0.99)
        profit = env.budget - 100000
        raw_score = 0.5 + min(0.49, (profit / 100000) * 0.49)

    return round(float(raw_score), 4)
