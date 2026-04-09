"""
Exhaustive edge case test for all 3 graders.
Simulates every possible boundary condition.
"""
import sys
sys.path.insert(0, '.')


# ── Mock objects ──────────────────────────────────────────────────────────────

class Team:
    def __init__(self, burnout=0.0, skill=0.5, size=3):
        self.burnout = burnout
        self.skill = skill
        self.size = size


class MockEnv:
    def __init__(self, budget, reputation, burnout=0.0, done=False, history=None, quarter=1):
        self.budget = budget
        self.reputation = reputation
        self.team = Team(burnout=burnout)
        self.done = done
        self.history = history or []
        self.quarter = quarter
        self.task_id = "test"


# ── Import graders ────────────────────────────────────────────────────────────

from src.tasks import task_easy, task_medium, task_hard


# ── Test cases ────────────────────────────────────────────────────────────────

def clamp_check(name, score):
    strict_ok = 0.0 < score < 1.0
    status = "OK   " if strict_ok else "FAIL *"
    print(f"  [{status}] {name}: {score}")
    return strict_ok


print("\n=== TASK EASY (single_quarter_survival) ===")
tests_easy = [
    ("budget=0 (bankrupt exact)",     MockEnv(0,       0.7)),
    ("budget=-1 (barely bankrupt)",   MockEnv(-1,      0.7)),
    ("budget=-49999",                 MockEnv(-49999,  0.7)),
    ("budget=-50000",                 MockEnv(-50000,  0.7)),
    ("budget=-100000 (total loss)",   MockEnv(-100000, 0.7)),
    ("budget=100000 (break-even)",    MockEnv(100000,  0.7)),
    ("budget=100001 (1 dollar up)",   MockEnv(100001,  0.7)),
    ("budget=150000 (mid)",           MockEnv(150000,  0.7)),
    ("budget=150000.0 (exact float)", MockEnv(150000.0,0.7)),
    ("budget=200000 (target)",        MockEnv(200000,  0.7)),
    ("budget=999999 (massive win)",   MockEnv(999999,  0.7)),
]
results_easy = [clamp_check(n, task_easy.grade(e)) for n, e in tests_easy]

print("\n=== TASK MEDIUM (four_quarter_growth) ===")
tests_medium = [
    ("no history, budget=100000",     MockEnv(100000, 0.7, history=[])),
    ("no history, budget=0",          MockEnv(0,      0.0, history=[])),
    ("no history, budget=200000",     MockEnv(200000, 1.0, history=[])),
    ("history, budget=100000",        MockEnv(100000, 0.7, history=["q1"])),
    ("history, budget=130000",        MockEnv(130000, 0.7, history=["q1"])),
    ("history, rep=0.0",              MockEnv(130000, 0.0, history=["q1"])),
    ("history, rep=1.0, budget=200k", MockEnv(200000, 1.0, history=["q1"])),
    ("history, budget=0, rep=0.0",    MockEnv(0,      0.0, history=["q1"])),
    ("history, budget=-50000",        MockEnv(-50000, 0.0, history=["q1"])),
]
results_medium = [clamp_check(n, task_medium.grade(e)) for n, e in tests_medium]

print("\n=== TASK HARD (adversarial_resilience) ===")
tests_hard = [
    ("budget=0, rep=0.0, burnout=1.0",   MockEnv(0,       0.0, burnout=1.0)),
    ("budget=0, rep=0.0, burnout=0.0",   MockEnv(0,       0.0, burnout=0.0)),
    ("budget=200000, rep=1.0, burnout=0",MockEnv(200000,  1.0, burnout=0.0)),
    ("budget=200000, rep=0.5, burnout=0",MockEnv(200000,  0.5, burnout=0.0)),
    ("budget=-1, rep=0.6, burnout=0.5",  MockEnv(-1,      0.6, burnout=0.5)),
    ("budget=100000, rep=0.7, burnout=0",MockEnv(100000,  0.7, burnout=0.0)),
    ("budget=100000, rep=0.0, burnout=0",MockEnv(100000,  0.0, burnout=0.0)),
    ("budget=100000, rep=1.0, burnout=1",MockEnv(100000,  1.0, burnout=1.0)),
    ("budget=999999, rep=1.0, burnout=0",MockEnv(999999,  1.0, burnout=0.0)),
    ("budget=-99999, rep=0.0, burnout=1",MockEnv(-99999,  0.0, burnout=1.0)),
]
results_hard = [clamp_check(n, task_hard.grade(e)) for n, e in tests_hard]

# ── Also test the server-level clamp via HTTP ─────────────────────────────────
print("\n=== LIVE /grade ENDPOINT (via HTTP) ===")
import httpx
BASE = "http://localhost:7860"
tasks = ["single_quarter_survival", "four_quarter_growth", "adversarial_resilience"]
results_live = []
for task_id in tasks:
    r = httpx.post(f"{BASE}/reset", params={"task_id": task_id})
    sid = r.json()["session_id"]
    rg = httpx.get(f"{BASE}/grade", params={"session_id": sid})
    score = rg.json()["score"]
    ok = clamp_check(f"{task_id} (after reset)", score)
    results_live.append(ok)

# Summary
all_results = results_easy + results_medium + results_hard + results_live
all_ok = all(all_results)
total = len(all_results)
passed = sum(all_results)

print(f"\n{'='*50}")
print(f"  {passed}/{total} edge cases passed")
if all_ok:
    print("  ALL EDGE CASES PASS - scores are strictly in (0.0, 1.0)")
else:
    print("  *** SOME EDGE CASES FAILED - scores are NOT strictly in (0.0, 1.0) ***")
print(f"{'='*50}")
