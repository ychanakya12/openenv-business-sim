"""
validate.py — Pre-submission validation script.

Checks all 5 automated judge gates before you submit.
Run this with the server already running:

    # Terminal 1:
    uvicorn src.server:app --host 0.0.0.0 --port 7860

    # Terminal 2:
    python validate.py
"""
import sys
import os
import httpx
import yaml

BASE = os.getenv("ENV_URL", "http://localhost:7860")
PASS = "✅"
FAIL = "❌"
results: list[bool] = []


def check(name: str, fn) -> bool:
    try:
        fn()
        print(f"{PASS}  {name}")
        results.append(True)
        return True
    except Exception as e:
        print(f"{FAIL}  {name}")
        print(f"    → {e}")
        results.append(False)
        return False


# ── Gate 1: HF Space responds ─────────────────────────────────────────────────
def gate_health():
    r = httpx.get(f"{BASE}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data.get("status") == "ok", f"Expected status=ok, got {data}"


# ── Gate 2: POST /reset returns valid CompanyObservation ─────────────────────
def gate_reset():
    r = httpx.post(
        f"{BASE}/reset",
        params={"task_id": "single_quarter_survival"},
        timeout=10,
    )
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
    data = r.json()
    required = ["session_id", "quarter", "budget", "team", "reputation",
                "available_projects", "market_phase"]
    for field in required:
        assert field in data, f"Missing field '{field}' in observation"


# ── Gate 3: POST /step returns reward + done ──────────────────────────────────
def gate_step():
    # Reset first
    r = httpx.post(
        f"{BASE}/reset",
        params={"task_id": "single_quarter_survival"},
        timeout=10,
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    # Step with no-op action
    r2 = httpx.post(
        f"{BASE}/step",
        json={
            "accept_project_id": None,
            "hire_count":        0,
            "fire_count":        0,
            "training_budget":   0.0,
            "tech_stack":        "standard",
            "reduce_workload":   False,
        },
        params={"session_id": sid},
        timeout=10,
    )
    assert r2.status_code == 200, f"Expected 200, got {r2.status_code}: {r2.text}"
    data = r2.json()
    assert "reward"      in data, "Missing 'reward' in StepResult"
    assert "done"        in data, "Missing 'done' in StepResult"
    assert "observation" in data, "Missing 'observation' in StepResult"
    assert isinstance(data["reward"], (int, float)), "reward must be numeric"
    assert isinstance(data["done"],   bool),         "done must be boolean"


# ── Gate 4: All 3 graders return score in [0.0, 1.0] ─────────────────────────
def gate_graders():
    for task_id in [
        "single_quarter_survival",
        "four_quarter_growth",
        "adversarial_resilience",
    ]:
        r = httpx.post(
            f"{BASE}/reset",
            params={"task_id": task_id},
            timeout=10,
        )
        assert r.status_code == 200, f"reset failed for {task_id}"
        sid = r.json()["session_id"]

        rg = httpx.get(
            f"{BASE}/grade",
            params={"session_id": sid},
            timeout=10,
        )
        assert rg.status_code == 200, f"grade failed for {task_id}"
        score = rg.json()["score"]
        assert 0.0 <= score <= 1.0, (
            f"Score {score} out of [0.0, 1.0] range for task '{task_id}'"
        )


# ── Gate 5: openenv.yaml is valid and has 3+ tasks ───────────────────────────
def gate_yaml():
    assert os.path.exists("openenv.yaml"), "openenv.yaml not found in root"
    with open("openenv.yaml") as f:
        cfg = yaml.safe_load(f)

    assert "name"              in cfg, "Missing 'name' in openenv.yaml"
    assert "tasks"             in cfg, "Missing 'tasks' in openenv.yaml"
    assert "action_space"      in cfg, "Missing 'action_space' in openenv.yaml"
    assert "observation_space" in cfg, "Missing 'observation_space' in openenv.yaml"
    assert len(cfg["tasks"])   >= 3,   f"Need 3+ tasks, found {len(cfg['tasks'])}"

    difficulties = {t["difficulty"] for t in cfg["tasks"]}
    assert "easy"   in difficulties, "Missing easy task"
    assert "medium" in difficulties, "Missing medium task"
    assert "hard"   in difficulties, "Missing hard task"


# ── Gate 6: GET /state returns FullState with history ────────────────────────
def gate_state():
    r = httpx.post(
        f"{BASE}/reset",
        params={"task_id": "four_quarter_growth"},
        timeout=10,
    )
    assert r.status_code == 200
    sid = r.json()["session_id"]

    rs = httpx.get(
        f"{BASE}/state",
        params={"session_id": sid},
        timeout=10,
    )
    assert rs.status_code == 200, f"Expected 200, got {rs.status_code}"
    data = rs.json()
    assert "observation"     in data
    assert "internal"        in data
    assert "episode_history" in data


# ── Gate 7: inference.py exists in root ──────────────────────────────────────
def gate_inference_exists():
    assert os.path.exists("inference.py"), "inference.py not found in root directory"
    with open("inference.py") as f:
        content = f.read()
    assert "API_BASE_URL" in content, "inference.py missing API_BASE_URL"
    assert "MODEL_NAME"   in content, "inference.py missing MODEL_NAME"
    assert "HF_TOKEN"     in content, "inference.py missing HF_TOKEN"
    assert "OpenAI"       in content, "inference.py must use OpenAI client"


# ── Run all gates ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  Pre-Submission Validator — Business Sim Env")
print(f"  Server: {BASE}")
print(f"{'='*55}\n")

check("Gate 1 — Server health check (GET /health)",         gate_health)
check("Gate 2 — POST /reset returns CompanyObservation",    gate_reset)
check("Gate 3 — POST /step returns reward + done",          gate_step)
check("Gate 4 — All 3 task graders return score in [0,1]",  gate_graders)
check("Gate 5 — openenv.yaml valid with easy/medium/hard",  gate_yaml)
check("Gate 6 — GET /state returns FullState",              gate_state)
check("Gate 7 — inference.py exists in root with env vars", gate_inference_exists)

passed = sum(results)
total  = len(results)

print(f"\n{'='*55}")
print(f"  {passed}/{total} gates passed")

if passed == total:
    print("  🚀 ALL GATES PASSED — ready to submit!")
else:
    print("  ⚠️  Fix the failing gates before submitting.")
    print("  Failing gates = automatic disqualification.")
print(f"{'='*55}\n")

sys.exit(0 if passed == total else 1)
