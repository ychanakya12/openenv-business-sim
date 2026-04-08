"""
server.py — FastAPI server exposing the OpenEnv standard API.

Endpoints (all required by OpenEnv spec):
  GET  /           — health check (judges ping this)
  GET  /health     — health check
  GET  /tasks      — list all tasks
  POST /reset      — start new episode, returns CompanyObservation
  POST /step       — take action, returns StepResult
  GET  /state      — full internal state + history
  GET  /grade      — current task score (0.0–1.0)
"""
from fastapi           import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.models                   import CEOAction, StepResult, FullState
from src.environment.company_env  import CompanyEnv
from src.tasks                    import task_easy, task_medium, task_hard

app = FastAPI(
    title       = "Business Simulation OpenEnv",
    description = (
        "Multi-agent business simulation where an AI CEO manages a software "
        "company across quarters under adversarial conditions."
    ),
    version     = "0.1.0",
)

# ── In-memory session store ───────────────────────────────────────────────────
_sessions: dict[str, CompanyEnv] = {}

TASK_CONFIG: dict[str, dict] = {
    "single_quarter_survival": dict(
        max_quarters = 1,
        difficulty   = "easy",
        grader       = task_easy,
        description  = "End the quarter with positive cash flow.",
    ),
    "four_quarter_growth": dict(
        max_quarters = 4,
        difficulty   = "medium",
        grader       = task_medium,
        description  = "Grow revenue 30% over 4 quarters amid market shifts.",
    ),
    "adversarial_resilience": dict(
        max_quarters = 8,
        difficulty   = "hard",
        grader       = task_hard,
        description  = "Survive 8 quarters with reputation >= 0.6 under shocks.",
    ),
}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Root health check — judges automated ping."""
    return {"status": "ok", "env": "business-sim-env", "version": "0.1.0"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Task listing ──────────────────────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """OpenEnv spec: enumerate available tasks."""
    return [
        {
            "id":           tid,
            "difficulty":   cfg["difficulty"],
            "max_quarters": cfg["max_quarters"],
            "description":  cfg["description"],
        }
        for tid, cfg in TASK_CONFIG.items()
    ]


# ── Core OpenEnv endpoints ────────────────────────────────────────────────────

@app.post("/reset")
def reset(task_id: str = "single_quarter_survival"):
    """
    POST /reset — start a new episode.
    Returns CompanyObservation (includes session_id for subsequent calls).
    """
    if task_id not in TASK_CONFIG:
        raise HTTPException(
            400,
            f"Unknown task_id '{task_id}'. "
            f"Valid options: {list(TASK_CONFIG.keys())}"
        )
    cfg = TASK_CONFIG[task_id]
    env = CompanyEnv(
        task_id      = task_id,
        max_quarters = cfg["max_quarters"],
        difficulty   = cfg["difficulty"],
    )
    result = env.reset()
    _sessions[env.session_id] = env
    return JSONResponse(result.observation.model_dump())


@app.post("/step", response_model=StepResult)
def step(action: CEOAction, session_id: str):
    """POST /step — submit CEO action, receive observation + reward."""
    env = _require_session(session_id)
    if env.done:
        raise HTTPException(400, "Episode is done. Call /reset to start a new one.")
    return env.step(action)


@app.get("/state", response_model=FullState)
def state(session_id: str):
    """GET /state — full internal state including history and counterfactuals."""
    return _require_session(session_id).get_full_state()


@app.get("/grade")
def grade(session_id: str):
    """GET /grade — current episode score in [0.0, 1.0]."""
    env   = _require_session(session_id)
    cfg   = TASK_CONFIG[env.task_id]
    score = cfg["grader"].grade(env)
    return {
        "task_id":    env.task_id,
        "score":      round(float(score), 4),
        "quarter":    env.quarter,
        "done":       env.done,
    }


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_session(session_id: str) -> CompanyEnv:
    if session_id not in _sessions:
        raise HTTPException(404, "Session not found. Call /reset first.")
    return _sessions[session_id]
