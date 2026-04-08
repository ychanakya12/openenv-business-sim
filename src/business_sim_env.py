"""
business_sim_env.py — Client-side wrapper for the Business Sim environment.

Mirrors the BrowserGymEnv.from_docker_image() pattern from the sample
inference script so inference.py can swap in cleanly.
"""
import httpx
from src.models import CEOAction, CompanyObservation, StepResult


class _Result:
    """Return type that mirrors BrowserGym's result objects."""
    def __init__(self, data: dict):
        self.observation = CompanyObservation(**data["observation"])
        self.reward      = data.get("reward", 0.0)
        self.done        = data.get("done", False)
        self.info        = data.get("info", {})


class _ResetResult:
    def __init__(self, obs_data: dict):
        self.observation = CompanyObservation(**obs_data)
        self.reward      = 0.0
        self.done        = False
        self.info        = {}


class BusinessSimEnv:
    """
    HTTP client that wraps the FastAPI environment server.
    Usage mirrors BrowserGymEnv so inference.py stays clean.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7860",
        task_id:  str = "single_quarter_survival",
    ):
        self.base_url   = base_url.rstrip("/")
        self.task_id    = task_id
        self.session_id: str | None = None
        self._client    = httpx.Client(timeout=30)

    # ── Mirrors BrowserGymEnv.from_docker_image() ─────────────────────────────
    @classmethod
    def from_docker_image(
        cls,
        image:    str,
        env_vars: dict,
    ) -> "BusinessSimEnv":
        task_id  = env_vars.get("BUSINESS_SIM_TASK", "single_quarter_survival")
        base_url = env_vars.get("BUSINESS_SIM_URL",  "http://localhost:7860")
        return cls(base_url=base_url, task_id=task_id)

    # ── OpenEnv API ────────────────────────────────────────────────────────────

    def reset(self) -> _ResetResult:
        r = self._client.post(
            f"{self.base_url}/reset",
            params={"task_id": self.task_id},
        )
        r.raise_for_status()
        data            = r.json()
        self.session_id = data.get("session_id")
        return _ResetResult(data)

    def step(self, action: CEOAction) -> _Result:
        r = self._client.post(
            f"{self.base_url}/step",
            json=action.model_dump(),
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return _Result(r.json())

    def get_full_state(self) -> dict:
        r = self._client.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return r.json()

    def grade(self) -> float:
        r = self._client.get(
            f"{self.base_url}/grade",
            params={"session_id": self.session_id},
        )
        r.raise_for_status()
        return float(r.json()["score"])

    def close(self):
        self._client.close()
