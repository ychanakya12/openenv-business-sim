"""
business_sim_env.py - Client-side wrapper for the Business Sim environment.

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

    # -- Mirrors BrowserGymEnv.from_docker_image() -----------------------------
    @classmethod
    def from_docker_image(
        cls,
        image:    str,
        env_vars: dict,
    ) -> "BusinessSimEnv":
        """Factory method using env vars similar to OpenEnv CI."""
        url     = env_vars.get("BUSINESS_SIM_URL", "http://localhost:7860")
        task_id = env_vars.get("BUSINESS_SIM_TASK", "single_quarter_survival")
        return cls(base_url=url, task_id=task_id)

    def reset(self) -> _ResetResult:
        """Call POST /reset?task_id=..."""
        resp = self._client.post(
            f"{self.base_url}/reset",
            params={"task_id": self.task_id}
        )
        resp.raise_for_status()
        data = resp.json()
        self.session_id = data["session_id"]
        return _ResetResult(data)

    def step(self, action: CEOAction) -> _Result:
        """Call POST /step?session_id=..."""
        if not self.session_id:
            raise RuntimeError("Must call reset() before step()")
            
        resp = self._client.post(
            f"{self.base_url}/step",
            params={"session_id": self.session_id},
            json=action.model_dump()
        )
        resp.raise_for_status()
        return _Result(resp.json())

    def grade(self) -> float:
        """Call GET /grade?session_id=..."""
        if not self.session_id:
            return 0.5
            
        resp = self._client.get(
            f"{self.base_url}/grade",
            params={"session_id": self.session_id}
        )
        resp.raise_for_status()
        return float(resp.json()["score"])

    def close(self):
        """Cleanup."""
        self._client.close()
