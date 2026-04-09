"""
inference.py - Business Simulation Environment
================================================
Runs the LLM agent through all 3 tasks and emits structured OpenEnv stdout logs.

Standard Log Format:
    [START] task=<task> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import re
import json
import textwrap
from typing import List, Optional

from openai import OpenAI
from src.business_sim_env import BusinessSimEnv
from src.models import CEOAction

# -- Configuration -------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK    = "business-sim"
MAX_STEPS    = 3
TEMPERATURE  = 0.2
MAX_TOKENS   = 300

SUCCESS_THRESHOLD = 0.5
FALLBACK_ACTION   = CEOAction()

TASKS = [
    "single_quarter_survival",
    "four_quarter_growth",
    "adversarial_resilience",
]

# -- Prompts -------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are the CEO of a software company. Each quarter you make strategic
    decisions to maximise profit, reputation, and team health.

    JSON schema:
    {
      "accept_project_id": "<8-char id or null>",
      "hire_count":        <int 0-3>,
      "fire_count":        <int 0-2>,
      "training_budget":   <float 0-50000>,
      "tech_stack":        "<cheap | standard | premium>",
      "reduce_workload":   <true | false>
    }
""").strip()

def build_user_prompt(step: int, observation) -> str:
    projects_text = "\n".join(
        f"- id={p.id} | {p.name} | profit=${p.base_profit:,.0f} | risk={p.base_risk:.2f}"
        for p in observation.available_projects
    ) or "(none available)"

    return textwrap.dedent(f"""
        Quarter: {observation.quarter} / {observation.max_quarters}
        Budget:  ${observation.budget:,.0f}
        Skill:   {observation.team.skill:.2f}
        Rep:     {observation.reputation:.2f}

        Available Projects:
        {projects_text}

        Reply with ONLY the JSON object.
    """).strip()

# -- Logging Helpers -----------------------------------------------------------

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = str(error) if error else "null"
    done_val  = str(done).lower()
    # Using simple action description for the log
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# -- Logic ---------------------------------------------------------------------

def parse_action(response_text: str, observation) -> CEOAction:
    action = FALLBACK_ACTION
    try:
        clean = re.sub(r"```(?:json)?|```", "", response_text).strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            data = json.loads(match.group())
            action = CEOAction(**data)
    except:
        pass
    
    # Force a project if none accepted but some available
    if not action.accept_project_id and observation.available_projects:
        action.accept_project_id = observation.available_projects[0].id

    return action

def run_task(client: OpenAI, task_id: str) -> None:
    log_start(task_id)
    
    env = BusinessSimEnv.from_docker_image(
        image    = "business-sim-env:latest",
        env_vars = {"BUSINESS_SIM_TASK": task_id, "BUSINESS_SIM_URL": ENV_URL}
    )

    rewards = []
    steps_taken = 0
    success = False
    score = 0.5

    try:
        result = env.reset()
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step
            try:
                completion = client.chat.completions.create(
                    model    = MODEL_NAME,
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(step, obs)},
                    ],
                    temperature = TEMPERATURE,
                )
                response = completion.choices[0].message.content or ""
                action = parse_action(response, obs)
                
                result = env.step(action)
                obs = result.observation
                reward = float(result.reward)
                done = bool(result.done)

                rewards.append(reward)
                log_step(step, "ceo_decision", reward, done, None)

                if done:
                    break
            except Exception as e:
                log_step(step, "error", 0.0, True, str(e))
                break

        # Final grade
        try:
            raw_score = env.grade()
            score = max(0.01, min(0.99, float(raw_score)))
        except:
            score = 0.5
        
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        # Emergency end
        score = 0.1
        success = False
    finally:
        log_end(success, steps_taken, score, rewards)
        env.close()

def main():
    if not API_KEY:
        print("[ERROR] HF_TOKEN / API_KEY missing.")
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_task(client, task_id)

if __name__ == "__main__":
    main()
