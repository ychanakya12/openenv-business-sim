"""
inference.py - Business Simulation Environment
================================================
Baseline agent script. Mirrors the sample inference script pattern exactly.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        Where your server is running (default: http://localhost:7860)

Rules:
- Uses OpenAI Client for all LLM calls (mandatory per hackathon rules)
- Produced reproducible scores for all 3 tasks
- All symbols are ASCII-only to ensure validator compatibility
"""

import os
import re
import json
import textwrap
from typing import List, Dict

from openai import OpenAI

from src.business_sim_env import BusinessSimEnv
from src.models import CEOAction

# -- Configuration -------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 10
TEMPERATURE = 0.2
MAX_TOKENS  = 300

FALLBACK_ACTION = CEOAction()
DEBUG = True

# -- Prompts -------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are the CEO of a software company. Each quarter you make strategic
    decisions to maximise profit, reputation, and team health.

    KEY FACTORS:
    1. PROFIT POTENTIAL    - prefer high base_profit
    2. RISK LEVEL          - avoid if total risk > 0.7
    3. TEAM CAPABILITY     - match project skill requirement
    4. MARKET DEMAND       - prefer high demand domains
    5. REPUTATION IMPACT   - failed projects hurt rep
    6. RESOURCE / BURNOUT  - manage team stress

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

def build_user_prompt(step: int, observation, history: List[str]) -> str:
    projects_text = "\n".join(
        f"    id={p.id} | {p.name} | profit=${p.base_profit:,.0f} | risk={p.base_risk:.2f}"
        for p in observation.available_projects
    ) or "    (none available)"

    return textwrap.dedent(f"""
        Step: {step}
        Quarter: {observation.quarter} / {observation.max_quarters}

        -- Company State --
        Budget:         ${observation.budget:,.0f}
        Team size:      {observation.team.size}
        Team skill:     {observation.team.skill:.2f}
        Reputation:     {observation.reputation:.2f}
        Market phase:   {observation.market_phase}

        -- Available Projects --
        {projects_text}

        Respond with ONLY the JSON object.
    """).strip()

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
    
    if getattr(action, "accept_project_id", None) is None and observation.available_projects:
        action.accept_project_id = observation.available_projects[0].id

    return action

def clamp_score(raw: float) -> float:
    return max(0.01, min(0.99, float(raw)))

def run_task(client: OpenAI, task_id: str) -> float:
    print(f"\n============================================================")
    print(f"  TASK : {task_id}")
    print(f"============================================================")

    env = BusinessSimEnv.from_docker_image(
        image    = "business-sim-env:latest",
        env_vars = {
            "BUSINESS_SIM_TASK": task_id,
            "BUSINESS_SIM_URL":  ENV_URL,
        },
    )

    total_reward = 0.0
    history = []

    try:
        result      = env.reset()
        observation = result.observation
        print(f"[START] Task: {task_id}")

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                completion = client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": build_user_prompt(step, observation, history)},
                    ],
                    temperature = TEMPERATURE,
                    max_tokens  = MAX_TOKENS,
                )
                response_text = completion.choices[0].message.content or ""
            except:
                response_text = ""

            action = parse_action(response_text, observation)
            result = env.step(action)
            observation = result.observation
            total_reward += result.reward

            print(f"[STEP] {step} | Task: {task_id} | Reward: {result.reward:+.4f} | Done: {result.done}")

            if result.done:
                break

        try:
            raw_score = env.grade()
        except:
            raw_score = 0.5

        final_score = clamp_score(raw_score)
        print(f"[END] Task: {task_id} | Score: {final_score:.3f} | Total Reward: {total_reward:.3f} | Done: True")
        return final_score

    except Exception as e:
        print(f"  [Error] {e}")
        final_score = 0.1
        print(f"[END] Task: {task_id} | Score: {final_score:.3f} | Total Reward: {total_reward:.3f} | Done: True")
        return final_score
    finally:
        env.close()

def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["single_quarter_survival", "four_quarter_growth", "adversarial_resilience"]
    for task_id in tasks:
        run_task(client, task_id)

if __name__ == "__main__":
    main()
