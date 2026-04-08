"""
inference.py — Business Simulation Environment
================================================
Baseline agent script. Mirrors the sample inference script pattern exactly.

MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_URL        Where your server is running (default: http://localhost:7860)

Rules:
- Uses OpenAI Client for all LLM calls (mandatory per hackathon rules)
- inference.py in root directory (mandatory)
- Runs in < 20 min on vcpu=2, memory=8GB
- Produces reproducible scores for all 3 tasks
"""

import os
import re
import json
import textwrap
from typing import List, Dict, Optional

from openai import OpenAI

from src.business_sim_env import BusinessSimEnv
from src.models import CEOAction

# ── Configuration (mirrors sample script pattern exactly) ─────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:7860")

MAX_STEPS   = 10      # safety cap (max_quarters per task is the real limit)
TEMPERATURE = 0.2
MAX_TOKENS  = 300

# Safe fallback — do nothing, spend nothing
FALLBACK_ACTION = CEOAction()

DEBUG = True

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are the CEO of a software company. Each quarter you make strategic
    decisions to maximise profit, reputation, and team health.

    KEY FACTORS to weigh every turn:
    1. PROFIT POTENTIAL    — prefer high base_profit when your skill matches
    2. RISK LEVEL          — base_risk + hidden_risk = danger. Avoid if total > 0.7
    3. TEAM CAPABILITY     — never accept if skill_required > team.skill + 0.15
    4. MARKET DEMAND       — check domain_demand dict; prefer demand > 1.0 domains
    5. REPUTATION IMPACT   — failed project loses more rep than a win gains
    6. RESOURCE / BURNOUT  — if team.burnout > 0.5, set reduce_workload=true first

    Decision heuristics:
    - Train if team.skill < 0.5 AND budget > 40000 → training_budget: 20000
    - Hire if team.size < 4 AND budget > 60000 → hire_count: 1
    - Use "premium" tech for AI/data projects with base_profit > 60000
    - Use "cheap" ONLY if budget < 20000 (accept tech debt risk)
    - Set reduce_workload=true if burnout > 0.55

    You MUST respond with ONLY a valid JSON object — no explanation, no markdown fences.

    JSON schema:
    {
      "accept_project_id": "<8-char id from available_projects, or null>",
      "hire_count":        <integer 0-3>,
      "fire_count":        <integer 0-2>,
      "training_budget":   <float 0.0 to 50000.0>,
      "tech_stack":        "<cheap | standard | premium>",
      "reduce_workload":   <true | false>
    }
""").strip()


# ── History helpers (mirrors sample script) ───────────────────────────────────

def build_history_lines(history: List[str]) -> str:
    if not history:
        return "None"
    return "\n".join(history[-4:])   # last 4 turns for context window economy


def build_user_prompt(step: int, observation, history: List[str]) -> str:
    projects_text = "\n".join(
        f"    id={p.id} | {p.name} | domain={p.domain} "
        f"| profit=${p.base_profit:,.0f} (±{p.profit_variance:.0%}) "
        f"| risk={p.base_risk:.2f} hidden={p.hidden_risk:.2f} "
        f"| skill_needed={p.skill_required:.2f} "
        f"| demand_sens={p.demand_sensitivity:.1f} "
        f"| rep_gain={p.reputation_gain:.2f} rep_loss={p.reputation_loss:.2f} "
        f"| deadline={'TIGHT' if p.deadline_tight else 'normal'}"
        for p in observation.available_projects
    ) or "    (none available)"

    domain_demand_text = " | ".join(
        f"{k}:{v:.1f}" for k, v in observation.domain_demand.items()
    )

    return textwrap.dedent(f"""
        Step: {step}
        Goal: {observation.goal}
        Quarter: {observation.quarter} / {observation.max_quarters}

        ── Company State ──────────────────────────────────
        Budget:         ${observation.budget:,.0f}
        Resource pool:  {observation.resource_pool:.0f} man-hrs
        Team size:      {observation.team.size} devs
        Team skill:     {observation.team.skill:.2f} / 1.0
        Team burnout:   {observation.team.burnout:.2f} / 1.0
        Reputation:     {observation.reputation:.2f} / 1.0
        Market phase:   {observation.market_phase}
        Domain demand:  {domain_demand_text}
        Active risks:   {observation.active_risks or 'none'}

        ── Available Projects ─────────────────────────────
        {projects_text}

        ── Feedback ───────────────────────────────────────
        Last result: {observation.last_action_result or 'none'}
        Last error:  {observation.last_action_error  or 'none'}

        ── Recent History ─────────────────────────────────
        {build_history_lines(history)}

        Respond with ONLY the JSON object.
    """).strip()


# ── Action parsing (mirrors sample script's parse_model_action) ───────────────

def parse_action(response_text: str) -> CEOAction:
    """
    Parse LLM JSON response → CEOAction.
    Falls back to safe FALLBACK_ACTION on any parse error.
    """
    if not response_text:
        return FALLBACK_ACTION
    try:
        # Strip markdown fences if model added them
        clean = re.sub(r"```(?:json)?|```", "", response_text).strip()
        # Extract first JSON object in response
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            if DEBUG:
                print(f"  [parse_action] No JSON found in: {response_text[:80]}")
            return FALLBACK_ACTION
        data = json.loads(match.group())
        return CEOAction(**data)
    except Exception as exc:
        if DEBUG:
            print(f"  [parse_action] Failed ({exc}). Using fallback.")
        return FALLBACK_ACTION


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_id: str) -> float:
    """
    Run one full episode for a task.
    Mirrors the main loop pattern from the sample inference script.
    Returns final score in [0.0, 1.0].
    """
    print(f"\n{'='*60}")
    print(f"  TASK : {task_id}")
    print(f"{'='*60}")

    # Mirrors: env = BrowserGymEnv.from_docker_image(image=..., env_vars=...)
    env = BusinessSimEnv.from_docker_image(
        image    = "business-sim-env:latest",
        env_vars = {
            "BUSINESS_SIM_TASK": task_id,
            "BUSINESS_SIM_URL":  ENV_URL,
        },
    )

    history:      List[str] = []
    total_reward: float     = 0.0

    try:
        # ── reset() ──────────────────────────────────────────────────────────
        result      = env.reset()
        observation = result.observation
        print(f"[START] Task: {task_id} | Goal: {observation.goal}")

        for step in range(1, MAX_STEPS + 1):
            # Mirrors sample: if result.done: break early
            if result.done:
                print("  Environment signalled done. Stopping early.")
                break

            # Build prompt and call LLM (OpenAI client — mandatory)
            user_prompt = build_user_prompt(step, observation, history)

            try:
                completion = client.chat.completions.create(
                    model       = MODEL_NAME,
                    messages    = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    temperature = TEMPERATURE,
                    max_tokens  = MAX_TOKENS,
                    stream      = False,
                )
                response_text = completion.choices[0].message.content or ""
            except Exception as exc:
                # Mirrors sample: model request failed → use fallback
                print(f"  [LLM error] {exc} — using fallback action")
                response_text = ""

            action = parse_action(response_text)

            if DEBUG:
                print(
                    f"  Q{step:02d} → project={action.accept_project_id} | "
                    f"hire={action.hire_count} | "
                    f"train=${action.training_budget:,.0f} | "
                    f"stack={action.tech_stack} | "
                    f"rest={action.reduce_workload}"
                )

            # ── step() ───────────────────────────────────────────────────────
            result      = env.step(action)
            observation = result.observation
            reward      = result.reward
            total_reward += reward

            # Build history line (mirrors sample pattern)
            error_flag   = " ERROR" if observation.last_action_error else ""
            history_line = (
                f"Q{step}: accepted={action.accept_project_id} "
                f"stack={action.tech_stack} rest={action.reduce_workload} "
                f"→ reward {reward:+.3f} "
                f"budget=${observation.budget:,.0f} "
                f"rep={observation.reputation:.2f} "
                f"burnout={observation.team.burnout:.2f}"
                + error_flag
            )
            history.append(history_line)

            print(
                f"[STEP] {step} | Task: {task_id} | Action: {action.accept_project_id or 'Skip'} | "
                f"Reward: {reward:+.4f} | Done: {result.done} | "
                f"Budget: ${observation.budget:,.0f} | Rep: {observation.reputation:.2f}"
            )

            if observation.last_action_error:
                print(f"[STEP] {step} ⚠ Error: {observation.last_action_error}")

            # Mirrors sample: if result.done → episode complete
            if result.done:
                print("  Episode complete.")
                break
        else:
            print(f"  Reached max steps ({MAX_STEPS}).")

        # ── Grade ─────────────────────────────────────────────────────────────
        final_score = env.grade()
        print(f"[END] Task: {task_id} | Score: {final_score:.4f} | Total Reward: {total_reward:.4f}")
        return final_score

    finally:
        env.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [
        "single_quarter_survival",
        "four_quarter_growth",
        "adversarial_resilience",
    ]

    scores: Dict[str, float] = {}
    for task_id in tasks:
        scores[task_id] = run_task(client, task_id)

    # ── Final score summary ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {task_id:<30} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average Score: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
