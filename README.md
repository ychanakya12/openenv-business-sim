# Business Simulation Environment

> **Meta × PyTorch Hackathon — Round 1 Submission**
> OpenEnv-compatible multi-agent business simulation environment.

An AI CEO agent manages a software company across quarters, making strategic
decisions under uncertainty, adversarial conditions, and dynamic market forces.

---

## 🎯 Environment Overview

The agent plays the role of a CEO choosing which projects to accept, how to
invest in the team, and how to manage risk — with every decision having
delayed, cascading consequences.

**Three autonomous agents:**
| Agent | Role |
|---|---|
| **Decision Agent** (LLM) | CEO — makes quarterly strategic decisions |
| **Market Agent** | Simulates boom/stable/recession cycles via Markov chain |
| **Adversarial Agent** | Injects realistic shocks — client disputes, dev quits, audits |

---

## ⚡ Key Factors Modelled

| # | Factor | Effect |
|---|---|---|
| 1 | **Profit Potential** | `base_profit × market_demand × tech_multiplier` |
| 2 | **Risk Level** | `base_risk + skill_gap + burnout + hidden_risk` |
| 3 | **Team Capability** | Skill gap drives failure probability |
| 4 | **Deadline Pressure** | Tight deadlines increase team burnout |
| 5 | **Market Demand** | Domain demand shifts each quarter (Markov chain) |
| 6 | **Reputation Impact** | Per-project gain/loss; affects future pipeline |
| + | **Burnout / Resources** | Overloaded team degrades skill, increases errors |

---

## 📋 Action Space — `CEOAction`

```json
{
  "accept_project_id": "abc12345 or null",
  "hire_count":        0,
  "fire_count":        0,
  "training_budget":   0.0,
  "tech_stack":        "cheap | standard | premium",
  "reduce_workload":   false
}
```

## 👁️ Observation Space — `CompanyObservation`

```json
{
  "session_id":         "uuid",
  "quarter":            1,
  "max_quarters":       4,
  "goal":               "...",
  "budget":             100000.0,
  "resource_pool":      500.0,
  "team": {
    "size":         5,
    "skill":        0.4,
    "burnout":      0.0,
    "domain_focus": "web"
  },
  "reputation":         0.7,
  "market_phase":       "stable",
  "domain_demand":      {"ai": 1.1, "web": 1.0, ...},
  "available_projects": [...],
  "active_risks":       [],
  "last_action_result": null,
  "last_action_error":  null
}
```

---

## 🗂️ Tasks

| Task ID | Difficulty | Quarters | Objective |
|---|---|---|---|
| `single_quarter_survival` | Easy | 1 | End with positive cash flow |
| `four_quarter_growth` | Medium | 4 | Grow revenue 30% |
| `adversarial_resilience` | Hard | 8 | Survive with reputation ≥ 0.6 |

All graders return a score in **[0.0, 1.0]** with partial progress credit.

---

## 🚀 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset?task_id=...` | Start new episode |
| `POST` | `/step?session_id=...` | Submit action |
| `GET` | `/state?session_id=...` | Full internal state |
| `GET` | `/grade?session_id=...` | Current score [0,1] |

---

## 🛠️ Setup & Running

### Local (no Docker)

```bash
pip install -r requirements.txt
uvicorn src.server:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
docker build -t business-sim-env .
docker run -p 7860:7860 business-sim-env
```

### Run inference agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="your_hf_token"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Pre-submission validation

```bash
# With server running in another terminal:
python validate.py
```

---

## 📁 Project Structure

```
business-sim-env/
├── inference.py              ← Baseline agent (mandatory, root level)
├── openenv.yaml              ← OpenEnv spec declaration
├── Dockerfile                ← Container definition
├── requirements.txt
├── validate.py               ← Pre-submission checker
├── README.md
└── src/
    ├── models.py             ← Pydantic typed models (CEOAction, CompanyObservation)
    ├── server.py             ← FastAPI server
    ├── business_sim_env.py   ← Client wrapper (mirrors BrowserGymEnv pattern)
    ├── environment/
    │   ├── company_env.py    ← Core simulation logic
    │   ├── market_agent.py   ← Markov chain market dynamics
    │   └── adversarial.py    ← Shock injection
    └── tasks/
        ├── task_easy.py      ← Single quarter survival grader
        ├── task_medium.py    ← Four quarter growth grader
        └── task_hard.py      ← Adversarial resilience grader
```

---

## 🌍 Real-World Relevance

Unlike toy environments, this simulation models:
- **Delayed consequences** — cheap tech stack causes bugs 2 quarters later
- **Compounding effects** — burnout degrades skill, which increases project risk
- **Market dynamics** — demand shifts force strategy adaptation
- **Hidden uncertainty** — `hidden_risk` simulates scope creep and client changes
- **Counterfactual reasoning** — `/state` returns what would have happened differently
