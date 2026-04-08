# 🛠️ SQL Debug & Optimize RL Environment

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi)
![SQLite](https://img.shields.io/badge/SQLite-In--Memory-003B57?style=for-the-badge&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface)

**An OpenEnv-compatible Reinforcement Learning environment for training AI agents to identify, fix, and optimise broken SQL queries.**

Built for the [Meta PyTorch × Scaler SST OpenEnv Hackathon 2026](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Database Schemas](#-database-schemas)
- [Task List](#-task-list)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Running the Project](#-running-the-project)
- [API Reference](#-api-reference)
- [Action & Observation Space](#-action--observation-space)
- [Reward Function](#-reward-function)
- [Docker](#-docker)
- [Deploy to HuggingFace Spaces](#-deploy-to-huggingface-spaces)
- [Security](#-security)
- [License](#-license)

---

## 🔍 Overview

The **SQL Debug & Optimize RL Environment** presents an AI agent with broken SQL queries and full database schemas. The agent must return corrected (and optionally optimised) SQL. Grading is **100% deterministic** for Easy and Medium tasks — queries are executed against an in-memory SQLite database and result sets are compared programmatically.

| Difficulty | Task | Grader | Max Reward |
|---|---|---|---|
| 🟢 **Easy** | Fix SQL syntax errors | Deterministic result-set comparison | 1.0 |
| 🟡 **Medium** | Fix semantic/logic bugs | Deterministic result-set comparison | 1.0 |
| 🔴 **Hard** | Fix bug + optimise query | Correctness (0–0.6) + LLM judge (0–0.4) | 1.0 |

**9 tasks total** across 3 real-world schemas: e-commerce, HR, and product analytics.

---

## ✨ Features

- ✅ **100% deterministic grading** for 6 out of 9 tasks — reproducible scores every run
- ✅ **Three distinct real-world database schemas** with realistic seed data
- ✅ **Hard tasks require genuine optimisation** — CTEs, window functions, eliminating correlated subqueries
- ✅ **Zero external dependencies** — SQLite runs fully in-memory, no DB server needed
- ✅ **OpenEnv-compatible** — plug-and-play with any OpenEnv-compatible RL framework
- ✅ **FastAPI server** — run as HTTP server or in-process
- ✅ **Docker ready** — one command to containerise and deploy

---

## 📁 Project Structure

```
meta/
└── sql_debug_env/
    ├── inference.py               # ← Main entry point (run this)
    ├── validate.py                # Pre-submission validator
    ├── pyproject.toml             # Project metadata & dependencies
    ├── README.md
    ├── .env                       # Your secrets (never commit this!)
    ├── .gitignore
    ├── outputs/
    │   └── evals/
    │       └── inference_results.json
    └── sql_debug_env/
        ├── __init__.py
        ├── models.py              # Action / Observation / State models
        ├── client.py              # HTTP client for server mode
        ├── openenv.yaml           # OpenEnv manifest
        └── server/
            ├── app.py             # FastAPI application
            ├── environment.py     # Core RL environment logic
            ├── tasks.py           # 9 task definitions
            ├── graders.py         # Deterministic + LLM graders
            ├── db_fixtures.py     # SQLite DB factories & helpers
            ├── requirements.txt   # Server dependencies
            └── Dockerfile
```

---

## 🗄️ Database Schemas

### 🛒 E-commerce
Tables: `customers`, `products`, `orders`, `order_items`
Real purchase funnel data: 5 customers, 7 orders, 10 order items, 5 products.

### 👥 HR
Tables: `departments`, `employees`, `salaries`, `projects`, `assignments`
Workforce data: 4 departments, 7 employees, current + historical salary records, project assignments.

### 📊 Analytics
Tables: `users`, `sessions`, `events`, `page_views`
Product analytics: 5 users across free/pro/enterprise plans, 8 sessions, event stream, page view log.

---

## 📋 Task List

| ID | Difficulty | Bug Type | Schema |
|---|---|---|---|
| `easy_001` | 🟢 Easy | `JOINN` typo | E-commerce |
| `easy_002` | 🟢 Easy | `GRUOP` typo | HR |
| `easy_003` | 🟢 Easy | `==` instead of `=` in ON clause | Analytics |
| `medium_001` | 🟡 Medium | Missing `c.name` in GROUP BY | E-commerce |
| `medium_002` | 🟡 Medium | JOIN on wrong column (`dept_id` vs `emp_id`) | HR |
| `medium_003` | 🟡 Medium | `COUNT` should be `COUNT(DISTINCT ...)` | Analytics |
| `hard_001` | 🔴 Hard | Typo `'deliverd'` in HAVING + correlated subqueries | E-commerce |
| `hard_002` | 🔴 Hard | Filters on inactive salaries + correlated AVG | HR |
| `hard_003` | 🔴 Hard | HAVING excludes users + `page_views` scanned 3x | Analytics |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/akshaypatni-lgtm/meta.git
cd meta/sql_debug_env
```

**2. Install dependencies**
```bash
pip install -r sql_debug_env/server/requirements.txt
pip install python-dotenv pyyaml
```

Or install as a package:
```bash
pip install -e .
```

---

## 🔧 Configuration

Create a `.env` file inside the `sql_debug_env/` folder:

```env
HF_TOKEN=hf_your_huggingface_token_here
API_BASE_URL=https://router.huggingface.co/hf-inference/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
```

> ⚠️ **Never commit your `.env` file.** It is already in `.gitignore`.

Get your HuggingFace token at: https://huggingface.co/settings/tokens

---

## 🚀 Running the Project

### Option 1 — Direct (in-process, recommended for local testing)
```bash
cd sql_debug_env
python inference.py
```

### Option 2 — Via HTTP server
```bash
# Terminal 1: Start the server
uvicorn sql_debug_env.server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run inference against the server
python inference.py --server-url http://localhost:8000
```

### Option 3 — Validate your setup first
```bash
python validate.py
```
You should see: `✅ ALL CHECKS PASSED — Ready to submit!`

### Windows Note
If you see `UnicodeEncodeError` with emojis on Windows CMD, the project already handles this automatically via `sys.stdout.reconfigure(encoding='utf-8')`.

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | Start a new episode, returns first observation |
| `POST` | `/step` | Submit `fixed_query`, receive graded observation |
| `GET` | `/state` | Get current episode state |
| `GET` | `/schema` | JSON schema for action/observation |
| `GET` | `/tasks` | List all 9 tasks |

### Example — Reset and Step

```python
import httpx

# Start episode
obs = httpx.post("http://localhost:8000/reset", json={}).json()
print(obs["task_id"], obs["buggy_query"])

# Submit fix
result = httpx.post("http://localhost:8000/step", json={
    "fixed_query": "SELECT * FROM customers WHERE id = 1"
}).json()
print(f"Reward: {result['reward']}")
```

---

## 🎮 Action & Observation Space

### Action
```python
class SQLDebugAction(BaseModel):
    fixed_query: str         # Corrected (and optionally optimised) SQL
    explanation: str | None  # Optional optimisation rationale (scored for HARD tasks)
```

### Observation
```python
class SQLDebugObservation(BaseModel):
    task_id: str
    difficulty: str                  # "easy" | "medium" | "hard"
    task_prompt: str                 # What the agent must fix
    buggy_query: str                 # The broken SQL
    db_schema: list[TableSchema]     # Full CREATE TABLE definitions
    reward: float                    # Reward for previous step
    done: bool                       # True when all 9 tasks complete
    feedback: str                    # Grader explanation
    success: bool                    # True if reward >= 0.5
```

---

## 🏆 Reward Function

```
EASY / MEDIUM:
  reward = 1.0  →  result set matches reference exactly
  reward = 0.3  →  row count matches but values differ (partial credit)
  reward = 0.0  →  query fails or wrong row count

HARD:
  reward = correctness × 0.6   (result set match, scaled 0–0.6)
         + optimisation × 0.4  (LLM judge scores query plan 0–10 → 0–0.4)
```

---

## 🐳 Docker

```bash
# Build image
docker build -t sql-debug-env:latest -f sql_debug_env/server/Dockerfile .

# Run container
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/hf-inference/v1" \
  -e MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" \
  -e HF_TOKEN="your_token_here" \
  sql-debug-env:latest
```

---

## 🤗 Deploy to HuggingFace Spaces

```bash
huggingface-cli login
huggingface-cli repo create sql-debug-env --type space --sdk docker
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/sql-debug-env
git push hf main
```

---

## 🔒 Security

- **Never hardcode API tokens** in `.py` files — always use environment variables
- **`.env` file is gitignored** — your secrets stay local
- **Rotate tokens immediately** if accidentally exposed — go to https://huggingface.co/settings/tokens

```
✅ Safe pattern:
.env file (never committed) → loaded by python-dotenv → read via os.environ
```

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ for the Meta PyTorch × Scaler SST OpenEnv Hackathon 2026
</div>
