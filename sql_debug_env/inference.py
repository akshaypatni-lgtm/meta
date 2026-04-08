#!/usr/bin/env python3
"""
inference.py — Mandatory root-level inference script for SQL Debug Env.

Runs an LLM agent through all 9 SQL tasks and reports per-task rewards.

Required env vars:  API_BASE_URL  MODEL_NAME  HF_TOKEN
Usage:
  python inference.py                          # in-process (default)
  python inference.py --server-url http://...  # via HTTP server
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()
import argparse
import json
import os
import re
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import textwrap
import time
from typing import Dict, List

# --- validate env vars ---
REQUIRED_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
missing = [v for v in REQUIRED_VARS if not os.environ.get(v)]
if missing:
    print(f"[ERROR] Missing env vars: {', '.join(missing)}")
    sys.exit(1)

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME   = os.environ["MODEL_NAME"]
HF_TOKEN     = os.environ["HF_TOKEN"]

from openai import OpenAI
llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a senior SQL developer. You will be given a broken SQL query and
    the database schema. Fix the query as instructed.
    For EASY/MEDIUM tasks: return ONLY the corrected SQL, wrapped in ```sql``` fences.
    For HARD tasks: return the corrected + optimised SQL in ```sql``` fences,
    then add a brief comment after the fence explaining the optimisation.
""").strip()


def agent_respond(task_prompt: str, buggy_query: str, schema_str: str, difficulty: str) -> Dict[str, str]:
    if difficulty == "hard":
        user_content = (
            f"DATABASE SCHEMA:\n{schema_str}\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"BUGGY QUERY:\n```sql\n{buggy_query}\n```\n\n"
            "Fix the bug AND optimise query performance. "
            "Return corrected+optimised SQL in ```sql``` fences, then explain your optimisation."
        )
    else:
        user_content = (
            f"DATABASE SCHEMA:\n{schema_str}\n\n"
            f"TASK:\n{task_prompt}\n\n"
            f"BUGGY QUERY:\n```sql\n{buggy_query}\n```\n\n"
            "Return ONLY the corrected SQL in ```sql``` fences."
        )
    try:
        resp = llm_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        full = resp.choices[0].message.content.strip()
        m = re.search(r"```(?:sql)?\n?(.*?)```", full, re.DOTALL | re.IGNORECASE)
        sql_part = m.group(1).strip() if m else full
        explanation = full[full.rfind("```") + 3:].strip() if "```" in full else None
        return {"fixed_query": sql_part, "explanation": explanation}
    except Exception as exc:
        print(f"  [WARN] LLM failed: {exc}")
        return {"fixed_query": buggy_query, "explanation": None}


def schema_to_str(schema_list: list) -> str:
    lines = []
    for t in schema_list:
        if isinstance(t, dict):
            cols = ", ".join(f"{c['name']} {c['type']}" for c in t.get("columns", []))
            lines.append(f"  {t['table_name']}({cols})")
        else:
            cols = ", ".join(f"{c.name} {c.type}" for c in t.columns)
            lines.append(f"  {t.table_name}({cols})")
    return "\n".join(lines)


def run_direct() -> List[Dict]:
    sys.path.insert(0, os.path.dirname(__file__))
    from sql_debug_env.server.environment import SQLDebugEnvironment
    from sql_debug_env.models import SQLDebugAction

    env = SQLDebugEnvironment()
    obs = env.reset()
    results = []
    step_num = 0

    while not obs.done:
        step_num += 1
        print(f"\n{'='*60}")
        print(f"Task {step_num}: [{obs.difficulty.upper()}] {obs.task_id}")
        print(f"Prompt: {obs.task_prompt[:100]}...")
        schema_str = schema_to_str(obs.db_schema)
        out = agent_respond(obs.task_prompt, obs.buggy_query, schema_str, obs.difficulty)
        print(f"Agent SQL (first 120): {out['fixed_query'][:120]}...")
        action = SQLDebugAction(**out)
        obs = env.step(action)
        print(f"Reward: {obs.reward:.4f} | {obs.feedback[:150]}")
        tid = obs.task_id if not obs.done else env._tasks[step_num-1]["task_id"]
        results.append({"task_id": tid, "difficulty": obs.difficulty if not obs.done else env._tasks[step_num-1]["difficulty"], "reward": obs.reward, "success": obs.success})

    results[-1]["episode_total_reward"] = env.state.total_reward
    return results


def run_via_server(server_url: str) -> List[Dict]:
    from sql_debug_env.client import SQLDebugEnv
    from sql_debug_env.models import SQLDebugAction

    with SQLDebugEnv(base_url=server_url) as env:
        print(f"Server: {env.health()}")
        obs = env.reset()
        results = []
        step_num = 0

        while not obs.done:
            step_num += 1
            print(f"\nTask {step_num}: [{obs.difficulty.upper()}] {obs.task_id}")
            schema_str = schema_to_str([s.model_dump() if hasattr(s, 'model_dump') else s for s in obs.db_schema])
            out = agent_respond(obs.task_prompt, obs.buggy_query, schema_str, obs.difficulty)
            action = SQLDebugAction(**out)
            obs = env.step(action)
            print(f"Reward: {obs.reward:.4f}")
            results.append({"task_id": obs.task_id, "difficulty": obs.difficulty, "reward": obs.reward, "success": obs.success})
    return results


def print_report(results: List[Dict]) -> None:
    print("\n" + "="*60)
    print("SQL DEBUG ENV — INFERENCE RESULTS")
    print("="*60)
    print(f"{'Task ID':<15} {'Difficulty':<10} {'Reward':>8} {'Pass?':>6}")
    print("-"*45)
    total = 0.0
    for r in results:
        reward = r.get("reward", 0.0)
        total += reward
        tick = "[PASS]" if r.get("success") else "[FAIL]"
        print(f"{r.get('task_id','?'):<15} {r.get('difficulty','?'):<10} {reward:>8.4f} {tick:>6}")
    mean = total / len(results) if results else 0
    print("-"*45)
    print(f"{'Total':<15} {'':10} {total:>8.4f}")
    print(f"{'Mean':<15} {'':10} {mean:>8.4f}")
    print("="*60)
    os.makedirs("outputs/evals", exist_ok=True)
    out_path = "outputs/evals/inference_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL_NAME, "api_base_url": API_BASE_URL, "total_reward": total, "mean_reward": mean, "num_tasks": len(results), "tasks": results}, f, indent=2)
    print(f"\nResults saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--direct", action="store_true")
    args = parser.parse_args()

    print("="*60)
    print("SQL Debug & Optimize RL Environment — Inference")
    print(f"Model:    {MODEL_NAME}")
    print(f"API Base: {API_BASE_URL}")
    print("="*60)

    t0 = time.time()
    if args.direct or args.server_url is None:
        print("\nMode: IN-PROCESS\n")
        results = run_direct()
    else:
        print(f"\nMode: HTTP SERVER ({args.server_url})\n")
        results = run_via_server(args.server_url)

    print(f"\nCompleted in {time.time()-t0:.1f}s")
    print_report(results)


if __name__ == "__main__":
    main()
