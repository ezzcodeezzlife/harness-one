"""Probe ONLY the planner across task types to judge decomposition quality.

Cheap: one LLM call per task. Lets us iterate on the planner prompt without
paying for full recursive runs.
"""
import sys, time
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from harness.llm import OllamaClient
from harness import prompts

TASKS = [
    ("trivial",      "What is 17 * 23?"),
    ("tiny-write",   "Write a haiku about autumn."),
    ("independent",  "Write a guide for CLI tool 'fractl': install, config, commands, troubleshooting."),
    ("ordered",      "Design a database schema for a blog, then write the SQL migration to "
                     "create those tables, then write 3 example queries that run against them."),
    ("analysis",     "Compare PostgreSQL, MySQL and SQLite across performance, features, and licensing."),
]

c = OllamaClient(model=sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:7b-instruct")
print(f"MODEL: {c.model}")
for name, task in TASKS:
    t0 = time.time()
    obj, _ = c.complete_json(
        prompts.PLANNER_SYSTEM,
        prompts.planner_user(task, "", 0, 3),
        schema=prompts.PLAN_SCHEMA,
    )
    subs = obj.get("subtasks", [])
    atomic = obj.get("is_atomic") or len(subs) <= 1
    print(f"\n### {name}  ({time.time()-t0:.0f}s)  -> {'ATOMIC' if atomic else f'{len(subs)} subtasks'}")
    print(f"    reason: {obj.get('reasoning','')[:100]}")
    n_edges = 0
    for s in subs:
        deps = s.get("depends_on", [])
        n_edges += len(deps)
        print(f"      - {s['title']:<28} deps={deps}")
    if subs:
        print(f"    edges={n_edges}  (0 = fully parallel)")
