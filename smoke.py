"""Quick smoke test: structured-output JSON via the client, end to end."""
import time
from harness.llm import OllamaClient
from harness import prompts

c = OllamaClient(model="qwen2.5:7b-instruct")
print("health:", c.health())

t0 = time.time()
obj, meta = c.complete_json(
    prompts.PLANNER_SYSTEM,
    prompts.planner_user(
        "Write a getting-started guide for a CLI tool 'fractl' covering install, config, commands, troubleshooting.",
        parent_goal="", depth=0, max_depth=3,
    ),
    schema=prompts.PLAN_SCHEMA,
)
print(f"\nplan in {time.time()-t0:.1f}s  meta={meta}")
print("is_atomic:", obj.get("is_atomic"))
print("reasoning:", obj.get("reasoning"))
for s in obj.get("subtasks", []):
    print(f"  - {s['id']}: {s['title']}  deps={s.get('depends_on')}")
    print(f"      {s['instruction'][:90]}")
