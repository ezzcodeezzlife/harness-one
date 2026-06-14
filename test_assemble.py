"""Verify the stitch-assembly fix on the existing 42-node run WITHOUT re-running
the whole tree: re-synthesize just the root from its children's stored artifacts.
"""
import sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from harness.llm import OllamaClient
from harness.store import Store
from harness.harness import Harness
from harness.context import total_tokens

RUN = sys.argv[1] if len(sys.argv) > 1 else "run-20260614-020653"
store = Store.load(f"runs/{RUN}")
llm = OllamaClient(model="qwen2.5:7b-instruct", num_ctx=8192)
h = Harness(llm, store)

root = store.get(store.root_id)
children = [store.get(c) for c in root["children"]]
entries = [(c["title"], store.load_artifact(c["id"]) or c.get("summary") or "") for c in children]

print(f"root has {len(children)} top-level sections")
print(f"combined section tokens ≈ {total_tokens(entries)}  (budget {h._budget()})")
print(f"old root output: {len(store.load_artifact(store.root_id) or '')} chars")

output, summary = h._assemble(store.root_id, root, children, entries)
print(f"\nNEW assembled output: {len(output)} chars")
print(f"summary: {summary[:200]}")
print("\n--- first 600 chars of assembled output ---")
print(output[:600])
