"""CLI entry point: run a task through the fractal harness.

    python run.py "your big task here"
    python run.py --file task.txt --model qwen2.5:7b-instruct --max-depth 3
    python run.py --resume run-20260614-010101   # resume an existing run

The on-disk run directory under ./runs is the source of truth.
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

# Windows consoles default to cp1252; force UTF-8 so tree glyphs don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from harness.llm import OllamaClient
from harness.events import EventBus
from harness.store import Store
from harness.harness import Harness

RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
PREFERRED = ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct", "qwen2.5:7b", "llama3.1:8b"]


def choose_model(client: OllamaClient, requested: str | None) -> str:
    available = client.list_models()
    if requested:
        if requested in available:
            return requested
        print(f"[warn] requested model '{requested}' not installed. Available: {available}")
    for p in PREFERRED:
        if p in available:
            return p
    if available:
        return available[0]
    sys.exit("No Ollama models installed. Run:  ollama pull qwen2.5:7b-instruct")


def print_tree(store: Store, node_id=None, indent=0):
    node_id = node_id or store.root_id
    n = store.get(node_id)
    icon = {"done": "✓", "failed": "✗", "skipped": "–"}.get(n["status"], "•")
    kind = n.get("kind") or "?"
    print("  " * indent + f"{icon} [{kind}] {n['title']}  ({n['status']})")
    if n.get("summary"):
        print("  " * indent + f"    ↳ {n['summary'][:120]}")
    for c in n["children"]:
        print_tree(store, c, indent + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="the task to run")
    ap.add_argument("--file", help="read the task from a file")
    ap.add_argument("--model", help="ollama model name")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-subtasks", type=int, default=6)
    ap.add_argument("--concurrency", type=int, default=2, help="max parallel LLM calls")
    ap.add_argument("--num-ctx", type=int, default=8192, help="context window tokens")
    ap.add_argument("--title", default="root")
    ap.add_argument("--resume", help="run id under ./runs to resume")
    ap.add_argument("--verbose", action="store_true", help="stream live harness log")
    args = ap.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            task = f.read().strip()
    else:
        task = args.task
    if not task and not args.resume:
        sys.exit("Provide a task (positional or --file) or --resume <run-id>.")

    import queue as _queue
    bus = EventBus()
    stop_printing = threading.Event()
    printer_thread = None
    if args.verbose:
        pq = bus.subscribe()

        def _printer():
            # stoppable: never blocks past shutdown, so no daemon-thread stdout
            # race at interpreter exit
            while not stop_printing.is_set():
                try:
                    ev = pq.get(timeout=0.2)
                except _queue.Empty:
                    continue
                if ev.get("type") == "log":
                    print(f"   · {ev.get('msg')}", flush=True)

        printer_thread = threading.Thread(target=_printer, daemon=True)
        printer_thread.start()
    probe = OllamaClient(model="", host=args.host)
    if not probe.health():
        sys.exit(f"Ollama not reachable at {args.host}. Is the server running?")
    model = choose_model(probe, args.model)
    print(f"[model] {model}")
    llm = OllamaClient(model=model, host=args.host, concurrency=args.concurrency,
                       num_ctx=args.num_ctx, bus=bus)

    os.makedirs(RUNS_DIR, exist_ok=True)
    if args.resume:
        run_dir = os.path.join(RUNS_DIR, args.resume)
        store = Store.load(run_dir, bus=bus)
        h = Harness(llm, store, bus, max_depth=args.max_depth, max_subtasks=args.max_subtasks)
        print(f"[resume] {run_dir}")
        h.run_node(store.root_id)
        root = store.get(store.root_id)
    else:
        run_id = "run-" + time.strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(RUNS_DIR, run_id)
        store = Store(run_dir, bus=bus)
        h = Harness(llm, store, bus, max_depth=args.max_depth, max_subtasks=args.max_subtasks)
        print(f"[run] {run_dir}\n")
        root = h.run(task, title=args.title)

    # stop the live printer and drain it before final output (avoids a
    # daemon-thread/stdout race at interpreter shutdown)
    stop_printing.set()
    if printer_thread:
        printer_thread.join(timeout=2)

    print("\n" + "=" * 70 + "\nTASK TREE\n" + "=" * 70)
    print_tree(store)
    tot = llm.token_totals()
    print(f"\n[tokens] prompt={tot['prompt']} eval={tot['eval']} calls={tot['calls']}")
    print(f"[status] {root['status']}")
    print("\n" + "=" * 70 + "\nFINAL OUTPUT\n" + "=" * 70)
    out = store.load_artifact(store.root_id)
    print(out or "(no output)")


if __name__ == "__main__":
    main()
