"""Analyze a finished run directory and print a quality scorecard.

    python report.py [run-id]      # defaults to the most recent run

Surfaces the things that matter for judging decomposition quality: tree shape,
how much actually ran in parallel, dependency edges, failures, tokens, time.
"""
import json, os, sys
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def load(run_id=None):
    if not run_id:
        runs = [d for d in os.listdir(RUNS) if d.startswith("run-")]
        run_id = sorted(runs)[-1]
    with open(os.path.join(RUNS, run_id, "state.json"), encoding="utf-8") as f:
        return run_id, json.load(f)


def parallelism(nodes):
    """For each branch, how many of its children's run-intervals overlapped."""
    max_concurrent = 0
    for n in nodes.values():
        kids = [nodes[c] for c in n["children"] if nodes.get(c)]
        ivals = [(k["started_at"], k["ended_at"]) for k in kids
                 if k.get("started_at") and k.get("ended_at")]
        for s, _ in ivals:
            conc = sum(1 for a, b in ivals if a <= s < b)
            max_concurrent = max(max_concurrent, conc)
    return max_concurrent


def tree(nodes, nid, depth, out):
    n = nodes[nid]
    icon = {"done": "✓", "failed": "✗", "skipped": "–"}.get(n["status"], "•")
    dep = f"  ⇠deps:{n['deps']}" if n["deps"] else ""
    tok = n["tokens"]["prompt"] + n["tokens"]["eval"]
    dur = (n["ended_at"] - n["started_at"]) if (n.get("ended_at") and n.get("started_at")) else 0
    out.append("  " * depth + f"{icon} [{n.get('kind') or '?'}] {n['title']}"
               f"  ({n['status']}, {dur:.0f}s, {tok}tok){dep}")
    for c in n["children"]:
        if nodes.get(c):
            tree(nodes, c, depth + 1, out)


def main():
    run_id, data = load(sys.argv[1] if len(sys.argv) > 1 else None)
    nodes = data["nodes"]
    root = nodes[data["root_id"]]
    by_status, by_kind = {}, {}
    edges = max_depth = total_tok = 0
    for n in nodes.values():
        by_status[n["status"]] = by_status.get(n["status"], 0) + 1
        by_kind[n.get("kind") or "?"] = by_kind.get(n.get("kind") or "?", 0) + 1
        edges += len(n["deps"])
        max_depth = max(max_depth, n["depth"])
        total_tok += n["tokens"]["prompt"] + n["tokens"]["eval"]
    wall = (root["ended_at"] - root["started_at"]) if root.get("ended_at") else 0
    out_path = os.path.join(RUNS, run_id, "nodes", data["root_id"], "output.md")
    out_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0

    print(f"RUN {run_id}")
    print(f"  status        {root['status']}")
    print(f"  nodes         {len(nodes)}   {by_status}")
    print(f"  kinds         {by_kind}")
    print(f"  max depth     {max_depth}")
    print(f"  dep edges     {edges}")
    print(f"  max parallel  {parallelism(nodes)} children running at once")
    print(f"  total tokens  {total_tok}")
    print(f"  wall time     {wall:.0f}s")
    print(f"  final output  {out_size} chars")
    print("\nTREE")
    lines = []
    try:
        tree(nodes, data["root_id"], 0, lines)
    except Exception as e:  # tolerate a mid-run (partially written) snapshot
        lines.append(f"(partial tree: {e})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
