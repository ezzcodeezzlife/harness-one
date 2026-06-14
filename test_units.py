"""Deterministic unit tests for the non-LLM logic: token packing and the
dependency-cycle breaker (the deadlock guard). No Ollama needed.

    python test_units.py
"""
import sys, types
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

from harness import context
from harness.harness import Harness


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        raise SystemExit(1)


# ---- context packing -------------------------------------------------
def test_pack():
    check("estimate grows with length",
          context.estimate_tokens("x" * 100) > context.estimate_tokens("x" * 10))
    entries = [("t", "word " * 200) for _ in range(6)]   # ~each > budget piece
    budget = 300
    groups = context.pack_entries(entries, budget)
    # every group must be under budget unless it's a single oversized entry
    for g in groups:
        check("group within budget or singleton",
              context.total_tokens(g) <= budget or len(g) == 1)
    check("packing kept all entries", sum(len(g) for g in groups) == len(entries))


# ---- cycle breaking --------------------------------------------------
class FakeStore:
    def __init__(self, nodes):
        self.nodes = {n["id"]: n for n in nodes}
    def get(self, i):
        return dict(self.nodes[i])
    def update(self, i, **f):
        self.nodes[i].update(f)


def has_cycle(deps):
    """deps: id -> list of ids it depends on. True if any cycle remains."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in deps}
    bad = [False]
    def dfs(u):
        color[u] = GRAY
        for v in deps[u]:
            if color[v] == GRAY:
                bad[0] = True
            elif color[v] == WHITE:
                dfs(v)
        color[u] = BLACK
    for i in deps:
        if color[i] == WHITE:
            dfs(i)
    return bad[0]


def make_harness(nodes):
    store = FakeStore(nodes)
    llm = types.SimpleNamespace(num_ctx=8192)
    return Harness(llm, store, bus=None)


def test_cycles():
    # 3-cycle: a<-b<-c<-a
    nodes = [{"id": "a", "deps": ["b"]}, {"id": "b", "deps": ["c"]}, {"id": "c", "deps": ["a"]}]
    h = make_harness(nodes)
    h._break_cycles([dict(n) for n in nodes])
    deps = {i: h.store.nodes[i]["deps"] for i in ("a", "b", "c")}
    check("3-cycle broken", not has_cycle(deps))

    # self-loop
    nodes = [{"id": "x", "deps": ["x"]}, {"id": "y", "deps": []}]
    h = make_harness(nodes)
    h._break_cycles([dict(n) for n in nodes])
    deps = {i: h.store.nodes[i]["deps"] for i in ("x", "y")}
    check("self-loop broken", not has_cycle(deps))

    # valid DAG must be left intact
    nodes = [{"id": "a", "deps": []}, {"id": "b", "deps": ["a"]}, {"id": "c", "deps": ["a", "b"]}]
    h = make_harness(nodes)
    h._break_cycles([dict(n) for n in nodes])
    deps = {i: h.store.nodes[i]["deps"] for i in ("a", "b", "c")}
    check("valid DAG unchanged", deps == {"a": [], "b": ["a"], "c": ["a", "b"]})
    check("valid DAG still acyclic", not has_cycle(deps))


if __name__ == "__main__":
    test_pack()
    test_cycles()
    print("\nALL UNIT TESTS PASSED")
