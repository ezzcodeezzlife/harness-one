"""On-disk state store: the source of truth for a run.

The task tree lives here, not in any context window. Nodes hold compact
summaries + a pointer to their full artifact on disk. Every mutation is
persisted atomically and broadcast to the event bus so the UI mirrors state.
"""
from __future__ import annotations

import json
import os
import time
import threading


# node lifecycle statuses (also used by the UI for colour)
PENDING = "pending"
PLANNING = "planning"
RUNNING = "running"        # leaf doing work
WAITING = "waiting"       # branch waiting on children
SYNTHESIZING = "synthesizing"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"


class Store:
    def __init__(self, run_dir: str, bus=None):
        self.run_dir = run_dir
        self.nodes_dir = os.path.join(run_dir, "nodes")
        os.makedirs(self.nodes_dir, exist_ok=True)
        self.bus = bus
        self.lock = threading.RLock()
        self.nodes = {}
        self.root_id = None
        self._counter = 0

    # ---- ids ----------------------------------------------------------
    def new_id(self) -> str:
        with self.lock:
            nid = f"n{self._counter}"
            self._counter += 1
            return nid

    # ---- nodes --------------------------------------------------------
    def add_node(
        self,
        title: str,
        task: str,
        parent_id=None,
        depth: int = 0,
        deps=None,
        parent_goal: str = "",
    ) -> dict:
        with self.lock:
            nid = self.new_id()
            node = {
                "id": nid,
                "parent_id": parent_id,
                "title": title,
                "task": task,
                "depth": depth,
                "deps": deps or [],          # sibling node ids this one depends on
                "parent_goal": parent_goal,
                "status": PENDING,
                "kind": None,                 # leaf | branch
                "children": [],
                "reasoning": None,
                "summary": None,
                "artifact": None,             # path to full output
                "error": None,
                "tokens": {"prompt": 0, "eval": 0},
                "created_at": time.time(),
                "started_at": None,
                "ended_at": None,
            }
            self.nodes[nid] = node
            if parent_id is None:
                self.root_id = nid
            else:
                self.nodes[parent_id]["children"].append(nid)
            self._persist()
        self._emit(nid)
        return node

    def update(self, node_id: str, **fields) -> dict:
        with self.lock:
            node = self.nodes[node_id]
            tokens = fields.pop("add_tokens", None)
            if tokens:
                node["tokens"]["prompt"] += tokens.get("prompt_tokens", 0)
                node["tokens"]["eval"] += tokens.get("eval_tokens", 0)
            node.update(fields)
            self._persist()
        self._emit(node_id)
        return node

    def get(self, node_id: str) -> dict:
        with self.lock:
            return dict(self.nodes[node_id])

    # ---- artifacts ----------------------------------------------------
    def save_artifact(self, node_id: str, content: str, name: str = "output.md") -> str:
        d = os.path.join(self.nodes_dir, node_id)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def load_artifact(self, node_id: str):
        node = self.nodes.get(node_id)
        if not node or not node.get("artifact"):
            return None
        try:
            with open(node["artifact"], "r", encoding="utf-8") as f:
                return f.read()
        except OSError:
            return None

    # ---- snapshots / persistence -------------------------------------
    def snapshot(self) -> dict:
        with self.lock:
            return {
                "root_id": self.root_id,
                "nodes": {k: dict(v) for k, v in self.nodes.items()},
            }

    def _persist(self):
        tmp = os.path.join(self.run_dir, "state.json.tmp")
        final = os.path.join(self.run_dir, "state.json")
        data = {
            "root_id": self.root_id,
            "counter": self._counter,
            "nodes": self.nodes,
        }
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, final)

    def _emit(self, node_id: str):
        if self.bus:
            self.bus.publish({"type": "node", "node": dict(self.nodes[node_id])})

    @classmethod
    def load(cls, run_dir: str, bus=None) -> "Store":
        s = cls(run_dir, bus=bus)
        with open(os.path.join(run_dir, "state.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        s.nodes = data["nodes"]
        s.root_id = data["root_id"]
        s._counter = data.get("counter", len(s.nodes))
        return s
