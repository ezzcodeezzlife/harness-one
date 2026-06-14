"""Web UI server for the fractal harness — stdlib http.server + SSE.

    python serve.py            # then open http://localhost:8765

Type a task in the browser, hit Run, and watch the task tree build live:
nodes light up as they plan / execute / wait / synthesize, edges show the DAG,
and you can click any node to read its task, reasoning, summary and full output.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

# Windows consoles default to cp1252; force UTF-8 so log glyphs don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from harness.llm import OllamaClient
from harness.events import EventBus
from harness.store import Store
from harness.harness import Harness

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(ROOT, "runs")
WEB_DIR = os.path.join(ROOT, "web")
HOST_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PORT = int(os.environ.get("PORT", "8765"))
PREFERRED = ["qwen2.5:14b-instruct", "qwen2.5:7b-instruct", "qwen2.5:7b", "llama3.1:8b"]

bus = EventBus()


class RunManager:
    """Holds the single active/last run. One run at a time keeps it simple."""

    def __init__(self):
        self.lock = threading.Lock()
        self.store = None
        self.llm = None
        self.thread = None
        self.run_id = None
        self.model = None

    def is_running(self):
        return self.thread is not None and self.thread.is_alive()

    def default_model(self):
        probe = OllamaClient(model="", host=HOST_OLLAMA)
        avail = probe.list_models()
        for p in PREFERRED:
            if p in avail:
                return p
        return avail[0] if avail else None

    def start(self, task, title, model, max_depth, max_subtasks, concurrency, num_ctx=8192):
        with self.lock:
            if self.is_running():
                return None, "a run is already in progress"
            run_id = "run-" + time.strftime("%Y%m%d-%H%M%S")
            run_dir = os.path.join(RUNS_DIR, run_id)
            os.makedirs(RUNS_DIR, exist_ok=True)
            self.run_id = run_id
            self.model = model
            self.store = Store(run_dir, bus=bus)
            self.llm = OllamaClient(model=model, host=HOST_OLLAMA,
                                    concurrency=concurrency, num_ctx=num_ctx, bus=bus)
            h = Harness(self.llm, self.store, bus,
                        max_depth=max_depth, max_subtasks=max_subtasks)
            bus.publish({"type": "run_started", "run_id": run_id, "model": model,
                         "task": task, "max_depth": max_depth})

            def worker():
                try:
                    h.run(task, title=title)
                except Exception as e:  # noqa: BLE001
                    bus.publish({"type": "log", "node": None, "msg": f"RUN ERROR: {e}"})
                    bus.publish({"type": "run_done", "node": None, "status": "failed"})

            self.thread = threading.Thread(target=worker, daemon=True)
            self.thread.start()
            return run_id, None


mgr = RunManager()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    # ---- helpers -----------------------------------------------------
    def _send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text, code=200, ctype="text/plain; charset=utf-8"):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- routing -----------------------------------------------------
    def do_GET(self):
        u = urlparse(self.path)
        path, qs = u.path, parse_qs(u.query)
        if path == "/" or path == "/index.html":
            self._serve_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
        elif path == "/api/health":
            probe = OllamaClient(model="", host=HOST_OLLAMA)
            up = probe.health()
            self._send_json({
                "ollama": up,
                "models": probe.list_models() if up else [],
                "default_model": mgr.default_model() if up else None,
                "running": mgr.is_running(),
                "run_id": mgr.run_id,
            })
        elif path == "/api/state":
            snap = mgr.store.snapshot() if mgr.store else {"root_id": None, "nodes": {}}
            snap["run_id"] = mgr.run_id
            snap["running"] = mgr.is_running()
            if mgr.llm:
                snap["tokens"] = mgr.llm.token_totals()
            self._send_json(snap)
        elif path == "/api/artifact":
            node = (qs.get("node") or [None])[0]
            text = mgr.store.load_artifact(node) if (mgr.store and node) else None
            self._send_text(text if text is not None else "(no output)")
        elif path == "/api/events":
            self._sse()
        else:
            self._send_text("not found", 404)

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/run":
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            task = (body.get("task") or "").strip()
            if not task:
                return self._send_json({"error": "empty task"}, 400)
            model = body.get("model") or mgr.default_model()
            if not model:
                return self._send_json({"error": "no model available"}, 400)
            run_id, err = mgr.start(
                task=task,
                title=body.get("title") or "root",
                model=model,
                max_depth=int(body.get("max_depth", 3)),
                max_subtasks=int(body.get("max_subtasks", 6)),
                concurrency=int(body.get("concurrency", 2)),
                num_ctx=int(body.get("num_ctx", 8192)),
            )
            if err:
                return self._send_json({"error": err}, 409)
            self._send_json({"run_id": run_id, "model": model})
        else:
            self._send_text("not found", 404)

    # ---- static ------------------------------------------------------
    def _serve_file(self, fpath, ctype):
        try:
            with open(fpath, "rb") as f:
                body = f.read()
        except OSError:
            return self._send_text("not found", 404)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---- SSE ---------------------------------------------------------
    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        q = bus.subscribe()
        try:
            # send a full snapshot first so a fresh client renders current state
            if mgr.store:
                snap = mgr.store.snapshot()
                snap["run_id"] = mgr.run_id
                snap["running"] = mgr.is_running()
                self._sse_write({"type": "snapshot", "state": snap})
            while True:
                try:
                    ev = q.get(timeout=15)
                    self._sse_write(ev)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            bus.unsubscribe(q)

    def _sse_write(self, obj):
        data = json.dumps(obj)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()


def main():
    probe = OllamaClient(model="", host=HOST_OLLAMA)
    up = probe.health()
    print(f"Ollama: {'UP' if up else 'DOWN'} @ {HOST_OLLAMA}")
    if up:
        print(f"Models: {probe.list_models()}")
        print(f"Default: {mgr.default_model()}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n  Harness UI →  http://localhost:{PORT}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
