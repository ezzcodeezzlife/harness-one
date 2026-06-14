"""Minimal Ollama client over the HTTP API using only the stdlib.

No `ollama` / `requests` package — keeps us robust on bleeding-edge Pythons.
The GPU effectively serializes generation, so concurrent LLM calls are gated
by a shared semaphore; the orchestration above stays genuinely parallel.
"""
from __future__ import annotations

import json
import time
import threading
import urllib.request
import urllib.error


class LLMError(RuntimeError):
    pass


def _first_json_block(text: str):
    """Tolerantly pull the first balanced {...} object out of a string."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 900,
        concurrency: int = 2,
        num_ctx: int = 8192,
        bus=None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx  # context window every call runs with
        self.sem = threading.Semaphore(max(1, concurrency))
        self.bus = bus
        self._tok_lock = threading.Lock()
        self.total_prompt_tokens = 0
        self.total_eval_tokens = 0
        self.total_calls = 0

    # ---- low level ----------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(f"Ollama request failed ({path}): {e}") from e

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.host + path, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise LLMError(f"Ollama GET failed ({path}): {e}") from e

    # ---- public -------------------------------------------------------
    def health(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except LLMError:
            return False

    def list_models(self):
        try:
            tags = self._get("/api/tags")
            return [m["name"] for m in tags.get("models", [])]
        except LLMError:
            return []

    def chat(self, messages, fmt=None, options=None, node_id=None):
        """Single non-streaming chat completion. Returns (text, meta)."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options or {"temperature": 0.2},
        }
        if fmt is not None:
            payload["format"] = fmt

        if self.bus:
            self.bus.publish({"type": "llm_start", "node": node_id, "model": self.model})

        t0 = time.time()
        with self.sem:  # GPU is the scarce resource: serialize calls
            resp = self._post("/api/chat", payload)
        dt = time.time() - t0

        text = (resp.get("message") or {}).get("content", "")
        meta = {
            "prompt_tokens": resp.get("prompt_eval_count", 0),
            "eval_tokens": resp.get("eval_count", 0),
            "seconds": round(dt, 2),
        }
        with self._tok_lock:
            self.total_prompt_tokens += meta["prompt_tokens"]
            self.total_eval_tokens += meta["eval_tokens"]
            self.total_calls += 1
            totals = self.token_totals()

        if self.bus:
            self.bus.publish(
                {"type": "llm_end", "node": node_id, "meta": meta, "totals": totals}
            )
        return text, meta

    def complete_json(self, system, user, schema=None, retries=3, options=None, node_id=None):
        """Chat that must return a JSON object. Retries on parse failure.

        Returns (obj, meta). Uses Ollama structured output when a schema is
        given, else format='json'.
        """
        fmt = schema if schema is not None else "json"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err = None
        for attempt in range(retries):
            text, meta = self.chat(messages, fmt=fmt, options=options, node_id=node_id)
            obj = None
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                block = _first_json_block(text)
                if block:
                    try:
                        obj = json.loads(block)
                    except json.JSONDecodeError as e:
                        last_err = e
            if isinstance(obj, dict):
                return obj, meta
            # nudge the model and retry
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "That was not valid JSON. Reply with ONLY a single JSON object, no prose.",
                }
            )
            last_err = last_err or ValueError("non-object JSON")
        raise LLMError(f"Could not get JSON after {retries} tries: {last_err}")

    def token_totals(self):
        return {
            "prompt": self.total_prompt_tokens,
            "eval": self.total_eval_tokens,
            "calls": self.total_calls,
        }
