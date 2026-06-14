"""Tiny thread-safe pub/sub for streaming harness events to the web UI."""
from __future__ import annotations

import threading
import queue


class EventBus:
    def __init__(self, history: int = 5000):
        self._subs = []
        self._lock = threading.Lock()
        self._history = []
        self._history_max = history
        self._seq = 0

    def publish(self, event: dict):
        with self._lock:
            self._seq += 1
            event = dict(event)
            event["seq"] = self._seq
            self._history.append(event)
            if len(self._history) > self._history_max:
                self._history = self._history[-self._history_max :]
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> "queue.Queue":
        q = queue.Queue(maxsize=10000)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q):
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def recent(self, after_seq: int = 0):
        with self._lock:
            return [e for e in self._history if e["seq"] > after_seq]
