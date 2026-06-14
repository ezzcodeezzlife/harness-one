"""Fractal task-decomposition harness.

A single recursive Node type that either does a task directly (leaf) or
decomposes it into a DAG of child nodes (branch), synthesizing their results.
The on-disk Store is the source of truth; the context window is just RAM.
"""

__all__ = ["llm", "store", "events", "prompts", "harness"]
