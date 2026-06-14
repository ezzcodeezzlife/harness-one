"""Context-budget accounting + packing for compaction.

We have no tokenizer dependency, so token counts are estimated conservatively
(slightly high) to avoid overflowing the model's window. Everything that
aggregates content (synthesis, dependency context) runs through here so a call's
input is checked against the window BEFORE it is sent — if it would overflow, the
caller compacts (map-reduce summarize) instead of truncating blindly.
"""
from __future__ import annotations

# ~3.5 chars/token for mixed English+code; dividing by a smaller number
# overestimates tokens, which is the safe direction here.
CHARS_PER_TOKEN = 3.3
ENTRY_OVERHEAD = 24  # tokens for the "===== SUBTASK n: title =====" scaffolding


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def entry_tokens(title: str, content: str) -> int:
    return estimate_tokens(title) + estimate_tokens(content) + ENTRY_OVERHEAD


def total_tokens(entries) -> int:
    return sum(entry_tokens(t, c) for t, c in entries)


def pack_entries(entries, budget: int):
    """Greedy-pack (title, content) entries into groups whose combined estimated
    tokens stay under `budget`. An entry larger than budget on its own becomes a
    singleton group (the caller is responsible for compacting it first)."""
    groups, cur, cur_tok = [], [], 0
    for title, content in entries:
        t = entry_tokens(title, content)
        if cur and cur_tok + t > budget:
            groups.append(cur)
            cur, cur_tok = [], 0
        cur.append((title, content))
        cur_tok += t
    if cur:
        groups.append(cur)
    return groups
