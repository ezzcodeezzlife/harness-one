"""Prompts + JSON schemas for the three node operations: plan, execute, synthesize.

This is the part most worth iterating on — decomposition quality lives here.
Schemas are passed to Ollama as structured-output `format` so the model is
constrained to valid JSON.
"""
from __future__ import annotations

# ---- JSON schemas (Ollama structured output) -------------------------

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "is_atomic": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "instruction": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "instruction", "depends_on"],
            },
        },
    },
    "required": ["is_atomic", "reasoning", "subtasks"],
}

WORK_SCHEMA = {
    "type": "object",
    "properties": {
        "output": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["output", "summary"],
}

ASSEMBLE_SCHEMA = {
    "type": "object",
    "properties": {
        "intro": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["intro", "summary"],
}


# ---- system prompts --------------------------------------------------

PLANNER_SYSTEM = """You are the PLANNER in a recursive task-decomposition system.

Decide whether a task should be done DIRECTLY or SPLIT into smaller subtasks,
then (if splitting) lay out those subtasks as a dependency graph.

Hard rules:
- Prefer doing it directly: set "is_atomic": true UNLESS the task is genuinely
  too large or too many-sided to answer well in one focused pass. A task that a
  capable worker could complete well in a single response is atomic. Do NOT
  split for the sake of splitting — over-decomposition degrades quality.
- If you split, produce at most N subtasks (N is given in the user message),
  each clearly SIMPLER than the whole. If the task naturally has MORE than N
  distinct parts, do NOT list them flat — GROUP related parts into a few broader
  subtasks. Each broader subtask gets decomposed further automatically, so
  grouping costs no coverage, whereas exceeding N silently DROPS the extra parts.
  (e.g. 12 API endpoints under a 6-cap -> a "Core API reference" subtask, not 12
  siblings.)
- Each subtask MUST be SELF-CONTAINED. The worker who does it sees ONLY the
  "instruction" you write here — never this parent task or the sibling subtasks.
  So put every concrete detail it needs inside the instruction.
- "depends_on" lists the ids of subtasks whose ACTUAL OUTPUT this subtask must
  read to do its own work. Be very conservative here: MOST subtasks are
  independent and should have an empty depends_on so they run IN PARALLEL.
  Add an edge ONLY when a subtask literally cannot be produced without the
  concrete content another subtask generates. Do NOT chain subtasks just
  because a human might do them in a certain order — sections of a document,
  independent components, or separate analyses are INDEPENDENT. Unnecessary
  edges destroy parallelism and are a serious error.
- Do NOT add a subtask that just merges/combines the others. The system
  synthesizes the children automatically.
- Keep "title" to 2-5 words. Keep "reasoning" to one or two sentences.

DEPENDENCY LITMUS TEST — apply to every potential edge:
  "To PRODUCE subtask B's output, must the worker READ the actual text/content
   that subtask A produces?"
  If NO -> there is NO dependency (empty depends_on), even if A would naturally
  happen first in real life. Temporal/logical order is NOT a dependency.

Examples:
  • Sections of a guide (Install / Configure / Commands / Troubleshooting): each
    can be written from the tool's spec alone. They do NOT read each other's
    text. -> ALL independent, depends_on=[] for every one.
  • "Design a schema", then "write the migration for that schema", then "write
    queries against it": the migration needs the actual schema; the queries need
    the actual tables. -> migration depends on schema; queries depend on
    migration. Real data dependencies.

Default to depends_on=[] and only add an edge when the litmus test says YES.

Respond with ONLY the JSON object."""

WORKER_SYSTEM = """You are a WORKER in a task-decomposition system.

Do the task FULLY and concretely. "output" must be the actual finished
deliverable (the real content/code/answer), not a plan or a description of what
you would do.

"summary" is 2-4 sentences capturing the essential result plus any facts a
parent task would need to use or build on your work. This summary is ALL that
propagates upward — the full output is stored but the parent reads the summary
first — so make it faithful and information-dense. Do not say "see output".

Respond with ONLY the JSON object."""

SYNTH_SYSTEM = """You are the SYNTHESIZER in a task-decomposition system.

You are given a parent task and the FULL OUTPUTS of its already-completed
subtasks. Assemble them into one complete, coherent deliverable that fulfils the
parent task, and put it in "output".

Critically: PRESERVE the substance of the subtask outputs. If the parent task is
to produce a document/guide/codebase, your "output" must contain the actual
combined content (all the sections/code/details), not a summary or overview of
it. Integrate the pieces: order them sensibly, remove duplication, smooth the
seams, fix contradictions, add brief connective tissue. Do not drop detail. If
something needed is clearly missing, note it briefly at the very end.

"summary" is a separate 2-4 sentence faithful description of the combined result
for the grandparent (this is the compact version that propagates up). Respond
with ONLY the JSON object."""


ASSEMBLE_SYSTEM = """You are assembling a large deliverable that is too big to
rewrite inside one context window. The full text of each section is ALREADY
written and will be concatenated verbatim right after your intro — so you must
NOT reproduce, summarize, or replace the sections.

Given the parent task and the list of section titles + summaries, produce JSON:
- "intro": a short integrative introduction (1-3 paragraphs) that frames the
  deliverable and orients the reader across the sections that follow. Base it
  ONLY on the sections actually listed below. Do NOT claim the deliverable
  covers topics that are not among these sections, even if the parent task asked
  for them — if the parent task asked for something no section covers, say so
  briefly instead of pretending it is included.
- "summary": a 2-4 sentence faithful description of the deliverable AS IT
  ACTUALLY IS (the sections present), for a parent task. Note any requested area
  that is missing.

Respond with ONLY the JSON object."""


COMPACT_SYSTEM = """You compress text so it fits a smaller context budget while
losing as little as possible. Preserve all essential facts, names, numbers,
code, commands, and the overall structure/headings. Drop redundancy and filler,
not substance. Output ONLY the compressed text — no preamble, no commentary."""


# ---- user-message builders ------------------------------------------

def planner_user(task: str, parent_goal: str, depth: int, max_depth: int,
                 context: str = "", max_subtasks: int = 6) -> str:
    parts = []
    if parent_goal:
        parts.append(f"This task serves a larger goal: {parent_goal}")
    if context:
        parts.append(f"Context from completed prerequisite subtasks:\n{context}")
    parts.append(f"Depth: {depth} of max {max_depth}. "
                 f"Produce AT MOST N={max_subtasks} subtasks; group if there are more.")
    if depth >= max_depth - 1:
        parts.append(
            "You are near the depth limit; strongly prefer is_atomic=true unless "
            "splitting is clearly necessary."
        )
    parts.append(f"TASK TO ASSESS:\n{task}")
    return "\n\n".join(parts)


def worker_user(task: str, parent_goal: str, context: str = "") -> str:
    parts = []
    if parent_goal:
        parts.append(f"This serves a larger goal: {parent_goal}")
    if context:
        parts.append(f"Outputs of prerequisite subtasks you should build on:\n{context}")
    parts.append(f"YOUR TASK:\n{task}")
    return "\n\n".join(parts)


def synth_user(task: str, children: list) -> str:
    """children: list of (title, content) in execution order (content = full output)."""
    lines = [f"PARENT TASK:\n{task}", "", "COMPLETED SUBTASK OUTPUTS:"]
    for i, (title, content) in enumerate(children, 1):
        lines.append(f"\n===== SUBTASK {i}: {title} =====\n{content}")
    return "\n".join(lines)


def assemble_user(task: str, children: list) -> str:
    """children: list of (title, summary) — full text is stitched separately."""
    lines = [f"PARENT TASK:\n{task}", "",
             "SECTIONS (already fully written; you only frame them):"]
    for i, (title, summary) in enumerate(children, 1):
        lines.append(f"{i}. {title}: {summary}")
    return "\n".join(lines)
