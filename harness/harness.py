"""The fractal harness.

One recursive operation, run_node, used identically at every depth:

    leaf   -> do the task in one LLM call, store output + summary
    branch -> plan subtasks (a DAG), run them (parallel where no edge),
              then synthesize their summaries into this node's result

Children are wired as a DAG: a child waits on its dependencies' done-events,
so independent children run concurrently while dependents block. Threads are
cheap; the genuinely scarce resource (the GPU) is gated inside the LLM client.
This makes the model deadlock-free under recursion (a parent blocking on its
children never starves a fixed worker pool).
"""
from __future__ import annotations

import threading
import time

from . import prompts, store as st
from .context import estimate_tokens, entry_tokens, total_tokens, pack_entries


class Harness:
    def __init__(self, llm, store, bus=None, max_depth=3, max_subtasks=6,
                 output_reserve=1500, prompt_overhead=700):
        self.llm = llm
        self.store = store
        self.bus = bus
        self.max_depth = max_depth
        self.max_subtasks = max_subtasks
        # token budget for an aggregating call's INPUT = window minus room for the
        # model's reply and the fixed system/scaffolding overhead.
        self.num_ctx = getattr(llm, "num_ctx", 8192)
        self.output_reserve = output_reserve
        self.prompt_overhead = prompt_overhead

    def _budget(self) -> int:
        return max(1024, self.num_ctx - self.output_reserve - self.prompt_overhead)

    # ---- logging -----------------------------------------------------
    def _log(self, node_id, msg):
        if self.bus:
            self.bus.publish({"type": "log", "node": node_id, "msg": msg})

    # ---- entry point -------------------------------------------------
    def run(self, task: str, title: str = "root") -> dict:
        root = self.store.add_node(title=title, task=task, depth=0)
        self._log(root["id"], f"Run started: {title}")
        self.run_node(root["id"])
        result = self.store.get(root["id"])
        self._log(root["id"], f"Run finished: {result['status']}")
        if self.bus:
            self.bus.publish({"type": "run_done", "node": root["id"],
                              "status": result["status"]})
        return result

    # ---- the recursive operation ------------------------------------
    def run_node(self, node_id: str):
        node = self.store.get(node_id)

        # resumability: a node already finished in a prior run is reused as-is
        if node["status"] in (st.DONE, st.SKIPPED):
            return

        self.store.update(node_id, status=st.PLANNING, started_at=time.time())
        context = self._gather_context(node)

        try:
            plan = self._plan(node, context)
        except Exception as e:  # noqa: BLE001 - planning failed: degrade to a leaf
            self._log(node_id, f"planning failed ({e}); executing directly as a leaf.")
            self._run_leaf(node_id, context, f"planning failed: {e}")
            return

        subtasks = plan["subtasks"] if not plan["is_atomic"] else []
        if not subtasks:
            self._run_leaf(node_id, context, plan.get("reasoning"))
        else:
            self._run_branch(node_id, subtasks, context, plan.get("reasoning"))

    # ---- planning ----------------------------------------------------
    def _plan(self, node, context):
        depth = node["depth"]
        # hard guard: at the depth limit we never decompose
        if depth >= self.max_depth:
            return {"is_atomic": True, "reasoning": "depth limit reached", "subtasks": []}

        user = prompts.planner_user(
            node["task"], node.get("parent_goal", ""), depth, self.max_depth, context,
            max_subtasks=self.max_subtasks,
        )
        self._log(node["id"], "Planning…")
        obj, meta = self.llm.complete_json(
            prompts.PLANNER_SYSTEM, user, schema=prompts.PLAN_SCHEMA, node_id=node["id"],
            options={"temperature": 0},  # planning is a decision: keep it consistent
        )
        self.store.update(node["id"], add_tokens=meta)

        subtasks = obj.get("subtasks") or []
        # gate: a single subtask is not a decomposition — treat as atomic
        if len(subtasks) <= 1:
            obj["is_atomic"] = True
            subtasks = []
        if len(subtasks) > self.max_subtasks:
            # honest about dropping work rather than silently truncating
            self._log(node["id"],
                      f"NOTE: planner proposed {len(subtasks)} subtasks; capping to "
                      f"{self.max_subtasks} (raise --max-subtasks to keep all).")
            subtasks = subtasks[: self.max_subtasks]
        obj["subtasks"] = subtasks
        return obj

    # ---- leaf --------------------------------------------------------
    def _run_leaf(self, node_id, context, reasoning):
        node = self.store.get(node_id)
        self.store.update(node_id, kind="leaf", status=st.RUNNING, reasoning=reasoning)
        self._log(node_id, "Executing (leaf)…")
        user = prompts.worker_user(node["task"], node.get("parent_goal", ""), context)
        try:
            obj, meta = self.llm.complete_json(
                prompts.WORKER_SYSTEM, user, schema=prompts.WORK_SCHEMA, node_id=node_id
            )
        except Exception as e:  # noqa: BLE001 - record any failure on the node
            self._fail(node_id, f"leaf execution failed: {e}")
            return
        artifact = self.store.save_artifact(node_id, obj.get("output", ""))
        self.store.update(
            node_id,
            status=st.DONE,
            summary=obj.get("summary", "").strip(),
            artifact=artifact,
            add_tokens=meta,
            ended_at=time.time(),
        )
        self._log(node_id, "Leaf done.")

    # ---- branch ------------------------------------------------------
    def _run_branch(self, node_id, subtasks, context, reasoning):
        node = self.store.get(node_id)
        # map planner-local ids -> created child node ids
        local_to_global = {}
        child_records = []
        parent_goal = node["task"]
        for s in subtasks:
            local_to_global[s["id"]] = None  # placeholder so deps can resolve
        for s in subtasks:
            deps_local = [d for d in s.get("depends_on", []) if d in local_to_global and d != s["id"]]
            child = self.store.add_node(
                title=s.get("title", s["id"]),
                task=s.get("instruction", ""),
                parent_id=node_id,
                depth=node["depth"] + 1,
                deps=[],  # filled below once all ids exist
                parent_goal=parent_goal,
            )
            local_to_global[s["id"]] = child["id"]
            child_records.append((child, deps_local))

        # resolve dependency ids now that every child exists
        children = []
        for child, deps_local in child_records:
            deps_global = [local_to_global[d] for d in deps_local if local_to_global.get(d)]
            self.store.update(child["id"], deps=deps_global)
            children.append(self.store.get(child["id"]))

        # a bad plan can contain dependency cycles, which would deadlock the
        # event-based scheduler. Break them (drop back-edges) before running.
        self._break_cycles(children)
        children = [self.store.get(c["id"]) for c in children]

        self.store.update(node_id, kind="branch", status=st.WAITING, reasoning=reasoning)
        self._log(node_id, f"Decomposed into {len(children)} subtasks; scheduling DAG.")

        self._run_children_dag(children)

        # collect results; tolerate partial failure — synthesize from whatever
        # succeeded rather than sinking the whole branch over one bad subtask.
        done_children = [self.store.get(c["id"]) for c in children]
        ok = [c for c in done_children if c["status"] == st.DONE]
        bad = [c for c in done_children if c["status"] in (st.FAILED, st.SKIPPED)]
        if not ok:
            self._fail(node_id, "all subtasks failed or were skipped")
            return
        if bad:
            self._log(node_id, f"{len(bad)} subtask(s) failed/skipped; synthesizing "
                               f"from the {len(ok)} that succeeded.")
        self._synthesize(node_id, ok)

    def _break_cycles(self, children):
        """Drop dependency edges that form cycles (DFS back-edge removal) so the
        scheduler can't deadlock. Each child's 'deps' lists the ids it waits on."""
        ids = {c["id"] for c in children}
        deps = {c["id"]: [d for d in c["deps"] if d in ids] for c in children}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {i: WHITE for i in ids}

        def dfs(u):
            color[u] = GRAY
            for v in list(deps[u]):
                if color[v] == GRAY:           # back edge -> cycle; cut it
                    deps[u].remove(v)
                    self._log(u, f"dropped cyclic dependency on {v}.")
                elif color[v] == WHITE:
                    dfs(v)
            color[u] = BLACK

        for i in ids:
            if color[i] == WHITE:
                dfs(i)
        for c in children:
            if deps[c["id"]] != c["deps"]:
                self.store.update(c["id"], deps=deps[c["id"]])

    def _run_children_dag(self, children):
        """Run children respecting DAG edges; independent ones run in parallel.

        Each child has a done-Event. A child worker waits on its deps' events,
        skips if any dep failed, then runs and sets its own event.
        """
        done_events = {c["id"]: threading.Event() for c in children}

        def worker(child):
            cid = child["id"]
            for dep in child["deps"]:
                ev = done_events.get(dep)
                if ev:
                    ev.wait()
            # if a dependency failed/skipped, skip this child
            dead = [d for d in child["deps"]
                    if self.store.get(d)["status"] in (st.FAILED, st.SKIPPED)]
            if dead:
                self.store.update(cid, status=st.SKIPPED,
                                  error=f"dependency not satisfied: {dead}")
                self._log(cid, "Skipped (dependency failed).")
                done_events[cid].set()
                return
            try:
                self.run_node(cid)
            except Exception as e:  # noqa: BLE001
                self._fail(cid, f"unhandled error: {e}")
            finally:
                done_events[cid].set()

        threads = [threading.Thread(target=worker, args=(c,), daemon=True)
                   for c in children]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ---- synthesis ---------------------------------------------------
    # Children's FULL outputs are retrieved from the store (retrieval over
    # compression). Two regimes:
    #   fits the window  -> INTEGRATE: one LLM call rewrites them into a single
    #                       coherent result (dedup, smooth) — output fits.
    #   exceeds window   -> STITCH: concatenate the full sections losslessly (a
    #                       big deliverable can't be regenerated inside one
    #                       window without compressing it), and use the LLM only
    #                       for a framing intro + the compact bubble-up summary.
    def _synthesize(self, node_id, children):
        node = self.store.get(node_id)
        self.store.update(node_id, status=st.SYNTHESIZING)
        entries = []
        for c in children:
            content = self.store.load_artifact(c["id"]) or c.get("summary") or "(no output)"
            entries.append((c["title"], content))
        try:
            if total_tokens(entries) <= self._budget():
                self._log(node_id, "Synthesizing (integrate) …")
                output, summary = self._synth_call(node_id, node["task"], entries)
            else:
                output, summary = self._assemble(node_id, node, children, entries)
        except Exception as e:  # noqa: BLE001
            self._fail(node_id, f"synthesis failed: {e}")
            return
        artifact = self.store.save_artifact(node_id, output)
        self.store.update(
            node_id, status=st.DONE, summary=summary, artifact=artifact,
            ended_at=time.time(),
        )
        self._log(node_id, "Branch synthesized.")

    def _synth_call(self, node_id, task, entries):
        """One synthesis LLM call over entries that already fit the budget."""
        user = prompts.synth_user(task, entries)
        obj, meta = self.llm.complete_json(
            prompts.SYNTH_SYSTEM, user, schema=prompts.WORK_SCHEMA,
            node_id=node_id, options={"temperature": 0.2},
        )
        self.store.update(node_id, add_tokens=meta)
        return obj.get("output", "").strip(), obj.get("summary", "").strip()

    def _assemble(self, node_id, node, children, entries):
        """Stitch full sections (lossless) + generate intro & bubble-up summary."""
        self._log(node_id, f"Content {total_tokens(entries)}tok exceeds "
                           f"{self._budget()}tok window — assembling by stitching "
                           f"full sections (lossless).")
        compact = [(c["title"], c.get("summary") or "") for c in children]
        # the compact (title+summary) list is what the framing call reasons over;
        # compact it further only in the rare case even that overflows.
        if sum(estimate_tokens(t) + estimate_tokens(s) for t, s in compact) > self._budget():
            compact = [(t, self._compact_text(node_id, s, 80)) for t, s in compact]
        user = prompts.assemble_user(node["task"], compact)
        obj, meta = self.llm.complete_json(
            prompts.ASSEMBLE_SYSTEM, user, schema=prompts.ASSEMBLE_SCHEMA,
            node_id=node_id, options={"temperature": 0.2},
        )
        self.store.update(node_id, add_tokens=meta)
        intro = obj.get("intro", "").strip()
        summary = obj.get("summary", "").strip()
        # a deterministic Contents list built from the ACTUAL sections — ground
        # truth for the reader even if the generated intro over-claims.
        toc = "## Contents\n" + "\n".join(f"- {t}" for t, _ in entries)
        body = "\n\n".join(f"## {t}\n\n{c}" for t, c in entries)
        output = "\n\n".join(x for x in (intro, toc, body) if x).strip()
        return output, summary

    def _compact_text(self, node_id, text, target_tokens):
        """Summarize text down toward target_tokens, chunking if it itself
        exceeds the window (recursive map-reduce summarization)."""
        budget = self._budget()
        if estimate_tokens(text) > budget:
            chunk_chars = int(budget * 3.0)  # leave room for the compaction prompt
            chunks = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
            per = max(60, target_tokens // max(1, len(chunks)))
            parts = [self._compact_one(node_id, ch, per) for ch in chunks]
            joined = "\n".join(parts)
            if estimate_tokens(joined) > target_tokens and len(chunks) > 1:
                return self._compact_text(node_id, joined, target_tokens)
            return joined
        return self._compact_one(node_id, text, target_tokens)

    def _compact_one(self, node_id, text, target_tokens):
        words = max(40, int(target_tokens * 0.7))
        user = (f"Compress the following to about {words} words, preserving all key "
                f"facts, names, numbers, code and structure:\n\n{text}")
        out, meta = self.llm.chat(
            [{"role": "system", "content": prompts.COMPACT_SYSTEM},
             {"role": "user", "content": user}],
            options={"temperature": 0}, node_id=node_id,
        )
        self.store.update(node_id, add_tokens=meta)
        return out.strip()

    # ---- helpers -----------------------------------------------------
    def _gather_context(self, node):
        """Context = summaries of this node's completed dependency siblings,
        compacted if the combined size would eat too much of the window."""
        ctx = []
        for dep in node.get("deps", []):
            d = self.store.get(dep)
            if d.get("summary"):
                ctx.append(f"- [{d['title']}] {d['summary']}")
        text = "\n".join(ctx)
        cap = self._budget() // 3  # context is supporting material, not the bulk
        if estimate_tokens(text) > cap:
            self._log(node["id"], "Dependency context over budget; compacting.")
            text = self._compact_text(node["id"], text, cap)
        return text

    def _fail(self, node_id, msg):
        self.store.update(node_id, status=st.FAILED, error=msg, ended_at=time.time())
        self._log(node_id, f"FAILED: {msg}")
