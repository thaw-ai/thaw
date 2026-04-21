#!/usr/bin/env python3
"""
pr_review_langgraph — A LangGraph multi-agent workflow that fans out a PR to
four specialist reviewers in parallel. Demonstrates ChatThaw as a drop-in
replacement for any LangChain chat model, with the coalescer routing the
Send() fan-out through ForkPool for prefill skip.

What the demo shows
-------------------
A single reviewer node receives a PR diff + ~8K-token codebase context,
then Send()s to four specialist nodes (security / performance / style /
correctness). Each specialist generates a perspective-specific review
against the full shared context. With the --mode thaw path, the four
concurrent ainvoke calls are coalesced and routed through ForkPool; with
--mode baseline, each call is routed single-path through the same parent
vLLM (equivalent to ChatOpenAI against a vLLM OpenAI-compatible server).

Produces a timing JSON receipt suitable for side-by-side comparison.

Usage
-----
    # Baseline (fork disabled — same as N concurrent vllm-openai requests)
    python demos/pr_review_langgraph.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --mode baseline \\
        --json-out pr_review_baseline.json

    # thaw (fork path enabled)
    python demos/pr_review_langgraph.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --mode thaw \\
        --json-out pr_review_thaw.json

Sizing
------
Parent LLM at gpu_memory_utilization=0.25 + 4 pool workers at 0.15 each
fits on a single 80 GB H100. For smaller GPUs use --workers 2.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

try:
    from langgraph.types import Send  # langgraph >= 1.0
except ImportError:
    from langgraph.constants import Send  # older langgraph


# ---------------------------------------------------------------------------
# The four reviewer personas
# ---------------------------------------------------------------------------

# A single shared system prompt. The per-specialist instruction lives in the
# final human message — that's the point that DIFFERS across Sends, so the
# coalescer can detect the shared prefix (system prompt + context + diff) and
# fork from there.
SHARED_SYSTEM_PROMPT = (
    "You are a senior staff engineer reviewing a pull request. Your team asks "
    "you for targeted reviews from specific angles: security, performance, "
    "style, or correctness. When asked for one angle, stay in that lane and "
    "give 3-5 concrete findings. Be concise; flag real issues, skip bikeshedding."
)

SPECIALISTS: List[tuple[str, str]] = [
    (
        "security",
        "Give 3-5 security findings for this PR. Look for injection "
        "vulnerabilities, unsafe deserialization, missing auth/authorization "
        "checks, and data exposure.",
    ),
    (
        "performance",
        "Give 3-5 performance findings for this PR. Look for N+1 queries, "
        "unnecessary allocations, hot-path regressions, O(n^2) loops, and "
        "synchronous I/O on request paths.",
    ),
    (
        "style",
        "Give 3-5 style/readability findings for this PR. Look for naming, "
        "organization, overly clever code, and inconsistencies with the "
        "surrounding codebase patterns.",
    ),
    (
        "correctness",
        "Give 3-5 correctness findings for this PR. Look for bugs, off-by-one "
        "errors, unhandled edge cases, race conditions, and logic errors.",
    ),
]


# ---------------------------------------------------------------------------
# LangGraph state shape
# ---------------------------------------------------------------------------


def _append(existing: list, new: list) -> list:
    return (existing or []) + new


class ReviewState(TypedDict, total=False):
    diff: str
    context: str
    reviews: Annotated[list, _append]
    perspective: str
    perspective_prompt: str


def _fanout(state: ReviewState):
    return [
        Send(
            "specialist",
            {
                "diff": state["diff"],
                "context": state["context"],
                "perspective": name,
                "perspective_prompt": prompt,
            },
        )
        for name, prompt in SPECIALISTS
    ]


async def _specialist(state: ReviewState, config):
    llm = config["configurable"]["llm"]
    # Messages are structured so the shared prefix is as long as possible: the
    # system prompt and the "here's the context+diff" human message are IDENTICAL
    # across all four specialists. Only the final human message (the perspective
    # ask) differs. That's what lets the coalescer detect a shared prefix and
    # route through the fork path.
    messages = [
        SystemMessage(content=SHARED_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Codebase context:\n\n{state['context']}\n\n"
                f"---\n\nPull request diff:\n\n{state['diff']}"
            )
        ),
        HumanMessage(content=state["perspective_prompt"]),
    ]
    t0 = time.perf_counter()
    response = await llm.ainvoke(messages)
    t1 = time.perf_counter()
    return {
        "reviews": [
            {
                "perspective": state["perspective"],
                "review": response.content,
                "elapsed_s": t1 - t0,
            }
        ]
    }


def _build_graph():
    g = StateGraph(ReviewState)
    g.add_node("specialist", _specialist)
    g.add_conditional_edges(START, _fanout, ["specialist"])
    g.add_edge("specialist", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------


def _build_thaw_llm(args):
    from thaw_vllm.langgraph import ChatThaw

    return ChatThaw(
        model=args.model,
        fork_window_ms=args.fork_window_ms,
        fork_min_prefix_tokens=args.fork_min_prefix_tokens,
        workers=args.workers,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


def _build_baseline_llm(args):
    # Baseline: same underlying parent vLLM, but force every call through the
    # single-invoke path by setting an unreachable prefix threshold. Equivalent
    # to ChatOpenAI pointed at a vLLM OpenAI-compatible server on this box.
    from thaw_vllm.langgraph import ChatThaw

    return ChatThaw(
        model=args.model,
        fork_window_ms=args.fork_window_ms,
        fork_min_prefix_tokens=10 ** 9,
        workers=args.workers,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        worker_gpu_memory_utilization=args.worker_gpu_memory_utilization,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class RoundResult:
    reviews: list
    wall_s: float
    per_branch_s: list[float] = field(default_factory=list)


async def _run_round(graph, llm, diff: str, context: str) -> RoundResult:
    t0 = time.perf_counter()
    result = await graph.ainvoke(
        {"diff": diff, "context": context, "reviews": []},
        config={"configurable": {"llm": llm}},
    )
    t1 = time.perf_counter()
    return RoundResult(
        reviews=result["reviews"],
        wall_s=t1 - t0,
        per_branch_s=[r["elapsed_s"] for r in result["reviews"]],
    )


async def _run(args) -> dict:
    diff = _load_text(args.diff_file, _DEFAULT_DIFF)
    context = _load_text(args.context_file, _DEFAULT_CONTEXT)

    if args.mode == "thaw":
        llm = _build_thaw_llm(args)
    elif args.mode == "baseline":
        llm = _build_baseline_llm(args)
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    graph = _build_graph()

    rounds: list[RoundResult] = []
    for r in range(args.rounds):
        print(f"[round {r + 1}/{args.rounds}] fanning out to {len(SPECIALISTS)} specialists...", flush=True)
        rr = await _run_round(graph, llm, diff, context)
        rounds.append(rr)
        print(
            f"  wall={rr.wall_s:.2f}s "
            f"per-branch=[{', '.join(f'{x:.2f}s' for x in rr.per_branch_s)}]",
            flush=True,
        )

    # Cleanup
    if hasattr(llm, "close"):
        llm.close()

    wall_times = [r.wall_s for r in rounds]
    wall_times_sorted = sorted(wall_times)
    median = wall_times_sorted[len(wall_times_sorted) // 2]

    return {
        "mode": args.mode,
        "model": args.model,
        "rounds": args.rounds,
        "branches_per_round": len(SPECIALISTS),
        "wall_times_s": wall_times,
        "wall_median_s": median,
        "wall_min_s": min(wall_times),
        "wall_max_s": max(wall_times),
        "round_details": [
            {
                "round": i,
                "wall_s": r.wall_s,
                "per_branch_s": r.per_branch_s,
                "sample_review": r.reviews[0]["review"][:240] if r.reviews else "",
            }
            for i, r in enumerate(rounds)
        ],
    }


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _load_text(path: str | None, fallback: str) -> str:
    if not path:
        return fallback
    with open(path, "r") as f:
        return f.read()


# A realistic but self-contained PR — real-looking code rather than "foo/bar".
_DEFAULT_DIFF = """
diff --git a/app/api/users.py b/app/api/users.py
index 1234567..abcdefg 100644
--- a/app/api/users.py
+++ b/app/api/users.py
@@ -45,12 +45,22 @@ class UserEndpoint:
     async def get_user(self, user_id: str) -> User:
         \"\"\"Fetch a user by id.\"\"\"
-        user = await self.db.users.find_one({"id": user_id})
-        if user is None:
-            raise HTTPException(404, "user not found")
+        query = f"SELECT * FROM users WHERE id = '{user_id}'"
+        rows = await self.db.execute(query)
+        if not rows:
+            raise HTTPException(404, "user not found")
+        user = User.from_row(rows[0])
         return user

     async def list_users(self, org_id: str, limit: int = 100) -> list[User]:
-        users = await self.db.users.find({"org_id": org_id}).limit(limit)
-        return [u async for u in users]
+        results = []
+        for user_id in await self._user_ids_for_org(org_id):
+            user = await self.get_user(user_id)
+            results.append(user)
+            if len(results) >= limit:
+                break
+        return results
+
+    async def _user_ids_for_org(self, org_id: str) -> list[str]:
+        rows = await self.db.execute(
+            f"SELECT id FROM users WHERE org_id = '{org_id}'"
+        )
+        return [r["id"] for r in rows]
"""


_DEFAULT_CONTEXT = """\
# Repository context

This is a FastAPI-based backend serving a multi-tenant CRM. The codebase uses:
  - SQLAlchemy 2.x as the ORM, with the async engine.
  - A thin `db` wrapper that exposes .execute(sql) for raw SQL and a collection
    API (.users, .orgs, etc.) that uses parameterized SQLAlchemy constructs.
  - All user input is expected to go through Pydantic models before reaching
    the endpoints; route handlers receive already-validated inputs.

# Style conventions
  - Endpoint classes use async def methods named after HTTP verbs (get_user,
    list_users, etc.).
  - Private helpers are prefixed with underscore.
  - SQL should go through the collection API when possible; raw .execute() is
    reserved for analytics/reporting paths that need custom joins.

# Performance expectations
  - List endpoints should be paginated (max 500) and should not issue N+1 queries.
  - p99 latency target for user reads is 50ms.
  - The users table has 20M rows and an index on (org_id, id).

# Known incidents
  - Q3 2025: a SQL injection bug in the legacy /search endpoint exposed
    customer emails. Postmortem mandated: NO raw f-string SQL in new code.
  - Q1 2026: a N+1 query in the org-listing endpoint caused a 40s p99 under
    load. The fix was to batch queries through the collection API.

# Data model
class User(BaseModel):
    id: str
    org_id: str
    email: str
    created_at: datetime
    ... (additional fields elided for brevity)

class Organization(BaseModel):
    id: str
    name: str
    ... (additional fields elided for brevity)

# Team
Code owner for app/api/users.py: @carol (off this week).
Code owner for app/db/*: @mike.
Reviewers for user-facing changes should also check app/audit/ — every
endpoint change must land an audit log entry.
""" * 3  # triple to push context into the 6-8K token range


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["thaw", "baseline"], required=True)
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--diff-file", default=None)
    p.add_argument("--context-file", default=None)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    p.add_argument("--worker-gpu-memory-utilization", type=float, default=0.15)
    p.add_argument("--fork-window-ms", type=float, default=2.0)
    p.add_argument("--fork-min-prefix-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--json-out", default=None)
    return p.parse_args()


def main():
    args = _parse_args()
    result = asyncio.run(_run(args))

    print("\n=== summary ===")
    print(f"mode={result['mode']} model={result['model']}")
    print(f"rounds={result['rounds']} branches={result['branches_per_round']}")
    print(
        f"wall median={result['wall_median_s']:.2f}s  "
        f"min={result['wall_min_s']:.2f}s  max={result['wall_max_s']:.2f}s"
    )

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nreceipt written to {args.json_out}")


if __name__ == "__main__":
    main()
