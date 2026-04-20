#!/usr/bin/env python3
"""
parallel_agents — N coding agents explore N approaches in parallel.

This is the coding-agent cousin of the RL rollout demo. Same primitive
(thaw.fork), different pitch:

    "One agent was mid-reasoning about how to implement a function.
     Fork into 8 parallel approaches. Let each finish. Pick the winner.
     In less time than a single cold start."

The loop
--------
  1. Load Llama-3-8B with enable_prefix_caching=True.
  2. Build a ~300-token reasoning trunk on a coding problem:
       "Implement an LRU cache with TTL expiry in Python."
  3. thaw_vllm.fork(llm) — snapshot at the pivot.
  4. Fork into 8 branches. Each branch is nudged toward a different data
     structure (OrderedDict, heapq, linked list, etc). Run them in
     parallel via fork_completions(workers=K).
  5. Extract each candidate's code, compile it, run it against a small
     test battery, compute pass rate.
  6. Print a ranked table + the winning code.

Runs on one H100 80GB in under 5 minutes. Shares the primitive with
rl_rollout_simulator.py — same fork API, different reward function.

Usage
-----
    python demos/parallel_agents.py
    python demos/parallel_agents.py --branches 4 --workers 2
"""

import argparse
import ast
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


# ─────────────────────────────────────────────────────────────────────
# The problem, the trunk, the branches
# ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a senior Python engineer. When asked to implement something:
1. Briefly think through the approach (2-3 sentences).
2. Write complete, runnable code in a single fenced code block.
3. Never include tests or examples — just the implementation.
4. Always export the class by the exact name requested."""

TASK = """\
Implement a thread-safe LRU cache with TTL (time-to-live) expiry. The class
must be named `LRUCache` and expose these methods:

    __init__(self, max_size: int, ttl_seconds: float)
    get(self, key) -> value | None      # None if missing or expired
    put(self, key, value) -> None       # evicts LRU if over capacity
    __len__(self) -> int                # number of unexpired entries
    clear(self) -> None

Rules:
  - Expired entries should not count toward max_size and should not
    be returned by get().
  - put() on an existing key updates the value and resets the TTL.
  - The implementation must work correctly under concurrent get/put
    from multiple threads (use threading.RLock internally)."""

TRUNK_PROMPT = (
    f"<|system|>\n{SYSTEM_PROMPT}\n"
    f"<|user|>\n{TASK}\n"
    f"<|assistant|>\n"
    f"Let me think through the design before writing code.\n\n"
    f"The key design decisions here are:\n"
)

# 8 different implementation hints. These become branch prompts appended
# to the trunk — each candidate is nudged toward a different approach.
BRANCH_HINTS = [
    "I'll use `collections.OrderedDict` and rely on move_to_end() for the LRU policy, with a separate dict mapping key to expiry timestamp.",
    "I'll use a dict plus a manually-maintained doubly-linked list of nodes for O(1) LRU reordering. TTL via per-node deadline.",
    "I'll use a dict for fast lookup plus `heapq` for TTL expiry (min-heap keyed by deadline). LRU via a secondary deque.",
    "I'll subclass `dict` and override __getitem__/__setitem__, storing (value, deadline) tuples. Iteration order gives me LRU.",
    "I'll compose using `collections.OrderedDict` but expire lazily: only prune on access, and only up to max_size.",
    "I'll use two parallel structures: one OrderedDict for LRU order and a dict mapping key to deadline. Purge lazily.",
    "I'll implement with a dict-of-nodes + intrusive linked list, and a second min-heap for TTL. Both updated under the same RLock.",
    "I'll lean on `functools` patterns — a dict with (value, deadline), and manual reinsertion on get() to maintain insertion-order as LRU order.",
]


# ─────────────────────────────────────────────────────────────────────
# Extract code + run tests
# ─────────────────────────────────────────────────────────────────────


_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\n(.*?)```", re.DOTALL | re.IGNORECASE
)


def extract_code(text: str):
    """Pull the first python code block from a model output."""
    m = _CODE_FENCE_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


TEST_CASES = [
    # (name, exec-inside-fresh-LRUCache → expected)
    ("basic put/get",
     "c = LRUCache(3, 60); c.put('a', 1); return c.get('a')", 1),
    ("missing key returns None",
     "c = LRUCache(3, 60); return c.get('nope')", None),
    ("LRU eviction",
     "c = LRUCache(2, 60); c.put('a', 1); c.put('b', 2); c.put('c', 3); return c.get('a')",
     None),
    ("put updates value",
     "c = LRUCache(3, 60); c.put('a', 1); c.put('a', 2); return c.get('a')", 2),
    ("len reflects stored",
     "c = LRUCache(5, 60); c.put('a', 1); c.put('b', 2); return len(c)", 2),
    ("clear empties",
     "c = LRUCache(3, 60); c.put('a', 1); c.clear(); return c.get('a')", None),
    ("TTL expiry",
     "import time; c = LRUCache(3, 0.05); c.put('a', 1); time.sleep(0.1); return c.get('a')",
     None),
    ("access moves to MRU",
     "c = LRUCache(2, 60); c.put('a', 1); c.put('b', 2); c.get('a'); c.put('c', 3); return c.get('b')",
     None),
]


def run_tests(code: str, timeout_s: float = 5.0):
    """Exec the candidate code in an isolated namespace, run each test case,
    return (passed, total, details).

    Safety note: this execs model-generated code. Running locally on a
    trusted machine is fine. If you're running this demo in a shared
    environment, wrap the exec in a subprocess sandbox.
    """
    ns = {"__builtins__": __builtins__}
    try:
        exec(code, ns)
    except Exception as e:
        return 0, len(TEST_CASES), [("compile/load", f"{type(e).__name__}: {e}")]

    if "LRUCache" not in ns:
        return 0, len(TEST_CASES), [("export", "LRUCache class not defined")]

    passed = 0
    details = []
    for name, expr, expected in TEST_CASES:
        # Wrap each test expression in a def so local 'c' doesn't collide
        # across tests, and so 'return' works inside exec.
        test_src = "def _t():\n"
        for line in expr.split(";"):
            test_src += "    " + line.strip() + "\n"
        try:
            test_ns = dict(ns)
            exec(test_src, test_ns)
            got = test_ns["_t"]()
            if got == expected:
                passed += 1
                details.append((name, "pass"))
            else:
                details.append((name, f"got {got!r}, expected {expected!r}"))
        except Exception as e:
            details.append((name, f"{type(e).__name__}: {e}"))
    return passed, len(TEST_CASES), details


def score_candidate(code: str | None):
    if not code:
        return {"pass_rate": 0.0, "passed": 0, "total": len(TEST_CASES),
                "has_code": False, "details": []}
    # Syntax-check first so we don't exec obviously broken code.
    try:
        ast.parse(code)
    except SyntaxError as e:
        return {"pass_rate": 0.0, "passed": 0, "total": len(TEST_CASES),
                "has_code": True, "syntax_error": str(e), "details": []}
    passed, total, details = run_tests(code)
    return {
        "pass_rate": passed / total if total else 0.0,
        "passed": passed,
        "total": total,
        "has_code": True,
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────


def run_branches_fork(llm, full_trunk, branches, workers, max_tokens, seed=42):
    """Fork + subprocess workers path — the thaw lane."""
    import thaw_vllm
    from vllm import SamplingParams

    # Prime the parent's KV cache with the trunk.
    llm.generate([full_trunk], SamplingParams(temperature=0.0, max_tokens=1))

    branch_prompts = [full_trunk + "\n" + BRANCH_HINTS[i % len(BRANCH_HINTS)]
                      for i in range(branches)]
    sp = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=seed,
    )

    t0 = time.perf_counter()
    with thaw_vllm.fork(llm, include_weights=True) as handle:
        fork_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        outputs = thaw_vllm.fork_completions(
            llm, branch_prompts, sp, workers=workers, handle=handle,
        )
        workers_s = time.perf_counter() - t1
    elapsed = time.perf_counter() - t0

    candidates = []
    for i, out in enumerate(outputs):
        code = extract_code(out.text)
        candidates.append({
            "branch": i,
            "hint": BRANCH_HINTS[i % len(BRANCH_HINTS)][:80],
            "text": out.text,
            "code": code,
            "worker": out.worker_index,
            "score": score_candidate(code),
        })
    return candidates, elapsed, fork_s, workers_s


def run_branches_native(llm, full_trunk, branches, max_tokens, seed=42):
    """vLLM's continuous batching does it natively — the baseline."""
    from vllm import SamplingParams

    branch_prompts = [full_trunk + "\n" + BRANCH_HINTS[i % len(BRANCH_HINTS)]
                      for i in range(branches)]
    sp = SamplingParams(
        temperature=0.7, top_p=0.95, max_tokens=max_tokens, seed=seed,
    )
    t0 = time.perf_counter()
    outs = llm.generate(branch_prompts, sp)
    elapsed = time.perf_counter() - t0

    candidates = []
    for i, out in enumerate(outs):
        text = out.outputs[0].text if out.outputs else ""
        code = extract_code(text)
        candidates.append({
            "branch": i,
            "hint": BRANCH_HINTS[i % len(BRANCH_HINTS)][:80],
            "text": text,
            "code": code,
            "worker": 0,
            "score": score_candidate(code),
        })
    return candidates, elapsed


# ─────────────────────────────────────────────────────────────────────
# Presentation
# ─────────────────────────────────────────────────────────────────────


def print_banner(msg, char="━", width=71):
    print()
    print(char * width)
    print(f"  {msg}")
    print(char * width)


def print_results(candidates, mode_label, elapsed):
    print_banner(f"{mode_label} — ranked results", char="─")
    # Sort by pass rate desc, then by has_code.
    ranked = sorted(
        enumerate(candidates),
        key=lambda x: (
            -x[1]["score"]["pass_rate"],
            -int(x[1]["score"]["has_code"]),
            x[0],
        ),
    )
    print(f"  Wall-clock: {elapsed:.2f}s  "
          f"({len(candidates)} candidates, "
          f"{len(candidates)/elapsed:.2f}/s)")
    print()
    print(f"  {'Rank':<6}{'Branch':<8}{'Pass':<10}{'Worker':<8}{'Hint'}")
    print(f"  {'────':<6}{'──────':<8}{'────':<10}{'──────':<8}{'────'}")
    for rank, (i, c) in enumerate(ranked, 1):
        pr = c["score"]["pass_rate"]
        pass_s = f"{c['score']['passed']}/{c['score']['total']} ({pr:.0%})"
        hint = c["hint"]
        print(f"  #{rank:<5}{i:<8}{pass_s:<10}{c['worker']:<8}{hint}")
    return ranked


def print_winner(ranked, candidates):
    if not ranked:
        return
    idx, winner = ranked[0]
    if not winner["score"]["has_code"]:
        print_banner("No candidate produced runnable code.", char="─")
        return
    print_banner(f"Winner: branch {idx} "
                 f"({winner['score']['passed']}/{winner['score']['total']} passed)",
                 char="─")
    if winner["code"]:
        code_lines = winner["code"].split("\n")
        # Truncate very long implementations for demo output.
        if len(code_lines) > 40:
            code_lines = code_lines[:40] + ["    # ...truncated..."]
        for line in code_lines:
            print(f"  {line}")
    print()
    print("  Test details:")
    for name, outcome in winner["score"]["details"]:
        marker = "✓" if outcome == "pass" else "✗"
        print(f"    {marker} {name}: {outcome}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--branches", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--skip-fork", action="store_true",
                    help="Only run the native-batch path.")
    ap.add_argument("--skip-native", action="store_true",
                    help="Only run the fork-subprocess path.")
    args = ap.parse_args()

    print_banner("thaw.fork parallel coding agents")
    print(f"Model:     {args.model}")
    print(f"Branches:  {args.branches}")
    print(f"Workers:   {args.workers}")
    print(f"Task:      Implement LRUCache with TTL + thread safety")

    # ── Load the parent engine ──
    print_banner("[1/3] Loading parent engine + building reasoning trunk",
                 char="─")
    from vllm import LLM, SamplingParams
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        dtype="float16",
        enable_prefix_caching=True,
        gpu_memory_utilization=0.90,
    )
    print(f"  parent ready in {time.perf_counter() - t0:.2f}s")

    sp_trunk = SamplingParams(temperature=0.0, max_tokens=300, seed=42)
    t0 = time.perf_counter()
    trunk_out = llm.generate([TRUNK_PROMPT], sp_trunk)
    trunk_s = time.perf_counter() - t0
    trunk_text = trunk_out[0].outputs[0].text if trunk_out[0].outputs else ""
    print(f"  trunk generated in {trunk_s:.2f}s "
          f"({len(trunk_out[0].outputs[0].token_ids)} tokens)")
    full_trunk = TRUNK_PROMPT + trunk_text

    # ── Native batch baseline ──
    if not args.skip_native:
        print_banner("[2/3] Native batch — vLLM's continuous batching", char="─")
        cands_n, elapsed_n = run_branches_native(
            llm, full_trunk, args.branches, args.max_tokens,
        )
        ranked_n = print_results(cands_n, "Native batch", elapsed_n)

    # ── Fork + workers ──
    if not args.skip_fork:
        print_banner(f"[3/3] Fork + {args.workers} subprocess workers", char="─")
        cands_f, elapsed_f, fork_s, workers_s = run_branches_fork(
            llm, full_trunk, args.branches, args.workers, args.max_tokens,
        )
        print(f"  fork freeze: {fork_s:.2f}s  |  workers end-to-end: {workers_s:.2f}s")
        ranked_f = print_results(cands_f, "Fork + workers", elapsed_f)
        print_winner(ranked_f, cands_f)

    print_banner("Same primitive, two lanes", char="━")
    print("  In-engine: vLLM's prefix cache does it (fast, one process).")
    print("  Cross-process: thaw.fork does it (process boundary, cluster-scale).")
    print("  The primitive is the same. Pick the lane by where the work has to run.")


if __name__ == "__main__":
    main()
