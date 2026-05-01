#!/usr/bin/env python3
"""
baseline_naive_fork — what every agent framework does today.

Four divergent branches off a shared 4000-token trunk. No prefix
caching, no fork primitive. Each branch redoes prefill from scratch
on the same engine — the bottleneck thaw's fork_completions exists
to eliminate.

Pair this with demos/fork_pool_rl.py for an apples-to-apples
side-by-side comparison: same model, same pod, same trunk, same
branches, same generation length.

Usage
-----
    python demos/baseline_naive_fork.py
    python demos/baseline_naive_fork.py --json-out baseline.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _trunk_padding import pad_trunk  # noqa: E402


_BASE_TRUNK = (
    "You are helping the user design a thread-safe LRU cache with TTL "
    "expiry. The user wants to understand the trade-offs between several "
    "implementation strategies before committing to one. Walk through the "
    "relevant design dimensions carefully and then answer the specific "
    "question that follows."
)

TASK_BRANCHES = [
    "\n\nQuestion: Give a minimal Python implementation.",
    "\n\nQuestion: List three edge cases you'd test.",
    "\n\nQuestion: What's the worst-case time complexity?",
    "\n\nQuestion: How would you parallelize this?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--trunk-tokens", type=int, default=4000)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print(f"=== Baseline: vanilla vLLM, no fork primitive ===")
    print(f"    model = {args.model}")
    print(f"    enable_prefix_caching = False  (each branch redoes prefill)\n")

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,  # the whole point of the baseline
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
    )
    load_s = time.perf_counter() - t0
    print(f"engine loaded in {load_s:.2f}s\n")

    tokenizer = llm.get_tokenizer()
    trunk, trunk_tokens = pad_trunk(
        tokenizer, _BASE_TRUNK, args.trunk_tokens, kind="coding",
    )
    print(f"trunk: {trunk_tokens} tokens\n")

    sp = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.7, seed=args.seed,
    )

    print(f"=== 4 branches sequentially (no shared cache) ===")
    per_branch: list[dict] = []
    total = 0.0
    for i, b in enumerate(TASK_BRANCHES):
        t0 = time.perf_counter()
        out = llm.generate(trunk + b, sp)
        dt = time.perf_counter() - t0
        total += dt
        text = out[0].outputs[0].text
        print(f"  branch {i}: {dt:6.2f}s  {text[:60]!r}")
        per_branch.append({
            "branch": i,
            "elapsed_s": round(dt, 3),
            "text_preview": text[:120],
        })

    per = total / len(TASK_BRANCHES)
    print(f"\n=== Summary ===")
    print(f"  TOTAL:      {total:6.2f}s for {len(TASK_BRANCHES)} branches")
    print(f"  PER BRANCH: {per:6.2f}s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "demo": "baseline_naive_fork",
            "model": args.model,
            "trunk_tokens": trunk_tokens,
            "load_s": round(load_s, 2),
            "total_s": round(total, 2),
            "per_branch_s": round(per, 2),
            "branches": per_branch,
        }, indent=2))
        print(f"\nWrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
