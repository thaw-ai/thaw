#!/usr/bin/env python3
"""
fork_smoke_test — minimal GPU validation for the fork primitive.

This is the cheapest real-GPU proof that ``thaw_vllm.fork`` works:
one parent engine, same-process fork_completions, 4 divergent branches.
Runs on any 16+ GB GPU in ~30s (plus model download).

Unlike parallel_agents.py and rl_rollout_simulator.py, this does NOT
spawn subprocess workers, so it fits on a 48 GB A6000 (where
parent + workers would OOM). Use this for CI or cheap validation.

Usage:
    python demos/fork_smoke_test.py --json-out receipt.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")


TRUNK = (
    "Let me think step by step about how to implement a thread-safe LRU "
    "cache with TTL expiry. First, I need a data structure that supports "
    "O(1) lookup and O(1) LRU eviction. Then I need TTL tracking. Then I "
    "need thread safety."
)

BRANCHES = [
    " Approach 1: use OrderedDict.",
    " Approach 2: use a doubly-linked list.",
    " Approach 3: use heapq for TTL.",
    " Approach 4: subclass dict.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--json-out", default=None,
                    help="Write sanitized receipt JSON to this path.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import thaw_vllm

    print("=== Load LLM ===")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
    )
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.2f}s")

    print("\n=== Warm trunk (populates prefix cache) ===")
    t0 = time.perf_counter()
    _ = llm.generate(TRUNK, SamplingParams(max_tokens=32, temperature=0.0))
    trunk_s = time.perf_counter() - t0
    print(f"  warmed in {trunk_s:.2f}s ({len(TRUNK)} chars)")

    print("\n=== thaw_vllm.fork(llm) ===")
    t0 = time.perf_counter()
    handle = thaw_vllm.fork(llm)
    fork_s = time.perf_counter() - t0
    print(f"  fork handle in {fork_s:.3f}s")
    print(f"  handle.model_id: {handle.model_id}")

    print("\n=== fork_completions(workers=None) — same-process ===")
    t0 = time.perf_counter()
    results = thaw_vllm.fork_completions(
        llm,
        prompts=[TRUNK + b for b in BRANCHES],
        sampling_params=SamplingParams(
            max_tokens=args.max_tokens, temperature=0.7, seed=42,
        ),
        handle=handle,
    )
    fc_s = time.perf_counter() - t0
    print(f"  fork_completions done in {fc_s:.2f}s ({len(results)} results)")

    print("\n=== Results ===")
    for i, r in enumerate(results):
        print(f"  [{i}] mode={r.mode} worker={r.worker_index} "
              f"elapsed={r.elapsed_s:.2f}s")
        print(f"      text: {r.text[:120]!r}")

    all_nonempty = all(len(r.text) > 0 for r in results)
    all_diverge = len(set(r.text for r in results)) == len(BRANCHES)

    print("\n=== Summary ===")
    print(json.dumps({
        "model": args.model,
        "trunk_chars": len(TRUNK),
        "branches": len(BRANCHES),
        "fork_handle_s": round(fork_s, 3),
        "fork_completions_s": round(fc_s, 3),
        "all_nonempty": all_nonempty,
        "all_diverge": all_diverge,
    }, indent=2))

    if args.json_out:
        from _receipt import build_receipt, write_receipt

        tok = llm.get_tokenizer()
        trunk_tokens = len(tok.encode(TRUNK, add_special_tokens=False))

        receipt = build_receipt(
            demo="fork_smoke_test",
            model=args.model,
            trunk_tokens=trunk_tokens,
            timings={
                "load_s": load_s,
                "trunk_warm_s": trunk_s,
                "fork_handle_s": fork_s,
                "fork_completions_s": fc_s,
            },
            modes={
                "same_process_fork": {
                    "ran": True,
                    "elapsed_s": round(fc_s, 3),
                    "count": len(results),
                    "per_branch_elapsed_s": [round(r.elapsed_s, 3) for r in results],
                },
            },
            checks={
                "all_nonempty": all_nonempty,
                "all_diverge": all_diverge,
                "prefix_cache_enabled": True,
            },
            samples=[
                {
                    "mode": "same_process",
                    "branch": i,
                    "prompt_suffix": BRANCHES[i],
                    "text_preview": r.text,
                }
                for i, r in enumerate(results)
            ],
            extra={
                "branches": len(BRANCHES),
                "max_tokens_per_branch": args.max_tokens,
                "max_model_len": args.max_model_len,
            },
        )
        write_receipt(args.json_out, receipt)
        print(f"\nWrote receipt to {args.json_out}")

    handle.close()
    print("\n=== OK ===")


if __name__ == "__main__":
    main()
