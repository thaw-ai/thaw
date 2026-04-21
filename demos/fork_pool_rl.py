#!/usr/bin/env python3
"""
fork_pool_rl — RL-shape validation of the ForkPool amortization.

Motivation
----------
``fork_completions(workers=N)`` spawns fresh subprocess workers for every
call. Each call pays the full vLLM cold-boot (~340s on H100). That's
fine for a one-off demo but unusable for RL: a real training loop forks
hundreds of times per epoch. ForkPool boots N workers once at startup
and hot-swaps snapshots into them, turning per-fork latency from minutes
into seconds.

This demo runs M sequential "rollout rounds" — each round is a fresh
fork + completions against a warm pool. We expect:

  - Round 0    → dominated by init_pool cost (paid once, outside loop).
  - Rounds 1+  → ~few seconds each (handle write + pipelined DMA + gen).

If the first round is ~5s and the tenth round is ~5s, ForkPool works.
If round 0 is 340s and round 1 is also 340s, something is wrong.

Usage
-----
    python demos/fork_pool_rl.py \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --workers 1 --rounds 5 --branches-per-round 4 \\
        --trunk-tokens 4000 --max-tokens 64 \\
        --json-out receipt.json

Sizing
------
Each worker holds a full dummy-weight vLLM engine resident on GPU. For
8B fp16 with gpu_memory_utilization=0.35 and max_model_len=8192 each
worker occupies ~20-25 GB. An 80 GB H100 fits parent (0.25) + one
worker (0.35) comfortably. For ``--workers ≥ 2`` use 2×H100 (TP=2) or
B200/H200.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
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
    "\n\nQuestion: What's a common bug pattern here?",
    "\n\nQuestion: How would you benchmark this?",
    "\n\nQuestion: What's the memory footprint?",
    "\n\nQuestion: What's the difference from a naive version?",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--workers", type=int, default=1,
                    help="Pool size. Each worker holds a full vLLM engine "
                         "on GPU. 80 GB H100 fits parent + 1 worker.")
    ap.add_argument("--rounds", type=int, default=5,
                    help="Number of fork_completions rounds. First round "
                         "proves hot-path; subsequent rounds prove "
                         "repeatability.")
    ap.add_argument("--branches-per-round", type=int, default=4,
                    help="Prompts per round (spread round-robin over "
                         "--workers).")
    ap.add_argument("--trunk-tokens", type=int, default=4000,
                    help="Shared prefix length to warm in the parent. "
                         "4K is a reasonable agent trunk; 8K+ exercises "
                         "prefix cache at scale.")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25,
                    help="Parent engine GPU fraction. 0.25 leaves room "
                         "for one worker at 0.55 on 80 GB.")
    ap.add_argument("--worker-gpu-memory", type=float, default=0.55,
                    help="Each worker's gpu_memory_utilization.")
    ap.add_argument("--worker-max-model-len", type=int, default=12288)
    ap.add_argument("--enforce-eager", action="store_true", default=True)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.branches_per_round > len(TASK_BRANCHES):
        raise SystemExit(
            f"--branches-per-round max is {len(TASK_BRANCHES)}"
        )

    from vllm import LLM, SamplingParams
    import thaw_vllm
    from thaw_vllm import ForkPool

    # ── Parent engine + trunk ───────────────────────────────────────
    print(f"=== Parent engine load: {args.model} ===")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="float16",
    )
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.2f}s")

    tokenizer = llm.get_tokenizer()
    trunk, trunk_tokens = pad_trunk(
        tokenizer, _BASE_TRUNK, args.trunk_tokens, kind="coding",
    )
    print(f"  trunk: {trunk_tokens} tokens")

    print("\n=== Warm trunk (populate parent prefix cache) ===")
    t0 = time.perf_counter()
    _ = llm.generate(
        trunk, SamplingParams(max_tokens=16, temperature=0.0, seed=args.seed),
    )
    trunk_warm_s = time.perf_counter() - t0
    print(f"  warmed in {trunk_warm_s:.2f}s")

    # ── ForkPool init (one-time cost) ──────────────────────────────
    # preload_weights=True: workers boot with real model weights
    # (load_format="auto"). Each fork then only snapshots + hydrates KV
    # cache, not the 16 GB of weights — appropriate for RL rollouts
    # where the policy is fixed across a round.
    print(f"\n=== ForkPool.init_pool(workers={args.workers}, "
          f"preload_weights=True) ===")
    pool = ForkPool()
    t0 = time.perf_counter()
    pool.init_pool(
        model=args.model,
        workers=args.workers,
        preload_weights=True,
        gpu_memory_utilization=args.worker_gpu_memory,
        max_model_len=args.worker_max_model_len,
        enforce_eager=True,
        dtype="float16",
    )
    init_pool_s = time.perf_counter() - t0
    per_worker_boot = [round(s.boot_s, 2) for s in pool.slots]
    print(f"  pool ready in {init_pool_s:.1f}s "
          f"(per-worker boot: {per_worker_boot})")

    # ── Rounds ─────────────────────────────────────────────────────
    print(f"\n=== {args.rounds} rounds × "
          f"{args.branches_per_round} branches ===")
    per_round: list[dict] = []
    for r in range(args.rounds):
        branches = TASK_BRANCHES[: args.branches_per_round]
        prompts = [trunk + b for b in branches]
        sp = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=0.7,
            seed=args.seed + r,
        )
        t0 = time.perf_counter()
        results = thaw_vllm.fork_completions(llm, prompts, sp, pool=pool)
        round_s = time.perf_counter() - t0

        nonempty = sum(1 for x in results if x.text)
        diverge = len(set(x.text for x in results)) == len(results)
        print(
            f"  round {r}: {round_s:6.2f}s  "
            f"nonempty={nonempty}/{len(results)}  diverge={diverge}  "
            f"sample={results[0].text[:60]!r}"
        )
        per_round.append(
            {
                "round": r,
                "elapsed_s": round(round_s, 3),
                "branches": len(results),
                "nonempty": nonempty,
                "all_diverge": diverge,
                "sample_worker_indices": [x.worker_index for x in results],
            }
        )

    pool.close()

    # ── Summary + receipt ──────────────────────────────────────────
    round_times = [pr["elapsed_s"] for pr in per_round]
    first_round_s = round_times[0] if round_times else 0.0
    tail_times = round_times[1:] if len(round_times) > 1 else round_times
    median_round_s = statistics.median(tail_times) if tail_times else 0.0
    mean_round_s = (sum(tail_times) / len(tail_times)) if tail_times else 0.0

    amortized_s = init_pool_s + sum(round_times)
    naive_s_estimate = (init_pool_s / max(args.workers, 1)) * args.rounds
    # Naive = each round pays one cold-boot; we estimate by dividing
    # init_pool cost by workers (serial boot pays one per worker) —
    # this is a floor on the cost per fresh worker. Real one-shot
    # subprocess cost was 340s on H100 (measured 2026-04-20).

    print("\n=== Summary ===")
    print(
        json.dumps(
            {
                "init_pool_s": round(init_pool_s, 2),
                "first_round_s": round(first_round_s, 3),
                "median_round_s": round(median_round_s, 3),
                "mean_round_s": round(mean_round_s, 3),
                "total_loop_s": round(sum(round_times), 3),
                "total_with_init_s": round(amortized_s, 2),
                "rounds": args.rounds,
                "workers": args.workers,
            },
            indent=2,
        )
    )

    if args.json_out:
        from _receipt import build_receipt, write_receipt

        receipt = build_receipt(
            demo="fork_pool_rl",
            model=args.model,
            trunk_tokens=trunk_tokens,
            timings={
                "load_s": load_s,
                "trunk_warm_s": trunk_warm_s,
                "init_pool_s": init_pool_s,
                "first_round_s": first_round_s,
                "median_round_s": median_round_s,
                "mean_round_s": mean_round_s,
                "total_loop_s": sum(round_times),
            },
            modes={
                "fork_pool": {
                    "workers": args.workers,
                    "rounds": args.rounds,
                    "branches_per_round": args.branches_per_round,
                    "per_round": per_round,
                    "per_worker_boot_s": per_worker_boot,
                },
            },
            checks={
                "all_rounds_nonempty": all(
                    pr["nonempty"] == pr["branches"] for pr in per_round
                ),
                "all_rounds_diverge": all(
                    pr["all_diverge"] for pr in per_round
                ),
                "amortization_visible": first_round_s <= init_pool_s
                    and median_round_s < first_round_s * 2
                    if first_round_s > 0
                    else None,
            },
            samples=[
                {
                    "round": 0,
                    "branch": i,
                    "text_preview": r.text,
                }
                for i, r in enumerate(results[: min(3, len(results))])
            ],
            extra={
                "workers": args.workers,
                "rounds": args.rounds,
                "branches_per_round": args.branches_per_round,
                "trunk_tokens": trunk_tokens,
                "max_tokens": args.max_tokens,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "worker_gpu_memory": args.worker_gpu_memory,
                "worker_max_model_len": args.worker_max_model_len,
                "amortization_note": (
                    "First subprocess-worker demo, pre-ForkPool, measured "
                    "~340s per one-shot fork on H100 80GB 2026-04-20. "
                    "ForkPool trades that for a one-time init_pool cost."
                ),
            },
        )
        write_receipt(args.json_out, receipt)
        print(f"\nWrote receipt to {args.json_out}")

    print("\n=== OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
