#!/usr/bin/env python3
"""
rl_training_step — one PPO/GRPO training step's rollout-collection phase.

What this script does
---------------------
Takes K math problems with known answers. For each problem, forks N
divergent rollouts via ``thaw_vllm.fork_completions(pool=pool)``. Scores
each rollout against ground truth. Reports pass@1, pass@N, and the
group-relative advantages a GRPO optimizer would consume.

This is the *rollout-collection* phase of a PPO/GRPO training step —
the part where the policy generates candidate completions and the
reward model scores them. It's ~80% of a training step's wall-clock
budget on most setups. The remaining 20% is the optimizer/weight-sync
step, which this demo deliberately does not cover: vLLM ↔ PyTorch
weight sync is a separate build.

Why this shape
--------------
HuggingFace's 2026 async-RL survey flagged KV pivot resampling as
unsupported by any current async-RL library. ``fork_completions`` is
that primitive. This script is the minimum-viable reproducer of how it
plugs into a real training loop:

    trainer.step():
        prompts  = dataset.sample(K)
        rollouts = thaw_vllm.fork_completions(llm, prompts, sp, pool=pool)
        rewards  = score(rollouts, ground_truth)
        adv      = groupwise_advantage(rewards)
        # ↑ this script produces everything up to here
        loss     = -(log_prob(rollouts) * adv).mean()
        loss.backward(); optimizer.step()
        sync_weights(llm, model)

Usage
-----
    pip install thaw-vllm
    python demos/rl_training_step.py
    python demos/rl_training_step.py --rollouts-per-problem 16 --json-out r.json
    python demos/rl_training_step.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

# Must be set before importing vLLM — fork path requires V1 inproc.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────────────
# The dataset — GSM8K-style word problems with integer ground truth.
# Kept inline so the demo runs with zero dataset setup.
# ─────────────────────────────────────────────────────────────────────

PROBLEMS: list[dict] = [
    {"q": "A bakery sells 12 cupcakes per tray. They baked 9 trays today "
          "and sold 87 cupcakes. How many cupcakes are left?",
     "answer": 21},
    {"q": "Tom reads 18 pages per day. If his book has 252 pages, how many "
          "days will it take him to finish?",
     "answer": 14},
    {"q": "A store marks up its cost by 40 percent. If an item costs $50 "
          "to make, what does it sell for (in dollars)?",
     "answer": 70},
    {"q": "A train travels at 60 mph for 2 hours, then at 45 mph for 3 "
          "hours. How many miles total?",
     "answer": 255},
    {"q": "Sarah has 3 times as many apples as Joe. Joe has 7 apples. "
          "How many apples do they have together?",
     "answer": 28},
    {"q": "A tank holds 180 gallons when full and is currently 2/3 full. "
          "How many gallons are in it right now?",
     "answer": 120},
    {"q": "A phone was originally $400 and is now 25% off. What is the "
          "sale price in dollars?",
     "answer": 300},
    {"q": "15 students split 120 candies equally. How many candies does "
          "each student get?",
     "answer": 8},
]

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your work. End your response with a line in the exact form: "
    "'Answer: <number>'."
)


# ─────────────────────────────────────────────────────────────────────
# Reward + advantages
# ─────────────────────────────────────────────────────────────────────

_ANSWER_RE = re.compile(
    r"Answer\s*:?\s*=?\s*\$?\s*([-+]?[0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

def score(text: str, ground_truth: float) -> float:
    """Return 1.0 if the rollout's final answer matches ground truth."""
    m = _ANSWER_RE.search(text or "")
    val: float | None = None
    if m:
        try:
            val = float(m.group(1))
        except ValueError:
            val = None
    if val is None:
        # Fallback — use the last number mentioned anywhere.
        nums = re.findall(r"[-+]?[0-9]+(?:\.[0-9]+)?", text or "")
        if not nums:
            return 0.0
        try:
            val = float(nums[-1])
        except ValueError:
            return 0.0
    return 1.0 if abs(val - ground_truth) < 0.01 else 0.0


def groupwise_advantages(rewards: list[float]) -> list[float]:
    """GRPO-style group-relative advantage: (r - mean_group) / (std + eps)."""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    m = statistics.mean(rewards)
    s = statistics.pstdev(rewards) or 1e-6
    return [(r - m) / s for r in rewards]


def pass_at_k(group_rewards: list[list[float]], k: int) -> float:
    """Fraction of problems where ANY of the first k rollouts was correct."""
    if not group_rewards or k <= 0:
        return 0.0
    hits = sum(1 for g in group_rewards if any(r > 0.5 for r in g[:k]))
    return hits / len(group_rewards)


# ─────────────────────────────────────────────────────────────────────
# Prompt building
# ─────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, question: str) -> str:
    """Use the model's native chat template."""
    try:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers that don't ship a chat template.
        return (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                    help="Ungated by default. Llama-3.2-1B-Instruct works too "
                         "if you've accepted the license.")
    ap.add_argument("--rollouts-per-problem", type=int, default=8,
                    help="N — divergent rollouts per problem. Bigger N = "
                         "higher pass@N ceiling + stronger GRPO signal.")
    ap.add_argument("--workers", type=int, default=1,
                    help="ForkPool worker count. Each holds a full engine. "
                         "Use 1 on 80 GB H100, 2+ on multi-GPU pods.")
    ap.add_argument("--max-tokens", type=int, default=192)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    ap.add_argument("--worker-gpu-memory", type=float, default=0.55)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json-out", default=None,
                    help="Write a sanitized receipt to this path.")
    ap.add_argument("--plot", default=None,
                    help="Optional PNG output path for pass@k curve.")
    ap.add_argument("--compare-naive", action="store_true",
                    help="Also run the naive llm.generate baseline for "
                         "wall-clock comparison. Fork's advantage shows at "
                         "long trunks and cross-process isolation — on short "
                         "prompts with enable_prefix_caching, naive is already "
                         "fast.")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import thaw_vllm
    from thaw_vllm import ForkPool

    K = len(PROBLEMS)
    N = args.rollouts_per_problem

    # ── Parent engine ─────────────────────────────────────────────
    print(f"=== Parent engine: {args.model} ===")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        enable_prefix_caching=True,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.2f}s")

    tokenizer = llm.get_tokenizer()
    prompts_by_problem = [build_prompt(tokenizer, p["q"]) for p in PROBLEMS]

    # ── ForkPool ───────────────────────────────────────────────────
    print(f"\n=== ForkPool.init_pool(workers={args.workers}, "
          f"preload_weights=True) ===")
    pool = ForkPool()
    t0 = time.perf_counter()
    pool.init_pool(
        model=args.model,
        workers=args.workers,
        preload_weights=True,
        gpu_memory_utilization=args.worker_gpu_memory,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        dtype="float16",
    )
    init_pool_s = time.perf_counter() - t0
    print(f"  pool ready in {init_pool_s:.2f}s")

    # ── Rollout collection via fork_completions ────────────────────
    print(f"\n=== Rollout phase: {K} problems × {N} branches ===")
    texts_fork: list[list[str]] = [["" for _ in range(N)] for _ in range(K)]
    per_problem_s: list[float] = []
    # Stochastic sampling per rollout — no per-call seed, so the N
    # copies of a given prompt diverge naturally via temperature. This
    # is how real RL rollout collection works. Pass@1 will vary ~±0.05
    # across runs; pass@N is more stable.
    sp = SamplingParams(
        temperature=args.temperature, top_p=0.95,
        max_tokens=args.max_tokens,
    )
    rollout_t0 = time.perf_counter()
    for i, base_prompt in enumerate(prompts_by_problem):
        t0 = time.perf_counter()
        results = thaw_vllm.fork_completions(
            llm, [base_prompt] * N, sp, pool=pool,
        )
        elapsed = time.perf_counter() - t0
        per_problem_s.append(elapsed)
        for j, r in enumerate(results):
            texts_fork[i][j] = r.text
        r0_preview = results[0].text.strip().replace("\n", " ")[:60]
        print(f"  problem {i}: {elapsed:5.2f}s  "
              f"gt={PROBLEMS[i]['answer']}  sample={r0_preview!r}")
    fork_total_s = time.perf_counter() - rollout_t0

    # ── Optional naive comparison ─────────────────────────────────
    naive_s: float | None = None
    texts_naive: list[list[str]] | None = None
    if args.compare_naive:
        print(f"\n=== Naive baseline: llm.generate on {K*N} prompts ===")
        flat_prompts: list[str] = []
        for p in prompts_by_problem:
            flat_prompts.extend([p] * N)
        sp_naive = SamplingParams(
            temperature=args.temperature, top_p=0.95,
            max_tokens=args.max_tokens, seed=args.seed,
        )
        t0 = time.perf_counter()
        outs = llm.generate(flat_prompts, sp_naive)
        naive_s = time.perf_counter() - t0
        texts_naive = [["" for _ in range(N)] for _ in range(K)]
        for idx, o in enumerate(outs):
            i, j = divmod(idx, N)
            texts_naive[i][j] = o.outputs[0].text if o.outputs else ""
        print(f"  {K*N} completions in {naive_s:.2f}s "
              f"({K*N / naive_s:.2f} completions/s)")

    pool.close()

    # ── Score + advantages ─────────────────────────────────────────
    group_rewards = [
        [score(t, PROBLEMS[i]["answer"]) for t in texts_fork[i]]
        for i in range(K)
    ]
    flat_rewards = [r for g in group_rewards for r in g]
    group_advantages = [groupwise_advantages(g) for g in group_rewards]
    flat_advantages = [a for g in group_advantages for a in g]

    p_at_1 = pass_at_k(group_rewards, 1)
    p_at_n = pass_at_k(group_rewards, N)
    mean_abs_adv = (
        statistics.mean(abs(a) for a in flat_advantages)
        if flat_advantages else 0.0
    )

    # ── Report ─────────────────────────────────────────────────────
    print("\n=== Results ===")
    print(f"  K = {K} problems, N = {N} rollouts per problem")
    print(f"  rollout phase wall-clock:  {fork_total_s:.2f}s  "
          f"(median {statistics.median(per_problem_s):.2f}s/problem)")
    print(f"  pool init (one-time):      {init_pool_s:.2f}s")
    print()
    print(f"  pass@1       = {p_at_1:.3f}  "
          f"(per-rollout accuracy)")
    print(f"  pass@{N:<6} = {p_at_n:.3f}  "
          f"(any-of-{N} accuracy = reward ceiling)")
    print(f"  lift         = {p_at_n - p_at_1:+.3f}  "
          f"(what best-of-{N} buys over greedy)")
    print(f"  mean |advantage| = {mean_abs_adv:.3f}  "
          f"(GRPO gradient-signal strength)")

    if naive_s is not None:
        speedup = naive_s / fork_total_s if fork_total_s > 0 else float("inf")
        print(f"\n  naive wall-clock:  {naive_s:.2f}s")
        print(f"  fork  wall-clock:  {fork_total_s:.2f}s")
        print(f"  ratio (naive/fork): {speedup:.2f}x")
        print(f"  (note: on short prompts with prefix caching, fork's "
              f"process-isolation advantage is not the headline. See "
              f"fork_pool_rl.py for the long-trunk receipt.)")

    # ── Optional plot ─────────────────────────────────────────────
    plot_path: str | None = None
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            ks = list(range(1, N + 1))
            ys = [pass_at_k(group_rewards, k) for k in ks]
            plt.figure(figsize=(6.5, 4))
            plt.plot(ks, ys, marker="o", linewidth=2, color="#0b1220")
            plt.xlabel("rollouts per problem (k)")
            plt.ylabel(f"pass@k across {K} problems")
            plt.title(f"pass@k scaling — {args.model}")
            plt.grid(alpha=0.3)
            plt.ylim(0, 1.05)
            plt.tight_layout()
            plt.savefig(args.plot, dpi=120)
            plot_path = args.plot
            print(f"\n  plot saved to {args.plot}")
        except ImportError:
            print("\n  (matplotlib not installed; skipping plot)")

    # ── Receipt ────────────────────────────────────────────────────
    if args.json_out:
        from _receipt import build_receipt, write_receipt
        samples = []
        for i in range(min(3, K)):
            for j in range(min(2, N)):
                samples.append({
                    "problem_idx": i,
                    "rollout_idx": j,
                    "question": PROBLEMS[i]["q"],
                    "ground_truth": PROBLEMS[i]["answer"],
                    "reward": group_rewards[i][j],
                    "advantage": round(group_advantages[i][j], 3),
                    "text": texts_fork[i][j],
                })
        modes = {
            "fork_pool_rollouts": {
                "workers": args.workers,
                "k_problems": K,
                "n_rollouts_per_problem": N,
                "wall_clock_s": round(fork_total_s, 3),
                "median_per_problem_s": round(
                    statistics.median(per_problem_s), 3,
                ),
                "pass_at_1": round(p_at_1, 3),
                "pass_at_n": round(p_at_n, 3),
                "mean_abs_advantage": round(mean_abs_adv, 3),
                "rewards_per_group": group_rewards,
                "advantages_per_group": [
                    [round(a, 3) for a in g] for g in group_advantages
                ],
            },
        }
        if naive_s is not None:
            modes["naive_generate"] = {
                "wall_clock_s": round(naive_s, 3),
                "completions": K * N,
                "note": ("llm.generate on flat K*N prompts with "
                         "enable_prefix_caching=True"),
            }
        checks = {
            "all_fork_nonempty": all(t for row in texts_fork for t in row),
            "at_least_one_correct": any(
                r > 0.5 for g in group_rewards for r in g
            ),
            "pass_at_n_gte_pass_at_1": p_at_n >= p_at_1,
        }
        receipt = build_receipt(
            demo="rl_training_step",
            model=args.model,
            trunk_tokens=None,
            timings={
                "load_s": load_s,
                "init_pool_s": init_pool_s,
                "fork_rollout_phase_s": fork_total_s,
                **({"naive_rollout_phase_s": naive_s} if naive_s else {}),
            },
            modes=modes,
            checks=checks,
            samples=samples,
            extra={
                "rollouts_per_problem": N,
                "k_problems": K,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "max_model_len": args.max_model_len,
                "plot_path": plot_path,
                "note": (
                    "Rollout-collection phase of a PPO/GRPO step. "
                    "Weights are not updated. The advantages_per_group "
                    "field is what the optimizer step would consume."
                ),
            },
        )
        write_receipt(args.json_out, receipt)
        print(f"\n  wrote receipt to {args.json_out}")

    print("\n=== OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
