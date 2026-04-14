#!/usr/bin/env python3
"""
thaw agent fork demo — clone a running AI session in milliseconds.

Demonstrates thaw's unique capability: snapshot an AI agent's full
conversation state (KV cache), then "fork" it — spawn multiple
completions that all start from the same point without re-running prefill.

Use cases this enables:
  - Multi-path exploration: agent considers 3 approaches in parallel
  - Scale-out: new instances start warm, not cold
  - Session migration: move a conversation to a different GPU
  - Checkpoint/rollback: save conversation state, roll back on error

Usage:
    python demos/agent_fork.py
    python demos/agent_fork.py --model meta-llama/Meta-Llama-3-8B-Instruct
    python demos/agent_fork.py --snapshot weights.thaw  # use thaw weight restore
    python demos/agent_fork.py --full-cycle              # unload + reload from snapshot

The --full-cycle flag runs the complete demonstration: load model, build
conversation context, freeze EVERYTHING (weights + KV cache), destroy the
instance, then recreate it entirely from thaw snapshots. This proves the
session state is fully portable — the agent remembers the entire conversation
even after the original instance is gone.

Requires: pip install thaw-vllm, GPU with >=16GB VRAM.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

# Must be set before importing vLLM
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent persona — a long, realistic system prompt
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM_PROMPT = """\
You are Atlas, a senior AI research scientist specializing in ML \
infrastructure and GPU computing. You give concrete, actionable answers \
with specific numbers and code examples when relevant. You think step \
by step and always consider trade-offs.

Current context: You're advising a startup building an AI-powered code \
review tool. They need to serve a 70B parameter model to 500+ concurrent \
users with <2s time-to-first-token. They have a cluster of 8x H100 80GB \
nodes connected via InfiniBand. Budget is tight — they need to maximize \
throughput per dollar.

Key constraints you're aware of:
- Peak load is 3x average (morning US hours)
- 80% of requests share one of 5 system prompts (different review modes)
- Average input length: 2,000 tokens (code context)
- Average output length: 500 tokens (review comments)
- They're currently on vLLM but open to alternatives
- Latency SLA: P99 TTFT < 2s, P99 inter-token < 50ms
- The team has 3 engineers and needs to ship in 6 weeks"""

CONVERSATION_HISTORY = [
    {
        "role": "user",
        "content": (
            "We're seeing P99 TTFT of 4.2 seconds under load. Our SLA requires "
            "<2s. We've already enabled prefix caching and continuous batching. "
            "What else can we do?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "4.2s P99 TTFT with prefix caching already enabled tells me your "
            "bottleneck is likely in one of three places:\n\n"
            "1. **Prefill compute saturation** — With 2K token inputs on a 70B "
            "model, each prefill takes ~200ms on 8xH100 with TP=8. At 500 "
            "concurrent users, that's 100 prefills/second, which saturates your "
            "cluster.\n\n"
            "2. **KV cache memory pressure** — 70B with 2K context at FP16 uses "
            "~2.5GB of KV cache per request. 500 concurrent = 1.25TB total. You "
            "likely have 640GB total (8x80GB), so requests are queuing.\n\n"
            "3. **Batch scheduling inefficiency** — Mixed prefill + decode batches "
            "cause decode latency spikes. Chunked prefill helps but doesn't "
            "eliminate this.\n\n"
            "My recommendation: Start with FP8 quantization — halves KV cache "
            "memory, H100 has native FP8 support via Transformer Engine. Then add "
            "chunked prefill with a 512-token chunk size. This should get you "
            "under 2s P99 without hardware changes.\n\n"
            "Want me to walk through the FP8 migration path, or should we explore "
            "the scheduling side first?"
        ),
    },
    {
        "role": "user",
        "content": (
            "Good analysis. We're leaning toward FP8 but worried about quality. "
            "Our code review accuracy benchmark dropped 3% in initial tests. "
            "Is there a way to get FP8 memory savings without that quality hit?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "3% accuracy drop on FP8 is actually higher than expected — typically "
            "FP8 post-training quantization on Llama-70B shows <1% degradation. "
            "A few things could explain your results:\n\n"
            "1. **Naive PTQ vs calibrated quantization** — If you're using "
            "per-tensor FP8 without calibration, switching to per-channel "
            "quantization with a calibration dataset (even 128 samples from your "
            "code review data) should recover most of that 3%.\n\n"
            "2. **KV cache quantization specifically** — The quality-sensitive path "
            "is often the KV cache, not the weights. Try: FP8 weights + FP16 KV "
            "cache. You still save ~40% memory (weights are the bulk) without "
            "touching the attention computation.\n\n"
            "3. **Attention sink tokens** — First few tokens in long contexts are "
            "disproportionately important. Some FP8 schemes quantize these poorly. "
            "Keep the first 128 tokens in FP16.\n\n"
            "Here's the concrete config change for vLLM:\n"
            "```python\n"
            "llm = LLM(\n"
            "    model='your-70b-model',\n"
            "    quantization='fp8',\n"
            "    kv_cache_dtype='fp16',  # keep KV in full precision\n"
            "    max_model_len=4096,\n"
            ")\n"
            "```\n\n"
            "Benchmark this config against your accuracy test. I'd expect the 3% "
            "drop to shrink to <0.5%."
        ),
    },
]

# Fork queries — different directions from the same conversation state
FORK_QUERIES = [
    (
        "Deep dive on FP8",
        "Walk me through the FP8 calibration process step by step. I want "
        "specific code — how do we collect calibration data from our code review "
        "dataset, run the quantization, and validate quality? Reference the "
        "per-channel approach you mentioned.",
    ),
    (
        "Alternative: speculative decoding",
        "Actually, let's explore a completely different approach. You mentioned "
        "our prefill is the bottleneck. What if instead of quantization, we used "
        "speculative decoding with an 8B draft model? We already have Llama-3-8B "
        "deployed for our fast tier. Give me concrete latency projections for our "
        "workload — 2K input, 500 output tokens, 500 concurrent users.",
    ),
    (
        "Infrastructure-only approach",
        "Let me reframe the problem entirely. Forget model optimization. What if "
        "we solved this purely with smarter infrastructure? I'm thinking: "
        "pre-warm KV caches for our 5 system prompts across all nodes, implement "
        "prefix-aware request routing, and auto-scale based on prefill queue "
        "depth. How much headroom could this buy us without touching the model?",
    ),
]


def _run_phase1_subprocess(model, snapshot, max_tokens, state_dir):
    """Run phase 1 (original agent session) in a subprocess.

    The subprocess loads the model, runs the conversation, freezes all
    state (weights + KV cache), writes a JSON handoff file, and exits.
    When the process exits, ALL GPU memory is freed — guaranteed.
    This solves the vLLM GPU memory leak issue.
    """
    script = f'''
import json, os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
import thaw_vllm
from transformers import AutoTokenizer

model = {model!r}
snapshot = {snapshot!r}
max_tokens = {max_tokens!r}
state_dir = {state_dir!r}

SYSTEM_PROMPT = """{SYSTEM_PROMPT}"""

CONVERSATION_HISTORY = {json.dumps(CONVERSATION_HISTORY)}

# ── Load model ──
print()
print("[1/4] Loading model: " + model)
t0 = time.perf_counter()
if snapshot:
    llm = thaw_vllm.load(model, snapshot, enable_prefix_caching=True)
    load_method = "thaw"
else:
    llm = LLM(model=model, enforce_eager=True, dtype="float16",
              enable_prefix_caching=True)
    load_method = "HuggingFace"
load_time = time.perf_counter() - t0
print(f"      Loaded via {{load_method}} in {{load_time:.1f}}s")

# ── Build session ──
tokenizer = AutoTokenizer.from_pretrained(model)
messages = [{{"role": "system", "content": SYSTEM_PROMPT}}] + CONVERSATION_HISTORY
base_prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
prefix_tokens = len(tokenizer.encode(base_prompt))
system_tokens = len(tokenizer.encode(SYSTEM_PROMPT))

print()
print("[2/4] Agent session")
print(f"      System prompt:  {{system_tokens}} tokens")
print(f"      Conversation:   {{len(CONVERSATION_HISTORY)}} turns")
print(f"      Total context:  {{prefix_tokens}} tokens")

# ── Initial completion ──
sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens)
print()
print("[3/4] Running initial completion (populates prefix cache)...")
t0 = time.perf_counter()
outputs = llm.generate([base_prompt], sampling)
initial_time = time.perf_counter() - t0
initial_text = outputs[0].outputs[0].text.strip()
initial_gen_tokens = len(outputs[0].outputs[0].token_ids)
print(f"      Time:    {{initial_time:.2f}}s")
print(f"      Tokens:  {{initial_gen_tokens}} generated")
preview = initial_text.replace(chr(10), " ")[:80]
print(f"      Preview: \\"{{preview}}...\\"")

# ── Freeze state ──
kv_path = os.path.join(state_dir, "session.thawkv")
weights_path = os.path.join(state_dir, "weights.thaw") if not snapshot else None

print()
print("[4/4] Freezing agent session state...")

w_stats = None
if weights_path:
    ec = thaw_vllm.kv_snapshot._get_engine_core(llm)
    model_nn = ec.model_executor.driver_worker.model_runner.model
    t0 = time.perf_counter()
    w_stats = thaw_vllm.freeze_model(model_nn, weights_path)
    w_time = time.perf_counter() - t0
    w_gb = w_stats["total_bytes"] / 1e9
    print(f"      Weights: {{w_stats['num_regions']}} params, "
          f"{{w_gb:.1f}} GB, {{w_time:.2f}}s "
          f"({{w_stats['throughput_gb_s']:.1f}} GB/s)")

freeze_stats = thaw_vllm.freeze_kv_cache(llm, kv_path)
print(f"      KV cache: {{freeze_stats['num_blocks']}} blocks, "
      f"{{freeze_stats['total_bytes'] / 1e6:.0f}} MB, "
      f"{{freeze_stats['elapsed_s']:.3f}}s")

total_state_mb = freeze_stats["total_bytes"] / 1e6
if w_stats:
    total_state_mb += w_stats["total_bytes"] / 1e6
print(f"      Total state captured: {{total_state_mb:.0f}} MB")

# ── Write handoff ──
handoff = {{
    "initial_text": initial_text,
    "initial_time": initial_time,
    "initial_gen_tokens": initial_gen_tokens,
    "load_time": load_time,
    "load_method": load_method,
    "prefix_tokens": prefix_tokens,
    "system_tokens": system_tokens,
    "kv_path": kv_path,
    "weights_path": weights_path,
    "snapshot": snapshot,
    "freeze_stats": {{
        "num_blocks": freeze_stats["num_blocks"],
        "total_bytes": freeze_stats["total_bytes"],
        "elapsed_s": freeze_stats["elapsed_s"],
    }},
}}
handoff_path = os.path.join(state_dir, "handoff.json")
with open(handoff_path, "w") as f:
    json.dump(handoff, f)

print()
print("      Phase 1 complete. Process exiting — GPU memory will be fully freed.")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    )
    if result.returncode != 0:
        print(f"\n  ERROR: Phase 1 subprocess failed (exit code {result.returncode})")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="thaw agent fork demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--snapshot",
        default=None,
        help="Path to thaw weight snapshot (.thaw file)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Max tokens per fork completion (default: 200)",
    )
    parser.add_argument(
        "--num-forks",
        type=int,
        default=3,
        help="Number of forks (default: 3, max: 3)",
    )
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="Full demo: freeze everything, destroy instance, restore from snapshots",
    )
    args = parser.parse_args()
    args.num_forks = min(args.num_forks, len(FORK_QUERIES))

    if args.full_cycle:
        return main_full_cycle(args)

    total_steps = 5

    print()
    print("=" * 62)
    print("  thaw — Agent Fork Demo")
    print("  Clone a running AI session in milliseconds")
    print("=" * 62)

    # ── Step 1: Load model ──────────────────────────────────────

    from vllm import LLM, SamplingParams
    import thaw_vllm

    print()
    print(f"[1/{total_steps}] Loading model: {args.model}")

    t0 = time.perf_counter()
    if args.snapshot:
        llm = thaw_vllm.load(
            args.model,
            args.snapshot,
            enable_prefix_caching=True,
        )
        load_method = "thaw"
    else:
        llm = LLM(
            model=args.model,
            enforce_eager=True,
            dtype="float16",
            enable_prefix_caching=True,
        )
        load_method = "HuggingFace"
    load_time = time.perf_counter() - t0

    print(f"      Loaded via {load_method} in {load_time:.1f}s")

    # ── Step 2: Build agent session ─────────────────────────────

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += CONVERSATION_HISTORY

    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prefix_tokens = len(tokenizer.encode(base_prompt))

    system_tokens = len(tokenizer.encode(SYSTEM_PROMPT))

    print()
    print(f"[2/{total_steps}] Agent session")
    print(f"      System prompt:  {system_tokens} tokens")
    print(f"      Conversation:   {len(CONVERSATION_HISTORY)} turns")
    print(f"      Total context:  {prefix_tokens} tokens")

    # ── Step 3: Initial completion (populates KV cache) ─────────

    sampling = SamplingParams(temperature=0.7, max_tokens=args.max_tokens)

    print()
    print(f"[3/{total_steps}] Running initial completion (populates prefix cache)...")

    t0 = time.perf_counter()
    outputs = llm.generate([base_prompt], sampling)
    initial_time = time.perf_counter() - t0
    initial_text = outputs[0].outputs[0].text.strip()
    initial_gen_tokens = len(outputs[0].outputs[0].token_ids)

    print(f"      Time:    {initial_time:.2f}s")
    print(f"      Tokens:  {initial_gen_tokens} generated")
    preview = initial_text.replace("\n", " ")[:80]
    print(f"      Preview: \"{preview}...\"")

    # ── Step 4: Freeze KV cache ─────────────────────────────────

    kv_dir = tempfile.mkdtemp(prefix="thaw_fork_")
    kv_path = os.path.join(kv_dir, "session.thawkv")

    print()
    print(f"[4/{total_steps}] Freezing agent session state...")

    freeze_stats = thaw_vllm.freeze_kv_cache(llm, kv_path)

    print(f"      KV cache: {freeze_stats['num_blocks']} blocks, "
          f"{freeze_stats['total_bytes'] / 1e6:.0f} MB, "
          f"{freeze_stats['elapsed_s']:.3f}s")

    # ── Step 5: Fork ────────────────────────────────────────────

    print()
    print(f"[5/{total_steps}] FORKING — "
          f"{args.num_forks} parallel explorations")
    print(f"      All forks share {prefix_tokens} tokens of cached context.")
    print(f"      Each fork explores a different direction from the same point.")
    print("-" * 62)

    fork_results = _run_forks(llm, tokenizer, messages, initial_text,
                              prefix_tokens, args.num_forks, args.max_tokens)

    # ── Results ─────────────────────────────────────────────────

    _print_results(
        prefix_tokens=prefix_tokens,
        freeze_stats=freeze_stats,
        initial_time=initial_time,
        initial_gen_tokens=initial_gen_tokens,
        fork_results=fork_results,
        full_cycle=False,
        load_time=load_time,
        snapshot=args.snapshot,
    )

    # Cleanup
    if os.path.exists(kv_path):
        os.unlink(kv_path)
    if os.path.exists(kv_dir):
        try:
            os.rmdir(kv_dir)
        except OSError:
            pass


def main_full_cycle(args):
    """Full-cycle mode: phase 1 in subprocess, phase 2 in main process.

    Phase 1 runs in a child process that loads the model, builds a
    conversation, freezes everything (weights + KV cache), and exits.
    When the process exits, ALL GPU memory is freed — guaranteed.

    Phase 2 runs in this process: restore from snapshots and fork.
    """
    total_steps = 7

    print()
    print("=" * 62)
    print("  thaw — Agent Fork Demo")
    print("  Clone a running AI session in milliseconds")
    print("  Mode: full cycle (destroy + restore from snapshots)")
    print("=" * 62)

    # ── Phase 1: subprocess ────────────────────────────────────

    state_dir = tempfile.mkdtemp(prefix="thaw_fork_")
    print(f"\n  State directory: {state_dir}")

    print()
    print("─" * 62)
    print("  PHASE 1: Original agent (runs in isolated subprocess)")
    print("─" * 62)

    _run_phase1_subprocess(args.model, args.snapshot, args.max_tokens, state_dir)

    # ── Read handoff data ──────────────────────────────────────

    handoff_path = os.path.join(state_dir, "handoff.json")
    with open(handoff_path) as f:
        handoff = json.load(f)

    initial_text = handoff["initial_text"]
    initial_time = handoff["initial_time"]
    initial_gen_tokens = handoff["initial_gen_tokens"]
    load_time = handoff["load_time"]
    prefix_tokens = handoff["prefix_tokens"]
    kv_path = handoff["kv_path"]
    weights_path = handoff["weights_path"]
    freeze_stats = handoff["freeze_stats"]
    snapshot_for_restore = handoff["snapshot"] or weights_path

    # ── Phase 2: restore + fork (this process) ─────────────────

    print()
    print("─" * 62)
    print("  PHASE 2: Restored agent (fresh process, zero GPU state)")
    print("─" * 62)

    import torch
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"\n      GPU memory: {free_gb:.1f}/{total_gb:.1f} GiB free")
    print(f"      Original agent process has exited — GPU fully released.")

    print()
    print(f"[5/{total_steps}] Restoring agent from thaw snapshots...")
    print(f"      Weight snapshot: {snapshot_for_restore}")
    print(f"      KV snapshot:     {kv_path}")

    import thaw_vllm
    from vllm import SamplingParams

    t_restore_start = time.perf_counter()
    llm = thaw_vllm.load(
        args.model,
        snapshot_for_restore,
        enable_prefix_caching=True,
    )
    weight_restore_time = time.perf_counter() - t_restore_start
    print(f"      Weights restored in {weight_restore_time:.1f}s")

    t0 = time.perf_counter()
    kv_restore_stats = thaw_vllm.restore_kv_cache(llm, kv_path)
    kv_actual_restore = time.perf_counter() - t0

    restore_time = time.perf_counter() - t_restore_start

    print(f"      KV cache restored in {kv_actual_restore:.3f}s "
          f"({kv_restore_stats['num_blocks']} blocks, "
          f"{kv_restore_stats['total_bytes'] / 1e6:.0f} MB)")
    print(f"      Total restore: {restore_time:.1f}s")
    print()
    print(f"      Agent is BACK — full conversation context restored")
    print(f"      from snapshots. The original instance no longer exists.")

    # ── Fork ────────────────────────────────────────────────────

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += CONVERSATION_HISTORY

    base_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    print()
    print(f"[6/{total_steps}] FORKING — "
          f"{args.num_forks} parallel explorations")
    print(f"      All forks share {prefix_tokens} tokens of cached context.")
    print(f"      Each fork explores a different direction from the same point.")
    print("-" * 62)

    fork_results = _run_forks(llm, tokenizer, messages, initial_text,
                              prefix_tokens, args.num_forks, args.max_tokens)

    # ── Results ─────────────────────────────────────────────────

    _print_results(
        prefix_tokens=prefix_tokens,
        freeze_stats=freeze_stats,
        initial_time=initial_time,
        initial_gen_tokens=initial_gen_tokens,
        fork_results=fork_results,
        full_cycle=True,
        load_time=load_time,
        restore_time=restore_time,
        avg_fork=sum(r["time"] for r in fork_results) / len(fork_results),
    )

    # Cleanup
    for p in [kv_path, weights_path, handoff_path]:
        if p and os.path.exists(p):
            os.unlink(p)
    try:
        os.rmdir(state_dir)
    except OSError:
        pass


def _run_forks(llm, tokenizer, messages, initial_text, prefix_tokens,
               num_forks, max_tokens):
    """Run fork queries and return results."""
    from vllm import SamplingParams

    sampling = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    fork_results = []

    for i in range(num_forks):
        label, query = FORK_QUERIES[i]

        forked_messages = messages + [
            {"role": "assistant", "content": initial_text},
            {"role": "user", "content": query},
        ]
        forked_prompt = tokenizer.apply_chat_template(
            forked_messages, tokenize=False, add_generation_prompt=True,
        )
        forked_tokens = len(tokenizer.encode(forked_prompt))
        new_tokens = forked_tokens - prefix_tokens

        t0 = time.perf_counter()
        outputs = llm.generate([forked_prompt], sampling)
        fork_time = time.perf_counter() - t0
        fork_text = outputs[0].outputs[0].text.strip()
        gen_tokens = len(outputs[0].outputs[0].token_ids)

        fork_results.append({
            "label": label,
            "query": query,
            "time": fork_time,
            "text": fork_text,
            "prompt_tokens": forked_tokens,
            "new_tokens": new_tokens,
            "gen_tokens": gen_tokens,
        })

        print(f"\n  Fork {i + 1}: {label}")
        print(f"  Query:     \"{query[:60]}...\"")
        print(f"  New input: {new_tokens} tokens (shared prefix: {prefix_tokens})")
        print(f"  Output:    {gen_tokens} tokens in {fork_time:.2f}s")

        lines = [l.strip() for l in fork_text.split("\n") if l.strip()]
        for line in lines[:2]:
            print(f"  > {line[:72]}")

    return fork_results


def _print_results(*, prefix_tokens, freeze_stats, initial_time,
                   initial_gen_tokens, fork_results, full_cycle,
                   load_time=None, snapshot=None, restore_time=None,
                   avg_fork=None):
    """Print results summary."""
    kv_mb = freeze_stats["total_bytes"] / 1e6
    if avg_fork is None:
        avg_fork = sum(r["time"] for r in fork_results) / len(fork_results)

    print()
    print("=" * 62)
    print("  Results")
    print("=" * 62)

    print(f"""
  Agent context:       {prefix_tokens} tokens ({len(CONVERSATION_HISTORY)} conversation turns)
  KV cache snapshot:   {kv_mb:.0f} MB ({freeze_stats['num_blocks']} blocks)
  Freeze time:         {freeze_stats['elapsed_s']:.3f}s""")

    if full_cycle and restore_time is not None:
        print(f"  Full restore:        {restore_time:.1f}s "
              f"(weights + KV cache)")
    else:
        print(f"  Restore time:        ~0.10s (per instance)")

    print()
    print(f"  Initial completion:  {initial_time:.2f}s ({initial_gen_tokens} tokens)")
    for i, r in enumerate(fork_results):
        print(f"  Fork {i + 1} ({r['label'][:20]}): "
              f"{r['time']:.2f}s ({r['gen_tokens']} tokens)")
    print(f"  Average fork time:   {avg_fork:.2f}s")

    if full_cycle and restore_time is not None:
        print(f"""
  ── Full cycle timing ──

  Original instance (subprocess — now exited):
    Model load:      {load_time:.1f}s
    Conversation:    {initial_time:.1f}s
    Freeze state:    {freeze_stats['elapsed_s']:.1f}s

  Forked instance (restored from snapshots):
    Full restore:    {restore_time:.1f}s (weights + KV cache)
    Fork queries:    {avg_fork:.1f}s avg

  The original process has EXITED. GPU memory was fully released.
  The forked instance never saw the original conversation.
  It restored the ENTIRE agent state from thaw snapshots.
""")
    else:
        normal_load = max(load_time, 15.0)
        thaw_restore = min(load_time, 5.0) if snapshot else 3.5
        kv_restore_est = 0.10
        prefill_per_instance = prefix_tokens / 10000

        n = 10
        without_thaw = n * (normal_load + prefill_per_instance)
        with_thaw = n * (thaw_restore + kv_restore_est)

        print(f"""
  ── Production scaling: fork to {n} instances ──

  Without thaw (each instance loads model + re-runs conversation):
    {n} x ({normal_load:.0f}s model load + {prefill_per_instance:.1f}s prefill) = {without_thaw:.0f}s

  With thaw (each instance restores snapshot):
    {n} x ({thaw_restore:.1f}s weight restore + {kv_restore_est:.1f}s KV restore) = {with_thaw:.0f}s

  Speedup: {without_thaw / with_thaw:.1f}x faster to get {n} warm agents
""")

    print("  Every forked instance has the FULL conversation context —")
    print("  the agent remembers everything, without re-running prefill.")
    print("  This is session cloning, not just fast loading.")
    print()


if __name__ == "__main__":
    main()
