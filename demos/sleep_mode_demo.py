#!/usr/bin/env python3
"""
sleep_mode_demo — bit-identity round trip through thaw as a sleep backend.

Run on a pod with GPU(s) available. Produces a JSON receipt comparing
inference outputs before sleep vs after wake_up, plus throughput for
both directions and GPU memory deltas.

Shape
-----
1. Create LLM with ``enable_sleep_mode=True``, warm one generation from
   a canonical prompt, record the completion tokens.
2. ``thaw_vllm.sleep_mode.sleep(llm, path)`` — freeze weights via thaw's
   pipelined DMA, then call ``llm.sleep(level=2)`` to free GPU memory
   via vLLM's ``CuMemAllocator``.
3. ``thaw_vllm.sleep_mode.wake_up(llm, path)`` — call ``llm.wake_up()``
   to re-allocate GPU tensors, then thaw's pipelined restore writes the
   weights back.
4. Run the same prompt with greedy decoding; tokens must match exactly.

Usage
-----
    python demos/sleep_mode_demo.py \\
        --model meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --snapshot /tmp/sleep.thaw \\
        --json-out site/receipts/$(date +%Y-%m-%d)_sleep_mode.json

For TP=2:
    python demos/sleep_mode_demo.py \\
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \\
        --snapshot /tmp/sleep70.thaw \\
        --tp 2

Receipt contents
----------------
The JSON output includes:
  - Greedy bit-identity (pre vs post token IDs must match).
  - ``timings``: load / sleep / wake wall-clock + thaw freeze/restore GB/s.
  - ``gpu_memory_bytes``: ``torch.cuda.memory_allocated()`` snapshots at
    four points (before load, after load, after sleep, after wake). The
    drop from after_load → after_sleep is the vLLM sleep mechanism
    actually freeing GPU memory.
  - ``sleep_stats`` / ``wake_stats``: the full dicts returned by
    ``thaw_vllm.sleep_mode``, including per-rank throughput for TP>1.

This receipt is what the vLLM RFC #34303 comment should cite.
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


PROMPT = (
    "Give a one-sentence definition of amortized complexity, then a "
    "concrete example where it matters."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--sleep-level", type=int, default=2,
                    help="vLLM sleep level (1 = keep KV cache on GPU, "
                         "2 = free all model memory, default).")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    import thaw_vllm

    try:
        import torch
        allocated_before_load = torch.cuda.memory_allocated()
    except Exception:
        torch = None  # type: ignore
        allocated_before_load = None

    print(f"=== Loading {args.model} (tp={args.tp}) ===")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        dtype="float16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_sleep_mode=True,  # required for llm.sleep() to free GPU
    )
    load_s = time.perf_counter() - t0
    print(f"  loaded in {load_s:.2f}s")

    allocated_after_load = (
        torch.cuda.memory_allocated() if torch else None
    )

    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, seed=42)

    print("\n=== Pre-sleep generation (greedy) ===")
    pre = llm.generate(PROMPT, sp)
    pre_text = pre[0].outputs[0].text
    pre_tokens = list(pre[0].outputs[0].token_ids)
    print(f"  {len(pre_tokens)} tokens: {pre_text[:120]!r}")

    print(f"\n=== sleep → {args.snapshot} (level={args.sleep_level}) ===")
    t0 = time.perf_counter()
    try:
        sleep_stats = thaw_vllm.sleep_mode.sleep(
            llm, args.snapshot, level=args.sleep_level,
        )
    except thaw_vllm.sleep_mode.SleepModeUnavailableError as e:
        print(f"  ! {e}", file=sys.stderr)
        print("  ! Rerun with enable_sleep_mode=True on the LLM.",
              file=sys.stderr)
        return 2
    sleep_s = time.perf_counter() - t0
    freeze_gb_s = sleep_stats.get("throughput_gb_s")
    freed_mb = (
        (sleep_stats.get("gpu_bytes_freed") or 0) / 1e6
        if sleep_stats.get("gpu_bytes_freed") is not None else None
    )
    print(f"  sleep in {sleep_s:.2f}s, freeze {freeze_gb_s:.2f} GB/s, "
          f"freed {freed_mb:.0f} MB of GPU memory"
          if freed_mb is not None and freeze_gb_s is not None
          else f"  sleep in {sleep_s:.2f}s")

    allocated_after_sleep = (
        torch.cuda.memory_allocated() if torch else None
    )

    print(f"\n=== wake_up ← {args.snapshot} ===")
    t0 = time.perf_counter()
    wake_stats = thaw_vllm.sleep_mode.wake_up(llm, args.snapshot)
    wake_s = time.perf_counter() - t0
    restore_gb_s = wake_stats.get("throughput_gb_s")
    populated_mb = (
        (wake_stats.get("gpu_bytes_populated") or 0) / 1e6
        if wake_stats.get("gpu_bytes_populated") is not None else None
    )
    print(f"  wake in {wake_s:.2f}s, restore {restore_gb_s:.2f} GB/s, "
          f"populated {populated_mb:.0f} MB of GPU memory"
          if populated_mb is not None and restore_gb_s is not None
          else f"  wake in {wake_s:.2f}s")

    allocated_after_wake = (
        torch.cuda.memory_allocated() if torch else None
    )

    print("\n=== Post-wake generation (greedy, same prompt + seed) ===")
    post = llm.generate(PROMPT, sp)
    post_text = post[0].outputs[0].text
    post_tokens = list(post[0].outputs[0].token_ids)
    print(f"  {len(post_tokens)} tokens: {post_text[:120]!r}")

    bit_identical = pre_tokens == post_tokens
    print(f"\n=== bit-identical: {bit_identical} ===")
    if not bit_identical:
        for i, (a, b) in enumerate(zip(pre_tokens, post_tokens)):
            if a != b:
                print(f"  first divergence at token {i}: {a} != {b}")
                break

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        receipt = {
            "demo": "sleep_mode",
            "model": args.model,
            "tp": args.tp,
            "max_tokens": args.max_tokens,
            "snapshot_path": args.snapshot,
            "sleep_level": args.sleep_level,
            "timings": {
                "load_s": round(load_s, 3),
                "sleep_s": round(sleep_s, 3),
                "wake_s": round(wake_s, 3),
                "freeze_gb_s": sleep_stats.get("throughput_gb_s"),
                "restore_gb_s": wake_stats.get("throughput_gb_s"),
            },
            "gpu_memory_bytes": {
                "before_load": allocated_before_load,
                "after_load": allocated_after_load,
                "after_sleep": allocated_after_sleep,
                "after_wake": allocated_after_wake,
                "freed_by_sleep": sleep_stats.get("gpu_bytes_freed"),
                "populated_by_wake": wake_stats.get("gpu_bytes_populated"),
            },
            "freed_gpu_memory": sleep_stats.get("freed_gpu_memory"),
            "vllm_wake_up_called": wake_stats.get("vllm_wake_up_called"),
            "bit_identical_greedy_output": bit_identical,
            "pre_token_count": len(pre_tokens),
            "post_token_count": len(post_tokens),
            "sleep_stats": sleep_stats,
            "wake_stats": wake_stats,
        }
        with open(args.json_out, "w") as fh:
            json.dump(receipt, fh, indent=2, default=str)
        print(f"\nWrote receipt to {args.json_out}")

    return 0 if bit_identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
