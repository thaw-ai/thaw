"""
kv_cache_demo.py — thaw KV cache snapshot/restore for vLLM V1.

Demonstrates prefix cache preservation across cold starts:
  1. Load model, generate with a shared prefix to populate KV cache
  2. Freeze model weights + KV cache to disk
  3. Cold restart with dummy weights, restore both
  4. Generate with same prefix — gets prefix cache hit, skips prefill

This is the competitive moat: nobody else snapshots KV cache.

Usage:
    python kv_cache_demo.py [--model MODEL] [--snapshot PATH] [--kv-snapshot PATH]

Requires: vllm >= 0.19, torch. Run on a GPU with enough VRAM.
"""

import argparse
import gc
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


SYSTEM_PREFIX = (
    "You are an expert AI assistant specializing in distributed systems, "
    "cloud infrastructure, and GPU computing. You provide detailed, technical "
    "answers with specific numbers and citations where possible. You always "
    "think step by step and consider edge cases. When discussing performance, "
    "you include throughput numbers, latency percentiles, and hardware specs. "
    "You are helping an engineer optimize their LLM inference infrastructure "
    "for minimum cold start latency and maximum GPU utilization across a "
    "fleet of heterogeneous GPUs including H100, A100, and L40S instances."
)

QUERY_1 = "What is the theoretical PCIe Gen5 x16 bandwidth and how does it compare to NVLink?"
QUERY_2 = "Explain the tradeoffs between tensor parallelism and pipeline parallelism for serving."


def find_model(llm):
    """Extract the nn.Module from a vLLM LLM instance."""
    engine = llm.llm_engine
    try:
        return engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    try:
        core = engine.engine_core
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        return core.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    raise RuntimeError("Could not locate nn.Module in vLLM LLM instance.")


def get_engine_core(llm):
    """Navigate to the real EngineCore."""
    ec = llm.llm_engine.engine_core
    if hasattr(ec, 'engine_core'):
        ec = ec.engine_core
    return ec


def count_cached_blocks(llm):
    """Count blocks with a block_hash in the block pool."""
    ec = get_engine_core(llm)
    block_pool = ec.scheduler.kv_cache_manager.block_pool
    count = 0
    for block in block_pool.blocks:
        if block._block_hash is not None and not block.is_null:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="thaw KV cache snapshot/restore demo")
    parser.add_argument("--model", default="facebook/opt-125m",
                        help="Model to use (default: opt-125m for fast testing)")
    parser.add_argument("--snapshot", default="/tmp/kv_demo_weights.thaw",
                        help="Path for weight snapshot")
    parser.add_argument("--kv-snapshot", default="/tmp/kv_demo_cache.thawkv",
                        help="Path for KV cache snapshot")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined
    from thaw_vllm.kv_snapshot import freeze_kv_cache, restore_kv_cache

    sampling = SamplingParams(temperature=0, max_tokens=50)

    # =================================================================
    # Phase 1: Load model, generate to populate prefix cache
    # =================================================================
    print("=" * 60)
    print("Phase 1: Load model + populate prefix cache")
    print("=" * 60)

    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.25,
    )

    # Generate with the shared system prefix + two different queries.
    # This populates the KV cache and the prefix cache hashes.
    prompt1 = f"{SYSTEM_PREFIX}\n\nUser: {QUERY_1}\nAssistant:"
    prompt2 = f"{SYSTEM_PREFIX}\n\nUser: {QUERY_2}\nAssistant:"

    out = llm.generate([prompt1, prompt2], sampling)
    ref_text1 = out[0].outputs[0].text
    ref_text2 = out[1].outputs[0].text

    cached_before = count_cached_blocks(llm)
    print(f"Cached blocks after generation: {cached_before}")
    print(f"Output 1: {ref_text1[:80]}...")
    print(f"Output 2: {ref_text2[:80]}...")

    # =================================================================
    # Phase 2: Freeze weights + KV cache
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Freeze weights + KV cache")
    print("=" * 60)

    model = find_model(llm)
    wstats = freeze_model_pipelined(model, args.snapshot)
    print(f"Weights: {wstats['num_regions']} regions, "
          f"{wstats['total_bytes'] / 1e9:.2f} GB in {wstats['elapsed_s']:.1f}s")

    kv_stats = freeze_kv_cache(llm, args.kv_snapshot)
    print(f"KV cache: {kv_stats['num_blocks']} blocks, "
          f"{kv_stats['total_bytes'] / 1e6:.1f} MB in {kv_stats['elapsed_s']:.3f}s")

    # Tear down
    del llm, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    # Phase 3: Cold start — restore weights + KV cache
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Thaw cold start (weights + KV cache)")
    print("=" * 60)

    t0 = time.perf_counter()
    llm_thaw = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.25,
        load_format="dummy",
    )
    init_time = time.perf_counter() - t0
    print(f"Dummy init: {init_time:.1f}s")

    # Restore weights
    model = find_model(llm_thaw)
    t0 = time.perf_counter()
    rstats = restore_model_pipelined(model, args.snapshot)
    weight_time = time.perf_counter() - t0
    print(f"Weight restore: {weight_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s)")

    # Restore KV cache
    t0 = time.perf_counter()
    kv_rstats = restore_kv_cache(llm_thaw, args.kv_snapshot)
    kv_time = time.perf_counter() - t0
    cached_after = count_cached_blocks(llm_thaw)
    print(f"KV restore: {kv_rstats['num_blocks']} blocks, "
          f"{kv_rstats['total_bytes'] / 1e6:.1f} MB in {kv_time:.3f}s")
    print(f"Cached blocks after restore: {cached_after}")

    total_thaw = init_time + weight_time + kv_time
    print(f"Total cold start: {total_thaw:.1f}s")

    # =================================================================
    # Phase 4: Generate with restored prefix cache
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Generate with restored prefix cache")
    print("=" * 60)

    # Same prompts — should hit the restored prefix cache
    t0 = time.perf_counter()
    out = llm_thaw.generate([prompt1, prompt2], sampling)
    gen_time = time.perf_counter() - t0
    thaw_text1 = out[0].outputs[0].text
    thaw_text2 = out[1].outputs[0].text

    print(f"Generation time: {gen_time:.2f}s")
    print(f"Output 1: {thaw_text1[:80]}...")
    print(f"Output 2: {thaw_text2[:80]}...")

    # =================================================================
    # Results
    # =================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  KV blocks frozen:         {kv_stats['num_blocks']}")
    print(f"  KV blocks restored:       {kv_rstats['num_blocks']}")
    print(f"  KV cache size:            {kv_stats['total_bytes'] / 1e6:.1f} MB")
    print(f"  KV freeze time:           {kv_stats['elapsed_s']:.3f}s")
    print(f"  KV restore time:          {kv_time:.3f}s")
    print(f"  Total cold start:         {total_thaw:.1f}s (init + weights + KV)")

    match1 = ref_text1 == thaw_text1
    match2 = ref_text2 == thaw_text2
    print(f"  Output match (query 1):   {'PASS' if match1 else 'FAIL'}")
    print(f"  Output match (query 2):   {'PASS' if match2 else 'FAIL'}")
    if not match1:
        print(f"    Expected: {ref_text1[:100]}")
        print(f"    Got:      {thaw_text1[:100]}")
    if not match2:
        print(f"    Expected: {ref_text2[:100]}")
        print(f"    Got:      {thaw_text2[:100]}")

    if match1 and match2 and cached_after > 0:
        print("\n  KV cache snapshot/restore: PASS")
        print("  Prefix cache preserved across cold start!")
    elif cached_after == 0:
        print("\n  WARNING: No cached blocks after restore — prefix cache may not be working")
    else:
        print("\n  FAIL: Output mismatch after KV restore")


if __name__ == "__main__":
    main()
