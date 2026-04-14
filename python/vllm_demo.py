"""
vllm_demo.py — thaw-powered vLLM cold start vs normal cold start.

Demonstrates the real use case:
  1. Normal vLLM cold start (download + load + init)
  2. Freeze the loaded model to a .thaw snapshot
  3. Thaw-powered cold start (dummy init + restore from snapshot)
  4. Both produce identical inference output (greedy decoding)

Usage:
    python vllm_demo.py [--model MODEL] [--snapshot PATH]

Requires: vllm, torch. Run on a GPU with enough VRAM for the model.
"""

import argparse
import gc
import os
import time

# vLLM v0.19 V1 engine runs the model in a subprocess by default.
# Disable multiprocessing so the model stays in-process and we can
# directly access the nn.Module for freeze/restore.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


def find_model(llm):
    """Extract the nn.Module from a vLLM LLM instance."""
    engine = llm.llm_engine

    # V0 engine: model_executor.driver_worker.model_runner.model
    try:
        return engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass

    # V1 serial engine: engine_core has the model runner in-process
    try:
        core = engine.engine_core
        if hasattr(core, 'model_runner'):
            return core.model_runner.model
        if hasattr(core, 'model_executor'):
            me = core.model_executor
            return me.driver_worker.model_runner.model
    except AttributeError:
        pass

    # V1 serial: might be nested differently
    try:
        return engine.engine_core.model_executor.model_runner.model
    except AttributeError:
        pass

    # get_model() API (PR #10353)
    try:
        return engine.get_model()
    except AttributeError:
        pass

    # Debug: print structure so we can find it
    print("\n[DEBUG] Could not find model. Dumping structure:")
    print(f"  llm.llm_engine type: {type(engine).__name__}")
    for attr in sorted(dir(engine)):
        if attr.startswith('_'):
            continue
        val = getattr(engine, attr, None)
        if not callable(val):
            print(f"  .{attr}: {type(val).__name__}")
    if hasattr(engine, 'engine_core'):
        core = engine.engine_core
        print(f"\n  engine_core type: {type(core).__name__}")
        for attr in sorted(dir(core)):
            if attr.startswith('_'):
                continue
            val = getattr(core, attr, None)
            if not callable(val):
                print(f"  .engine_core.{attr}: {type(val).__name__}")

    raise RuntimeError(
        "Could not locate nn.Module in vLLM LLM instance. "
        "See debug output above."
    )


def main():
    parser = argparse.ArgumentParser(description="thaw + vLLM cold start demo")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--snapshot", default="snapshot.thaw")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined, restore_model_from_ram

    prompt = "The future of artificial intelligence is"
    sampling = SamplingParams(temperature=0, max_tokens=50)

    # =================================================================
    # Phase 1: Normal vLLM cold start
    # =================================================================
    print("=" * 60)
    print("Phase 1: Normal vLLM cold start")
    print("=" * 60)

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,      # skip CUDA graph compilation
        tensor_parallel_size=1,  # single GPU
        gpu_memory_utilization=0.25,  # conservative — vLLM doesn't fully free on del
    )
    normal_time = time.perf_counter() - t0
    print(f"Normal load time: {normal_time:.1f}s")

    # Generate reference output (greedy = deterministic)
    out = llm.generate([prompt], sampling)
    ref_text = out[0].outputs[0].text
    print(f"Output: {ref_text}")

    # =================================================================
    # Phase 2: Freeze model weights to .thaw snapshot (pipelined)
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Freeze to .thaw snapshot (pipelined)")
    print("=" * 60)

    model = find_model(llm)
    stats = freeze_model_pipelined(model, args.snapshot)
    size_gb = stats["total_bytes"] / 1e9
    backend = stats.get("backend", "python")
    print(f"Frozen: {stats['num_regions']} regions, {size_gb:.2f} GB")
    print(f"Freeze: {stats['elapsed_s']:.1f}s ({stats['throughput_gb_s']:.2f} GB/s) [{backend}]")

    # Tear down to free GPU memory for the next load.
    # vLLM doesn't fully release CUDA allocations on del, so we
    # synchronize + double gc to catch weak references.
    del llm, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    # Phase 3: Thaw-powered vLLM cold start
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Thaw-powered vLLM cold start")
    print("=" * 60)

    # Step A: dummy init — creates model architecture with random weights,
    # skipping the expensive weight download + deserialization.
    t0 = time.perf_counter()
    llm_thaw = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        load_format="dummy",
    )
    init_time = time.perf_counter() - t0
    print(f"Dummy init: {init_time:.1f}s")

    # Step B: restore real weights from .thaw snapshot.
    t0 = time.perf_counter()
    model = find_model(llm_thaw)
    rstats = restore_model_pipelined(model, args.snapshot)
    restore_time = time.perf_counter() - t0
    backend = rstats.get("backend", "python")
    thaw_total = init_time + restore_time
    print(f"Restore: {restore_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s) [{backend}]")
    print(f"Total thaw cold start: {thaw_total:.1f}s")

    # Generate and compare
    out = llm_thaw.generate([prompt], sampling)
    thaw_text = out[0].outputs[0].text
    print(f"Output: {thaw_text}")

    # =================================================================
    # Phase 4: Thaw-powered cold start from RAM
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Thaw-powered cold start from RAM")
    print("=" * 60)

    # Tear down Phase 3 model to free GPU memory.
    del llm_thaw, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # Step A: dummy init (same as Phase 3).
    t0 = time.perf_counter()
    llm_ram = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.25,
        load_format="dummy",
    )
    ram_init_time = time.perf_counter() - t0
    print(f"Dummy init: {ram_init_time:.1f}s")

    # Step B: restore from RAM (snapshot pre-loaded into memory).
    model = find_model(llm_ram)
    ram_rstats = restore_model_from_ram(model, args.snapshot)
    ram_dma_time = ram_rstats['elapsed_s']  # DMA only, excludes file read
    ram_read_time = ram_rstats.get('read_time_s', 0)
    ram_backend = ram_rstats.get("backend", "python")
    ram_hot_total = ram_init_time + ram_dma_time   # production: snapshot already in RAM
    ram_cold_total = ram_init_time + ram_read_time + ram_dma_time  # includes file read
    print(f"File read to RAM: {ram_read_time:.2f}s (not in hot path)")
    print(f"DMA to GPU:       {ram_dma_time:.1f}s ({ram_rstats['throughput_gb_s']:.2f} GB/s) [{ram_backend}]")
    print(f"Hot path total:   {ram_hot_total:.1f}s (init + DMA, snapshot pre-loaded)")
    print(f"Cold path total:  {ram_cold_total:.1f}s (init + read + DMA)")

    # Generate and compare.
    out = llm_ram.generate([prompt], sampling)
    ram_text = out[0].outputs[0].text
    print(f"Output: {ram_text}")

    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Normal vLLM cold start:    {normal_time:.1f}s")
    print(f"  Thaw (disk):               {thaw_total:.1f}s  [{backend}]")
    print(f"    - dummy init:            {init_time:.1f}s")
    print(f"    - weight restore:        {restore_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s)")
    print(f"  Thaw (RAM, hot path):      {ram_hot_total:.1f}s  [{ram_backend}]")
    print(f"    - dummy init:            {ram_init_time:.1f}s")
    print(f"    - DMA to GPU:            {ram_dma_time:.1f}s ({ram_rstats['throughput_gb_s']:.2f} GB/s)")
    print(f"  Thaw (RAM, cold):          {ram_cold_total:.1f}s  (includes {ram_read_time:.1f}s file read)")
    print(f"  Freeze:                    {stats['elapsed_s']:.1f}s ({stats['throughput_gb_s']:.2f} GB/s) [{stats.get('backend', 'python')}]")
    print(f"  Speedup (disk):            {normal_time / thaw_total:.1f}x")
    print(f"  Speedup (RAM, hot path):   {normal_time / ram_hot_total:.1f}x")
    print(f"  Speedup (RAM, cold):       {normal_time / ram_cold_total:.1f}x")
    match = ref_text == thaw_text
    ram_match = ref_text == ram_text
    print(f"  Output match (disk):     {'PASS' if match else 'FAIL'}")
    print(f"  Output match (RAM):      {'PASS' if ram_match else 'FAIL'}")
    if not match:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got:      {thaw_text[:100]}")
    if not ram_match:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got (RAM): {ram_text[:100]}")


if __name__ == "__main__":
    main()
