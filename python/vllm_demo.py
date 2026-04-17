"""
vllm_demo.py — thaw-powered vLLM cold start vs normal cold start.

Demonstrates the real use case:
  1. Normal vLLM cold start (download + load + init)
  2. Freeze the loaded model to a .thaw snapshot
  3. Thaw-powered cold start with COLD page cache (honest NVMe read)
  3b. Thaw-powered cold start with WARM page cache (post-warm, optional)
  4. Thaw-powered cold start from pre-staged RAM (upper-bound scenario)
  5. All produce identical inference output (greedy decoding)

Methodology:
  - Between Phase 2 (freeze) and Phase 3 (disk restore) we flush the file
    from the Linux page cache via posix_fadvise(POSIX_FADV_DONTNEED). This
    forces Phase 3 to read from NVMe instead of RAM — otherwise the fresh
    freeze leaves the file warm in cache and "disk restore" is a lie.
  - Phase 3b optionally re-runs restore without dropping cache, showing the
    warm-cache ceiling.
  - Phase 4 uses restore_model_from_ram with an explicitly pre-read buffer,
    representing "snapshot pre-staged in memory" (e.g. thaw serve daemon).

Usage:
    python vllm_demo.py [--model MODEL] [--snapshot PATH] [--skip-warm]

Requires: vllm, torch. Run on a GPU with enough VRAM for the model.
"""

import argparse
import gc
import logging
import os
import time

# vLLM v0.19 V1 engine runs the model in a subprocess by default.
# Disable multiprocessing so the model stays in-process and we can
# directly access the nn.Module for freeze/restore.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


class InitProfiler(logging.Handler):
    """Capture timestamps of vLLM log messages to profile LLM() construction.

    vLLM emits INFO-level log lines at each major init phase (engine setup,
    model init, KV cache profiling, CUDA graph capture, etc.). Attaching this
    handler lets us reconstruct a timeline without monkey-patching vLLM internals.
    """

    def __init__(self):
        super().__init__()
        self.events = []
        self.start = None

    def emit(self, record):
        t = time.perf_counter()
        if self.start is None:
            self.start = t
        self.events.append((t - self.start, record.name, record.levelname, record.getMessage()))

    def reset(self):
        self.events = []
        self.start = None

    def waterfall(self, total_elapsed: float, label: str) -> str:
        """Render a compact waterfall of init events."""
        if not self.events:
            return f"[profile] no events captured for {label}"
        lines = [f"[profile] {label} breakdown (total {total_elapsed:.2f}s):"]
        prev_t = 0.0
        for t, name, lvl, msg in self.events:
            delta = t - prev_t
            msg_short = msg.replace("\n", " ")[:90]
            lines.append(f"  +{t:6.2f}s (Δ{delta:5.2f}s) [{name}] {msg_short}")
            prev_t = t
        tail = total_elapsed - prev_t
        if tail > 0.1:
            lines.append(f"  +{total_elapsed:6.2f}s (Δ{tail:5.2f}s) [tail — after last log event]")
        return "\n".join(lines)


def attach_profiler(profiler: "InitProfiler | None"):
    """Attach the profiler to the vllm + torch loggers. Returns the list of loggers touched."""
    if profiler is None:
        return []
    touched = []
    for name in ("vllm", "vllm.engine", "vllm.worker", "vllm.executor",
                 "vllm.v1", "vllm.model_executor", "torch", "thaw"):
        lg = logging.getLogger(name)
        lg.addHandler(profiler)
        if lg.level == 0 or lg.level > logging.INFO:
            lg.setLevel(logging.INFO)
        touched.append(lg)
    return touched


def drop_file_from_cache(path: str) -> str:
    """Drop file from Linux page cache without root.

    Tries vmtouch -e first (hard eviction, verified via mincore). Falls back
    to posix_fadvise(POSIX_FADV_DONTNEED) — advisory, the kernel may ignore
    it on hosts with abundant free RAM.

    Returns a human-readable description of what happened.
    """
    if not os.path.exists(path):
        return f"skip (file missing: {path})"

    size_gb = os.path.getsize(path) / 1e9

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    except OSError:
        pass

    import shutil
    import subprocess

    vmtouch = shutil.which("vmtouch")
    if vmtouch is not None:
        try:
            res = subprocess.run(
                [vmtouch, "-e", path],
                capture_output=True, text=True, timeout=30,
            )
            if res.returncode == 0:
                verify = subprocess.run(
                    [vmtouch, "-v", path],
                    capture_output=True, text=True, timeout=10,
                )
                resident = "0%" if " 0%" in verify.stdout or "0/" in verify.stdout else "unknown"
                return f"evicted {size_gb:.2f} GB via vmtouch -e (resident: {resident})"
            return f"vmtouch -e failed ({res.returncode}): {res.stderr.strip()}"
        except (subprocess.TimeoutExpired, OSError) as e:
            return f"vmtouch error: {e}"

    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            fadv = getattr(os, "POSIX_FADV_DONTNEED", None)
            if hasattr(os, "posix_fadvise") and fadv is not None:
                os.posix_fadvise(fd, 0, 0, fadv)
                return (
                    f"fadvise hint sent for {size_gb:.2f} GB (advisory — "
                    "kernel may ignore). Install vmtouch for hard eviction."
                )
            return "posix_fadvise unavailable (likely macOS) — cache remains warm"
        finally:
            os.close(fd)
    except OSError as e:
        return f"cache drop failed: {e}"


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
    parser.add_argument(
        "--skip-warm",
        action="store_true",
        help="Skip Phase 3b (warm-cache measurement). Default runs it.",
    )
    parser.add_argument(
        "--skip-cache-drop",
        action="store_true",
        help="DO NOT drop page cache before Phase 3. Produces dishonest warm numbers; for debugging only.",
    )
    parser.add_argument(
        "--profile-init",
        action="store_true",
        help="Profile vLLM LLM() construction via log-event timestamps. "
             "Prints a waterfall breakdown of where init time goes.",
    )
    args = parser.parse_args()

    profiler = InitProfiler() if args.profile_init else None
    attach_profiler(profiler)

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

    if profiler:
        profiler.reset()
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
    if profiler:
        print(profiler.waterfall(normal_time, "Phase 1 (normal LLM init)"))

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

    # Drop the snapshot file from the Linux page cache so Phase 3 really
    # reads from NVMe. Without this, the file is still warm from the freeze
    # we just did, and "disk restore" numbers are lies.
    if args.skip_cache_drop:
        print("\n[WARN] --skip-cache-drop: Phase 3 will read from warm cache (NOT a real cold start)")
        cache_msg = "skipped (--skip-cache-drop)"
    else:
        cache_msg = drop_file_from_cache(args.snapshot)
        print(f"\n[cache] {cache_msg}")

    # =================================================================
    # Phase 3: Thaw-powered vLLM COLD-CACHE restore (honest NVMe read)
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Thaw-powered cold start — COLD page cache (NVMe)")
    print("=" * 60)

    # Step A: dummy init — creates model architecture with random weights,
    # skipping the expensive weight download + deserialization.
    if profiler:
        profiler.reset()
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
    if profiler:
        print(profiler.waterfall(init_time, "Phase 3 (dummy LLM init)"))

    # Step B: restore real weights from .thaw snapshot (cold cache).
    t0 = time.perf_counter()
    model = find_model(llm_thaw)
    rstats = restore_model_pipelined(model, args.snapshot)
    restore_time = time.perf_counter() - t0
    backend = rstats.get("backend", "python")
    thaw_total = init_time + restore_time
    print(f"Restore (cold): {restore_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s) [{backend}]")
    print(f"Total thaw cold start: {thaw_total:.1f}s")

    # Generate and compare
    out = llm_thaw.generate([prompt], sampling)
    thaw_text = out[0].outputs[0].text
    print(f"Output: {thaw_text}")

    # =================================================================
    # Phase 3b: Warm-cache restore (optional, for comparison)
    # =================================================================
    warm_restore_time = None
    warm_throughput = None
    if not args.skip_warm:
        print("\n" + "=" * 60)
        print("Phase 3b: Warm-cache restore (after Phase 3 warmed the cache)")
        print("=" * 60)
        print("Upper-bound scenario: file is now in page cache from Phase 3's read.")

        # Tear down Phase 3 model.
        del llm_thaw, model
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        # Rebuild dummy engine (cache is still warm).
        t0 = time.perf_counter()
        llm_warm = LLM(
            model=args.model,
            dtype="float16",
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.25,
            load_format="dummy",
        )
        warm_init_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        model = find_model(llm_warm)
        warm_stats = restore_model_pipelined(model, args.snapshot)
        warm_restore_time = time.perf_counter() - t0
        warm_throughput = warm_stats["throughput_gb_s"]
        print(f"Restore (warm): {warm_restore_time:.1f}s ({warm_throughput:.2f} GB/s)")

        # Sanity-check output matches.
        out = llm_warm.generate([prompt], sampling)
        warm_text = out[0].outputs[0].text
        warm_match = warm_text == ref_text

        # Swap references so Phase 4 teardown uses the current engine.
        llm_thaw = llm_warm
    else:
        warm_match = None

    # =================================================================
    # Phase 4: Pre-staged RAM (upper-bound scenario)
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Pre-staged RAM (upper-bound — snapshot already in memory)")
    print("=" * 60)
    print("This is the ceiling for a warm-RAM hot path (e.g. thaw serve daemon).")

    # Tear down previous model to free GPU memory.
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

    # Step B: restore from RAM (snapshot already resident in page cache).
    model = find_model(llm_ram)
    ram_rstats = restore_model_from_ram(model, args.snapshot)
    ram_dma_time = ram_rstats['elapsed_s']  # DMA only, excludes file read
    ram_read_time = ram_rstats.get('read_time_s', 0)
    ram_backend = ram_rstats.get("backend", "python")
    ram_hot_total = ram_init_time + ram_dma_time   # production: snapshot pre-staged
    print(f"File read (page cache): {ram_read_time:.2f}s (excluded — pre-staged scenario)")
    print(f"DMA to GPU:             {ram_dma_time:.1f}s ({ram_rstats['throughput_gb_s']:.2f} GB/s) [{ram_backend}]")
    print(f"Pre-staged total:       {ram_hot_total:.1f}s (init + DMA)")

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
    print(f"  Normal vLLM cold start:          {normal_time:.1f}s")
    print()
    print(f"  HEADLINE — Thaw cold-cache NVMe: {thaw_total:.1f}s  [{backend}]")
    print(f"    - dummy init:                  {init_time:.1f}s")
    print(f"    - weight restore (cold):       {restore_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s)")
    print(f"    - cache state: {cache_msg}")
    print()
    if warm_restore_time is not None:
        print(f"  Thaw warm-cache restore:         {warm_restore_time:.1f}s  ({warm_throughput:.2f} GB/s)")
        print(f"    (upper-bound for disk path when file is already in page cache)")
        print()
    print(f"  Thaw pre-staged RAM:             {ram_hot_total:.1f}s  [{ram_backend}]")
    print(f"    - dummy init:                  {ram_init_time:.1f}s")
    print(f"    - DMA to GPU:                  {ram_dma_time:.1f}s ({ram_rstats['throughput_gb_s']:.2f} GB/s)")
    print(f"    (scenario: thaw serve has snapshot pre-loaded into RAM)")
    print()
    print(f"  Freeze:                          {stats['elapsed_s']:.1f}s ({stats['throughput_gb_s']:.2f} GB/s) [{stats.get('backend', 'python')}]")
    print()
    print(f"  Speedup (cold-cache NVMe):       {normal_time / thaw_total:.1f}x  <-- headline")
    if warm_restore_time is not None:
        warm_total = init_time + warm_restore_time
        print(f"  Speedup (warm-cache):            {normal_time / warm_total:.1f}x")
    print(f"  Speedup (pre-staged RAM):        {normal_time / ram_hot_total:.1f}x")
    match = ref_text == thaw_text
    ram_match = ref_text == ram_text
    print()
    print(f"  Output match (cold NVMe):  {'PASS' if match else 'FAIL'}")
    if warm_match is not None:
        print(f"  Output match (warm):       {'PASS' if warm_match else 'FAIL'}")
    print(f"  Output match (RAM):        {'PASS' if ram_match else 'FAIL'}")
    if not match:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got:      {thaw_text[:100]}")
    if not ram_match:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got (RAM): {ram_text[:100]}")


if __name__ == "__main__":
    main()
