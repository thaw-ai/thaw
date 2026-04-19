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
import datetime
import gc
import json
import logging
import os
import platform
import subprocess
import time

# vLLM v0.19 V1 engine runs the model in a subprocess by default.
# Disable multiprocessing so the model stays in-process and we can
# directly access the nn.Module for freeze/restore.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch


def hardware_fingerprint() -> dict:
    """Collect a stable fingerprint of the host for reproducibility.

    Captures GPU name + driver + CUDA version, CPU, RAM, OS, and Python
    info. Queried once at the start of a run and embedded in the JSON
    report so a claim can be traced back to the exact hardware it was
    measured on. Silent on missing fields — this never fails the run.
    """
    fp = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()}",
        "arch": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        fp["torch"] = torch.__version__
        fp["cuda"] = torch.version.cuda
        fp["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["gpu_count"] = torch.cuda.device_count()
            props = torch.cuda.get_device_properties(0)
            fp["gpu_mem_gb"] = round(props.total_memory / 1e9, 2)
            fp["gpu_capability"] = f"{props.major}.{props.minor}"
    except Exception as e:
        fp["torch_probe_error"] = str(e)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            fp["nvidia_driver"] = out.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    fp["ram_gb"] = round(kb / 1_000_000, 1)
                    break
    except OSError:
        pass
    try:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if sha.returncode == 0:
            fp["thaw_git_sha"] = sha.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        import thaw
        fp["thaw_native_version"] = getattr(thaw, "__version__", "unknown")
    except ImportError:
        fp["thaw_native_version"] = "not installed"
    return fp


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
    parser.add_argument(
        "--json-out",
        default=None,
        help="Path to write a structured JSON report with all timings, "
             "throughput, hardware fingerprint, and bit-identity checks. "
             "Intended for downstream aggregation by benchmarks/run_validation.py.",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="tensor_parallel_size. When >1, freeze/restore dispatch via "
             "collective_rpc (freeze_model_tp / restore_model_tp) and Phase 4 "
             "pre-staged RAM is skipped (restore_model_from_ram is per-worker; "
             "TP variant not wired yet — documented gap, not a test failure).",
    )
    parser.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.25,
        help="gpu_memory_utilization. Raise for larger models (70B TP=2 needs ~0.85).",
    )
    args = parser.parse_args()

    # Accumulate structured results alongside human-readable stdout. Written
    # at the end of main() iff --json-out was supplied. We seed with hardware
    # fingerprint and config so a crashed run still emits partial context.
    results = {
        "schema_version": 1,
        "hardware": hardware_fingerprint(),
        "config": {
            "model": args.model,
            "snapshot": args.snapshot,
            "skip_warm": args.skip_warm,
            "skip_cache_drop": args.skip_cache_drop,
            "tp": args.tp,
            "gpu_mem_util": args.gpu_mem_util,
        },
        "status": "started",
    }

    def _dump_json():
        if args.json_out:
            try:
                with open(args.json_out, "w") as fh:
                    json.dump(results, fh, indent=2, default=str)
            except OSError as e:
                print(f"[json-out] write failed: {e}")

    profiler = InitProfiler() if args.profile_init else None
    attach_profiler(profiler)

    from vllm import LLM, SamplingParams
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined, restore_model_from_ram
    from thaw_vllm.snapshot import freeze_model_tp, restore_model_tp

    tp = args.tp
    gpu_mem = args.gpu_mem_util
    tp_mode = tp > 1

    def _freeze(llm_obj, model_obj, path):
        # TP>1: dispatch to every worker via collective_rpc. Each rank writes
        # rank_snapshot_path(base, rank). TP=1: direct in-process freeze of the
        # single model instance (same as before).
        if tp_mode:
            return freeze_model_tp(llm_obj, path)
        return freeze_model_pipelined(model_obj, path)

    def _restore(llm_obj, model_obj, path):
        if tp_mode:
            return restore_model_tp(llm_obj, path)
        return restore_model_pipelined(model_obj, path)

    prompt = "The future of artificial intelligence is"
    sampling = SamplingParams(temperature=0, max_tokens=50)
    results["config"]["prompt"] = prompt
    results["config"]["max_tokens"] = 50
    results["phases"] = {}

    def _run():
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
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,  # vLLM doesn't fully free on del
        )
        normal_time = time.perf_counter() - t0
        print(f"Normal load time: {normal_time:.1f}s")
        if profiler:
            print(profiler.waterfall(normal_time, "Phase 1 (normal LLM init)"))

        # Generate reference output (greedy = deterministic)
        out = llm.generate([prompt], sampling)
        ref_text = out[0].outputs[0].text
        print(f"Output: {ref_text}")
        results["phases"]["normal_cold_start"] = {
            "elapsed_s": normal_time,
            "output": ref_text,
        }
        _dump_json()

        # =================================================================
        # Phase 2: Freeze model weights to .thaw snapshot (pipelined)
        # =================================================================
        print("\n" + "=" * 60)
        print("Phase 2: Freeze to .thaw snapshot (pipelined)")
        print("=" * 60)

        model = None if tp_mode else find_model(llm)
        stats = _freeze(llm, model, args.snapshot)
        size_gb = stats["total_bytes"] / 1e9
        backend = stats.get("backend", "python")
        print(f"Frozen: {stats['num_regions']} regions, {size_gb:.2f} GB")
        print(f"Freeze: {stats['elapsed_s']:.1f}s ({stats['throughput_gb_s']:.2f} GB/s) [{backend}]")
        results["phases"]["freeze"] = {
            "elapsed_s": stats["elapsed_s"],
            "throughput_gb_s": stats["throughput_gb_s"],
            "total_bytes": stats["total_bytes"],
            "num_regions": stats["num_regions"],
            "backend": backend,
            "snapshot_size_gb": size_gb,
        }
        _dump_json()

        # Tear down to free GPU memory for the next load.
        # vLLM doesn't fully release CUDA allocations on del, so we
        # synchronize + double gc to catch weak references.
        del llm
        if model is not None:
            del model
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        # Drop the snapshot file(s) from the Linux page cache so Phase 3 really
        # reads from NVMe. TP>1 writes rank-suffixed files alongside the base
        # path (weights.rank1.thaw, etc). Without dropping all of them, Phase 3
        # silently reads from RAM and the "disk restore" numbers are lies.
        def _all_rank_paths(base: str) -> list[str]:
            paths = [base]
            if tp_mode:
                import glob
                root, ext = os.path.splitext(base)
                paths.extend(sorted(glob.glob(f"{root}.rank*{ext}")))
            return paths

        if args.skip_cache_drop:
            print("\n[WARN] --skip-cache-drop: Phase 3 will read from warm cache (NOT a real cold start)")
            cache_msg = "skipped (--skip-cache-drop)"
        else:
            msgs = [drop_file_from_cache(p) for p in _all_rank_paths(args.snapshot)]
            cache_msg = " | ".join(msgs)
            print(f"\n[cache] {cache_msg}")
        results["phases"]["cache_drop"] = cache_msg

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
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            load_format="dummy",
        )
        init_time = time.perf_counter() - t0
        print(f"Dummy init: {init_time:.1f}s")
        if profiler:
            print(profiler.waterfall(init_time, "Phase 3 (dummy LLM init)"))

        # Step B: restore real weights from .thaw snapshot (cold cache).
        t0 = time.perf_counter()
        model = None if tp_mode else find_model(llm_thaw)
        rstats = _restore(llm_thaw, model, args.snapshot)
        restore_time = time.perf_counter() - t0
        backend = rstats.get("backend", "python")
        thaw_total = init_time + restore_time
        print(f"Restore (cold): {restore_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s) [{backend}]")
        print(f"Total thaw cold start: {thaw_total:.1f}s")

        # Generate and compare
        out = llm_thaw.generate([prompt], sampling)
        thaw_text = out[0].outputs[0].text
        print(f"Output: {thaw_text}")
        cold_match = thaw_text == ref_text
        results["phases"]["thaw_cold_nvme"] = {
            "dummy_init_s": init_time,
            "restore_s": restore_time,
            "restore_throughput_gb_s": rstats["throughput_gb_s"],
            "total_s": thaw_total,
            "backend": backend,
            "output": thaw_text,
            "output_match_ref": cold_match,
        }
        _dump_json()

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
            del llm_thaw
            if model is not None:
                del model
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
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_mem,
                load_format="dummy",
            )
            warm_init_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            model = None if tp_mode else find_model(llm_warm)
            warm_stats = _restore(llm_warm, model, args.snapshot)
            warm_restore_time = time.perf_counter() - t0
            warm_throughput = warm_stats["throughput_gb_s"]
            print(f"Restore (warm): {warm_restore_time:.1f}s ({warm_throughput:.2f} GB/s)")

            # Sanity-check output matches.
            out = llm_warm.generate([prompt], sampling)
            warm_text = out[0].outputs[0].text
            warm_match = warm_text == ref_text
            results["phases"]["thaw_warm_cache"] = {
                "dummy_init_s": warm_init_time,
                "restore_s": warm_restore_time,
                "restore_throughput_gb_s": warm_throughput,
                "total_s": warm_init_time + warm_restore_time,
                "output": warm_text,
                "output_match_ref": warm_match,
            }
            _dump_json()

            # Swap references so Phase 4 teardown uses the current engine.
            llm_thaw = llm_warm
        else:
            warm_match = None

        # =================================================================
        # Phase 4: Pre-staged RAM (upper-bound scenario) — TP=1 ONLY
        # =================================================================
        # restore_model_from_ram is per-worker and there is no TP dispatcher
        # wired yet. Running it under TP>1 would only pre-stage rank 0.
        # Rather than produce a misleading number, we skip Phase 4 for TP>1
        # and mark it explicitly in the JSON so the consumer knows it's a
        # documented gap, not a failure.
        ram_match = None
        ram_hot_total = None
        ram_backend = None
        ram_dma_time = None
        ram_rstats = None
        ram_init_time = None
        if tp_mode:
            print("\n" + "=" * 60)
            print("Phase 4: SKIPPED under TP>1 (no TP dispatcher for restore_model_from_ram)")
            print("=" * 60)
            results["phases"]["thaw_prestaged_ram"] = {"skipped": True, "reason": "tp>1"}
            _dump_json()
        else:
            print("\n" + "=" * 60)
            print("Phase 4: Pre-staged RAM (upper-bound — snapshot already in memory)")
            print("=" * 60)
            print("This is the ceiling for a warm-RAM hot path (e.g. thaw serve daemon).")

            # Tear down previous model to free GPU memory.
            del llm_thaw
            if model is not None:
                del model
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
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_mem,
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
            ram_match = ref_text == ram_text
            results["phases"]["thaw_prestaged_ram"] = {
                "dummy_init_s": ram_init_time,
                "dma_s": ram_dma_time,
                "read_s": ram_read_time,
                "dma_throughput_gb_s": ram_rstats["throughput_gb_s"],
                "total_s": ram_hot_total,
                "backend": ram_backend,
                "output": ram_text,
                "output_match_ref": ram_match,
            }
            _dump_json()

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
        if ram_hot_total is not None:
            print(f"  Thaw pre-staged RAM:             {ram_hot_total:.1f}s  [{ram_backend}]")
            print(f"    - dummy init:                  {ram_init_time:.1f}s")
            print(f"    - DMA to GPU:                  {ram_dma_time:.1f}s ({ram_rstats['throughput_gb_s']:.2f} GB/s)")
            print(f"    (scenario: thaw serve has snapshot pre-loaded into RAM)")
        else:
            print(f"  Thaw pre-staged RAM:             SKIPPED (TP>1 — no dispatcher wired)")
        print()
        print(f"  Freeze:                          {stats['elapsed_s']:.1f}s ({stats['throughput_gb_s']:.2f} GB/s) [{stats.get('backend', 'python')}]")
        print(f"  TP size:                         {tp}")
        print()
        speedup_cold = normal_time / thaw_total
        speedup_ram = (normal_time / ram_hot_total) if ram_hot_total is not None else None
        print(f"  Speedup (cold-cache NVMe):       {speedup_cold:.1f}x  <-- headline")
        speedup_warm = None
        if warm_restore_time is not None:
            warm_total = init_time + warm_restore_time
            speedup_warm = normal_time / warm_total
            print(f"  Speedup (warm-cache):            {speedup_warm:.1f}x")
        if speedup_ram is not None:
            print(f"  Speedup (pre-staged RAM):        {speedup_ram:.1f}x")
        match = ref_text == thaw_text
        print()
        print(f"  Output match (cold NVMe):  {'PASS' if match else 'FAIL'}")
        if warm_match is not None:
            print(f"  Output match (warm):       {'PASS' if warm_match else 'FAIL'}")
        if ram_match is not None:
            print(f"  Output match (RAM):        {'PASS' if ram_match else 'FAIL'}")
        if not match:
            print(f"    Expected: {ref_text[:100]}")
            print(f"    Got:      {thaw_text[:100]}")
        if ram_match is False:
            print(f"    Expected: {ref_text[:100]}")
            print(f"    Got (RAM): {ram_text[:100]}")

        # bit-identity rollup treats a skipped phase as neutral (None), not a
        # failure — the failure would be a phase that ran and produced wrong
        # output, not a phase we deliberately didn't run.
        all_match = match
        if warm_match is not None:
            all_match = all_match and warm_match
        if ram_match is not None:
            all_match = all_match and ram_match
        results["summary"] = {
            "tp": tp,
            "speedup_cold_nvme": speedup_cold,
            "speedup_warm_cache": speedup_warm,
            "speedup_prestaged_ram": speedup_ram,
            "all_outputs_match_ref": bool(all_match),
        }

    try:
        _run()
    except BaseException as e:
        results["status"] = "error"
        results["error"] = f"{type(e).__name__}: {e}"
        _dump_json()
        raise
    results["status"] = "ok"
    _dump_json()


if __name__ == "__main__":
    main()
