#!/usr/bin/env python3
"""Diagnose thaw restore throughput bottleneck.

Runs targeted benchmarks to isolate:
1. Raw pread speed from /dev/shm (RAM)
2. Raw cudaMemcpy H2D from pinned memory (PyTorch)
3. Raw cudaMemcpy H2D from pageable memory (PyTorch)
4. Rust pipelined restore from /dev/shm (file-based)
5. Rust pipelined restore from RAM (bytes-based)
6. Rust pipelined restore with different chunk sizes

Usage:
    python demos/diagnose_throughput.py
"""

import os
import time
import sys

# Suppress vLLM spam
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

SIZE_GB = 2  # 2 GB test payload
SIZE = SIZE_GB * 1024 * 1024 * 1024
TMPDIR = "/dev/shm/thaw_diag"
TESTFILE = f"{TMPDIR}/test.bin"

os.makedirs(TMPDIR, exist_ok=True)


def banner(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def test_pread_speed():
    """Raw pread speed from /dev/shm."""
    banner("Test 1: Raw pread from /dev/shm")

    # Write test file
    print(f"  Writing {SIZE_GB} GB test file to {TESTFILE}...")
    data = os.urandom(64 * 1024 * 1024)  # 64 MB random data
    with open(TESTFILE, "wb") as f:
        for _ in range(SIZE // len(data)):
            f.write(data)

    # Read with Python (buffered)
    print("  Reading back (Python buffered read)...")
    t0 = time.perf_counter()
    with open(TESTFILE, "rb") as f:
        while f.read(64 * 1024 * 1024):
            pass
    elapsed = time.perf_counter() - t0
    print(f"  Python buffered read: {SIZE/1e9/elapsed:.2f} GB/s ({elapsed:.3f}s)")

    # Read with os.pread (direct syscall)
    print("  Reading back (os.pread, 64 MB chunks)...")
    fd = os.open(TESTFILE, os.O_RDONLY)
    t0 = time.perf_counter()
    offset = 0
    chunk = 64 * 1024 * 1024
    while offset < SIZE:
        os.pread(fd, chunk, offset)
        offset += chunk
    elapsed = time.perf_counter() - t0
    os.close(fd)
    print(f"  os.pread (64 MB chunks): {SIZE/1e9/elapsed:.2f} GB/s ({elapsed:.3f}s)")


def test_cuda_memcpy():
    """Raw cudaMemcpy H2D throughput."""
    import torch

    banner("Test 2: Raw cudaMemcpy H2D")

    # Pageable memory → GPU
    print(f"  Allocating {SIZE_GB} GB pageable host tensor...")
    host_pageable = torch.randn(SIZE // 4, dtype=torch.float32)
    gpu = torch.empty_like(host_pageable, device="cuda:0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu.copy_(host_pageable)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Pageable → GPU: {SIZE/1e9/elapsed:.2f} GB/s ({elapsed:.3f}s)")

    del gpu

    # Pinned memory → GPU
    print(f"  Allocating {SIZE_GB} GB pinned host tensor...")
    host_pinned = torch.randn(SIZE // 4, dtype=torch.float32).pin_memory()
    gpu = torch.empty_like(host_pinned, device="cuda:0")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu.copy_(host_pinned)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Pinned → GPU: {SIZE/1e9/elapsed:.2f} GB/s ({elapsed:.3f}s)")

    # Pinned memory → GPU with non_blocking
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gpu.copy_(host_pinned, non_blocking=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  Pinned → GPU (non_blocking): {SIZE/1e9/elapsed:.2f} GB/s ({elapsed:.3f}s)")

    del host_pageable, host_pinned, gpu
    torch.cuda.empty_cache()


def test_rust_pipelined():
    """Rust pipelined restore with different configs."""
    import torch

    banner("Test 3: Rust pipelined restore")

    try:
        import thaw as _thaw
        if not hasattr(_thaw, "restore_from_file_pipelined"):
            print("  ERROR: Rust restore_from_file_pipelined not found")
            return
        print(f"  Rust thaw module loaded: {_thaw}")
    except ImportError:
        print("  ERROR: Rust thaw module not available")
        return

    # Create a real thaw snapshot via freeze
    from vllm import LLM, SamplingParams
    import thaw_vllm
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    snap_path = f"{TMPDIR}/diag.thaw"

    print(f"\n  Loading {model_id} for freeze...")
    llm = LLM(
        model=model_id,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.40,
    )

    # Find the model module
    engine = llm.llm_engine
    for path_fn in [
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_executor.model_runner.model,
    ]:
        try:
            model = path_fn()
            break
        except (AttributeError, TypeError):
            continue

    # Count total params
    total_bytes = sum(p.data.nbytes for p in model.parameters() if p.is_cuda)
    print(f"  Model params: {total_bytes/1e9:.2f} GB")

    # Freeze
    print("  Freezing to /dev/shm...")
    fstats = freeze_model_pipelined(model, snap_path)
    print(f"  Freeze: {fstats['throughput_gb_s']:.2f} GB/s ({fstats['elapsed_s']:.2f}s)")
    print(f"  Freeze backend: {fstats.get('backend', 'unknown')}")

    # Teardown
    import gc
    del llm, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    # Restore with dummy init — test different chunk sizes
    for chunk_mb in [32, 64, 128, 256, 512]:
        print(f"\n  --- Restore with chunk_size={chunk_mb} MB ---")

        llm2 = LLM(
            model=model_id,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.40,
            load_format="dummy",
        )

        engine2 = llm2.llm_engine
        for path_fn in [
            lambda: engine2.model_executor.driver_worker.model_runner.model,
            lambda: engine2.engine_core.model_runner.model,
            lambda: engine2.engine_core.model_executor.driver_worker.model_runner.model,
            lambda: engine2.engine_core.model_executor.model_runner.model,
        ]:
            try:
                model2 = path_fn()
                break
            except (AttributeError, TypeError):
                continue

        rstats = restore_model_pipelined(
            model2, snap_path, chunk_size_mb=chunk_mb
        )
        print(f"  Restore: {rstats['throughput_gb_s']:.2f} GB/s ({rstats['elapsed_s']:.2f}s)")
        print(f"  Backend: {rstats.get('backend', 'unknown')}")

        del llm2, model2
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(1)

    # Test restore_from_bytes (RAM path) if available
    if hasattr(_thaw, "restore_from_bytes_pipelined"):
        print(f"\n  --- Restore from RAM (restore_from_bytes_pipelined) ---")

        llm3 = LLM(
            model=model_id,
            dtype="float16",
            enforce_eager=True,
            gpu_memory_utilization=0.40,
            load_format="dummy",
        )
        engine3 = llm3.llm_engine
        for path_fn in [
            lambda: engine3.model_executor.driver_worker.model_runner.model,
            lambda: engine3.engine_core.model_runner.model,
            lambda: engine3.engine_core.model_executor.driver_worker.model_runner.model,
            lambda: engine3.engine_core.model_executor.model_runner.model,
        ]:
            try:
                model3 = path_fn()
                break
            except (AttributeError, TypeError):
                continue

        from thaw_vllm.snapshot import restore_model_from_ram
        rstats = restore_model_from_ram(model3, snap_path)
        print(f"  RAM restore: {rstats['throughput_gb_s']:.2f} GB/s ({rstats['elapsed_s']:.2f}s)")
        print(f"  Backend: {rstats.get('backend', 'unknown')}")

        del llm3, model3
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    else:
        print("\n  restore_from_bytes_pipelined not available, skipping RAM test")


def test_direct_torch_restore():
    """Baseline: load snapshot with pure Python/PyTorch (no Rust)."""
    import torch

    banner("Test 4: Pure Python restore (torch memcpy baseline)")

    from vllm import LLM

    model_id = "microsoft/Phi-3-mini-4k-instruct"
    snap_path = f"{TMPDIR}/diag.thaw"

    if not os.path.exists(snap_path):
        print("  Skipping — no snapshot file (run test 3 first)")
        return

    llm = LLM(
        model=model_id,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.40,
        load_format="dummy",
    )
    engine = llm.llm_engine
    for path_fn in [
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_executor.model_runner.model,
    ]:
        try:
            model = path_fn()
            break
        except (AttributeError, TypeError):
            continue

    # Force Python-only restore
    from thaw_vllm.snapshot import restore_model
    t0 = time.perf_counter()
    rstats = restore_model(model, snap_path)
    elapsed = time.perf_counter() - t0
    print(f"  Python restore: {rstats['throughput_gb_s']:.2f} GB/s ({rstats['elapsed_s']:.2f}s)")

    import gc
    del llm, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("thaw throughput diagnostic")
    print(f"  Test payload: {SIZE_GB} GB")
    print(f"  Snapshot dir: {TMPDIR}")

    test_pread_speed()
    test_cuda_memcpy()
    test_rust_pipelined()
    test_direct_torch_restore()

    banner("Summary")
    print("  Compare the numbers above to identify the bottleneck.")
    print("  Expected on H100 SXM:")
    print("    pread /dev/shm: 10-20 GB/s")
    print("    Pinned H2D:     20-26 GB/s")
    print("    Pageable H2D:   3-6 GB/s")
    print("    Rust pipelined: 10-15 GB/s")
    print()
    print("  If Rust pipelined matches pageable H2D (~4 GB/s),")
    print("  the pinned memory allocation may be failing silently.")

    # Cleanup
    print(f"\n  Cleaning up {TMPDIR}...")
    import shutil
    shutil.rmtree(TMPDIR, ignore_errors=True)
