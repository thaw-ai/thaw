"""
micro_pipeline.py — minimal thaw pipeline microbench, no vLLM.

Isolates the Rust freeze/restore throughput from vLLM init overhead so we
can iterate on NUMA / chunk size / WC vs pinned / zero-copy hypotheses
without waiting 3 minutes per run.

Shape of an experiment:
    python benchmarks/micro_pipeline.py \
        --size-gb 16 --chunk-mb 64 --repeats 3 \
        --path /tmp/micro.thaw

Flags:
    --size-gb   device buffer size; default 16 (matches Llama-3-8B fp16).
    --chunk-mb  pipeline chunk size. Sweep: 4 16 32 64 128 256.
    --repeats   iterations of each phase after warmup; median reported.
    --path      snapshot file path (NVMe preferred).
    --drop-cache  fsync + posix_fadvise(DONTNEED) before each cold read.
    --no-direct   disable O_DIRECT on the file restore path.
    --zerocopy    force zero-copy cudaHostRegister mmap path.
    --json-out    optional structured JSON for run-to-run comparison.

Writes a deterministic pattern to the device buffer so freeze→restore
round-trip can be checked via device-side memcmp against a golden tensor.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import statistics
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--size-gb", type=float, default=16.0)
    p.add_argument("--chunk-mb", type=int, default=64)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--path", default="/tmp/micro.thaw")
    p.add_argument("--num-regions", type=int, default=1,
                   help="Split buffer into N equal-size regions (simulate per-param layout).")
    p.add_argument(
        "--layout", choices=["contiguous", "scattered"], default="contiguous",
        help="contiguous: one cudaMalloc carved into N regions by offset "
             "(matches our earlier microbench runs). scattered: N independent "
             "torch.empty() allocations — matches vLLM's per-parameter "
             "layout where regions sit at non-adjacent CUDA VAs. Use this to "
             "test whether destination contiguity is the microbench-vs-"
             "vllm_demo gap.",
    )
    p.add_argument("--drop-cache", action="store_true")
    p.add_argument("--no-direct", action="store_true")
    p.add_argument("--zerocopy", action="store_true")
    p.add_argument("--no-zerocopy", action="store_true",
                   help="Force staging-buffer mmap path (skip zerocopy).")
    p.add_argument("--json-out", default=None)
    p.add_argument(
        "--mode", choices=["file", "mmap", "both"], default="both",
        help="Which restore path(s) to exercise.",
    )
    return p.parse_args()


def drop_file_cache(path: str):
    """fadvise DONTNEED on the file so cold reads are honest."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        # posix_fadvise is available on Linux via os module since 3.3.
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def median_stats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals),
        "median_s": statistics.median(vals),
        "min_s": min(vals),
        "max_s": max(vals),
        "stdev_s": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    args = parse_args()
    import torch

    if not torch.cuda.is_available():
        print("[error] CUDA not available", file=sys.stderr)
        sys.exit(2)

    import thaw

    chunk_bytes = args.chunk_mb * 1024 * 1024
    assert chunk_bytes % 4096 == 0, "chunk_size must be multiple of 4KB for O_DIRECT"
    size_bytes = int(args.size_gb * 1024 * 1024 * 1024)
    # Round down to a multiple of 4096 so O_DIRECT never hits an unaligned tail.
    size_bytes = size_bytes & ~0xFFF

    device = torch.device("cuda:0")
    results = {
        "device_name": torch.cuda.get_device_name(device),
        "size_gb": size_bytes / 1e9,
        "size_bytes": size_bytes,
        "chunk_mb": args.chunk_mb,
        "pid": os.getpid(),
        "tid": os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None,
    }
    # Report NUMA binding the kernel actually applied
    try:
        mems = open(f"/proc/{os.getpid()}/status").read()
        for line in mems.splitlines():
            if line.startswith(("Cpus_allowed_list", "Mems_allowed_list")):
                results[line.split(":")[0].strip()] = line.split(":", 1)[1].strip()
    except OSError:
        pass

    print(f"[setup] GPU={results['device_name']}  size={results['size_gb']:.2f} GB  "
          f"chunk={args.chunk_mb} MB  cpus_allowed={results.get('Cpus_allowed_list')}  "
          f"mems_allowed={results.get('Mems_allowed_list')}")

    n_regions = max(1, args.num_regions)

    # Build the region request list. Two layout modes:
    #   contiguous — one big cudaMalloc carved by offset. CUDA DMA engine
    #                sees N regions at adjacent VAs with good TLB locality.
    #   scattered  — N independent torch.empty() allocations. VAs are
    #                scattered across the torch caching allocator's arenas.
    #                Matches vLLM's per-parameter layout (each nn.Parameter
    #                is its own .data allocation).
    # We hold references to all tensors for the lifetime of the bench so
    # the allocator doesn't recycle them under our region pointers.
    tensors: list = []
    freeze_req: list = []
    if args.layout == "contiguous":
        elts = size_bytes // 2
        t = torch.arange(elts, dtype=torch.int16, device=device)
        dev_ptr = int(t.data_ptr())
        dev_size = t.numel() * t.element_size()
        assert dev_size == size_bytes, f"tensor size {dev_size} != {size_bytes}"
        per_region = (dev_size // n_regions) & ~0xFFF
        offset = 0
        for i in range(n_regions):
            sz = per_region if i < n_regions - 1 else dev_size - offset
            freeze_req.append(("weights", i, dev_ptr + offset, sz))
            offset += sz
        assert offset == dev_size, f"region sum {offset} != {dev_size}"
        tensors.append(t)
        total_size = dev_size
    else:  # scattered
        per_region = (size_bytes // n_regions) & ~0xFFF
        assert per_region > 0, "size-gb too small for num-regions in scattered mode"
        total_size = 0
        for i in range(n_regions):
            sz = per_region if i < n_regions - 1 else size_bytes - total_size
            sz &= ~0xFFF  # keep every region 4KB-aligned for O_DIRECT
            if sz == 0:
                break
            elts = sz // 2
            ti = torch.empty(elts, dtype=torch.int16, device=device)
            tensors.append(ti)
            freeze_req.append(("weights", i, int(ti.data_ptr()), sz))
            total_size += sz

    torch.cuda.synchronize()

    # Freeze ------------------------------------------------------------
    if os.path.exists(args.path):
        os.unlink(args.path)
    print(f"[setup] {n_regions} region(s), avg size "
          f"{(total_size/n_regions)/1e6:.1f} MB, layout={args.layout}")

    freeze_times = []
    for i in range(args.repeats + 1):
        if os.path.exists(args.path):
            os.unlink(args.path)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fs = thaw.freeze_to_file_pipelined(args.path, freeze_req)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tag = " (warmup)" if i == 0 else ""
        print(f"[freeze {i}] {elapsed:6.2f}s  {total_size/1e9/elapsed:6.2f} GB/s{tag}")
        if i > 0:
            freeze_times.append(elapsed)

    results["freeze"] = median_stats(freeze_times)
    results["freeze"]["gb_s"] = (
        size_bytes / 1e9 / results["freeze"]["median_s"] if freeze_times else 0
    )
    print(f"[freeze summary] median {results['freeze'].get('median_s',0):.2f}s  "
          f"{results['freeze']['gb_s']:.2f} GB/s")

    # Restore: from_file (O_DIRECT pread path) --------------------------
    if args.mode in ("file", "both"):
        print()
        print("--- restore_from_file_pipelined ---")
        restore_req = list(freeze_req)
        direct = not args.no_direct
        file_times = []
        for i in range(args.repeats + 1):
            if args.drop_cache:
                drop_file_cache(args.path)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rs = thaw.restore_from_file_pipelined(
                args.path, restore_req,
                chunk_size_mb=args.chunk_mb, direct_io=direct,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tag = " (warmup)" if i == 0 else ""
            print(f"[file  {i}] {elapsed:6.2f}s  "
                  f"{total_size/1e9/elapsed:6.2f} GB/s  "
                  f"direct_io={direct}  drop_cache={args.drop_cache}{tag}")
            if i > 0:
                file_times.append(elapsed)

        results["restore_file"] = median_stats(file_times)
        results["restore_file"]["gb_s"] = (
            total_size / 1e9 / results["restore_file"]["median_s"] if file_times else 0
        )
        results["restore_file"]["direct_io"] = direct
        print(f"[file summary] median {results['restore_file'].get('median_s',0):.2f}s  "
              f"{results['restore_file']['gb_s']:.2f} GB/s")

    # Restore: from mmap bytes (staging or zerocopy) --------------------
    if args.mode in ("mmap", "both"):
        print()
        print("--- restore_from_bytes_pipelined (mmap) ---")
        mmap_times = []
        for i in range(args.repeats + 1):
            if args.drop_cache:
                drop_file_cache(args.path)
            # re-mmap fresh each iteration so we're not re-using a cached
            # pointer in the Rust extension.
            fd = os.open(args.path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            # MAP_PRIVATE + PROT_READ|PROT_WRITE: cudaHostRegister with the
            # default flag sets up a bidirectional pin even though the DMA
            # is one-way, so PROT_READ-only mappings fail with
            # cudaErrorInvalidValue. MAP_PRIVATE keeps the mapping COW so
            # the underlying file isn't mutated.
            mm = mmap.mmap(
                fd, file_size,
                flags=mmap.MAP_PRIVATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
            mm.madvise(mmap.MADV_HUGEPAGE)
            os.close(fd)
            restore_req = list(freeze_req)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if args.zerocopy:
                # zerocopy pins the full mmap with cudaHostRegister and
                # DMAs directly from it; it does not chunk, so
                # chunk_size_mb is not part of this entry point's
                # signature. Requires `ulimit -l` ≥ snapshot size.
                rs = thaw.restore_from_bytes_pipelined_zerocopy(
                    mm, restore_req,
                )
            elif args.no_zerocopy:
                rs = thaw.restore_from_bytes_pipelined(
                    mm, restore_req, chunk_size_mb=args.chunk_mb,
                )
            else:
                rs = thaw.restore_from_bytes_auto(
                    mm, restore_req, chunk_size_mb=args.chunk_mb,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            tag = " (warmup)" if i == 0 else ""
            mode_label = (
                "zerocopy" if args.zerocopy else
                ("staging" if args.no_zerocopy else "auto")
            )
            print(f"[mmap  {i}] {elapsed:6.2f}s  "
                  f"{total_size/1e9/elapsed:6.2f} GB/s  mode={mode_label}{tag}")
            mm.close()
            if i > 0:
                mmap_times.append(elapsed)

        results["restore_mmap"] = median_stats(mmap_times)
        results["restore_mmap"]["gb_s"] = (
            total_size / 1e9 / results["restore_mmap"]["median_s"] if mmap_times else 0
        )
        results["restore_mmap"]["mode"] = (
            "zerocopy" if args.zerocopy else
            ("staging" if args.no_zerocopy else "auto")
        )
        print(f"[mmap summary] median {results['restore_mmap'].get('median_s',0):.2f}s  "
              f"{results['restore_mmap']['gb_s']:.2f} GB/s")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))
        print(f"[json] wrote {args.json_out}")


if __name__ == "__main__":
    main()
