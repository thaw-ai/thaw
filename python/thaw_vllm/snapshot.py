"""
thaw_vllm.snapshot — freeze/restore PyTorch model weights to .thaw format.

This is a pure-Python implementation of thaw's binary file format. It
produces files that are byte-compatible with the Rust implementation
(thaw-core), so a file written here can be read by the Rust side and
vice versa.

Performance strategy (from DESIGN.md §3.1 and READING_NOTES §2.2-2.3):
  - All DMA uses pinned memory (cudaMallocHost equivalent via PyTorch).
    Pageable memory causes cudaMemcpyAsync to silently fall back to
    synchronous transfer — the single worst performance trap in CUDA.
  - Freeze: async D2H copies into a pre-allocated pinned buffer, then
    write directly from that buffer (no Python heap copy).
  - Restore: read file directly into pinned buffer (readinto, zero-copy),
    single DMA to GPU, then GPU-internal scatter.
  - Bottleneck should be disk I/O, not PCIe. If PCIe is the bottleneck,
    something is wrong with the pinned memory path.
"""

import mmap
import os
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# -- thaw file format constants, must match crates/thaw-core --

MAGIC = b"THAW"
VERSION = 1
HEADER_SIZE = 4096
REGION_ENTRY_SIZE = 32

# Region kind discriminants
KIND_WEIGHTS = 0
KIND_KV_LIVE_BLOCK = 1
KIND_METADATA = 2


def _write_header(
    f,
    num_regions: int,
    vllm_commit: Optional[bytes] = None,
):
    """Write a 4096-byte thaw header."""
    buf = bytearray(HEADER_SIZE)
    buf[0:4] = MAGIC
    struct.pack_into("<I", buf, 4, VERSION)
    struct.pack_into("<Q", buf, 8, num_regions)
    struct.pack_into("<Q", buf, 16, HEADER_SIZE)  # region_table_offset
    if vllm_commit is not None:
        if len(vllm_commit) != 40:
            raise ValueError(f"vllm_commit must be 40 bytes, got {len(vllm_commit)}")
        buf[24:64] = vllm_commit
    f.write(buf)


def _write_region_entry(
    f,
    kind: int,
    logical_id: int,
    size: int,
    file_offset: int,
):
    """Write a 32-byte region table entry."""
    buf = bytearray(REGION_ENTRY_SIZE)
    struct.pack_into("<I", buf, 0, kind)
    struct.pack_into("<I", buf, 4, logical_id)
    struct.pack_into("<Q", buf, 8, size)
    struct.pack_into("<Q", buf, 16, file_offset)
    f.write(buf)


def _read_header(f):
    """Read and validate a thaw header. Returns (num_regions, vllm_commit)."""
    buf = f.read(HEADER_SIZE)
    if len(buf) < HEADER_SIZE:
        raise ValueError(f"file too short for header: {len(buf)} bytes")
    if buf[0:4] != MAGIC:
        raise ValueError(f"bad magic: {buf[0:4]!r}")
    version = struct.unpack_from("<I", buf, 4)[0]
    if version != VERSION:
        raise ValueError(f"unsupported version {version}, expected {VERSION}")
    num_regions = struct.unpack_from("<Q", buf, 8)[0]
    vllm_commit = bytes(buf[24:64])
    return num_regions, vllm_commit


def _read_region_entry(f):
    """Read a 32-byte region entry. Returns (kind, logical_id, size, file_offset)."""
    buf = f.read(REGION_ENTRY_SIZE)
    if len(buf) < REGION_ENTRY_SIZE:
        raise ValueError("truncated region entry")
    kind = struct.unpack_from("<I", buf, 0)[0]
    logical_id = struct.unpack_from("<I", buf, 4)[0]
    size = struct.unpack_from("<Q", buf, 8)[0]
    file_offset = struct.unpack_from("<Q", buf, 16)[0]
    return kind, logical_id, size, file_offset


def freeze_model(
    model: nn.Module,
    path: str,
    vllm_commit: Optional[str] = None,
) -> dict:
    """Freeze all GPU parameters of a model to a .thaw file.

    Uses pinned host memory as a staging buffer. Each parameter is copied
    from GPU to the pinned buffer via async DMA (non_blocking=True),
    then the entire buffer is written to disk in one shot.
    """
    commit_bytes = None
    if vllm_commit is not None:
        commit_bytes = vllm_commit.encode("ascii")

    # Collect all CUDA parameters in deterministic order.
    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            p = param.data.contiguous()
            params.append((name, p))

    num_regions = len(params)
    table_start = HEADER_SIZE
    payload_start = table_start + num_regions * REGION_ENTRY_SIZE
    sizes = [p.nbytes for _, p in params]
    offsets = []
    off = payload_start
    for s in sizes:
        offsets.append(off)
        off += s

    total_bytes = sum(sizes)

    if num_regions == 0:
        with open(path, "wb") as f:
            _write_header(f, 0, commit_bytes)
        return {
            "num_regions": 0, "total_bytes": 0,
            "elapsed_s": 0, "throughput_gb_s": 0, "params": [],
        }

    t0 = time.perf_counter()

    # Allocate pinned host buffer once. This is the staging area for
    # all D2H transfers — no pageable memory in the hot path.
    flat_pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)

    # Async copy each parameter's raw bytes from GPU to pinned buffer.
    # With pinned memory + non_blocking=True, each copy is truly async:
    # the CPU thread queues the DMA and returns immediately.
    offset = 0
    for _, p in params:
        nbytes = p.nbytes
        flat_pinned[offset:offset + nbytes].copy_(
            p.reshape(-1).view(torch.uint8), non_blocking=True
        )
        offset += nbytes
    torch.cuda.synchronize()  # wait for all D2H transfers

    # Write header + region table + payload. The payload write goes
    # directly from the pinned numpy buffer — no .tobytes() copy.
    with open(path, "wb") as f:
        _write_header(f, num_regions, commit_bytes)
        for i, (name, p) in enumerate(params):
            _write_region_entry(f, KIND_WEIGHTS, i, sizes[i], offsets[i])
        f.write(flat_pinned.numpy())

    del flat_pinned

    elapsed = time.perf_counter() - t0
    return {
        "num_regions": num_regions,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        "params": [(name, s) for (name, _), s in zip(params, sizes)],
    }


def restore_model(
    model: nn.Module,
    path: str,
) -> dict:
    """Restore model weights from a .thaw file onto GPU.

    Uses pinned host memory as a staging buffer. The entire payload is
    read directly into pinned memory (readinto, zero Python-heap copy),
    then transferred to GPU in a single DMA, then scattered to
    individual parameters via fast GPU-internal copies.
    """
    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            params.append((name, param))

    t0 = time.perf_counter()

    with open(path, "rb") as f:
        num_regions, vllm_commit = _read_header(f)

        if num_regions != len(params):
            raise ValueError(
                f"snapshot has {num_regions} regions but model has "
                f"{len(params)} CUDA parameters"
            )

        entries = []
        for _ in range(num_regions):
            entries.append(_read_region_entry(f))

        # Validate all entries before doing any I/O.
        total_bytes = 0
        for i, (name, param) in enumerate(params):
            kind, logical_id, size, file_offset = entries[i]
            if size != param.data.nbytes:
                raise ValueError(
                    f"size mismatch for {name}: snapshot has {size} bytes, "
                    f"parameter has {param.data.nbytes} bytes"
                )
            total_bytes += size

        # Allocate pinned host buffer. This is where ALL the data lives
        # between disk and GPU — no pageable memory in the hot path.
        flat_pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)

        # Read entire payload directly into pinned memory.
        # readinto() uses the buffer protocol — data goes straight from
        # the kernel's read buffer into our pinned pages. No Python
        # bytes object, no bytearray copy, no .pin_memory() re-copy.
        payload_start = entries[0][3] if entries else 0
        f.seek(payload_start)
        np_buf = flat_pinned.numpy()
        bytes_read = f.readinto(np_buf)
        if bytes_read != total_bytes:
            raise ValueError(
                f"truncated payload: expected {total_bytes}, got {bytes_read}"
            )

        # Single DMA: pinned CPU → GPU. With pinned memory this is a
        # true async transfer that saturates PCIe bandwidth.
        device = params[0][1].device
        flat_gpu = flat_pinned.to(device, non_blocking=True)
        torch.cuda.synchronize()
        del flat_pinned

        # Scatter from flat GPU buffer to individual parameters.
        # These are GPU-internal copies (~2 TB/s on A100) — essentially
        # free compared to the PCIe transfer.
        offset = 0
        for name, param in params:
            nbytes = param.data.nbytes
            param.data.copy_(
                flat_gpu[offset:offset + nbytes].view(param.dtype).reshape(param.shape)
            )
            offset += nbytes
        del flat_gpu

    elapsed = time.perf_counter() - t0

    return {
        "num_regions": num_regions,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
    }


def freeze_model_tp(
    llm,
    base_path: str,
    vllm_commit: Optional[str] = None,
) -> dict:
    """Freeze model weights from all tensor parallel workers.

    With TP > 1, each worker runs in a separate process. This function
    uses vLLM's collective_rpc to dispatch freeze_model to each worker.
    Each worker saves its own shard to a per-rank file:
      base_path (rank 0), base_path.rank1.thaw (rank 1), etc.

    With TP = 1, falls back to the standard single-GPU freeze.
    """
    from thaw_vllm.loader import _rank_snapshot_path

    ec = _get_engine_core_from_llm(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        model = ec.model_executor.driver_worker.model_runner.model
        try:
            return freeze_model_pipelined(model, base_path, vllm_commit)
        except Exception:
            return freeze_model(model, base_path, vllm_commit)

    # TP > 1: dispatch freeze to each worker via collective_rpc.
    # Define the function that each worker will execute.
    def _worker_freeze(worker, base_path, vllm_commit):
        """Runs inside each worker process."""
        import os
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        rank_path = _rank_snapshot_path(base_path, rank)
        model = worker.model_runner.model

        # Try Rust pipelined first (fast), fall back to pure Python
        from thaw_vllm.snapshot import freeze_model_pipelined, freeze_model
        try:
            stats = freeze_model_pipelined(model, rank_path, vllm_commit)
        except Exception:
            stats = freeze_model(model, rank_path, vllm_commit)
        stats['rank'] = rank
        stats['path'] = rank_path
        return stats

    results = ec.model_executor.collective_rpc(
        _worker_freeze,
        args=(base_path, vllm_commit),
    )

    # Aggregate stats from all workers
    total_bytes = sum(r['total_bytes'] for r in results)
    total_regions = sum(r['num_regions'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    return {
        'num_regions': total_regions,
        'total_bytes': total_bytes,
        'elapsed_s': total_elapsed,
        'throughput_gb_s': (total_bytes / 1e9) / total_elapsed if total_elapsed > 0 else 0,
        'tensor_parallel_size': tp_size,
        'per_rank': results,
    }


def _get_engine_core_from_llm(llm):
    """Navigate to an object with model_executor and vllm_config.

    Works with both V0 (no engine_core) and V1 (nested engine_core).
    """
    engine = llm.llm_engine
    # V1 engine: engine_core wraps the real core
    if hasattr(engine, 'engine_core'):
        ec = engine.engine_core
        if hasattr(ec, 'engine_core'):
            ec = ec.engine_core
        return ec
    # V0 engine: model_executor lives directly on the engine
    return engine


def restore_model_tp(
    llm,
    base_path: str,
    chunk_size_mb: int = 64,
) -> dict:
    """Restore model weights to all tensor parallel workers.

    With TP > 1, each worker runs in a separate process. This function
    uses vLLM's collective_rpc to dispatch restore to each worker.
    Each worker loads from its per-rank snapshot file:
      base_path (rank 0), base_path.rank1.thaw (rank 1), etc.

    With TP = 1, falls back to the standard single-GPU restore_model_from_ram.
    """
    from thaw_vllm.loader import _rank_snapshot_path

    ec = _get_engine_core_from_llm(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        # Single-GPU: find model and restore directly
        model = ec.model_executor.driver_worker.model_runner.model
        return restore_model_from_ram(model, base_path, chunk_size_mb)

    # TP > 1: dispatch restore to each worker via collective_rpc.
    def _worker_restore(worker, base_path, chunk_size_mb):
        """Runs inside each worker process."""
        import os
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()

        # Compute per-rank path (same logic as _rank_snapshot_path)
        if rank == 0:
            rank_path = base_path
        else:
            stem, ext = os.path.splitext(base_path)
            rank_path = f"{stem}.rank{rank}{ext}"

        model = worker.model_runner.model

        # Try RAM restore (fastest), then file pipelined, then pure Python
        from thaw_vllm.snapshot import (
            restore_model_from_ram,
            restore_model_pipelined,
            restore_model,
        )
        try:
            stats = restore_model_from_ram(model, rank_path, chunk_size_mb)
        except Exception:
            try:
                stats = restore_model_pipelined(model, rank_path, chunk_size_mb)
            except Exception:
                stats = restore_model(model, rank_path)

        stats['rank'] = rank
        stats['path'] = rank_path
        return stats

    results = ec.model_executor.collective_rpc(
        _worker_restore,
        args=(base_path, chunk_size_mb),
    )

    # Aggregate stats from all workers
    total_bytes = sum(r['total_bytes'] for r in results)
    total_regions = sum(r['num_regions'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    return {
        'num_regions': total_regions,
        'total_bytes': total_bytes,
        'elapsed_s': total_elapsed,
        'throughput_gb_s': (total_bytes / 1e9) / total_elapsed if total_elapsed > 0 else 0,
        'tensor_parallel_size': tp_size,
        'per_rank': results,
    }


def freeze_model_pipelined(
    model: nn.Module,
    path: str,
    vllm_commit: Optional[str] = None,
) -> dict:
    """Freeze all GPU parameters using the Rust pipelined path.

    Uses double-buffered async D2H DMA to overlap GPU copies with disk
    writes. Falls back to the pure-Python path if the Rust extension is
    not available.
    """
    try:
        import thaw as _thaw
        if not hasattr(_thaw, 'freeze_to_file_pipelined'):
            raise ImportError("freeze_to_file_pipelined not found")
    except ImportError:
        return freeze_model(model, path, vllm_commit)

    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            p = param.data.contiguous()
            params.append((name, p))

    if not params:
        return freeze_model(model, path, vllm_commit)

    mapping = [
        ("weights", i, param.data.data_ptr(), param.data.nbytes)
        for i, (name, param) in enumerate(params)
    ]

    t0 = time.perf_counter()
    result = _thaw.freeze_to_file_pipelined(
        path, mapping, vllm_commit=vllm_commit,
    )
    elapsed = time.perf_counter() - t0

    total_bytes = result['bytes_copied']
    return {
        "num_regions": result['regions_frozen'],
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        "backend": "rust_pipelined",
        "params": [(name, p.nbytes) for name, p in params],
    }


def restore_model_from_ram(
    model: nn.Module,
    path: str,
    chunk_size_mb: int = 64,
) -> dict:
    """Restore model weights from a .thaw snapshot pre-loaded into RAM.

    Reads the entire snapshot file into memory first, then uses the
    Rust pipelined path to DMA directly from host memory to GPU.
    Eliminates disk I/O from the hot path — throughput is limited
    only by PCIe bandwidth.

    For production use, the caller would keep the bytes in memory
    (e.g., mmap'd from tmpfs) and pass them directly. This function
    reads from disk for convenience, but the restore itself is
    RAM-speed.
    """
    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            params.append((name, param))

    use_rust = False
    try:
        import thaw as _thaw
        if hasattr(_thaw, 'restore_from_bytes_pipelined'):
            use_rust = True
    except ImportError:
        pass

    if use_rust:
        mapping = [
            ("weights", i, param.data.data_ptr(), param.data.nbytes)
            for i, (name, param) in enumerate(params)
        ]

        # mmap the snapshot file. On /dev/shm (tmpfs) this is zero-copy:
        # the kernel maps the same physical RAM pages into our address space.
        # No allocation, no memcpy, no page cache pressure.
        # On real filesystems, mmap still avoids the Python bytes allocation
        # and lets the kernel page in data on demand.
        t_read = time.perf_counter()
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        if sys.platform == "linux":
            import ctypes
            import ctypes.util
            MAP_POPULATE = 0x08000
            MADV_HUGEPAGE = 14
            mm = mmap.mmap(fd, file_size, flags=mmap.MAP_PRIVATE | MAP_POPULATE,
                            prot=mmap.PROT_READ | mmap.PROT_WRITE)
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                buf = (ctypes.c_char * file_size).from_buffer(mm)
                libc.madvise(ctypes.addressof(buf), file_size, MADV_HUGEPAGE)
                del buf
            except Exception:
                pass
        else:
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        os.close(fd)
        read_time = time.perf_counter() - t_read

        t0 = time.perf_counter()
        result = _thaw.restore_from_bytes_pipelined(
            mm, mapping, chunk_size_mb=chunk_size_mb,
        )
        dma_time = time.perf_counter() - t0
        mm.close()

        total_bytes = result['bytes_copied']
        elapsed = read_time + dma_time
    else:
        # Pure-Python fallback: region-by-region pinned CPU → GPU copy.
        # Copies directly into existing parameter tensors — no extra GPU
        # memory allocation needed (unlike restore_model which allocates
        # a full second copy).
        t0 = time.perf_counter()

        with open(path, "rb") as f:
            num_regions, vllm_commit = _read_header(f)

            if num_regions != len(params):
                raise ValueError(
                    f"snapshot has {num_regions} regions but model has "
                    f"{len(params)} CUDA parameters"
                )

            entries = []
            for _ in range(num_regions):
                entries.append(_read_region_entry(f))

            total_bytes = 0
            for i, (name, param) in enumerate(params):
                kind, logical_id, size, file_offset = entries[i]
                if size != param.data.nbytes:
                    raise ValueError(
                        f"size mismatch for {name}: snapshot has {size} bytes, "
                        f"parameter has {param.data.nbytes} bytes"
                    )
                total_bytes += size

            # Copy region-by-region: read each parameter's data into a
            # small pinned buffer, then DMA directly into the existing
            # GPU parameter tensor. Peak host memory = largest single
            # parameter (~512 MB for large models) instead of the full
            # model size.
            payload_start = entries[0][3] if entries else 0
            f.seek(payload_start)
            for i, (name, param) in enumerate(params):
                nbytes = param.data.nbytes
                pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                np_buf = pinned.numpy()
                bytes_read = f.readinto(np_buf)
                if bytes_read != nbytes:
                    raise ValueError(
                        f"truncated read for {name}: expected {nbytes}, got {bytes_read}"
                    )
                param.data.copy_(pinned.view(param.dtype).reshape(param.shape))
                del pinned

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    num_regions = result['regions_restored'] if use_rust else len(params)
    backend = "rust_pipelined_mmap" if use_rust else "python_region_copy"
    stats = {
        "num_regions": num_regions,
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        "backend": backend,
    }
    if use_rust:
        stats["mmap_time_s"] = read_time
        stats["dma_time_s"] = dma_time
        stats["dma_throughput_gb_s"] = (total_bytes / 1e9) / dma_time if dma_time > 0 else 0
    return stats


def restore_model_pipelined(
    model: nn.Module,
    path: str,
    chunk_size_mb: int = 64,
    direct_io: bool = True,
) -> dict:
    """Restore model weights using the Rust pipelined path.

    Uses double-buffered async DMA with O_DIRECT for maximum throughput.
    Falls back to the pure-Python path if the Rust extension is not available.
    """
    try:
        import thaw as _thaw
        if not hasattr(_thaw, 'restore_from_file_pipelined'):
            raise ImportError("restore_from_file_pipelined not found")
    except ImportError:
        return restore_model(model, path)

    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            params.append((name, param))

    mapping = [
        ("weights", i, param.data.data_ptr(), param.data.nbytes)
        for i, (name, param) in enumerate(params)
    ]

    t0 = time.perf_counter()
    result = _thaw.restore_from_file_pipelined(
        path, mapping, chunk_size_mb=chunk_size_mb, direct_io=direct_io,
    )
    elapsed = time.perf_counter() - t0

    total_bytes = result['bytes_copied']
    return {
        "num_regions": result['regions_restored'],
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        "backend": "rust_pipelined",
    }
