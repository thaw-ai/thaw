"""
thaw_common.snapshot — freeze/restore PyTorch model weights to .thaw format.

Engine-agnostic: works with any nn.Module. Engine-specific packages
(thaw_vllm, thaw_sglang) build TP and KV cache support on top.

Performance strategy:
  - All DMA uses pinned memory (cudaMallocHost equivalent via PyTorch).
    Pageable memory causes cudaMemcpyAsync to silently fall back to
    synchronous transfer — the single worst performance trap in CUDA.
  - Freeze: async D2H copies into a pre-allocated pinned buffer, then
    write directly from that buffer (no Python heap copy).
  - Restore: read file directly into pinned buffer (readinto, zero-copy),
    single DMA to GPU, then GPU-internal scatter.
"""

import mmap
import os
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

from thaw_common.format import (
    HEADER_SIZE,
    REGION_ENTRY_SIZE,
    KIND_WEIGHTS,
    write_header,
    write_region_entry,
    read_header,
    read_region_entry,
)
from thaw_common.telemetry import (
    fallback_warning,
    strict_mode,
    check_pinned,
    logger as _log,
)


def freeze_model(
    model: nn.Module,
    path: str,
    engine_commit: Optional[str] = None,
) -> dict:
    """Freeze all GPU parameters of a model to a .thaw file.

    Uses pinned host memory as a staging buffer. Each parameter is copied
    from GPU to the pinned buffer via async DMA (non_blocking=True),
    then the entire buffer is written to disk in one shot.
    """
    commit_bytes = None
    if engine_commit is not None:
        commit_bytes = engine_commit.encode("ascii")

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
            write_header(f, 0, commit_bytes)
        return {
            "num_regions": 0, "total_bytes": 0,
            "elapsed_s": 0, "throughput_gb_s": 0, "params": [],
        }

    t0 = time.perf_counter()

    flat_pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
    check_pinned(flat_pinned, "freeze_model flat staging buffer")

    offset = 0
    for _, p in params:
        nbytes = p.nbytes
        flat_pinned[offset:offset + nbytes].copy_(
            p.reshape(-1).view(torch.uint8), non_blocking=True
        )
        offset += nbytes
    torch.cuda.synchronize()

    with open(path, "wb") as f:
        write_header(f, num_regions, commit_bytes)
        for i, (name, p) in enumerate(params):
            write_region_entry(f, KIND_WEIGHTS, i, sizes[i], offsets[i])
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
        num_regions, engine_commit = read_header(f)

        if num_regions != len(params):
            raise ValueError(
                f"snapshot has {num_regions} regions but model has "
                f"{len(params)} CUDA parameters"
            )

        entries = []
        for _ in range(num_regions):
            entries.append(read_region_entry(f))

        total_bytes = 0
        for i, (name, param) in enumerate(params):
            kind, logical_id, size, file_offset = entries[i]
            if size != param.data.nbytes:
                raise ValueError(
                    f"size mismatch for {name}: snapshot has {size} bytes, "
                    f"parameter has {param.data.nbytes} bytes"
                )
            total_bytes += size

        flat_pinned = torch.empty(total_bytes, dtype=torch.uint8, pin_memory=True)
        check_pinned(flat_pinned, "restore_model flat staging buffer")

        payload_start = entries[0][3] if entries else 0
        f.seek(payload_start)
        np_buf = flat_pinned.numpy()
        bytes_read = f.readinto(np_buf)
        if bytes_read != total_bytes:
            raise ValueError(
                f"truncated payload: expected {total_bytes}, got {bytes_read}"
            )

        device = params[0][1].device
        flat_gpu = flat_pinned.to(device, non_blocking=True)
        torch.cuda.synchronize()
        del flat_pinned

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


def freeze_model_pipelined(
    model: nn.Module,
    path: str,
    engine_commit: Optional[str] = None,
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
    except ImportError as e:
        fallback_warning("freeze_model_pipelined (rust ext not loaded)", e,
                         dst="freeze_model (pure python)")
        if strict_mode():
            raise
        return freeze_model(model, path, engine_commit)

    params = []
    for name, param in model.named_parameters():
        if param.is_cuda:
            p = param.data.contiguous()
            params.append((name, p))

    if not params:
        return freeze_model(model, path, engine_commit)

    mapping = [
        ("weights", i, param.data.data_ptr(), param.data.nbytes)
        for i, (name, param) in enumerate(params)
    ]

    t0 = time.perf_counter()
    result = _thaw.freeze_to_file_pipelined(
        path, mapping, vllm_commit=engine_commit,
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
    except ImportError as e:
        fallback_warning("restore_model_from_ram (rust ext not loaded)", e,
                         dst="python region-by-region copy")
        if strict_mode():
            raise

    if use_rust:
        mapping = [
            ("weights", i, param.data.data_ptr(), param.data.nbytes)
            for i, (name, param) in enumerate(params)
        ]

        t_read = time.perf_counter()
        fd = os.open(path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size
        # MAP_PRIVATE + PROT_READ|PROT_WRITE: cudaHostRegister needs
        # writable pages under the default flag (it sets up a bidirectional
        # pin). A read-only mapping returns cudaErrorInvalidValue on
        # registration even though we only DMA from it. MAP_PRIVATE is
        # safe because we never actually write — the zero-copy restore
        # path only reads these pages, so no COW fault is triggered and
        # we DMA directly from the original page-cache pages.
        if sys.platform == "linux":
            import ctypes
            import ctypes.util
            MAP_POPULATE = 0x08000
            MADV_HUGEPAGE = 14
            mm = mmap.mmap(fd, file_size,
                           flags=mmap.MAP_PRIVATE | MAP_POPULATE,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE)
            try:
                libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
                buf = (ctypes.c_char * file_size).from_buffer(mm)
                if libc.madvise(ctypes.addressof(buf), file_size, MADV_HUGEPAGE) != 0:
                    _log.debug(
                        "madvise(MADV_HUGEPAGE) unavailable (errno=%d); "
                        "zero-copy cudaHostRegister path doesn't need it.",
                        ctypes.get_errno(),
                    )
                del buf
            except Exception as e:
                _log.debug("madvise setup skipped (%s: %s)",
                           type(e).__name__, e)
        else:
            mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)
        os.close(fd)
        read_time = time.perf_counter() - t_read

        # Zero-copy path (cudaHostRegister + direct DMA) is opt-in via
        # THAW_ZEROCOPY_MMAP=1. cudaHostRegister is O(pages) — pinning a
        # 16 GB mmap can take several seconds, which dominates the
        # restore budget for a one-shot load. The chunked pinned-staging
        # path is faster when registration can't be amortized. The
        # zero-copy path is designed for `thaw serve`, which registers
        # the mmap once at slot warm-up and reuses it across restores.
        backend_label = "rust_pipelined_mmap"
        zerocopy_fn = getattr(_thaw, "restore_from_bytes_pipelined_zerocopy", None)
        zerocopy_enabled = os.environ.get("THAW_ZEROCOPY_MMAP", "0") == "1"
        used_zerocopy = False
        t0 = time.perf_counter()
        if zerocopy_fn is not None and zerocopy_enabled:
            try:
                result = zerocopy_fn(mm, mapping)
                backend_label = "rust_pipelined_mmap_zerocopy"
                used_zerocopy = True
            except RuntimeError as e:
                _log.warning(
                    "zero-copy restore failed (%s); falling back to "
                    "chunked memcpy path. Check `ulimit -l` on this host.",
                    e,
                )
        if not used_zerocopy:
            result = _thaw.restore_from_bytes_pipelined(
                mm, mapping, chunk_size_mb=chunk_size_mb,
            )
        dma_time = time.perf_counter() - t0
        mm.close()

        total_bytes = result['bytes_copied']
        elapsed = read_time + dma_time
    else:
        t0 = time.perf_counter()

        with open(path, "rb") as f:
            num_regions, engine_commit = read_header(f)

            if num_regions != len(params):
                raise ValueError(
                    f"snapshot has {num_regions} regions but model has "
                    f"{len(params)} CUDA parameters"
                )

            entries = []
            for _ in range(num_regions):
                entries.append(read_region_entry(f))

            total_bytes = 0
            for i, (name, param) in enumerate(params):
                kind, logical_id, size, file_offset = entries[i]
                if size != param.data.nbytes:
                    raise ValueError(
                        f"size mismatch for {name}: snapshot has {size} bytes, "
                        f"parameter has {param.data.nbytes} bytes"
                    )
                total_bytes += size

            payload_start = entries[0][3] if entries else 0
            f.seek(payload_start)
            for i, (name, param) in enumerate(params):
                nbytes = param.data.nbytes
                pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                check_pinned(pinned, f"restore_model_from_ram region {i} ({name})")
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
    backend = backend_label if use_rust else "python_region_copy"
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
    except ImportError as e:
        fallback_warning("restore_model_pipelined (rust ext not loaded)", e,
                         dst="restore_model (pure python)")
        if strict_mode():
            raise
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
