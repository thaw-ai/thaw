"""
thaw_vllm.kv_snapshot — freeze/restore vLLM V1 prefix-cached KV blocks.

Captures the KV cache blocks that vLLM's prefix caching has retained
after generation, along with their block hashes. On restore, writes
the block data back to GPU and reconstructs the prefix cache mappings
so new requests with the same prefix get cache hits (skip prefill).

This is the competitive moat: nobody else snapshots KV cache.

Requires vLLM V1 engine with VLLM_ENABLE_V1_MULTIPROCESSING=0.

Performance strategy
--------------------
The KV blob can be tens of GB. Python-level per-block loops serializing
each K/V slab through a staging tensor bottleneck at O(100 MB/s). The
fast path builds a flat `(kind, logical_id, device_ptr, nbytes)` mapping
over every K and V slab of every cached block and hands the entire batch
to the Rust pipelined writer/reader (`_thaw.freeze_to_file_pipelined` /
`_thaw.restore_from_file_pipelined`) — the same double-buffered, O_DIRECT
path weights use. Per-block `.contiguous()` GPU copies disappear: in
the vLLM KV cache layout ``[2, num_blocks, block_size, heads, head_size]``,
``kv_layer[k, bid]`` with ``k`` fixed is already contiguous, so we DMA
directly from it.

The on-disk format is a standard THAW file (via `freeze_to_file_pipelined`)
holding every K/V slab as a `kv_live_block` region, paired with a small
sidecar ``<path>.meta`` JSON file carrying block_ids, block_hashes, and
shape/dtype info needed to reconstruct the prefix cache on restore.
"""

import base64
import json
import os
import struct
import time
from pathlib import Path

import torch

from thaw_common.telemetry import (
    fallback_warning,
    strict_mode,
    check_pinned,
)

# Legacy (pre-pipelined) single-file format, still recognized for
# backward-compatible reads:
#   [8 bytes: magic "THAWKV\x00\x00"]
#   [4 bytes: metadata length (little-endian u32)]
#   [metadata_length bytes: JSON metadata]
#   [payload: concatenated block data, ordered by (layer, block_index)]
#
# New format (the fast path):
#   <path>        : standard THAW file written by the Rust pipelined
#                   freeze. Each region is a `kv_live_block` containing
#                   either the K or V slab of one (layer, block) pair.
#                   Regions are ordered (layer_idx, slot_idx, kv_idx)
#                   with logical_id = layer*N*2 + slot*2 + kv.
#   <path>.meta   : small JSON sidecar; [KV_MAGIC | u32 len | JSON].
#                   Carries block_ids, block_hashes, num_layers, block
#                   shape/dtype so prefix-cache state can be rebuilt.
KV_MAGIC = b"THAWKV\x00\x00"


def _meta_path(path: str) -> str:
    """Sidecar metadata path for a given KV snapshot path."""
    return path + ".meta"


def _write_meta_sidecar(path: str, metadata: dict) -> None:
    """Write the KV sidecar metadata file: [KV_MAGIC | u32 len | JSON]."""
    meta_bytes = json.dumps(metadata).encode()
    with open(_meta_path(path), "wb") as f:
        f.write(KV_MAGIC)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)


def _read_meta_sidecar(path: str) -> dict:
    """Read the KV sidecar metadata file. Raises if magic is wrong."""
    with open(_meta_path(path), "rb") as f:
        magic = f.read(8)
        if magic != KV_MAGIC:
            raise ValueError(f"Bad KV meta magic: {magic!r}")
        meta_len = struct.unpack("<I", f.read(4))[0]
        return json.loads(f.read(meta_len))


def _read_legacy_single_file(path: str):
    """Peek at a legacy single-file snapshot. Returns (metadata, payload_offset)
    if the file is a legacy THAWKV file, else (None, None)."""
    try:
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != KV_MAGIC:
                return None, None
            meta_len = struct.unpack("<I", f.read(4))[0]
            metadata = json.loads(f.read(meta_len))
            payload_offset = 8 + 4 + meta_len
            return metadata, payload_offset
    except (OSError, ValueError):
        return None, None


def _get_engine_core(llm):
    """Navigate to the real EngineCore from a vLLM LLM instance."""
    ec = llm.llm_engine.engine_core
    # InprocClient wraps the real EngineCore
    if hasattr(ec, 'engine_core'):
        ec = ec.engine_core
    return ec


def _serialize_block_hash(block_hash):
    """Serialize a block hash to a JSON-safe string.

    In vLLM v0.19+, _block_hash is raw bytes (e.g. 36 bytes).
    We base64-encode for JSON storage.
    """
    if block_hash is None:
        return None
    if isinstance(block_hash, bytes):
        return base64.b64encode(block_hash).decode('ascii')
    # Fallback for older vLLM with tuple-based hashes
    return repr(block_hash)


def _deserialize_block_hash(data):
    """Deserialize a block hash from JSON."""
    if data is None:
        return None
    if isinstance(data, str):
        return base64.b64decode(data)
    return data


def _rank_kv_path(base_path: str, rank: int) -> str:
    """Get per-rank KV snapshot path. Rank 0 uses base path."""
    if rank == 0:
        return base_path
    stem, ext = os.path.splitext(base_path)
    return f"{stem}.rank{rank}{ext}"


def _collect_kv_slab_requests(kv_caches, block_ids):
    """Build the pipelined freeze/restore mapping list for every K and V slab
    of every (layer, block) pair.

    vLLM's layout is ``[2, num_blocks, block_size, num_kv_heads, head_size]``.
    With K/V split at dim 0, each ``kv_layer[k, bid]`` (k in {0,1}) is a
    single contiguous memory region, which is the PCIe-DMA-friendly unit.

    Returns ``(mapping, slab_nbytes)`` where ``mapping`` is a list of
    ``("kv_live_block", logical_id, device_ptr, nbytes)`` tuples and
    ``slab_nbytes`` is the byte size of one K or V slab (half of
    ``block_bytes``). Iteration order is ``(layer, slot, kv)`` and
    ``logical_id = layer * num_slots * 2 + slot * 2 + kv``.
    """
    num_layers = len(kv_caches)
    num_slots = len(block_ids)

    # kv_layer[0, bid] is contiguous: the last three dims are packed.
    sample = kv_caches[0][0, 0]
    slab_nbytes = sample.numel() * sample.element_size()

    mapping = []
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]
        # Per-layer pointer + stride query. Indexing kv_layer[kv_idx, bid]
        # in the inner loop would create ~10k Tensor views and call
        # .data_ptr() just as many times — microsecond overhead that
        # adds up to ~10-20 ms per mapping build. Here we query once
        # per layer and compute slab pointers with plain byte arithmetic.
        base_ptr = kv_layer.data_ptr()
        strides = kv_layer.stride()
        esize = kv_layer.element_size()
        kv_stride_b = strides[0] * esize     # bytes between K and V halves
        block_stride_b = strides[1] * esize  # bytes between consecutive blocks
        base = layer_idx * num_slots * 2
        for slot_idx, bid in enumerate(block_ids):
            k_ptr = base_ptr + bid * block_stride_b
            v_ptr = k_ptr + kv_stride_b
            logical_id_k = base + slot_idx * 2
            mapping.append(("kv_live_block", logical_id_k, k_ptr, slab_nbytes))
            mapping.append(("kv_live_block", logical_id_k + 1, v_ptr, slab_nbytes))
    return mapping, slab_nbytes


def _freeze_kv_python_fallback(path, kv_caches, metadata, block_ids):
    """Pure-Python KV freeze: staged pinned buffer, legacy single-file format.

    Used only when the Rust pipelined extension is unavailable. Produces a
    legacy THAWKV file (no sidecar) so read-path compatibility is preserved
    without a rust build.
    """
    num_layers = len(kv_caches)
    # block_bytes in metadata is the full [2, B, H, D] slab pair size.
    block_bytes = metadata["block_bytes"]

    total_payload = len(block_ids) * num_layers * block_bytes

    if total_payload == 0:
        with open(path, "wb") as f:
            f.write(KV_MAGIC)
            meta_empty = json.dumps({"num_blocks": 0}).encode()
            f.write(struct.pack("<I", len(meta_empty)))
            f.write(meta_empty)
        return 0

    flat_buf = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)
    check_pinned(flat_buf, "kv_snapshot freeze fallback staging buffer")

    offset = 0
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]
        for bid in block_ids:
            block_tensor = kv_layer[:, bid].contiguous()
            src = block_tensor.view(-1).view(torch.uint8)
            flat_buf[offset:offset + block_bytes].copy_(src, non_blocking=True)
            offset += block_bytes
    torch.cuda.synchronize()

    meta_bytes = json.dumps(metadata).encode()
    with open(path, "wb") as f:
        f.write(KV_MAGIC)
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)
        f.write(flat_buf.numpy().tobytes())

    del flat_buf
    return total_payload


def _restore_kv_python_fallback_legacy(path, kv_caches, metadata, payload_offset):
    """Pure-Python KV restore for legacy single-file snapshots."""
    block_ids = metadata["block_ids"]
    num_layers = metadata["num_layers"]
    block_bytes = metadata["block_bytes"]
    block_shape = metadata["block_shape"]
    dtype_str = metadata["dtype"]

    total_payload = len(block_ids) * num_layers * block_bytes
    flat_pinned = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)
    check_pinned(flat_pinned, "kv_snapshot restore fallback staging buffer")

    with open(path, "rb") as f:
        f.seek(payload_offset)
        np_buf = flat_pinned.numpy()
        bytes_read = f.readinto(np_buf)
        if bytes_read != total_payload:
            raise ValueError(
                f"Truncated KV payload: expected {total_payload}, got {bytes_read}"
            )

    dtype = getattr(torch, dtype_str.replace("torch.", ""))
    device = kv_caches[0].device
    flat_gpu = flat_pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()
    del flat_pinned

    offset = 0
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]
        for bid in block_ids:
            block_data = (
                flat_gpu[offset:offset + block_bytes]
                .view(dtype)
                .reshape(block_shape)
            )
            kv_layer[:, bid].copy_(block_data)
            offset += block_bytes
    del flat_gpu
    return total_payload


def freeze_kv_cache(llm, path: str) -> dict:
    """Freeze prefix-cached KV blocks to a file.

    After vLLM has processed requests with prefix caching enabled,
    completed requests release their blocks but the blocks retain
    their hash in the prefix cache. This function captures those
    cached blocks so they can be restored on a fresh instance.

    Writes a standard THAW file (via the Rust pipelined writer) plus a
    sidecar ``<path>.meta`` JSON file. Falls back to the legacy
    single-file format when the Rust extension is unavailable.

    Args:
        llm: A vLLM LLM instance (V1 engine, in-process).
        path: Output file path for the KV cache snapshot.

    Returns:
        Stats dict with num_blocks, total_bytes, elapsed_s.
    """
    t0 = time.perf_counter()

    ec = _get_engine_core(llm)
    kvm = ec.scheduler.kv_cache_manager
    block_pool = kvm.block_pool
    mr = ec.model_executor.driver_worker.model_runner
    kv_caches = mr.kv_caches  # list[Tensor], one per layer

    # Find all blocks that have a block_hash set (prefix-cached blocks).
    # These are blocks that were used by completed requests and retained
    # in the cache for future prefix hits.
    cached_blocks = []
    for block in block_pool.blocks:
        if block._block_hash is not None and not block.is_null:
            cached_blocks.append(block)

    if not cached_blocks:
        # Write empty snapshot (legacy single-file form; no KV payload
        # means no reason to drag in the Rust path).
        with open(path, 'wb') as f:
            f.write(KV_MAGIC)
            meta = json.dumps({"num_blocks": 0}).encode()
            f.write(struct.pack('<I', len(meta)))
            f.write(meta)
        # Ensure no stale sidecar lingers from a previous run.
        try:
            os.remove(_meta_path(path))
        except FileNotFoundError:
            pass
        return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}

    block_ids = [b.block_id for b in cached_blocks]
    block_hashes = [_serialize_block_hash(b._block_hash) for b in cached_blocks]

    num_layers = len(kv_caches)
    # Shape of one block in one layer: [2, block_size, num_kv_heads, head_size]
    block_shape = list(kv_caches[0][:, 0].shape)
    block_bytes = kv_caches[0][:, 0].nbytes
    dtype_str = str(kv_caches[0].dtype)

    metadata = {
        "num_blocks": len(cached_blocks),
        "block_ids": block_ids,
        "block_hashes": block_hashes,
        "num_layers": num_layers,
        "block_shape": block_shape,
        "block_bytes": block_bytes,
        "dtype": dtype_str,
        "block_size": ec.scheduler.block_size,
    }

    # --- Fast path: Rust pipelined freeze over every K/V slab ---
    use_rust = True
    try:
        import thaw as _thaw
        if not hasattr(_thaw, "freeze_to_file_pipelined"):
            raise ImportError("freeze_to_file_pipelined not found")
    except ImportError as e:
        fallback_warning(
            "freeze_kv_cache (rust ext not loaded)", e,
            dst="pure-python pinned-staging single-file",
        )
        if strict_mode():
            raise
        use_rust = False

    if use_rust:
        mapping, slab_nbytes = _collect_kv_slab_requests(kv_caches, block_ids)
        metadata["slab_nbytes"] = slab_nbytes
        metadata["num_slots"] = len(block_ids)
        try:
            _thaw.freeze_to_file_pipelined(path, mapping, vllm_commit=None)
            _write_meta_sidecar(path, metadata)
            total_payload = slab_nbytes * len(mapping)
        except Exception as e:
            fallback_warning(
                "freeze_kv_cache (rust pipelined failed)", e,
                dst="pure-python pinned-staging single-file",
            )
            if strict_mode():
                raise
            use_rust = False

    if not use_rust:
        total_payload = _freeze_kv_python_fallback(
            path, kv_caches, metadata, block_ids,
        )

    elapsed = time.perf_counter() - t0

    return {
        "num_blocks": len(cached_blocks),
        "total_bytes": total_payload,
        "elapsed_s": elapsed,
        "block_ids": block_ids,
    }


def restore_kv_cache(llm, path: str) -> dict:
    """Restore prefix-cached KV blocks from a snapshot file.

    Writes block data back to GPU and reconstructs the prefix cache
    hash mappings in the block pool so new requests with matching
    prefixes get cache hits.

    Recognizes both the new THAW + sidecar format and the legacy
    single-file THAWKV format.

    Args:
        llm: A vLLM LLM instance (V1 engine, in-process). Should already
            have weights restored (via restore_model or similar).
        path: Path to the KV cache snapshot file.

    Returns:
        Stats dict with num_blocks, total_bytes, elapsed_s.
    """
    t0 = time.perf_counter()

    ec = _get_engine_core(llm)
    kvm = ec.scheduler.kv_cache_manager
    block_pool = kvm.block_pool
    mr = ec.model_executor.driver_worker.model_runner
    kv_caches = mr.kv_caches

    legacy_meta, legacy_offset = _read_legacy_single_file(path)
    has_sidecar = os.path.exists(_meta_path(path))

    if legacy_meta is not None and not has_sidecar:
        # Legacy format — read metadata from the file itself.
        metadata = legacy_meta
        if metadata.get("num_blocks", 0) == 0:
            return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}
        _validate_metadata(metadata, kv_caches)
        total_payload = _restore_kv_python_fallback_legacy(
            path, kv_caches, metadata, legacy_offset,
        )
    else:
        # New format with sidecar metadata.
        metadata = _read_meta_sidecar(path)
        if metadata.get("num_blocks", 0) == 0:
            return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}
        _validate_metadata(metadata, kv_caches)
        total_payload = _restore_kv_rust_or_fallback(
            path, kv_caches, metadata,
        )

    block_ids = metadata["block_ids"]
    block_hashes = metadata["block_hashes"]

    # Validate block IDs fit in this instance's block pool
    max_block_id = len(block_pool.blocks) - 1
    for bid in block_ids:
        if bid > max_block_id:
            raise ValueError(
                f"Block ID {bid} exceeds pool size {len(block_pool.blocks)}. "
                f"Ensure gpu_memory_utilization matches the snapshot instance."
            )

    # Reconstruct prefix cache state in the block pool.
    # Each restored block needs:
    # 1. Its _block_hash set (so the cache knows it's cached)
    # 2. An entry in cached_block_hash_to_block (so lookups find it)
    # The block stays in the free queue with ref_cnt=0 (eviction candidate),
    # which is correct — it will be "touched" when a request hits it.
    #
    # BlockHashToBlockMap uses .insert(), not __setitem__.
    for i, bid in enumerate(block_ids):
        block = block_pool.blocks[bid]
        hash_data = _deserialize_block_hash(block_hashes[i])
        block._block_hash = hash_data
        block_pool.cached_block_hash_to_block.insert(hash_data, block)

    elapsed = time.perf_counter() - t0

    return {
        "num_blocks": metadata["num_blocks"],
        "total_bytes": total_payload,
        "elapsed_s": elapsed,
    }


def _validate_metadata(metadata, kv_caches):
    """Shared metadata vs. live model sanity checks."""
    num_layers = metadata["num_layers"]
    block_shape = metadata["block_shape"]
    if num_layers != len(kv_caches):
        raise ValueError(
            f"KV snapshot has {num_layers} layers but model has "
            f"{len(kv_caches)} layers"
        )
    current_block_shape = list(kv_caches[0][:, 0].shape)
    if block_shape != current_block_shape:
        raise ValueError(
            f"Block shape mismatch: snapshot {block_shape} vs "
            f"model {current_block_shape}"
        )


def _restore_kv_rust_or_fallback(path, kv_caches, metadata):
    """Restore a new-format (THAW + sidecar) KV snapshot.

    Fast path is `_thaw.restore_from_file_pipelined` with one mapping entry
    per K/V slab. Falls back to a staged pinned read+copy if the Rust
    extension isn't loaded.
    """
    block_ids = metadata["block_ids"]

    try:
        import thaw as _thaw
        if not hasattr(_thaw, "restore_from_file_pipelined"):
            raise ImportError("restore_from_file_pipelined not found")
    except ImportError as e:
        fallback_warning(
            "restore_kv_cache (rust ext not loaded)", e,
            dst="pure-python readinto + GPU scatter",
        )
        if strict_mode():
            raise
        return _restore_kv_python_fallback_new(path, kv_caches, metadata)

    mapping, _slab_nbytes = _collect_kv_slab_requests(kv_caches, block_ids)
    try:
        _thaw.restore_from_file_pipelined(path, mapping)
    except Exception as e:
        fallback_warning(
            "restore_kv_cache (rust pipelined failed)", e,
            dst="pure-python readinto + GPU scatter",
        )
        if strict_mode():
            raise
        return _restore_kv_python_fallback_new(path, kv_caches, metadata)

    num_layers = metadata["num_layers"]
    return len(block_ids) * num_layers * metadata["block_bytes"]


def _restore_kv_python_fallback_new(path, kv_caches, metadata):
    """Pure-Python restore for the new THAW-format KV file.

    Reads THAW regions directly (bypassing the Rust ext) and scatters
    them into kv_caches. Slow — intended only as a safety net.
    """
    from thaw_common.format import read_header, read_region_entry

    block_ids = metadata["block_ids"]
    num_layers = metadata["num_layers"]
    slab_nbytes = metadata.get("slab_nbytes")
    dtype_str = metadata["dtype"]
    dtype = getattr(torch, dtype_str.replace("torch.", ""))

    if slab_nbytes is None:
        # Legacy-schema sidecar (shouldn't happen, but be defensive).
        slab_nbytes = metadata["block_bytes"] // 2

    with open(path, "rb") as f:
        num_regions, _engine_commit = read_header(f)
        entries = [read_region_entry(f) for _ in range(num_regions)]

        total_payload = sum(size for _k, _lid, size, _off in entries)
        flat_pinned = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)
        check_pinned(flat_pinned, "restore_kv_cache fallback staging buffer")

        payload_start = entries[0][3] if entries else 0
        f.seek(payload_start)
        np_buf = flat_pinned.numpy()
        bytes_read = f.readinto(np_buf)
        if bytes_read != total_payload:
            raise ValueError(
                f"Truncated KV payload: expected {total_payload}, got {bytes_read}"
            )

    device = kv_caches[0].device
    flat_gpu = flat_pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()
    del flat_pinned

    # Region layout is (layer, slot, kv) flattened; reconstruct the
    # matching K/V slab view and copy into place.
    num_slots = len(block_ids)
    offset = 0
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]
        for slot_idx, bid in enumerate(block_ids):
            for kv_idx in (0, 1):
                slab = kv_layer[kv_idx, bid]
                slab_shape = list(slab.shape)
                slice_gpu = (
                    flat_gpu[offset:offset + slab_nbytes]
                    .view(dtype)
                    .reshape(slab_shape)
                )
                slab.copy_(slice_gpu)
                offset += slab_nbytes

    del flat_gpu
    return total_payload


def freeze_kv_cache_tp(llm, base_path: str) -> dict:
    """Freeze KV cache from all tensor parallel workers.

    With TP > 1, each worker's KV cache holds a shard of the KV heads.
    This dispatches freeze_kv_cache logic to each worker via collective_rpc.
    Each worker saves to a per-rank file.

    The block pool and block hashes are global (managed by the scheduler
    in the main process), so we pass them to each worker.

    With TP = 1, falls back to the standard single-GPU freeze.
    """
    ec = _get_engine_core(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        return freeze_kv_cache(llm, base_path)

    # Get block info from the global scheduler (runs in main process)
    kvm = ec.scheduler.kv_cache_manager
    block_pool = kvm.block_pool

    cached_blocks = []
    for block in block_pool.blocks:
        if block._block_hash is not None and not block.is_null:
            cached_blocks.append(block)

    if not cached_blocks:
        return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}

    block_ids = [b.block_id for b in cached_blocks]
    block_hashes = [_serialize_block_hash(b._block_hash) for b in cached_blocks]
    block_size = ec.scheduler.block_size

    # Dispatch KV cache freeze to each worker
    def _worker_freeze_kv(worker, base_path, block_ids, block_hashes, block_size):
        """Runs inside each worker process."""
        import time

        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        # Import helpers from the outer module so workers use the same
        # fast-path logic. Workers run in the same interpreter under V1
        # in-proc, so a plain import is fine here.
        from thaw_vllm.kv_snapshot import (
            _collect_kv_slab_requests,
            _freeze_kv_python_fallback,
            _rank_kv_path,
            _write_meta_sidecar,
            KV_MAGIC,
        )
        from thaw_common.telemetry import fallback_warning, strict_mode

        rank = get_tensor_model_parallel_rank()
        rank_path = _rank_kv_path(base_path, rank)

        t0 = time.perf_counter()

        mr = worker.model_runner
        kv_caches = mr.kv_caches
        num_layers = len(kv_caches)
        block_shape = list(kv_caches[0][:, 0].shape)
        block_bytes = kv_caches[0][:, 0].nbytes
        dtype_str = str(kv_caches[0].dtype)

        metadata = {
            "num_blocks": len(block_ids),
            "block_ids": block_ids,
            "block_hashes": block_hashes,
            "num_layers": num_layers,
            "block_shape": block_shape,
            "block_bytes": block_bytes,
            "dtype": dtype_str,
            "block_size": block_size,
            "tensor_parallel_rank": rank,
        }

        use_rust = True
        try:
            import thaw as _thaw
            if not hasattr(_thaw, "freeze_to_file_pipelined"):
                raise ImportError("freeze_to_file_pipelined not found")
        except ImportError as e:
            fallback_warning(
                f"freeze_kv_cache_tp rank {rank} (rust ext not loaded)", e,
                dst="pure-python pinned-staging single-file",
            )
            if strict_mode():
                raise
            use_rust = False

        if use_rust:
            mapping, slab_nbytes = _collect_kv_slab_requests(kv_caches, block_ids)
            metadata["slab_nbytes"] = slab_nbytes
            metadata["num_slots"] = len(block_ids)
            try:
                _thaw.freeze_to_file_pipelined(rank_path, mapping, vllm_commit=None)
                _write_meta_sidecar(rank_path, metadata)
                total_payload = slab_nbytes * len(mapping)
            except Exception as e:
                fallback_warning(
                    f"freeze_kv_cache_tp rank {rank} (rust pipelined failed)", e,
                    dst="pure-python pinned-staging single-file",
                )
                if strict_mode():
                    raise
                use_rust = False

        if not use_rust:
            total_payload = _freeze_kv_python_fallback(
                rank_path, kv_caches, metadata, block_ids,
            )

        elapsed = time.perf_counter() - t0

        return {
            "num_blocks": len(block_ids),
            "total_bytes": total_payload,
            "elapsed_s": elapsed,
            "rank": rank,
            "path": rank_path,
        }

    results = ec.model_executor.collective_rpc(
        _worker_freeze_kv,
        args=(base_path, block_ids, block_hashes, block_size),
    )

    total_bytes = sum(r['total_bytes'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    return {
        "num_blocks": len(block_ids),
        "total_bytes": total_bytes,
        "elapsed_s": total_elapsed,
        "block_ids": block_ids,
        "tensor_parallel_size": tp_size,
        "per_rank": results,
    }


def restore_kv_cache_tp(llm, base_path: str) -> dict:
    """Restore KV cache to all tensor parallel workers.

    With TP > 1, dispatches restore to each worker. Each worker reads
    from its per-rank file and writes KV data to its local GPU.
    Block pool reconstruction happens in the main process (scheduler side).

    With TP = 1, falls back to the standard single-GPU restore.
    """
    ec = _get_engine_core(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        return restore_kv_cache(llm, base_path)

    # Dispatch KV restore to each worker (they write data to their GPU)
    def _worker_restore_kv(worker, base_path):
        """Runs inside each worker process."""
        import os
        import time

        from vllm.distributed import get_tensor_model_parallel_rank

        from thaw_vllm.kv_snapshot import (
            _collect_kv_slab_requests,
            _rank_kv_path,
            _read_legacy_single_file,
            _read_meta_sidecar,
            _meta_path,
            _restore_kv_python_fallback_legacy,
            _restore_kv_python_fallback_new,
            _validate_metadata,
        )
        from thaw_common.telemetry import fallback_warning, strict_mode

        rank = get_tensor_model_parallel_rank()
        rank_path = _rank_kv_path(base_path, rank)

        t0 = time.perf_counter()

        mr = worker.model_runner
        kv_caches = mr.kv_caches

        legacy_meta, legacy_offset = _read_legacy_single_file(rank_path)
        has_sidecar = os.path.exists(_meta_path(rank_path))

        if legacy_meta is not None and not has_sidecar:
            metadata = legacy_meta
            if metadata.get("num_blocks", 0) == 0:
                return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0,
                        "rank": rank, "block_ids": [], "block_hashes": []}
            _validate_metadata(metadata, kv_caches)
            total_payload = _restore_kv_python_fallback_legacy(
                rank_path, kv_caches, metadata, legacy_offset,
            )
        else:
            metadata = _read_meta_sidecar(rank_path)
            if metadata.get("num_blocks", 0) == 0:
                return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0,
                        "rank": rank, "block_ids": [], "block_hashes": []}
            _validate_metadata(metadata, kv_caches)

            use_rust = True
            try:
                import thaw as _thaw
                if not hasattr(_thaw, "restore_from_file_pipelined"):
                    raise ImportError("restore_from_file_pipelined not found")
            except ImportError as e:
                fallback_warning(
                    f"restore_kv_cache_tp rank {rank} (rust ext not loaded)", e,
                    dst="pure-python readinto + GPU scatter",
                )
                if strict_mode():
                    raise
                use_rust = False

            if use_rust:
                mapping, _slab_nbytes = _collect_kv_slab_requests(
                    kv_caches, metadata["block_ids"],
                )
                try:
                    _thaw.restore_from_file_pipelined(rank_path, mapping)
                    total_payload = (
                        len(metadata["block_ids"])
                        * metadata["num_layers"]
                        * metadata["block_bytes"]
                    )
                except Exception as e:
                    fallback_warning(
                        f"restore_kv_cache_tp rank {rank} (rust pipelined failed)", e,
                        dst="pure-python readinto + GPU scatter",
                    )
                    if strict_mode():
                        raise
                    use_rust = False

            if not use_rust:
                total_payload = _restore_kv_python_fallback_new(
                    rank_path, kv_caches, metadata,
                )

        elapsed = time.perf_counter() - t0

        return {
            "num_blocks": metadata["num_blocks"],
            "total_bytes": total_payload,
            "elapsed_s": elapsed,
            "rank": rank,
            "block_ids": metadata["block_ids"],
            "block_hashes": metadata["block_hashes"],
        }

    results = ec.model_executor.collective_rpc(
        _worker_restore_kv,
        args=(base_path,),
    )

    # Reconstruct prefix cache state in the global block pool
    # (runs in main process — the scheduler manages block state)
    kvm = ec.scheduler.kv_cache_manager
    block_pool = kvm.block_pool

    # Use block info from rank 0 (block_ids and hashes are the same across ranks)
    rank0 = results[0]
    block_ids = rank0['block_ids']
    block_hashes = rank0['block_hashes']

    max_block_id = len(block_pool.blocks) - 1
    for i, bid in enumerate(block_ids):
        if bid > max_block_id:
            raise ValueError(
                f"Block ID {bid} exceeds pool size {len(block_pool.blocks)}."
            )
        block = block_pool.blocks[bid]
        hash_data = _deserialize_block_hash(block_hashes[i])
        block._block_hash = hash_data
        block_pool.cached_block_hash_to_block.insert(hash_data, block)

    total_bytes = sum(r['total_bytes'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    return {
        "num_blocks": len(block_ids),
        "total_bytes": total_bytes,
        "elapsed_s": total_elapsed,
        "tensor_parallel_size": tp_size,
    }
