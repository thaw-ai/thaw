"""
thaw_vllm.kv_snapshot — freeze/restore vLLM V1 prefix-cached KV blocks.

Captures the KV cache blocks that vLLM's prefix caching has retained
after generation, along with their block hashes. On restore, writes
the block data back to GPU and reconstructs the prefix cache mappings
so new requests with the same prefix get cache hits (skip prefill).

This is the competitive moat: nobody else snapshots KV cache.

Requires vLLM V1 engine with VLLM_ENABLE_V1_MULTIPROCESSING=0.
"""

import base64
import json
import os
import struct
import time
from pathlib import Path

import torch

# KV cache snapshot file format:
# [8 bytes: magic "THAWKV\x00\x00"]
# [4 bytes: metadata length (little-endian u32)]
# [metadata_length bytes: JSON metadata]
# [payload: concatenated block data, ordered by (layer, block_index)]
#
# The metadata contains everything needed to reconstruct the prefix cache:
# - block_ids: list of cached block IDs
# - block_hashes: serialized BlockHashWithGroupId for each block
# - num_layers: number of model layers
# - block_shape: [2, block_size, num_kv_heads, head_size] per layer per block
# - dtype: tensor dtype string

KV_MAGIC = b"THAWKV\x00\x00"


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


def freeze_kv_cache(llm, path: str) -> dict:
    """Freeze prefix-cached KV blocks to a file.

    After vLLM has processed requests with prefix caching enabled,
    completed requests release their blocks but the blocks retain
    their hash in the prefix cache. This function captures those
    cached blocks so they can be restored on a fresh instance.

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
        # Write empty snapshot
        with open(path, 'wb') as f:
            f.write(KV_MAGIC)
            meta = json.dumps({"num_blocks": 0}).encode()
            f.write(struct.pack('<I', len(meta)))
            f.write(meta)
        return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}

    block_ids = [b.block_id for b in cached_blocks]
    block_hashes = [_serialize_block_hash(b._block_hash) for b in cached_blocks]

    num_layers = len(kv_caches)
    # Shape of one block in one layer: [2, block_size, num_kv_heads, head_size]
    block_shape = list(kv_caches[0][:, 0].shape)
    block_bytes = kv_caches[0][:, 0].nbytes
    dtype_str = str(kv_caches[0].dtype)

    # Build metadata
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
    meta_bytes = json.dumps(metadata).encode()

    # Extract block data from GPU — all layers, all cached blocks.
    # We gather into a flat CPU buffer for sequential write.
    total_payload = len(cached_blocks) * num_layers * block_bytes
    flat_buf = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)

    offset = 0
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]  # [2, num_blocks, block_size, heads, head_size]
        for bid in block_ids:
            block_tensor = kv_layer[:, bid].contiguous()
            src = block_tensor.view(-1).view(torch.uint8)
            flat_buf[offset:offset + block_bytes].copy_(src, non_blocking=True)
            offset += block_bytes

    torch.cuda.synchronize()

    # Write to file
    with open(path, 'wb') as f:
        f.write(KV_MAGIC)
        f.write(struct.pack('<I', len(meta_bytes)))
        f.write(meta_bytes)
        f.write(flat_buf.numpy().tobytes())

    del flat_buf
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

    with open(path, 'rb') as f:
        magic = f.read(8)
        if magic != KV_MAGIC:
            raise ValueError(f"Bad KV cache magic: {magic!r}")

        meta_len = struct.unpack('<I', f.read(4))[0]
        metadata = json.loads(f.read(meta_len))

        num_blocks = metadata["num_blocks"]
        if num_blocks == 0:
            return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0}

        block_ids = metadata["block_ids"]
        block_hashes = metadata["block_hashes"]
        num_layers = metadata["num_layers"]
        block_bytes = metadata["block_bytes"]
        block_shape = metadata["block_shape"]
        dtype_str = metadata["dtype"]

        # Validate against current model
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

        # Read payload into pinned memory
        total_payload = num_blocks * num_layers * block_bytes
        flat_pinned = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)
        np_buf = flat_pinned.numpy()
        bytes_read = f.readinto(np_buf)
        if bytes_read != total_payload:
            raise ValueError(
                f"Truncated KV payload: expected {total_payload}, got {bytes_read}"
            )

    # Resolve dtype
    dtype = getattr(torch, dtype_str.replace('torch.', ''))

    # Write block data to GPU
    device = kv_caches[0].device
    flat_gpu = flat_pinned.to(device, non_blocking=True)
    torch.cuda.synchronize()
    del flat_pinned

    offset = 0
    for layer_idx in range(num_layers):
        kv_layer = kv_caches[layer_idx]
        for bid in block_ids:
            block_data = flat_gpu[offset:offset + block_bytes].view(dtype).reshape(block_shape)
            kv_layer[:, bid].copy_(block_data)
            offset += block_bytes

    del flat_gpu

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
        "num_blocks": num_blocks,
        "total_bytes": total_payload,
        "elapsed_s": elapsed,
    }


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
        import json
        import os
        import struct
        import time

        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

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
        meta_bytes = json.dumps(metadata).encode()

        total_payload = len(block_ids) * num_layers * block_bytes
        flat_buf = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)

        offset = 0
        for layer_idx in range(num_layers):
            kv_layer = kv_caches[layer_idx]
            for bid in block_ids:
                block_tensor = kv_layer[:, bid].contiguous()
                src = block_tensor.view(-1).view(torch.uint8)
                flat_buf[offset:offset + block_bytes].copy_(src, non_blocking=True)
                offset += block_bytes

        torch.cuda.synchronize()

        with open(rank_path, 'wb') as f:
            f.write(KV_MAGIC)
            f.write(struct.pack('<I', len(meta_bytes)))
            f.write(meta_bytes)
            f.write(flat_buf.numpy().tobytes())

        del flat_buf
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
        import json
        import os
        import struct
        import time

        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
        rank_path = _rank_kv_path(base_path, rank)

        t0 = time.perf_counter()

        mr = worker.model_runner
        kv_caches = mr.kv_caches

        with open(rank_path, 'rb') as f:
            magic = f.read(8)
            if magic != KV_MAGIC:
                raise ValueError(f"Bad KV cache magic: {magic!r}")

            meta_len = struct.unpack('<I', f.read(4))[0]
            metadata = json.loads(f.read(meta_len))

            num_blocks = metadata["num_blocks"]
            if num_blocks == 0:
                return {"num_blocks": 0, "total_bytes": 0, "elapsed_s": 0,
                        "rank": rank, "block_ids": [], "block_hashes": []}

            block_ids = metadata["block_ids"]
            block_hashes = metadata["block_hashes"]
            num_layers = metadata["num_layers"]
            block_bytes = metadata["block_bytes"]
            block_shape = metadata["block_shape"]
            dtype_str = metadata["dtype"]

            if num_layers != len(kv_caches):
                raise ValueError(
                    f"KV snapshot has {num_layers} layers but model has "
                    f"{len(kv_caches)} layers (rank {rank})"
                )

            total_payload = num_blocks * num_layers * block_bytes
            flat_pinned = torch.empty(total_payload, dtype=torch.uint8, pin_memory=True)
            np_buf = flat_pinned.numpy()
            bytes_read = f.readinto(np_buf)
            if bytes_read != total_payload:
                raise ValueError(
                    f"Truncated KV payload: expected {total_payload}, got {bytes_read}"
                )

        dtype = getattr(torch, dtype_str.replace('torch.', ''))
        device = kv_caches[0].device
        flat_gpu = flat_pinned.to(device, non_blocking=True)
        torch.cuda.synchronize()
        del flat_pinned

        offset = 0
        for layer_idx in range(num_layers):
            kv_layer = kv_caches[layer_idx]
            for bid in block_ids:
                block_data = flat_gpu[offset:offset + block_bytes].view(dtype).reshape(block_shape)
                kv_layer[:, bid].copy_(block_data)
                offset += block_bytes

        del flat_gpu
        elapsed = time.perf_counter() - t0

        return {
            "num_blocks": num_blocks,
            "total_bytes": total_payload,
            "elapsed_s": elapsed,
            "rank": rank,
            "block_ids": block_ids,
            "block_hashes": block_hashes,
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
