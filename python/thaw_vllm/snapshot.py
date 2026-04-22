"""
thaw_vllm.snapshot — vLLM-specific tensor parallel freeze/restore.

Engine-agnostic functions (freeze_model, restore_model, etc.) live in
thaw_common.snapshot. This module adds vLLM-specific TP support using
collective_rpc, which works transparently under V0, V1 inproc, and V1 MP.

For backward compatibility, all functions are re-exported from here.
"""

import os
import time
from typing import Optional

# Re-export engine-agnostic functions for backward compatibility.
from thaw_common.snapshot import (  # noqa: F401
    freeze_model,
    freeze_model_pipelined,
    restore_model,
    restore_model_pipelined,
    restore_model_from_ram,
)
from thaw_common.format import (  # noqa: F401
    MAGIC,
    VERSION,
    HEADER_SIZE,
    REGION_ENTRY_SIZE,
    KIND_WEIGHTS,
    KIND_KV_LIVE_BLOCK,
    KIND_METADATA,
)
from thaw_common.format import (
    write_header as _write_header,
    write_region_entry as _write_region_entry,
    read_header as _read_header,
    read_region_entry as _read_region_entry,
)
from thaw_common.util import rank_snapshot_path as _rank_snapshot_path


def _get_engine_core_from_llm(llm):
    """Navigate to an object with model_executor and vllm_config.

    Works under V1 inproc mode (single process) and V0. Under V1 MP,
    model_executor lives in a child process and is not accessible from
    the parent — use llm.collective_rpc for weight ops instead. Kept
    for kv_snapshot.py, which still reaches into scheduler state and
    so requires VLLM_ENABLE_V1_MULTIPROCESSING=0.
    """
    engine = llm.llm_engine
    if hasattr(engine, 'engine_core'):
        ec = engine.engine_core
        if hasattr(ec, 'engine_core'):
            ec = ec.engine_core
        return ec
    return engine


def _worker_freeze(worker, base_path, vllm_commit):
    """Freeze this worker's model shard to a rank-specific path.

    Runs inside a vLLM worker process via collective_rpc. Each TP rank
    writes to rank_snapshot_path(base_path, rank); for TP=1 that is
    just base_path.
    """
    import os as _os
    import tempfile
    from vllm.distributed import get_tensor_model_parallel_rank
    from thaw_common.snapshot import freeze_model_pipelined, freeze_model
    from thaw_common.util import rank_snapshot_path
    from thaw_common.cloud import is_remote, upload_snapshot
    from thaw_common.telemetry import fallback_warning, strict_mode

    rank = get_tensor_model_parallel_rank()
    target_uri = rank_snapshot_path(base_path, rank)
    model = worker.model_runner.model

    # For remote targets, freeze to a local tempfile then upload.
    if is_remote(target_uri):
        fd, local_path = tempfile.mkstemp(suffix=".thaw")
        _os.close(fd)
    else:
        local_path = target_uri

    try:
        stats = freeze_model_pipelined(model, local_path, vllm_commit)
    except Exception as e:
        fallback_warning(
            f"_worker_freeze(rank={rank}).freeze_model_pipelined", e,
            dst="freeze_model (pure python)",
        )
        if strict_mode():
            raise
        stats = freeze_model(model, local_path, vllm_commit)

    if is_remote(target_uri):
        upload_snapshot(local_path, target_uri)
        _os.unlink(local_path)

    stats['rank'] = rank
    stats['path'] = target_uri
    return stats


def _worker_restore(worker, base_path, chunk_size_mb):
    """Restore this worker's model shard from its rank-specific snapshot.

    Under TP>1, two ranks issuing O_DIRECT pread concurrently contend for
    one NVMe controller and degrade aggregate throughput. Prefer RAM-mmap
    first (shared page cache is read-parallel), fall back to pipelined
    pread, then pure-Python. Matches TP=1 cascade in loader.py.
    """
    import os as _os
    # Per-chunk CRC fold is CPU-serial work on the pread path; on TP>1 it
    # blocks the other rank's DMA. Safe to skip — magic + region-size
    # checks still run.
    _os.environ.setdefault("THAW_VERIFY", "0")

    from vllm.distributed import get_tensor_model_parallel_rank
    from thaw_common.snapshot import (
        restore_model_from_ram,
        restore_model_pipelined,
        restore_model,
    )
    from thaw_common.util import rank_snapshot_path
    from thaw_common.cloud import resolve_snapshot_path
    from thaw_common.telemetry import fallback_warning, strict_mode

    rank = get_tensor_model_parallel_rank()
    rank_path = resolve_snapshot_path(rank_snapshot_path(base_path, rank))
    model = worker.model_runner.model

    try:
        stats = restore_model_from_ram(model, rank_path, chunk_size_mb)
    except Exception as e_ram:
        fallback_warning(
            f"_worker_restore(rank={rank}).restore_model_from_ram", e_ram,
            dst="restore_model_pipelined",
        )
        if strict_mode():
            raise
        try:
            stats = restore_model_pipelined(model, rank_path, chunk_size_mb)
        except Exception as e_pipe:
            fallback_warning(
                f"_worker_restore(rank={rank}).restore_model_pipelined", e_pipe,
                dst="restore_model (pure python)",
            )
            if strict_mode():
                raise
            stats = restore_model(model, rank_path)

    stats['rank'] = rank
    stats['path'] = rank_path
    return stats


def freeze_model_tp(
    llm,
    base_path: str,
    vllm_commit: Optional[str] = None,
) -> dict:
    """Freeze model weights from all tensor parallel workers.

    Dispatches to each worker via llm.collective_rpc, which works under
    V0, V1 inproc, and V1 MP. With TP=1 the result list has one entry;
    with TP>1 each rank writes rank_snapshot_path(base_path, rank).
    """
    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size

    results = llm.collective_rpc(
        _worker_freeze,
        args=(base_path, vllm_commit),
    )

    total_bytes = sum(r['total_bytes'] for r in results)
    total_regions = sum(r['num_regions'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    # Forward the backend label if all ranks agree; otherwise "mixed".
    # Without this, the aggregate dict lacks `backend` and downstream
    # reporters default-label it "[python]" even when Rust ran on every
    # rank — cost us hours of diagnosis on 2026-04-19.
    backends = {r.get('backend') for r in results if r.get('backend')}
    agg_backend = backends.pop() if len(backends) == 1 else ('mixed' if backends else 'unknown')

    return {
        'num_regions': total_regions,
        'total_bytes': total_bytes,
        'elapsed_s': total_elapsed,
        'throughput_gb_s': (total_bytes / 1e9) / total_elapsed if total_elapsed > 0 else 0,
        'tensor_parallel_size': tp_size,
        'backend': agg_backend,
        'per_rank': results,
    }


def restore_model_tp(
    llm,
    base_path: str,
    chunk_size_mb: int = 64,
) -> dict:
    """Restore model weights into all tensor parallel workers.

    Dispatches to each worker via llm.collective_rpc, which works under
    V0, V1 inproc, and V1 MP. Each rank loads from rank_snapshot_path
    with the RAM→pipelined→pure-Python fallback chain.
    """
    tp_size = llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size

    results = llm.collective_rpc(
        _worker_restore,
        args=(base_path, chunk_size_mb),
    )

    total_bytes = sum(r['total_bytes'] for r in results)
    total_regions = sum(r['num_regions'] for r in results)
    total_elapsed = max(r['elapsed_s'] for r in results)

    # Forward the backend label if all ranks agree; otherwise "mixed".
    backends = {r.get('backend') for r in results if r.get('backend')}
    agg_backend = backends.pop() if len(backends) == 1 else ('mixed' if backends else 'unknown')

    return {
        'num_regions': total_regions,
        'total_bytes': total_bytes,
        'elapsed_s': total_elapsed,
        'throughput_gb_s': (total_bytes / 1e9) / total_elapsed if total_elapsed > 0 else 0,
        'tensor_parallel_size': tp_size,
        'backend': agg_backend,
        'per_rank': results,
    }
