"""
thaw_vllm.snapshot — vLLM-specific tensor parallel freeze/restore.

Engine-agnostic functions (freeze_model, restore_model, etc.) live in
thaw_common.snapshot. This module adds vLLM-specific TP support using
collective_rpc.

For backward compatibility, all functions are re-exported from here.
"""

import os
import time
from typing import Optional

# Re-export engine-agnostic functions for backward compatibility.
# Existing code that does `from thaw_vllm.snapshot import freeze_model`
# continues to work.
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
    ec = _get_engine_core_from_llm(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        from thaw_common.cloud import is_remote, upload_snapshot
        from thaw_common.telemetry import fallback_warning, strict_mode
        model = ec.model_executor.driver_worker.model_runner.model

        if is_remote(base_path):
            import tempfile
            fd, local_path = tempfile.mkstemp(suffix=".thaw")
            os.close(fd)
        else:
            local_path = base_path

        try:
            stats = freeze_model_pipelined(model, local_path, vllm_commit)
        except Exception as e:
            fallback_warning("freeze_model_tp(TP=1).freeze_model_pipelined", e,
                             dst="freeze_model (pure python)")
            if strict_mode():
                raise
            stats = freeze_model(model, local_path, vllm_commit)

        if is_remote(base_path):
            upload_snapshot(local_path, base_path)
            os.unlink(local_path)

        return stats

    def _worker_freeze(worker, base_path, vllm_commit):
        """Runs inside each worker process."""
        import os
        import tempfile
        from vllm.distributed import get_tensor_model_parallel_rank
        from thaw_common.snapshot import freeze_model_pipelined, freeze_model
        from thaw_common.util import rank_snapshot_path
        from thaw_common.cloud import is_remote, upload_snapshot
        from thaw_common.telemetry import fallback_warning, strict_mode

        rank = get_tensor_model_parallel_rank()
        target_uri = rank_snapshot_path(base_path, rank)
        model = worker.model_runner.model

        # For S3 targets, freeze locally then upload. Freeze is slow
        # (~1.6 GB/s) so sequential upload is fine for MVP.
        if is_remote(target_uri):
            fd, local_path = tempfile.mkstemp(suffix=".thaw")
            os.close(fd)
        else:
            local_path = target_uri

        try:
            stats = freeze_model_pipelined(model, local_path, vllm_commit)
        except Exception as e:
            fallback_warning(f"_worker_freeze(rank={rank}).freeze_model_pipelined", e,
                             dst="freeze_model (pure python)")
            if strict_mode():
                raise
            stats = freeze_model(model, local_path, vllm_commit)

        if is_remote(target_uri):
            upload_snapshot(local_path, target_uri)
            os.unlink(local_path)

        stats['rank'] = rank
        stats['path'] = target_uri
        return stats

    results = ec.model_executor.collective_rpc(
        _worker_freeze,
        args=(base_path, vllm_commit),
    )

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
    ec = _get_engine_core_from_llm(llm)
    tp_size = ec.vllm_config.parallel_config.tensor_parallel_size

    if tp_size == 1:
        from thaw_common.cloud import resolve_snapshot_path
        model = ec.model_executor.driver_worker.model_runner.model
        return restore_model_from_ram(model, resolve_snapshot_path(base_path), chunk_size_mb)

    def _worker_restore(worker, base_path, chunk_size_mb):
        """Runs inside each worker process."""
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
            fallback_warning(f"_worker_restore(rank={rank}).restore_model_from_ram", e_ram,
                             dst="restore_model_pipelined")
            if strict_mode():
                raise
            try:
                stats = restore_model_pipelined(model, rank_path, chunk_size_mb)
            except Exception as e_pipe:
                fallback_warning(f"_worker_restore(rank={rank}).restore_model_pipelined", e_pipe,
                                 dst="restore_model (pure python)")
                if strict_mode():
                    raise
                stats = restore_model(model, rank_path)

        stats['rank'] = rank
        stats['path'] = rank_path
        return stats

    results = ec.model_executor.collective_rpc(
        _worker_restore,
        args=(base_path, chunk_size_mb),
    )

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
