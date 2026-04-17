"""
thaw_vllm._pool_worker — in-worker slot-warm state for EnginePool.

Runs inside vLLM worker processes (via llm.collective_rpc). Module-level
state persists across collective_rpc calls within a single worker
process, which is where the pinned-mmap registration physically lives
— cudaHostRegister state cannot cross the IPC boundary into the parent
under V1 MP.

State is keyed by EnginePool slot_id so that V1-inproc multi-slot
configurations (two LLMs sharing one Python process) don't collide.
Under V1 MP each LLM spawns its own worker process, so the dict ends
up with exactly one entry per worker.
"""

# Per-slot state: slot_id -> {"path": str | None, "pinned_mmap": object | None}
_slot_state: dict[int, dict] = {}


def swap_model(worker, slot_id: int, base_snapshot_path: str) -> dict:
    """Swap weights into this worker's model.

    Tries, in order:
      1. Slot-persistent pinned mmap (reused across same-path reloads)
      2. RAM restore (per-call mmap + pipelined DMA)
      3. Pipelined file restore (double-buffered DMA + O_DIRECT)
      4. Pure-Python region-by-region
    """
    from vllm.distributed import get_tensor_model_parallel_rank
    from thaw_common.snapshot import (
        make_pinned_mmap,
        restore_model_from_pinned_mmap,
        restore_model_from_ram,
        restore_model_pipelined,
        restore_model,
    )
    from thaw_common.util import rank_snapshot_path
    from thaw_common.telemetry import fallback_warning, strict_mode

    rank = get_tensor_model_parallel_rank()
    snapshot_path = rank_snapshot_path(base_snapshot_path, rank)
    model = worker.model_runner.model

    state = _slot_state.setdefault(slot_id, {"path": None, "pinned_mmap": None})

    # Drop stale pinned mmap if the snapshot changed on this slot.
    if state["path"] != snapshot_path and state["pinned_mmap"] is not None:
        state["pinned_mmap"] = None
        state["path"] = None

    stats = None

    # Path 1: slot-persistent pinned mmap.
    if state["pinned_mmap"] is None:
        try:
            state["pinned_mmap"] = make_pinned_mmap(snapshot_path)
            state["path"] = snapshot_path
        except Exception as e_pin:
            fallback_warning(
                f"pool_worker.slot{slot_id}.rank{rank}.make_pinned_mmap",
                e_pin,
                dst="restore_model_from_ram",
            )
            if strict_mode():
                raise

    if state["pinned_mmap"] is not None:
        try:
            stats = restore_model_from_pinned_mmap(model, state["pinned_mmap"])
        except Exception as e_reuse:
            fallback_warning(
                f"pool_worker.slot{slot_id}.rank{rank}.restore_model_from_pinned_mmap",
                e_reuse,
                dst="restore_model_from_ram",
            )
            state["pinned_mmap"] = None
            state["path"] = None
            if strict_mode():
                raise

    # Path 2–4: fallback chain.
    if stats is None:
        try:
            stats = restore_model_from_ram(model, snapshot_path)
        except Exception as e_ram:
            fallback_warning(
                f"pool_worker.slot{slot_id}.rank{rank}.restore_model_from_ram",
                e_ram,
                dst="restore_model_pipelined",
            )
            if strict_mode():
                raise
            try:
                stats = restore_model_pipelined(model, snapshot_path)
            except Exception as e_pipe:
                fallback_warning(
                    f"pool_worker.slot{slot_id}.rank{rank}.restore_model_pipelined",
                    e_pipe,
                    dst="restore_model (pure python)",
                )
                if strict_mode():
                    raise
                stats = restore_model(model, snapshot_path)

    stats['rank'] = rank
    stats['path'] = snapshot_path
    return stats


def drop_pinned(worker, slot_id: int) -> None:
    """Release the slot-persistent pinned registration inside this worker."""
    state = _slot_state.get(slot_id)
    if state is not None:
        state["pinned_mmap"] = None
        state["path"] = None
