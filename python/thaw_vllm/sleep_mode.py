"""thaw as a vLLM sleep-mode backend.

Wires thaw's freeze/restore into vLLM's ``LLM.sleep()`` / ``LLM.wake_up()``
API so an engine can be put to sleep by serializing its state to disk
(or S3) via thaw's pipelined DMA, then resumed by restoring from that
artifact.

Motivation
----------
vLLM RFC #34303 proposes adding pluggable sleep-mode backends. Today
vLLM ships two levels built on ``CuMemAllocator``:

- **level 1** — offload weights to host RAM, keep KV cache on GPU. Fast
  wake, but needs host RAM to fit all weights.
- **level 2** — free weights and KV cache entirely. Wake requires
  reloading from the original source (HF / safetensors). Slow.

thaw slots in as a **level-2 backend that serializes to disk or S3**
instead of going back to the original source. Wake re-routes through
thaw's pipelined DMA restore rather than safetensors.

Tradeoffs vs. the other backends:

- **CuMemAllocator level 1** requires host RAM to fit the full model.
  thaw streams to disk/S3 with O_DIRECT + pinned double-buffers — 140 GB
  of 70B weights don't need 140 GB of host RAM as a staging area.
- **cuda-checkpoint + CRIU** captures the full CUDA context (kernels,
  CUDA graphs, NCCL state). thaw does not — weights re-enter through the
  model's own ``load_weights`` path, so fp8/fp4 quantized weights that
  miss swizzling on L2 wake work correctly.
- **Cold start from HF** is the default level-2 behavior. thaw wins any
  scenario that can pre-freeze and amortize — serverless scale-to-zero,
  multi-model hot-swap pools, RL rollout loops.

How it composes with vLLM
-------------------------
Under the hood, ``sleep(llm, path)`` does three things in order:

1. ``thaw_vllm.freeze_model_tp(llm, path)`` — snapshot weights via the
   pipelined DMA path. Each TP rank writes its shard.
2. ``llm.sleep(level=2)`` — vLLM's native mechanism that frees the GPU
   memory held by model weights (requires ``enable_sleep_mode=True`` on
   ``LLM`` construction).
3. Record ``torch.cuda.memory_allocated()`` deltas in the returned stats.

``wake_up(llm, path)`` reverses it:

1. ``llm.wake_up()`` — re-allocates GPU tensors for the model (does NOT
   reload weights; they come back as uninitialized/zero memory).
2. ``thaw_vllm.restore_model_tp(llm, path)`` — blits the saved weights
   back into the freshly-allocated GPU tensors via pipelined DMA.

Usage
-----

    import thaw_vllm
    from vllm import LLM

    # enable_sleep_mode=True is REQUIRED for the GPU-freeing side to work.
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",
              enable_sleep_mode=True,
              enforce_eager=True)

    # ... run some inference ...

    thaw_vllm.sleep_mode.sleep(llm, "/snapshots/llama8b.thaw")
    # → GPU weight memory freed, snapshot on disk

    # ... later, same process ...

    thaw_vllm.sleep_mode.wake_up(llm, "/snapshots/llama8b.thaw")
    # → GPU weight memory re-allocated and populated from snapshot

For a fresh process (not the one that slept), use ``thaw_vllm.load``
directly — this module's contract is "same ``LLM`` instance in, same
instance out, sleeping in between."
"""
from __future__ import annotations

from typing import Any

from thaw_vllm.snapshot import freeze_model_tp, restore_model_tp


class SleepModeUnavailableError(RuntimeError):
    """Raised when vLLM was not initialized with ``enable_sleep_mode=True``.

    Without vLLM's sleep-mode allocator, there is no supported way to
    free GPU memory from inside an existing engine. The freeze side
    (serialization) still works, but the caller should not expect the
    GPU pool to shrink after ``sleep()`` returns — pass ``strict=False``
    to opt into freeze-only behavior.
    """


def _cuda_allocated() -> int | None:
    """Return ``torch.cuda.memory_allocated()`` or ``None`` off-GPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return int(torch.cuda.memory_allocated())
    except Exception:
        return None


def _vllm_sleep_available(llm: Any) -> bool:
    """True iff the engine was initialized with ``enable_sleep_mode=True``.

    The ``CuMemAllocator`` path only activates when the sleep-mode flag
    was passed to ``LLM.__init__`` — otherwise ``llm.sleep()`` raises
    from inside vLLM and we should surface that to the caller *before*
    we freeze the weights.
    """
    try:
        cfg = llm.llm_engine.vllm_config
    except AttributeError:
        return False
    # Sleep-mode lives on ``model_config.enable_sleep_mode`` in 0.19.
    mc = getattr(cfg, "model_config", None)
    return bool(getattr(mc, "enable_sleep_mode", False))


def sleep(
    llm: Any,
    path: str,
    *,
    level: int = 2,
    strict: bool = True,
) -> dict:
    """Put a vLLM engine to sleep by freezing weights to ``path``.

    Order of operations:

    1. Record ``torch.cuda.memory_allocated()`` (before).
    2. ``thaw_vllm.freeze_model_tp(llm, path)`` → writes one ``.thaw``
       file per rank (``path`` for rank 0, ``path.rankN.thaw`` for N>0).
    3. ``llm.sleep(level=level)`` → vLLM frees the GPU weight memory.
       Requires ``enable_sleep_mode=True`` on the LLM.
    4. Record ``torch.cuda.memory_allocated()`` (after).

    Args:
        llm: a ``vllm.LLM`` instance.
        path: destination snapshot path. For TP>1, sibling per-rank files
            are derived automatically. ``s3://`` URIs are supported when
            thaw-cloud is installed.
        level: vLLM sleep level to apply after freezing. Level 2 frees
            all model memory (appropriate when we've just serialized
            everything). Level 1 keeps KV cache on GPU.
        strict: if True (default) and the engine was not created with
            ``enable_sleep_mode=True``, raise ``SleepModeUnavailableError``
            *before* any work is done. Set False to run as freeze-only
            (snapshot lands on disk, GPU is not freed — equivalent to a
            durable checkpoint).

    Returns:
        The aggregate stats dict from ``freeze_model_tp`` plus:
          - ``sleep_path``: the ``path`` argument
          - ``sleep_level``: the level passed through
          - ``gpu_bytes_before_sleep``, ``gpu_bytes_after_sleep``,
            ``gpu_bytes_freed`` (or None if CUDA unavailable)
          - ``freed_gpu_memory``: True iff vLLM's sleep ran successfully

    Raises:
        SleepModeUnavailableError: if ``strict=True`` and vLLM's sleep
            mode wasn't enabled on the engine.
    """
    sleep_available = _vllm_sleep_available(llm)
    if strict and not sleep_available:
        raise SleepModeUnavailableError(
            "vLLM was not initialized with enable_sleep_mode=True. "
            "Either rebuild the LLM with enable_sleep_mode=True, or "
            "call sleep(..., strict=False) to freeze-only without "
            "freeing GPU memory."
        )

    gpu_before = _cuda_allocated()
    stats = freeze_model_tp(llm, path)

    freed = False
    if sleep_available:
        try:
            llm.sleep(level=level)
            freed = True
        except Exception as e:
            # Freeze succeeded; just report that GPU-free step didn't run.
            stats["sleep_error"] = repr(e)

    gpu_after = _cuda_allocated()
    gpu_freed = (
        gpu_before - gpu_after
        if (gpu_before is not None and gpu_after is not None)
        else None
    )

    stats["sleep_path"] = path
    stats["sleep_level"] = level
    stats["gpu_bytes_before_sleep"] = gpu_before
    stats["gpu_bytes_after_sleep"] = gpu_after
    stats["gpu_bytes_freed"] = gpu_freed
    stats["freed_gpu_memory"] = freed
    return stats


def wake_up(
    llm: Any,
    path: str,
    *,
    chunk_size_mb: int = 64,
) -> dict:
    """Wake a vLLM engine by restoring weights from ``path``.

    If the engine was slept via vLLM's ``llm.sleep()``, we first call
    ``llm.wake_up()`` to re-allocate GPU tensors. Then thaw's pipelined
    restore blits the saved weights into those freshly-allocated tensors.

    If ``enable_sleep_mode=False`` (freeze-only sleep), we skip the
    vLLM wake-up call — the GPU tensors were never freed and
    ``restore_model_tp`` can write directly.

    Returns the restore stats dict plus before/after GPU-memory figures.
    """
    gpu_before = _cuda_allocated()

    woke = False
    if _vllm_sleep_available(llm):
        try:
            llm.wake_up()
            woke = True
        except Exception:
            # If the engine wasn't actually asleep, wake_up raises.
            # Swallow and proceed — restore will handle whatever shape
            # the GPU tensors are in.
            pass

    stats = restore_model_tp(llm, path, chunk_size_mb=chunk_size_mb)

    gpu_after = _cuda_allocated()
    gpu_populated = (
        gpu_after - gpu_before
        if (gpu_before is not None and gpu_after is not None)
        else None
    )

    stats["wake_path"] = path
    stats["gpu_bytes_before_wake"] = gpu_before
    stats["gpu_bytes_after_wake"] = gpu_after
    stats["gpu_bytes_populated"] = gpu_populated
    stats["vllm_wake_up_called"] = woke
    return stats


__all__ = ["sleep", "wake_up", "SleepModeUnavailableError"]
