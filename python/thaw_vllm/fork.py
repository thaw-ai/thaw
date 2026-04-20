"""
thaw_vllm.fork — make a running vLLM session portable.

vLLM's prefix cache dies with the process. That makes cheap work impossible:

  - RL pivot-point resampling (Tree-GRPO, DEEP-GRPO) — fork at a pivot,
    branch N rollouts, no reprefill of the trunk.
  - Coding agents exploring N parallel implementations from one reasoning
    trunk — 8 engineers in the time of 1 cold start.
  - Session migration across GPU / pod / node — move a live conversation
    without recompute.
  - Crash-safe checkpoints for long-running agents — resume mid-turn.

`fork()` captures a live session's state (optional weights + KV cache
blocks + prefix-cache hash table + metadata) into a portable handle.
`ForkHandle.hydrate()` restores it into a fresh or different engine.
The shared prefix's prefill cost becomes a one-time memcpy, not a
recompute-per-worker.

HuggingFace's async-RL landscape survey (2026) flagged this primitive as
missing from every shipping library:

    "DEEP-GRPO's pivot resampling… requires saving KV cache state at
     pivot points, which no current async library supports out of the box."

This module is the `out of the box` answer.

Design
------
Two layers:

  1. Primitive — `fork(llm) -> ForkHandle`, `ForkHandle.hydrate(llm)`.
     Stable, minimal. Snapshots to a state directory; hydrates into any
     vLLM engine with matching architecture.

  2. Convenience — `fork_completions(llm, prompts, sp, workers=N)`.
     N=None delegates to `llm.generate` (vLLM's own continuous batching
     handles it). N>0 spawns N subprocess workers that each hydrate and
     run one prompt — that's where fork is load-bearing.

Constraints in v1
-----------------
- KV cache operations require `VLLM_ENABLE_V1_MULTIPROCESSING=0` (scheduler
  state is only reachable in-proc). This module sets it via `os.environ.
  setdefault` at import-time guards in `fork()` / `hydrate()`.
- `fork_completions(workers=N)` requires TP=1 and `include_weights=True`
  (children are fresh processes).
- Parent must have no unfinished requests; child must have
  `enable_prefix_caching=True`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

HANDLE_VERSION = 1
HANDLE_FILENAME = "handle.json"
WEIGHTS_FILENAME = "weights.thaw"
KV_FILENAME = "kv.thawkv"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ForkError(RuntimeError):
    """Base class for fork errors."""


class ModelMismatchError(ForkError):
    """Parent model_id differs from the engine being hydrated."""


class BlockShapeMismatchError(ForkError):
    """Parent and child KV cache block shapes differ."""


class BlockPoolTooSmallError(ForkError):
    """Child block pool is smaller than the largest block_id in the snapshot."""


class PrefixCachingDisabledError(ForkError):
    """Child engine was loaded without enable_prefix_caching=True."""


class UnfinishedRequestsError(ForkError):
    """Parent has pending requests; a fork here would be inconsistent."""


class HandleClosedError(ForkError):
    """Operation attempted on a ForkHandle whose state_dir is gone."""


# ---------------------------------------------------------------------------
# ForkHandle
# ---------------------------------------------------------------------------


@dataclass
class ForkHandle:
    """Portable reference to a frozen vLLM session.

    A handle is a directory on disk containing:

      - ``handle.json`` — fork-level metadata (this dataclass, serialized)
      - ``kv.thawkv`` + ``kv.thawkv.meta`` — KV cache payload + sidecar
      - ``weights.thaw`` — optional; present when ``include_weights=True``

    Attributes:
        model_id: HF repo name or local path the parent was loaded from.
                  Hydrate will refuse to restore into a different model.
        state_dir: Directory holding the handle's files. Deleted on
                   ``close()`` if this handle owns the directory.
        kv_path: Absolute path to ``kv.thawkv``.
        weights_path: Absolute path to ``weights.thaw`` or None.
        prefix_tokens: ``num_cached_blocks * block_size`` — upper bound
                       on tokens stashed in the KV snapshot.
        block_shape: Per-block shape from the KV sidecar, used for
                     compatibility checks on hydrate.
        num_layers: Number of KV cache layers (must match on hydrate).
        max_block_id: Largest block_id in the snapshot — child's block
                      pool must be strictly larger.
        num_kv_blocks: Number of prefix-cached blocks captured.
        tensor_parallel_size: TP degree of the parent engine.
        vllm_version: vLLM version string (informational).
        created_at: Unix timestamp when the handle was written.
        version: Handle format version.
    """

    model_id: str
    state_dir: str
    kv_path: str
    weights_path: Optional[str]
    prefix_tokens: int
    block_shape: list
    num_layers: int
    max_block_id: int
    num_kv_blocks: int
    tensor_parallel_size: int = 1
    vllm_version: Optional[str] = None
    created_at: float = 0.0
    version: int = HANDLE_VERSION
    _owns_state_dir: bool = field(default=False, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)

    # ---- serialization ---------------------------------------------------

    def _to_json_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Strip private bookkeeping fields from the on-disk form.
        d.pop("_owns_state_dir", None)
        d.pop("_closed", None)
        return d

    def _write_manifest(self) -> None:
        manifest_path = os.path.join(self.state_dir, HANDLE_FILENAME)
        with open(manifest_path, "w") as f:
            json.dump(self._to_json_dict(), f, indent=2)

    @classmethod
    def load(cls, state_dir: str) -> "ForkHandle":
        """Read a handle back from a directory written by ``fork()`` or ``save()``."""
        state_dir = os.path.abspath(state_dir)
        manifest_path = os.path.join(state_dir, HANDLE_FILENAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"No {HANDLE_FILENAME} in {state_dir}")
        with open(manifest_path) as f:
            data = json.load(f)
        version = data.get("version", 0)
        if version != HANDLE_VERSION:
            raise ForkError(
                f"Handle version {version} incompatible with reader {HANDLE_VERSION}"
            )
        # Paths in the manifest may be absolute (parent's state_dir) or
        # just filenames. Rebase to the directory we loaded from so the
        # handle is relocation-safe.
        data["state_dir"] = state_dir
        data["kv_path"] = os.path.join(state_dir, KV_FILENAME)
        if data.get("weights_path"):
            data["weights_path"] = os.path.join(state_dir, WEIGHTS_FILENAME)
        # Filter to known fields so older/newer manifests don't crash loading.
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in field_names}
        h = cls(**filtered)
        h._owns_state_dir = False
        return h

    def save(self, target_dir: str) -> "ForkHandle":
        """Copy this handle's state to ``target_dir`` and return a new handle.

        The returned handle does NOT own the target dir — it won't be
        deleted by ``close()``. Use this to persist a handle across
        process restarts or ship it to another machine.
        """
        self._assert_open()
        target_dir = os.path.abspath(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        shutil.copy2(self.kv_path, os.path.join(target_dir, KV_FILENAME))
        meta_src = self.kv_path + ".meta"
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, os.path.join(target_dir, KV_FILENAME + ".meta"))
        if self.weights_path and os.path.exists(self.weights_path):
            shutil.copy2(
                self.weights_path, os.path.join(target_dir, WEIGHTS_FILENAME)
            )
            # TP>1 rank shards. Copy any siblings that look like rank shards.
            parent_dir = os.path.dirname(self.weights_path)
            stem = Path(WEIGHTS_FILENAME).stem
            for entry in os.listdir(parent_dir):
                if entry.startswith(f"{stem}.rank") and entry.endswith(".thaw"):
                    shutil.copy2(
                        os.path.join(parent_dir, entry),
                        os.path.join(target_dir, entry),
                    )

        copied = ForkHandle(
            model_id=self.model_id,
            state_dir=target_dir,
            kv_path=os.path.join(target_dir, KV_FILENAME),
            weights_path=(
                os.path.join(target_dir, WEIGHTS_FILENAME)
                if self.weights_path
                else None
            ),
            prefix_tokens=self.prefix_tokens,
            block_shape=list(self.block_shape),
            num_layers=self.num_layers,
            max_block_id=self.max_block_id,
            num_kv_blocks=self.num_kv_blocks,
            tensor_parallel_size=self.tensor_parallel_size,
            vllm_version=self.vllm_version,
            created_at=self.created_at,
        )
        copied._write_manifest()
        return copied

    # ---- lifecycle -------------------------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise HandleClosedError(f"ForkHandle at {self.state_dir} is closed")

    def close(self) -> None:
        """Delete the state directory if this handle owns it. Idempotent."""
        if self._closed:
            return
        self._closed = True
        if self._owns_state_dir and os.path.isdir(self.state_dir):
            shutil.rmtree(self.state_dir, ignore_errors=True)

    def __enter__(self) -> "ForkHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        # Best-effort cleanup; do not mask exceptions from attr access
        # during interpreter teardown.
        try:
            self.close()
        except Exception:
            pass

    # ---- hydrate ---------------------------------------------------------

    def hydrate(self, llm) -> dict:
        """Restore this handle's state into an existing vLLM engine.

        The engine must:
          - Be loaded with the same ``model_id``.
          - Have ``enable_prefix_caching=True``.
          - Have a KV cache block pool at least as large as the snapshot.
          - Have matching KV block shape (dtype, heads, head_size, block_size).

        Validators run BEFORE any GPU work. Failures raise a typed
        subclass of ``ForkError`` so callers can react without scraping
        strings.

        Returns a stats dict from ``restore_kv_cache`` (plus a
        ``weights_restored`` flag when weights were also loaded).
        """
        self._assert_open()
        # KV path is unreachable under V1 MP default — the scheduler
        # lives in a child process. Defensively set V1 MP=0 for any
        # callers who bypassed thaw_vllm.load's setdefault.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        _validate_child(llm, self)

        from thaw_vllm.kv_snapshot import restore_kv_cache, restore_kv_cache_tp
        from thaw_vllm.snapshot import restore_model_tp

        tp_size = _tp_size(llm)
        stats: dict[str, Any] = {}

        if self.weights_path:
            # When the child was launched with dummy weights, the handle
            # must carry the parent's weights. Restore via collective_rpc
            # so TP=1 and TP>1 share one code path.
            weights_base = self.weights_path
            ws = restore_model_tp(llm, weights_base)
            stats["weights_restored"] = True
            stats["weights"] = ws
        else:
            stats["weights_restored"] = False

        if tp_size > 1:
            kv_stats = restore_kv_cache_tp(llm, self.kv_path)
        else:
            kv_stats = restore_kv_cache(llm, self.kv_path)
        stats["kv"] = kv_stats
        return stats


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tp_size(llm) -> int:
    return llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size


def _model_id(llm) -> str:
    return llm.llm_engine.vllm_config.model_config.model


def _vllm_version() -> Optional[str]:
    try:
        import vllm

        return getattr(vllm, "__version__", None)
    except ImportError:
        return None


def _assert_parent_quiescent(llm) -> None:
    """Fork is a pure read — but the snapshot races with in-flight writes.
    Reject any parent with outstanding requests."""
    try:
        has_unfinished = llm.llm_engine.has_unfinished_requests()
    except (AttributeError, Exception):
        # Older vLLM versions or mocked engines — don't refuse if we
        # can't check. This is a forward-compat hedge, not a loosening
        # of the guarantee.
        has_unfinished = False
    if has_unfinished:
        raise UnfinishedRequestsError(
            "fork() requires the parent engine to be idle. "
            "Wait for pending generate() calls to finish before forking."
        )


def _assert_prefix_caching_enabled(llm) -> None:
    """Parent must have prefix caching on, else there's nothing to snapshot."""
    try:
        from thaw_vllm.kv_snapshot import _get_engine_core

        ec = _get_engine_core(llm)
        block_pool = ec.scheduler.kv_cache_manager.block_pool
    except Exception as e:
        raise PrefixCachingDisabledError(
            f"Could not reach block_pool on parent engine: {e!r}. "
            f"Load with enable_prefix_caching=True."
        ) from e
    if not hasattr(block_pool, "cached_block_hash_to_block"):
        raise PrefixCachingDisabledError(
            "Parent block pool has no cached_block_hash_to_block — "
            "load the engine with enable_prefix_caching=True."
        )


def _load_meta(kv_path: str) -> dict:
    """Read the KV sidecar .meta JSON produced by freeze_kv_cache."""
    from thaw_vllm.kv_snapshot import _read_meta_sidecar

    return _read_meta_sidecar(kv_path)


def _validate_child(llm, handle: ForkHandle) -> None:
    """Gate on handle-vs-child compatibility BEFORE any GPU mutation."""
    from thaw_vllm.kv_snapshot import _get_engine_core

    child_model = _model_id(llm)
    if child_model != handle.model_id:
        raise ModelMismatchError(
            f"ForkHandle was created from {handle.model_id!r}; "
            f"cannot hydrate into {child_model!r}."
        )

    # Child engine must have prefix caching enabled — else block_pool
    # exists but cached_block_hash_to_block is missing and restore_kv_cache
    # will fail deep inside the mutation.
    try:
        ec = _get_engine_core(llm)
        block_pool = ec.scheduler.kv_cache_manager.block_pool
        kv_caches = ec.model_executor.driver_worker.model_runner.kv_caches
    except Exception as e:
        raise PrefixCachingDisabledError(
            f"Could not reach child block_pool / kv_caches: {e!r}. "
            f"Ensure the child is loaded with enable_prefix_caching=True."
        ) from e

    if not hasattr(block_pool, "cached_block_hash_to_block"):
        raise PrefixCachingDisabledError(
            "Child block pool has no cached_block_hash_to_block — "
            "load with enable_prefix_caching=True."
        )

    pool_size = len(block_pool.blocks)
    if handle.max_block_id >= pool_size:
        raise BlockPoolTooSmallError(
            f"Snapshot references block_id up to {handle.max_block_id}, "
            f"but child pool has only {pool_size} blocks. Increase the "
            f"child's gpu_memory_utilization or max_model_len."
        )

    # Compare per-block shape. Sidecar stores [2, block_size, H, D];
    # child kv_caches[0][:, 0] gives the same slice.
    if len(kv_caches) != handle.num_layers:
        raise BlockShapeMismatchError(
            f"Snapshot has {handle.num_layers} layers; child has "
            f"{len(kv_caches)}. Model mismatch or different quantization."
        )

    try:
        child_block_shape = list(kv_caches[0][:, 0].shape)
    except Exception as e:
        raise BlockShapeMismatchError(
            f"Could not determine child block shape: {e!r}."
        ) from e
    if list(handle.block_shape) != child_block_shape:
        raise BlockShapeMismatchError(
            f"Snapshot block_shape {handle.block_shape} != "
            f"child block_shape {child_block_shape}. Different dtype, "
            f"head count, head dim, or block size."
        )


# ---------------------------------------------------------------------------
# fork()
# ---------------------------------------------------------------------------


def fork(
    llm,
    *,
    include_weights: bool = False,
    state_dir: Optional[str] = None,
) -> ForkHandle:
    """Freeze a running vLLM session into a portable handle.

    Captures prefix-cached KV blocks (always) and weights (optional).
    The parent engine is untouched — fork is a pure read on GPU state.

    Args:
        llm: A vLLM LLM instance loaded with ``enable_prefix_caching=True``
             that has processed at least one request (the trunk).
        include_weights: If True, also snapshot model weights. Required
                         when the handle will be hydrated into fresh
                         engines that haven't loaded weights themselves
                         (e.g., ``fork_completions(workers=N)``).
        state_dir: Directory to write the handle into. If None, a unique
                   temp directory is created and will be deleted by
                   ``ForkHandle.close()``.

    Returns:
        A ``ForkHandle`` pointing at ``state_dir``. Use as a context
        manager or call ``close()`` when done.

    Raises:
        UnfinishedRequestsError: parent has pending generate() calls.
        PrefixCachingDisabledError: parent lacks a reachable prefix cache.

    Notes:
        Sets ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` via setdefault because
        the KV path needs in-proc scheduler access. If the caller needs
        V1 MP on, they cannot use KV fork; this is a vLLM architectural
        constraint (see vllm #34303).
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    _assert_parent_quiescent(llm)
    _assert_prefix_caching_enabled(llm)

    from thaw_vllm.kv_snapshot import freeze_kv_cache, freeze_kv_cache_tp
    from thaw_vllm.snapshot import freeze_model_tp

    tp_size = _tp_size(llm)
    owns_dir = state_dir is None
    if state_dir is None:
        state_dir = tempfile.mkdtemp(prefix="thaw_fork_")
    else:
        state_dir = os.path.abspath(state_dir)
        os.makedirs(state_dir, exist_ok=True)

    kv_path = os.path.join(state_dir, KV_FILENAME)
    weights_path: Optional[str] = None

    try:
        if include_weights:
            weights_path = os.path.join(state_dir, WEIGHTS_FILENAME)
            freeze_model_tp(llm, weights_path)

        if tp_size > 1:
            freeze_kv_cache_tp(llm, kv_path)
        else:
            freeze_kv_cache(llm, kv_path)

        meta = _load_meta(kv_path)
        num_blocks = int(meta.get("num_blocks", 0))
        block_ids = list(meta.get("block_ids", []))
        block_size = int(meta.get("block_size", 0) or 0)
        num_layers = int(meta.get("num_layers", 0))
        block_shape = list(meta.get("block_shape", []))
        max_block_id = max(block_ids) if block_ids else -1
        prefix_tokens = num_blocks * block_size

        handle = ForkHandle(
            model_id=_model_id(llm),
            state_dir=state_dir,
            kv_path=kv_path,
            weights_path=weights_path,
            prefix_tokens=prefix_tokens,
            block_shape=block_shape,
            num_layers=num_layers,
            max_block_id=max_block_id,
            num_kv_blocks=num_blocks,
            tensor_parallel_size=tp_size,
            vllm_version=_vllm_version(),
            created_at=time.time(),
        )
        handle._owns_state_dir = owns_dir
        handle._write_manifest()
        return handle
    except Exception:
        # Don't leak the temp dir on freeze failure.
        if owns_dir and os.path.isdir(state_dir):
            shutil.rmtree(state_dir, ignore_errors=True)
        raise


# ---------------------------------------------------------------------------
# fork_completions()
# ---------------------------------------------------------------------------


@dataclass
class ForkCompletionResult:
    """One worker's output from ``fork_completions(workers=N)``.

    In same-process mode (``workers=None``), ``text`` and ``token_ids``
    come straight from the corresponding entry of ``llm.generate``. In
    subprocess mode, each ``ForkCompletionResult`` is hydrated from the
    worker's stdout JSON.
    """

    prompt: str
    text: str
    token_ids: list
    worker_index: int
    elapsed_s: float
    mode: str  # "same_process" | "subprocess"
    raw: Optional[dict] = None


def fork_completions(
    llm,
    prompts: Sequence[str],
    sampling_params,
    *,
    workers: Optional[int] = None,
    state_dir: Optional[str] = None,
    handle: Optional[ForkHandle] = None,
    extra_env: Optional[dict] = None,
) -> list[ForkCompletionResult]:
    """Run N completions from the parent's cached prefix.

    Two modes:

      - ``workers=None`` (default) — delegate to ``llm.generate``. vLLM's
        continuous batching + prefix cache handle N parallel branches
        inside the same engine. This is the right default — fork is
        only load-bearing when the parent process can't host them.

      - ``workers=N`` (N>0) — snapshot the parent, spawn N subprocess
        workers that each hydrate and run one prompt. Prompts are
        distributed round-robin across workers; one worker may own
        multiple prompts.

    Args:
        llm: parent vLLM engine.
        prompts: list of prompt strings. Usually the trunk conversation
                 + N divergent queries.
        sampling_params: ``vllm.SamplingParams``.
        workers: None for same-process; positive int for subprocess mode.
        state_dir: override for the snapshot directory (subprocess mode).
        handle: reuse an already-created handle (subprocess mode); must
                have ``weights_path`` set.
        extra_env: extra environment variables to pass to children.

    Returns:
        List of ``ForkCompletionResult`` with the same length as prompts.

    Raises:
        ForkError: TP>1 with ``workers>0`` (not yet supported).
    """
    if workers is not None and workers <= 0:
        raise ForkError(
            f"workers must be None (same-process) or a positive int, got {workers!r}"
        )
    mode = "same_process" if workers is None else "subprocess"

    if mode == "same_process":
        t0 = time.perf_counter()
        outputs = llm.generate(list(prompts), sampling_params)
        elapsed = time.perf_counter() - t0
        results: list[ForkCompletionResult] = []
        for i, out in enumerate(outputs):
            text = out.outputs[0].text if out.outputs else ""
            token_ids = list(out.outputs[0].token_ids) if out.outputs else []
            results.append(
                ForkCompletionResult(
                    prompt=prompts[i],
                    text=text,
                    token_ids=token_ids,
                    worker_index=0,
                    elapsed_s=elapsed / max(len(prompts), 1),
                    mode=mode,
                )
            )
        return results

    # Subprocess mode --------------------------------------------------------
    if _tp_size(llm) > 1:
        raise ForkError(
            "fork_completions(workers>0) currently requires TP=1. "
            "Use workers=None or run the subprocess path under TP=1 "
            "and hydrate into a TP>1 child separately."
        )
    # Validated above; at this point workers is a positive int.
    owns_handle = handle is None
    if handle is None:
        handle = fork(llm, include_weights=True, state_dir=state_dir)
    else:
        if not handle.weights_path:
            raise ForkError(
                "fork_completions(workers>0) needs a handle with "
                "include_weights=True so subprocess children can load."
            )

    try:
        return _run_subprocess_workers(
            handle=handle,
            prompts=list(prompts),
            sampling_params=sampling_params,
            workers=int(workers),
            extra_env=extra_env or {},
        )
    finally:
        if owns_handle:
            handle.close()


# ---------------------------------------------------------------------------
# Subprocess fan-out
# ---------------------------------------------------------------------------


_WORKER_SCRIPT = r'''
import json
import os
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

payload_path = sys.argv[1]
with open(payload_path) as f:
    payload = json.load(f)

model_id = payload["model_id"]
weights_path = payload["weights_path"]
kv_path = payload["kv_path"]
prompts = payload["prompts"]
prompt_indices = payload["prompt_indices"]
worker_index = payload["worker_index"]
sp_kwargs = payload["sampling_params"]
result_path = payload["result_path"]

from vllm import SamplingParams
import thaw_vllm

sp = SamplingParams(**sp_kwargs)

t0 = time.perf_counter()
llm = thaw_vllm.load(
    model_id,
    weights_path,
    kv_snapshot=kv_path,
    enable_prefix_caching=True,
)
load_s = time.perf_counter() - t0

t0 = time.perf_counter()
outputs = llm.generate(prompts, sp)
gen_s = time.perf_counter() - t0

result = {
    "worker_index": worker_index,
    "load_s": load_s,
    "gen_s": gen_s,
    "outputs": [],
}
for i, out in enumerate(outputs):
    first = out.outputs[0] if out.outputs else None
    result["outputs"].append({
        "prompt_index": prompt_indices[i],
        "text": first.text if first else "",
        "token_ids": list(first.token_ids) if first else [],
    })

with open(result_path, "w") as f:
    json.dump(result, f)
'''


def _run_subprocess_workers(
    *,
    handle: ForkHandle,
    prompts: list,
    sampling_params,
    workers: int,
    extra_env: dict,
) -> list[ForkCompletionResult]:
    """Spawn N subprocess children, distribute prompts round-robin, collect."""
    if handle.weights_path is None:
        raise ForkError("subprocess workers require a handle with weights")

    # Round-robin prompt → worker assignment.
    assignments: list[list[int]] = [[] for _ in range(workers)]
    for i in range(len(prompts)):
        assignments[i % workers].append(i)

    sp_kwargs = _sampling_params_to_dict(sampling_params)

    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    env.update(extra_env)

    workdir = tempfile.mkdtemp(prefix="thaw_fork_workers_")
    procs: list[tuple[int, subprocess.Popen, str, float]] = []
    try:
        for w in range(workers):
            prompt_indices = assignments[w]
            if not prompt_indices:
                continue
            worker_prompts = [prompts[i] for i in prompt_indices]
            result_path = os.path.join(workdir, f"worker_{w}.json")
            payload_path = os.path.join(workdir, f"worker_{w}_payload.json")
            payload = {
                "model_id": handle.model_id,
                "weights_path": handle.weights_path,
                "kv_path": handle.kv_path,
                "prompts": worker_prompts,
                "prompt_indices": prompt_indices,
                "worker_index": w,
                "sampling_params": sp_kwargs,
                "result_path": result_path,
            }
            with open(payload_path, "w") as f:
                json.dump(payload, f)
            t0 = time.perf_counter()
            p = subprocess.Popen(
                [sys.executable, "-c", _WORKER_SCRIPT, payload_path],
                env=env,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            procs.append((w, p, result_path, t0))

        # Wait for all workers, collect outputs.
        results_by_prompt: dict[int, ForkCompletionResult] = {}
        for w, p, result_path, t0 in procs:
            rc = p.wait()
            elapsed = time.perf_counter() - t0
            if rc != 0:
                raise ForkError(
                    f"fork worker {w} exited with rc={rc}. "
                    f"Check the worker's stderr output."
                )
            if not os.path.exists(result_path):
                raise ForkError(
                    f"fork worker {w} produced no result at {result_path}"
                )
            with open(result_path) as f:
                data = json.load(f)
            for out in data["outputs"]:
                pi = out["prompt_index"]
                results_by_prompt[pi] = ForkCompletionResult(
                    prompt=prompts[pi],
                    text=out["text"],
                    token_ids=out["token_ids"],
                    worker_index=w,
                    elapsed_s=elapsed,
                    mode="subprocess",
                    raw=data,
                )

        return [results_by_prompt[i] for i in range(len(prompts))]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _sampling_params_to_dict(sp) -> dict:
    """Best-effort conversion of vllm.SamplingParams to a JSON-safe dict.

    vLLM's SamplingParams has many attributes; we forward the common
    ones. Anything more exotic should be set by the caller via
    ``extra_env`` or a dedicated wrapper.
    """
    fields = [
        "n",
        "best_of",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "min_tokens",
        "stop",
        "seed",
        "logprobs",
        "ignore_eos",
    ]
    out = {}
    for name in fields:
        if hasattr(sp, name):
            val = getattr(sp, name)
            # Skip None for fields where None is "unset" in vLLM.
            if val is None:
                continue
            try:
                json.dumps(val)
                out[name] = val
            except TypeError:
                # Not JSON-serializable; skip.
                pass
    return out


__all__ = [
    "fork",
    "fork_completions",
    "ForkHandle",
    "ForkCompletionResult",
    "ForkError",
    "ModelMismatchError",
    "BlockShapeMismatchError",
    "BlockPoolTooSmallError",
    "PrefixCachingDisabledError",
    "UnfinishedRequestsError",
    "HandleClosedError",
]
