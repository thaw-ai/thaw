"""
thaw_vllm.fork_pool — pre-warmed subprocess pool for ``fork_completions``.

``fork_completions(workers=N)`` without a pool pays the ~340s vLLM
cold-boot *per call*. That one-shot subprocess pattern is fine for a
single demo run but unusable for RL training loops or agent systems
that want to fork hundreds of times. ForkPool boots N vLLM subprocess
workers once, keeps them resident on GPU (dummy weights), and hot-swaps
weights + KV into them on each fork via thaw's pipelined DMA paths.

Rough budget on an H100 80 GB per fork, after the pool is warm:

  - fork(llm, include_weights=True)     ~2-6 s  (NVMe write-bound)
  - worker.hydrate                      ~2-5 s  (pipelined DMA)
  - worker.generate                      real work
  - worker.reset                         <0.1 s

Comparable path without the pool: every fork re-pays ~340 s of vLLM
engine construction (torch.compile/CUDA graphs/weights init/NCCL
bring-up). The pool amortizes that cost to a one-time cost at
init_pool time.

Usage
-----
>>> from thaw_vllm import ForkPool, fork_completions
>>> pool = ForkPool()
>>> pool.init_pool(
...     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
...     workers=4,
...     gpu_memory_utilization=0.35,
... )
>>> results = fork_completions(llm, prompts, sp, pool=pool)
>>> pool.close()

Constraints in v1
-----------------
- TP=1 only. Subprocess workers are single-engine; TP>1 fork across
  processes needs a pool-of-pools layout that lands in Phase 2.
- Same model id across all forks dispatched to one pool. Different
  models need separate pools (future: ForkPool.register for multi-model,
  mirroring EnginePool).
- Synchronous API. Async wrapper is Phase 2.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from thaw_vllm.fork import (
    ForkCompletionResult,
    ForkError,
    ForkHandle,
    _sampling_params_to_dict,
    _tp_size,
    fork as _fork,
)

logger = logging.getLogger("thaw.fork_pool")

# Launch the worker by file path rather than ``-m thaw_vllm._fork_pool_worker``.
# The -m form imports ``thaw_vllm/__init__.py`` during module resolution (to
# find the submodule), which triggers ``_register_loader()`` and emits a vLLM
# INFO line onto stdout *before* the worker script's code runs — so the pipe's
# first line is garbage from the parent's perspective and the boot handshake
# fails. Running the file directly keeps fd 1 clean until the worker's own
# dup2 redirects it.
_WORKER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "_fork_pool_worker.py"
)


class ForkPoolError(ForkError):
    """Base class for ForkPool-specific failures."""


class WorkerBootTimeout(ForkPoolError):
    """A worker did not emit its ``ready`` line before the boot deadline."""


class WorkerDead(ForkPoolError):
    """A worker process exited unexpectedly."""


class WorkerProtocolError(ForkPoolError):
    """Worker returned a malformed message or unexpected op response."""


# ---------------------------------------------------------------------------
# WorkerSlot — one live subprocess
# ---------------------------------------------------------------------------


@dataclass
class _WorkerSlot:
    id: int
    proc: Any  # subprocess.Popen
    lock: threading.Lock = field(default_factory=threading.Lock)
    boot_s: float = 0.0
    current_weights_path: Optional[str] = None
    current_kv_path: Optional[str] = None
    dead: bool = False

    @property
    def pid(self) -> int:
        return self.proc.pid if self.proc else -1


# ---------------------------------------------------------------------------
# ForkPool
# ---------------------------------------------------------------------------


class ForkPool:
    """Pool of pre-warmed vLLM subprocess workers for agent fork.

    All workers share the same base model architecture. Each ``dispatch``
    call:

      1. The parent snapshots the live session to disk (one ``ForkHandle``).
      2. The handle's weights.thaw + kv.thawkv are DMA'd into each
         idle worker via ``restore_model_tp`` + ``restore_kv_cache``.
      3. Each worker runs ``llm.generate`` on its prompt slice.
      4. Each worker resets its prefix cache, ready for the next dispatch.

    Thread-safe: ``dispatch`` may be called concurrently from multiple
    threads. Internal locks serialize per-worker IPC.
    """

    def __init__(self, *, worker_cmd: Optional[list[str]] = None) -> None:
        self.slots: list[_WorkerSlot] = []
        self.model: str = ""
        self.worker_kwargs: dict = {}
        self.preload_weights: bool = False
        self._closed = False
        self._claim_lock = threading.Lock()
        # Override for tests — a script that speaks the same JSON
        # protocol but doesn't import vLLM. Production callers leave
        # this None and the real worker script is launched by path.
        self._worker_cmd: list[str] = worker_cmd or [
            sys.executable, _WORKER_SCRIPT,
        ]

    # ---- lifecycle -------------------------------------------------------

    def init_pool(
        self,
        model: str,
        workers: int = 1,
        *,
        preload_weights: bool = False,
        boot_timeout_s: float = 1800.0,
        env: Optional[dict] = None,
        **llm_kwargs,
    ) -> None:
        """Spawn ``workers`` subprocess workers each running a dummy-weight engine.

        Args:
            model: HuggingFace repo id (or local path). Must match the
                ``model_id`` of any ForkHandle later dispatched through
                this pool.
            workers: Number of subprocess workers. Each holds a full
                vLLM engine resident on GPU — sizing is your decision.
                An 80 GB H100 fits parent + 1 worker comfortably for
                8B fp16; 2× H100 (or H200/B200) fits parent + multiple
                workers.
            preload_weights: If True, each worker boots with the real
                model weights (``load_format="auto"``) and subsequent
                forks only snapshot + restore KV cache. Correct for RL
                rollouts where the policy is constant across forks of
                the same epoch. Default False matches the original
                weight-patch flow where every fork ships fresh weights.
            boot_timeout_s: Per-worker boot budget. vLLM cold-start is
                2-6 minutes on H100; default 30 min is generous.
            env: Extra environment variables passed to every worker.
            **llm_kwargs: Forwarded to the worker's ``LLM()`` call.
                Recommended: ``gpu_memory_utilization``, ``max_model_len``,
                ``enforce_eager``, ``dtype``.

        Raises:
            WorkerBootTimeout: a worker failed to emit ``ready`` in time.
            WorkerDead: a worker exited before boot completed.
            ForkPoolError: any protocol error during boot handshake.
        """
        if self.slots:
            raise ForkPoolError("init_pool called twice; close() first")
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers!r}")

        self.model = model
        self.worker_kwargs = dict(llm_kwargs)
        self.preload_weights = bool(preload_weights)

        base_env = os.environ.copy()
        base_env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        base_env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        if env:
            base_env.update(env)

        boot_config = {
            "op": "boot",
            "model": model,
            "llm_kwargs": dict(llm_kwargs),
            "preload_weights": self.preload_weights,
            "gpu_memory_utilization": llm_kwargs.get(
                "gpu_memory_utilization", 0.35
            ),
            "max_model_len": llm_kwargs.get("max_model_len", 24576),
            "enforce_eager": llm_kwargs.get("enforce_eager", True),
            "dtype": llm_kwargs.get("dtype", "float16"),
        }

        logger.info(
            "ForkPool: booting %d worker(s) for %s (this takes several minutes)",
            workers, model,
        )

        # Boot workers serially. Parallel boot risks GPU contention
        # during torch.compile / weights init; serial is slower wall-
        # clock but predictable and survivable on single-GPU pods.
        for i in range(workers):
            slot = self._spawn_slot(i, boot_config, base_env, boot_timeout_s)
            self.slots.append(slot)
            logger.info(
                "ForkPool slot %d ready in %.1fs (pid=%d)",
                slot.id, slot.boot_s, slot.pid,
            )

    def _spawn_slot(
        self,
        slot_id: int,
        boot_config: dict,
        env: dict,
        boot_timeout_s: float,
    ) -> _WorkerSlot:
        import subprocess

        proc = subprocess.Popen(
            self._worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # pass worker logs through
            text=True,
            bufsize=1,  # line-buffered
            env=env,
        )
        slot = _WorkerSlot(id=slot_id, proc=proc)

        # Send the boot config.
        try:
            self._raw_send(slot, boot_config)
        except BrokenPipeError as e:
            slot.dead = True
            raise WorkerDead(
                f"worker slot {slot_id} died before accepting boot config"
            ) from e

        # Wait for the ready line with a deadline. readline blocks; we
        # poll with a thread that kills the proc on timeout — simpler
        # than non-blocking IO in Python.
        ready = self._recv_with_timeout(slot, boot_timeout_s)
        if ready is None:
            self._terminate_slot(slot)
            raise WorkerBootTimeout(
                f"worker slot {slot_id} did not emit ready in {boot_timeout_s:.0f}s"
            )
        if ready.get("status") != "ready":
            self._terminate_slot(slot)
            raise ForkPoolError(
                f"worker slot {slot_id} boot failed: {ready!r}"
            )

        slot.boot_s = float(ready.get("boot_s", 0.0))
        return slot

    def close(self, timeout_s: float = 30.0) -> None:
        """Shutdown all workers. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for slot in self.slots:
            if slot.dead or slot.proc.poll() is not None:
                continue
            try:
                self._raw_send(slot, {"op": "shutdown"})
            except Exception:
                pass
            try:
                slot.proc.wait(timeout=timeout_s)
            except Exception:
                self._terminate_slot(slot)
        self.slots = []

    def __enter__(self) -> "ForkPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close(timeout_s=2.0)
        except Exception:
            pass

    # ---- dispatch --------------------------------------------------------

    def fork_completions(
        self,
        llm,
        prompts: Sequence[str],
        sampling_params,
        *,
        state_dir: Optional[str] = None,
        handle: Optional[ForkHandle] = None,
        extra_op_fields: Optional[dict] = None,
    ) -> list[ForkCompletionResult]:
        """Snapshot the parent, hydrate into the pool, generate.

        Equivalent in spirit to ``thaw_vllm.fork_completions(workers=N)``
        but dispatched across the pool's live workers instead of spawning
        fresh ones.

        Args:
            llm: parent vLLM engine.
            prompts: list of prompt strings.
            sampling_params: ``vllm.SamplingParams``.
            state_dir: override for the snapshot directory. If None, a
                temp dir is used and cleaned up after all workers finish.
            handle: reuse an already-created handle. Must have
                ``weights_path`` set.
            extra_op_fields: extra fields merged into every hydrate op
                (reserved for future pool features).

        Returns:
            ``list[ForkCompletionResult]``, same length and order as
            ``prompts``.

        Raises:
            ForkPoolError / WorkerDead / subclasses: on IPC or worker failure.
        """
        self._assert_open()
        if not prompts:
            return []
        if _tp_size(llm) > 1:
            raise ForkError(
                "ForkPool.fork_completions currently requires parent TP=1."
            )

        owns_handle = handle is None
        include_weights = not self.preload_weights
        if handle is None:
            handle = _fork(llm, include_weights=include_weights,
                           state_dir=state_dir)
        elif include_weights and not handle.weights_path:
            raise ForkError(
                "ForkPool.fork_completions requires a handle with "
                "include_weights=True when the pool was booted without "
                "preload_weights=True."
            )

        try:
            return self._dispatch(handle, list(prompts), sampling_params,
                                  extra_op_fields or {})
        finally:
            if owns_handle:
                handle.close()

    def _dispatch(
        self,
        handle: ForkHandle,
        prompts: list[str],
        sampling_params,
        extra_op_fields: dict,
    ) -> list[ForkCompletionResult]:
        """Round-robin prompts across live slots, run concurrently, collect."""
        slot_count = len(self.slots)
        if slot_count == 0:
            raise ForkPoolError("dispatch with no workers; init_pool first")

        active = min(slot_count, len(prompts))
        assignments: list[list[int]] = [[] for _ in range(active)]
        for i in range(len(prompts)):
            assignments[i % active].append(i)

        sp_kwargs = _sampling_params_to_dict(sampling_params)

        results_by_prompt: dict[int, ForkCompletionResult] = {}
        errors: list[BaseException] = []

        def _run_on_slot(slot_idx: int, prompt_indices: list[int]) -> None:
            slot = self.slots[slot_idx]
            worker_prompts = [prompts[i] for i in prompt_indices]
            t0 = time.perf_counter()
            try:
                with slot.lock:
                    self._send_and_check(
                        slot,
                        {
                            "op": "hydrate",
                            "weights_path": handle.weights_path,
                            "kv_path": handle.kv_path,
                            **extra_op_fields,
                        },
                    )
                    gen_resp = self._send_and_check(
                        slot,
                        {
                            "op": "generate",
                            "prompts": worker_prompts,
                            "sampling_params": sp_kwargs,
                        },
                    )
                    # Reset is fire-and-forget quality-of-life; if it
                    # errors we still return results. Raises propagate.
                    self._send_and_check(slot, {"op": "reset"})
            except BaseException as e:
                errors.append(e)
                return

            elapsed = time.perf_counter() - t0
            outputs = gen_resp.get("outputs", [])
            for local_i, pi in enumerate(prompt_indices):
                out = outputs[local_i] if local_i < len(outputs) else {}
                results_by_prompt[pi] = ForkCompletionResult(
                    prompt=prompts[pi],
                    text=out.get("text", ""),
                    token_ids=out.get("token_ids", []),
                    worker_index=slot_idx,
                    elapsed_s=elapsed,
                    mode="subprocess",
                    raw=gen_resp,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=active) as pool:
            futures = [
                pool.submit(_run_on_slot, i, assignments[i])
                for i in range(active)
            ]
            for f in concurrent.futures.as_completed(futures):
                # Re-raise any unhandled exception from a worker thread.
                f.result()

        if errors:
            # Surface the first error; the rest will be referenced in
            # the traceback chain for debugging.
            raise errors[0]

        return [results_by_prompt[i] for i in range(len(prompts))]

    # ---- IPC plumbing ----------------------------------------------------

    def _assert_open(self) -> None:
        if self._closed:
            raise ForkPoolError("ForkPool is closed")

    def _raw_send(self, slot: _WorkerSlot, obj: dict) -> None:
        if slot.proc.stdin is None or slot.proc.stdin.closed:
            raise WorkerDead(f"worker slot {slot.id} stdin is closed")
        line = json.dumps(obj, separators=(",", ":"))
        slot.proc.stdin.write(line + "\n")
        slot.proc.stdin.flush()

    def _recv_with_timeout(
        self, slot: _WorkerSlot, timeout_s: float
    ) -> Optional[dict]:
        """Read one line from the worker with a deadline.

        Uses a background thread to readline; if it hasn't returned by
        the deadline we assume the worker is wedged, kill it, and
        return None so the caller can raise.
        """
        result: list[Optional[str]] = [None]
        done = threading.Event()

        def _read() -> None:
            try:
                line = slot.proc.stdout.readline()
                result[0] = line
            except Exception:
                result[0] = None
            finally:
                done.set()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        if not done.wait(timeout_s):
            return None
        line = result[0]
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise WorkerProtocolError(
                f"worker slot {slot.id} returned non-JSON: {line!r}"
            ) from e

    def _recv(self, slot: _WorkerSlot) -> dict:
        line = slot.proc.stdout.readline()
        if not line:
            rc = slot.proc.poll()
            slot.dead = True
            raise WorkerDead(
                f"worker slot {slot.id} closed stdout (rc={rc})"
            )
        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            raise WorkerProtocolError(
                f"worker slot {slot.id} returned non-JSON: {line!r}"
            ) from e

    def _send_and_check(self, slot: _WorkerSlot, op: dict) -> dict:
        """Send an op, receive the response, raise on error status."""
        self._raw_send(slot, op)
        resp = self._recv(slot)
        if resp.get("status") != "ok":
            msg = resp.get("message", "")
            tb = resp.get("traceback", "")
            raise ForkPoolError(
                f"worker slot {slot.id} op={op.get('op')!r} failed: "
                f"{resp.get('type', 'Error')}: {msg}\n{tb}"
            )
        return resp

    def _terminate_slot(self, slot: _WorkerSlot) -> None:
        slot.dead = True
        try:
            slot.proc.terminate()
            slot.proc.wait(timeout=5)
        except Exception:
            try:
                slot.proc.kill()
            except Exception:
                pass

    # ---- introspection ---------------------------------------------------

    def status(self) -> dict:
        return {
            "model": self.model,
            "closed": self._closed,
            "workers": [
                {
                    "id": s.id,
                    "pid": s.pid,
                    "boot_s": round(s.boot_s, 2),
                    "busy": s.lock.locked(),
                    "dead": s.dead or (s.proc.poll() is not None),
                }
                for s in self.slots
            ],
        }


__all__ = [
    "ForkPool",
    "ForkPoolError",
    "WorkerBootTimeout",
    "WorkerDead",
    "WorkerProtocolError",
]
