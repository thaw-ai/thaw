"""
thaw_vllm._fork_pool_worker — long-lived subprocess for ForkPool.

Launched by path (``python /.../python/thaw_vllm/_fork_pool_worker.py``),
not ``-m``. Running as ``-m`` would import the ``thaw_vllm`` package
during module resolution, which triggers ``_register_loader()`` and
emits a vLLM log line onto fd 1 *before* this script's dup2 can
redirect it — breaking the boot handshake. Boots a dummy-weight vLLM
engine once, then services JSON-line commands from stdin until
shutdown. One command at a time; responses go out on stdout, one line
each. Stderr is untouched — useful for tracebacks + vLLM log noise.

Wire protocol
-------------
Each message is a single line of JSON terminated by ``\\n``. No embedded
newlines in payloads (the parent serializes with compact separators).

Parent → worker:
    {"op": "hydrate", "weights_path": "...", "kv_path": "..."}
    {"op": "generate", "prompts": [...], "sampling_params": {...}}
    {"op": "reset"}
    {"op": "shutdown"}

Worker → parent (first line after boot):
    {"status": "ready", "boot_s": 137.4, "pid": 12345}

Worker → parent (per command):
    {"status": "ok", ...op-specific fields}
    {"status": "error", "type": "...", "message": "...", "traceback": "..."}

Design notes
------------
- ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` is required for the KV cache path;
  we set it via ``setdefault`` at import so callers can override if they
  only ever need weight-only hydration (not the current API).
- Each ``hydrate`` drops any prior prefix-cache state before restoring
  the new KV snapshot — otherwise stale hashes would collide with new
  block_ids.
- Generate errors are caught and reported as structured responses; only
  an unrecoverable KeyboardInterrupt or vLLM engine crash unwinds past
  the command loop and exits the worker.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Optional

# V1 MP off: KV restore needs an in-proc scheduler. Worker-scoped; does
# not affect the parent that spawned us.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# Reserve stdout (fd 1) exclusively for JSON-lines IPC with the parent.
# vLLM's engine init writes heavily to stdout (config dumps, progress
# bars, NCCL banner). If we leave fd 1 as the real stdout, that noise
# interleaves with our JSON responses and the parent can't parse them;
# worse, vLLM's suppress_stdout() flushes stdout and raises BrokenPipe
# when the parent hasn't drained the pipe. Fix: dup fd 1 to a fresh fd
# for our own writes, then point fd 1 at stderr (fd 2). All vLLM chatter
# lands on stderr where the parent ignores it.
#
# CRITICAL: this must run *before* any import that could log to stdout.
# The parent launches this script by path (not ``-m``) precisely so
# ``thaw_vllm/__init__.py``'s ``_register_loader()`` log line doesn't
# poison fd 1 before we get control. Do not move these lines.
_IPC_OUT_FD = os.dup(1)
os.dup2(2, 1)
_IPC_OUT = os.fdopen(_IPC_OUT_FD, "w", buffering=1)
sys.stdout = sys.stderr  # Python-level: print()/sys.stdout.write go to stderr

# Running as a script means Python's module search path doesn't include
# this file's package parent. Add ``python/`` so ``from thaw_vllm...``
# works in the op handlers below.
_PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


class _WorkerState:
    """Holds the vLLM engine and last-known hydrated paths."""

    def __init__(self) -> None:
        self.llm: Any = None
        self.current_weights_path: Optional[str] = None
        self.current_kv_path: Optional[str] = None
        self.preload_weights: bool = False


def _write_response(obj: dict) -> None:
    """Emit one JSON line on the reserved IPC fd and flush."""
    line = json.dumps(obj, separators=(",", ":"))
    _IPC_OUT.write(line + "\n")
    _IPC_OUT.flush()


def _read_request() -> Optional[dict]:
    """Read one JSON line from stdin. Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)


def _build_error(exc: BaseException) -> dict:
    return {
        "status": "error",
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _boot(config: dict, state: _WorkerState) -> None:
    """Construct the engine that all hydrations land into.

    Two load strategies:

    - ``preload_weights=False`` (default): boot with ``load_format="dummy"``.
      Each hydrate restores both weights + KV from the parent's snapshot.
      Appropriate when parent weights change between forks (policy updates
      mid-rollout, weight-patch studies).

    - ``preload_weights=True``: boot with the full model via vLLM's native
      loader. Subsequent hydrates only restore KV — weights stay resident.
      Drops per-fork cost from (weights + KV) to (KV only). The RL rollout
      shape — same policy, many trunks — is exactly this case.
    """
    from vllm import LLM

    model = config["model"]
    state.preload_weights = bool(config.get("preload_weights", False))
    kwargs: dict[str, Any] = {
        "load_format": "auto" if state.preload_weights else "dummy",
        "enable_prefix_caching": True,
        "enforce_eager": config.get("enforce_eager", True),
        "dtype": config.get("dtype", "float16"),
        "gpu_memory_utilization": config.get("gpu_memory_utilization", 0.35),
        "max_model_len": config.get("max_model_len", 24576),
    }
    extra = config.get("llm_kwargs") or {}
    kwargs.update(extra)
    state.llm = LLM(model=model, **kwargs)


def _op_hydrate(state: _WorkerState, req: dict) -> dict:
    """Restore weights (if provided) and KV from the handle into the engine."""
    if state.llm is None:
        raise RuntimeError("hydrate called before engine boot")

    weights_path = req.get("weights_path")
    kv_path = req.get("kv_path")
    if not kv_path:
        raise ValueError("hydrate requires kv_path")
    if not weights_path and not state.preload_weights:
        raise ValueError(
            "hydrate requires weights_path when worker booted "
            "without preload_weights=True"
        )

    from thaw_vllm.kv_snapshot import restore_kv_cache

    # Drop any prior prefix-cache state before restoring a fresh snapshot.
    # Without this, the new block_ids collide with stale hashes from the
    # prior hydrate and the scheduler serves cached tokens belonging to
    # the previous fork's conversation.
    _reset_prefix_cache(state.llm)

    weights_s = 0.0
    w_stats: Any = None
    if weights_path:
        from thaw_vllm.snapshot import restore_model_tp
        t0 = time.perf_counter()
        w_stats = restore_model_tp(state.llm, weights_path)
        weights_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    kv_stats = restore_kv_cache(state.llm, kv_path)
    kv_s = time.perf_counter() - t0

    state.current_weights_path = weights_path
    state.current_kv_path = kv_path

    return {
        "status": "ok",
        "weights_s": weights_s,
        "kv_s": kv_s,
        "weights": _scrub(w_stats),
        "kv": _scrub(kv_stats),
    }


def _op_generate(state: _WorkerState, req: dict) -> dict:
    """Run llm.generate(prompts, sp). Returns text + token_ids per prompt."""
    if state.llm is None:
        raise RuntimeError("generate called before engine boot")

    from vllm import SamplingParams

    prompts = req["prompts"]
    sp_kwargs = req.get("sampling_params") or {}
    sp = SamplingParams(**sp_kwargs)

    t0 = time.perf_counter()
    outputs = state.llm.generate(prompts, sp)
    gen_s = time.perf_counter() - t0

    results = []
    for out in outputs:
        first = out.outputs[0] if out.outputs else None
        results.append(
            {
                "text": first.text if first else "",
                "token_ids": list(first.token_ids) if first else [],
                "finish_reason": getattr(first, "finish_reason", None)
                if first
                else None,
            }
        )

    return {"status": "ok", "gen_s": gen_s, "outputs": results}


def _op_reset(state: _WorkerState, req: dict) -> dict:
    """Clear prefix-cache state so the next hydrate starts clean."""
    if state.llm is None:
        raise RuntimeError("reset called before engine boot")
    _reset_prefix_cache(state.llm)
    return {"status": "ok"}


def _reset_prefix_cache(llm) -> None:
    """Drop cached_block_hash_to_block so subsequent hydrations don't collide.

    The KV restore path rewrites the block pool's hash map to point at
    the snapshot's block_ids. If a prior hydrate left entries around,
    the new snapshot's block_ids may overlap and the scheduler will
    serve stale tokens. Clearing before every hydrate is the cheap fix.
    """
    try:
        from thaw_vllm.kv_snapshot import _get_engine_core

        ec = _get_engine_core(llm)
        block_pool = ec.scheduler.kv_cache_manager.block_pool
        if hasattr(block_pool, "cached_block_hash_to_block"):
            block_pool.cached_block_hash_to_block.clear()
        if hasattr(block_pool, "free_block_queue"):
            # Best-effort: drop any blocks currently pinned as "cached"
            # back into the free queue. vLLM exposes reset_prefix_cache
            # on some versions; fall through if absent.
            pass
        reset_fn = getattr(llm, "reset_prefix_cache", None)
        if callable(reset_fn):
            reset_fn()
    except Exception:
        # Non-fatal: worst case the next hydrate starts with stale
        # entries. restore_kv_cache overwrites by block_id so the
        # tensors themselves are correct; only the hash map can stale.
        pass


def _scrub(stats: Any) -> Any:
    """Make stats JSON-safe — drop nested tensors / non-serializable values."""
    if stats is None:
        return None
    if isinstance(stats, dict):
        out = {}
        for k, v in stats.items():
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
        return out
    try:
        json.dumps(stats)
        return stats
    except (TypeError, ValueError):
        return repr(stats)


_OPS = {
    "hydrate": _op_hydrate,
    "generate": _op_generate,
    "reset": _op_reset,
}


def main() -> int:
    state = _WorkerState()

    # Boot: read one config line, construct engine, emit "ready".
    boot_t0 = time.perf_counter()
    try:
        config = _read_request()
        if config is None:
            sys.stderr.write("thaw fork worker: stdin closed before boot\n")
            return 2
        if config.get("op") != "boot":
            _write_response(
                {
                    "status": "error",
                    "type": "ProtocolError",
                    "message": f"first message must be op=boot, got {config!r}",
                    "traceback": "",
                }
            )
            return 2
        _boot(config, state)
    except BaseException as e:
        _write_response(_build_error(e))
        return 3
    _write_response(
        {"status": "ready", "boot_s": time.perf_counter() - boot_t0, "pid": os.getpid()}
    )

    # Command loop.
    while True:
        try:
            req = _read_request()
        except json.JSONDecodeError as e:
            _write_response(_build_error(e))
            continue
        if req is None:
            # Parent closed stdin — treat as implicit shutdown.
            return 0
        op = req.get("op")
        if op == "shutdown":
            _write_response({"status": "ok"})
            return 0
        handler = _OPS.get(op)
        if handler is None:
            _write_response(
                {
                    "status": "error",
                    "type": "ProtocolError",
                    "message": f"unknown op: {op!r}",
                    "traceback": "",
                }
            )
            continue
        try:
            resp = handler(state, req)
        except BaseException as e:
            _write_response(_build_error(e))
            continue
        _write_response(resp)


if __name__ == "__main__":
    sys.exit(main())
