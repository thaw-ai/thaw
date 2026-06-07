import os as _os

# vLLM 0.19+ defaults to msgspec for engine IPC, which rejects function
# callables. `collective_rpc(fn, args=...)` relies on passing worker
# callables by reference, so opt into the cloudpickle fallback. Setdefault
# so callers can still override if they need hardened serialization.
_os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

# thaw_vllm — GPU snapshot/restore for vLLM model weights + KV cache, plus
# the fork() primitive that makes a live session portable.
#
# Lazy imports (PEP 562 __getattr__): `import thaw_vllm` must NOT pull in
# torch / vLLM. The heavy submodules (snapshot, kv_snapshot, pool,
# fork_pool, sleep_mode, loader) are imported only when one of their
# symbols is actually accessed. This is what lets the lightweight CLI
# (`thaw info` / `inspect` / `diff`) and `ForkHandle` inspection run on a
# laptop with no GPU stack installed. Heavy work still imports torch/vLLM
# on demand, exactly as before — just not at package-import time.

# Exported name -> submodule that defines it. Resolved on first access.
_LAZY = {
    # thaw_common.snapshot (engine-agnostic weight freeze/restore)
    "freeze_model": "thaw_common.snapshot",
    "freeze_model_pipelined": "thaw_common.snapshot",
    "restore_model": "thaw_common.snapshot",
    "restore_model_from_ram": "thaw_common.snapshot",
    "restore_model_pipelined": "thaw_common.snapshot",
    # thaw_vllm.snapshot (vLLM TP weight freeze/restore)
    "freeze_model_tp": "thaw_vllm.snapshot",
    "restore_model_tp": "thaw_vllm.snapshot",
    # thaw_vllm.kv_snapshot (KV cache freeze/restore — the moat)
    "freeze_kv_cache": "thaw_vllm.kv_snapshot",
    "freeze_kv_cache_tp": "thaw_vllm.kv_snapshot",
    "restore_kv_cache": "thaw_vllm.kv_snapshot",
    "restore_kv_cache_tp": "thaw_vllm.kv_snapshot",
    # NOTE: thaw_vllm.fork names are NOT lazy — see the eager import below.
    # `fork` is both a submodule (fork.py) and a function; only an eager
    # binding makes `from thaw_vllm import fork` resolve to the callable
    # rather than the module. fork.py is stdlib-only at load (torch is
    # imported lazily inside its functions), so this stays GPU-free.
    # thaw_vllm.pool (pre-warmed engine pool + OpenAI server)
    "EnginePool": "thaw_vllm.pool",
    "create_pool_app": "thaw_vllm.pool",
    # thaw_vllm.fork_pool (pre-warmed subprocess workers)
    "ForkPool": "thaw_vllm.fork_pool",
    "ForkPoolError": "thaw_vllm.fork_pool",
    "WorkerBootTimeout": "thaw_vllm.fork_pool",
    "WorkerDead": "thaw_vllm.fork_pool",
    "WorkerProtocolError": "thaw_vllm.fork_pool",
    # thaw_vllm.loader (load_format="thaw" ModelLoader)
    "ThawModelLoader": "thaw_vllm.loader",
}


def __getattr__(name):
    """PEP 562 lazy attribute access. Imports the owning submodule on
    first touch, caches the result in module globals, and returns it."""
    import importlib

    if name == "sleep_mode":
        mod = importlib.import_module("thaw_vllm.sleep_mode")
        globals()["sleep_mode"] = mod
        return mod

    if name == "rewind":
        # Laptop-side rollout inspection (logprob diff / pivot). stdlib-only,
        # no GPU — same discipline as agentfs.
        mod = importlib.import_module("thaw_vllm.rewind")
        globals()["rewind"] = mod
        return mod

    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'thaw_vllm' has no attribute {name!r}")
    mod = importlib.import_module(target)
    val = getattr(mod, name)
    globals()[name] = val  # cache: subsequent access skips __getattr__
    return val


def __dir__():
    return sorted(set(globals()) | set(_LAZY) | {"sleep_mode", "rewind", "load"})


# Eager (but GPU-free) — fork.py imports only stdlib at module load; torch/vLLM
# are imported lazily inside its functions. Eager binding is REQUIRED so the
# `fork` *function* shadows the `fork` *submodule* for `from thaw_vllm import fork`.
from thaw_vllm.fork import (  # noqa: E402
    fork,
    checkpoint,
    checkout,
    capture_rollouts,
    fork_completions,
    ForkHandle,
    ForkCompletionResult,
    ForkError,
    ModelMismatchError,
    BlockShapeMismatchError,
    BlockPoolTooSmallError,
    PrefixCachingDisabledError,
    UnfinishedRequestsError,
    HandleClosedError,
)


# Register load_format="thaw" with vLLM when available. vLLM also
# auto-discovers this via the `vllm.general_plugins` entrypoint in
# pyproject.toml, so `LLM(load_format="thaw", ...)` works without an
# explicit `import thaw_vllm` once thaw-native is installed. This block
# preserves the existing `import thaw_vllm` side effect; register() is
# idempotent. The ImportError guard makes a torch/vLLM-free import a
# no-op rather than a crash.
try:
    from thaw_vllm.loader import register as _register_loader

    _register_loader()
except ImportError:
    # vLLM not installed — loader registration is optional.
    pass


def load(model: str, snapshot: str, kv_snapshot: str = None, **kwargs):
    """One-line thaw-powered model loading.

    Usage:
        import thaw_vllm
        llm = thaw_vllm.load("meta-llama/Meta-Llama-3-8B", "/path/to/weights.thaw")

    Multi-GPU:
        llm = thaw_vllm.load("meta-llama/Meta-Llama-3-70B", "/path/to/weights.thaw",
                             tensor_parallel_size=4)

    KV-cache restore requires V1-inproc mode because it touches scheduler
    state that isn't reachable across V1 MP's IPC. Weights-only loads run
    under V1 MP default.
    """
    import os

    if kv_snapshot:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm import LLM

    kwargs.setdefault("enforce_eager", True)
    kwargs.setdefault("dtype", "float16")

    tp_size = kwargs.get("tensor_parallel_size", 1)

    if tp_size > 1:
        # TP > 1: vLLM spawns worker processes that don't have thaw_vllm
        # imported, so load_format="thaw" fails. Instead, init with dummy
        # weights then restore via collective_rpc to each worker.
        llm = LLM(
            model=model,
            load_format="dummy",
            **kwargs,
        )
        from thaw_vllm.snapshot import restore_model_tp

        restore_model_tp(llm, snapshot)
    else:
        llm = LLM(
            model=model,
            load_format="thaw",
            model_loader_extra_config={"snapshot": snapshot},
            **kwargs,
        )

    if kv_snapshot:
        from thaw_vllm.kv_snapshot import restore_kv_cache, restore_kv_cache_tp

        if tp_size > 1:
            restore_kv_cache_tp(llm, kv_snapshot)
        else:
            restore_kv_cache(llm, kv_snapshot)

    return llm


__all__ = [
    "freeze_model",
    "freeze_model_pipelined",
    "freeze_model_tp",
    "restore_model",
    "restore_model_from_ram",
    "restore_model_pipelined",
    "restore_model_tp",
    "freeze_kv_cache",
    "freeze_kv_cache_tp",
    "restore_kv_cache",
    "restore_kv_cache_tp",
    "load",
    "ThawModelLoader",
    "EnginePool",
    "create_pool_app",
    # Fork primitive — make GPU session state portable.
    "fork",
    "checkpoint",
    "checkout",
    "capture_rollouts",
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
    # ForkPool — pre-warmed subprocess workers for repeated forks.
    "ForkPool",
    "ForkPoolError",
    "WorkerBootTimeout",
    "WorkerDead",
    "WorkerProtocolError",
    # sleep-mode backend wrapper (vLLM RFC #34303 integration).
    "sleep_mode",
    # rewind — laptop-side RL rollout inspection (logprob diff / pivot).
    "rewind",
]
