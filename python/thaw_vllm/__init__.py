# thaw_vllm — GPU snapshot/restore for vLLM model weights.
#
# This module provides two operations:
#
#   freeze_model(model, path)   — snapshot all model weights to a .thaw file
#   restore_model(model, path)  — load weights from a .thaw file back onto GPU
#
# The file format is thaw's binary format (see crates/thaw-core). This
# Python implementation reads/writes the same byte layout as the Rust
# side, so files are interchangeable. The Python path uses PyTorch's
# own CUDA memory operations under the hood — no custom FFI required.
#
# This is the MVP integration layer. In production, the Rust
# implementation (via thaw-py/PyO3) replaces this for higher
# throughput. The file format is identical either way.

# Engine-agnostic functions from thaw_common
from thaw_common.snapshot import (
    freeze_model,
    freeze_model_pipelined,
    restore_model,
    restore_model_from_ram,
    restore_model_pipelined,
)

# vLLM-specific TP functions
from thaw_vllm.snapshot import (
    freeze_model_tp,
    restore_model_tp,
)

from thaw_vllm.kv_snapshot import (
    freeze_kv_cache,
    freeze_kv_cache_tp,
    restore_kv_cache,
    restore_kv_cache_tp,
)
from thaw_vllm.pool import EnginePool, create_pool_app

# Register load_format="thaw" with vLLM when available.
try:
    from thaw_vllm.loader import ThawModelLoader  # noqa: F401
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
    """
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
        restore_model_tp(llm, snapshot)
    else:
        llm = LLM(
            model=model,
            load_format="thaw",
            model_loader_extra_config={"snapshot": snapshot},
            **kwargs,
        )

    if kv_snapshot:
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
]
