"""
thaw_sglang — GPU snapshot/restore for SGLang inference.

Same .thaw file format as thaw_vllm. Snapshots are interchangeable
between engines — freeze with vLLM, restore with SGLang, or vice versa.

Usage:
    import thaw_sglang
    engine = thaw_sglang.load("meta-llama/Meta-Llama-3-8B", "/path/to/weights.thaw")

Multi-GPU:
    engine = thaw_sglang.load("meta-llama/Meta-Llama-3-70B", "/path/to/weights.thaw",
                              tp_size=4)
"""

# Engine-agnostic functions from thaw_common
from thaw_common.snapshot import (
    freeze_model,
    freeze_model_pipelined,
    restore_model,
    restore_model_from_ram,
    restore_model_pipelined,
)

# SGLang-specific loaders. The import is lazy: if SGLang isn't installed,
# attribute access raises a clear ImportError instead of silently returning
# None (which would later crash with a confusing "NoneType is not callable"
# inside vLLM/SGLang's load_format dispatch).
_SGLANG_IMPORT_ERROR = None
try:
    from thaw_sglang.loader import ThawSGLangModelLoader  # noqa: F401
    from thaw_sglang.loader import ThawSGLangFreezeLoader  # noqa: F401
except ImportError as _e:
    _SGLANG_IMPORT_ERROR = _e


def __getattr__(name):
    if name in ("ThawSGLangModelLoader", "ThawSGLangFreezeLoader"):
        raise ImportError(
            f"thaw_sglang.{name} requires the 'sglang' package. "
            f"Install with: pip install thaw-vllm[sglang]\n"
            f"Original import error: {_SGLANG_IMPORT_ERROR}"
        )
    raise AttributeError(f"module 'thaw_sglang' has no attribute {name!r}")


def freeze(model: str, output: str, **kwargs):
    """Freeze model weights to .thaw snapshot using SGLang.

    Loads the model normally from HuggingFace via SGLang, then snapshots
    the GPU tensors to a .thaw file. TP is handled automatically.

    Usage:
        import thaw_sglang
        thaw_sglang.freeze("meta-llama/Meta-Llama-3-8B",
                           "/path/to/weights.thaw")

    Multi-GPU:
        thaw_sglang.freeze("meta-llama/Meta-Llama-3-70B",
                           "/path/to/weights.thaw", tp_size=4)

    Args:
        model: HuggingFace model ID or local path.
        output: Path to write the .thaw snapshot file.
        **kwargs: Additional arguments passed to sglang.Engine().
    """
    import asyncio
    import sglang
    from thaw_sglang.loader import ThawSGLangFreezeLoader

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    kwargs.setdefault("dtype", "float16")

    engine = sglang.Engine(
        model_path=model,
        load_format=ThawSGLangFreezeLoader,
        model_loader_extra_config={"snapshot": output},
        **kwargs,
    )
    engine.shutdown()


def load(model: str, snapshot: str, **kwargs):
    """One-line thaw-powered model loading via SGLang.

    Usage:
        import thaw_sglang
        engine = thaw_sglang.load("meta-llama/Meta-Llama-3-8B",
                                  "/path/to/weights.thaw")

    Multi-GPU:
        engine = thaw_sglang.load("meta-llama/Meta-Llama-3-70B",
                                  "/path/to/weights.thaw", tp_size=4)

    Args:
        model: HuggingFace model ID or local path.
        snapshot: Path to .thaw snapshot file.
        **kwargs: Additional arguments passed to sglang.Engine().
    """
    import asyncio
    import sglang
    from thaw_sglang.loader import ThawSGLangModelLoader

    # Workaround for SGLang v0.5.2 + uvloop >= 0.22.1 event loop bug.
    # Fixed upstream in SGLang v0.5.3+ (PR #11746).
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    kwargs.setdefault("dtype", "float16")

    engine = sglang.Engine(
        model_path=model,
        load_format=ThawSGLangModelLoader,
        model_loader_extra_config={"snapshot": snapshot},
        **kwargs,
    )
    return engine


__all__ = [
    "freeze_model",
    "freeze_model_pipelined",
    "restore_model",
    "restore_model_from_ram",
    "restore_model_pipelined",
    "freeze",
    "load",
    "ThawSGLangModelLoader",
    "ThawSGLangFreezeLoader",
]
