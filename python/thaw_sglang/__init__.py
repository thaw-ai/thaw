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

# SGLang-specific loader (import conditionally — SGLang may not be installed)
try:
    from thaw_sglang.loader import ThawSGLangModelLoader  # noqa: F401
except ImportError:
    ThawSGLangModelLoader = None


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
    import sglang
    from thaw_sglang.loader import ThawSGLangModelLoader

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
    "load",
    "ThawSGLangModelLoader",
]
