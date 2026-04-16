"""
thaw_common — engine-agnostic GPU snapshot/restore primitives.

This package contains the core freeze/restore functions that work with
any PyTorch nn.Module, regardless of the inference engine (vLLM, SGLang,
etc.). Engine-specific packages (thaw_vllm, thaw_sglang) build on top.
"""

from thaw_common.format import (
    MAGIC,
    VERSION,
    HEADER_SIZE,
    REGION_ENTRY_SIZE,
    KIND_WEIGHTS,
    KIND_KV_LIVE_BLOCK,
    KIND_METADATA,
)
from thaw_common.snapshot import (
    freeze_model,
    freeze_model_pipelined,
    restore_model,
    restore_model_pipelined,
    restore_model_from_ram,
)
from thaw_common.util import rank_snapshot_path

__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER_SIZE",
    "REGION_ENTRY_SIZE",
    "KIND_WEIGHTS",
    "KIND_KV_LIVE_BLOCK",
    "KIND_METADATA",
    "freeze_model",
    "freeze_model_pipelined",
    "restore_model",
    "restore_model_pipelined",
    "restore_model_from_ram",
    "rank_snapshot_path",
]
