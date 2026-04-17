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
    make_pinned_mmap,
    restore_model_from_pinned_mmap,
)
from thaw_common.util import rank_snapshot_path
from thaw_common.cloud import (
    is_remote,
    resolve_snapshot_path,
    upload_snapshot,
)
from thaw_common.telemetry import (
    fallback_warning,
    strict_mode,
    check_pinned,
)

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
    "make_pinned_mmap",
    "restore_model_from_pinned_mmap",
    "rank_snapshot_path",
    "is_remote",
    "resolve_snapshot_path",
    "upload_snapshot",
]
