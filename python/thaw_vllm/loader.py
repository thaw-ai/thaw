"""
thaw_vllm.loader — vLLM ModelLoader integration.

Registers load_format="thaw" with vLLM so users can do:

    import thaw_vllm  # registers the loader
    llm = LLM(model="meta-llama/Meta-Llama-3-8B",
              load_format="thaw",
              model_loader_extra_config={"snapshot": "/path/to/weights.thaw"})

The loader initializes the model with empty weights, then restores
from a .thaw snapshot using pipelined DMA.

Multi-GPU / tensor parallel:
    With tensor_parallel_size > 1, each worker process loads from a
    per-rank snapshot file. If snapshot="/path/to/weights.thaw" and
    TP=4, rank 0 loads from weights.thaw, rank 1 from weights.rank1.thaw,
    rank 2 from weights.rank2.thaw, etc. (Rank 0 uses the base path for
    backward compatibility with single-GPU snapshots.)

    Freeze with: thaw freeze --model MODEL --output weights.thaw --tensor-parallel 4
    This produces: weights.thaw, weights.rank1.thaw, weights.rank2.thaw, weights.rank3.thaw
"""

import logging
import os
import torch.nn as nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader


from thaw_common.util import rank_snapshot_path as _rank_snapshot_path
from thaw_common.cloud import resolve_snapshot_path as _resolve_snapshot_path
from thaw_common.telemetry import fallback_warning as _fallback_warning, strict_mode as _strict_mode

logger = logging.getLogger(__name__)


def _get_tp_rank() -> int:
    """Get this worker's tensor parallel rank. Returns 0 for single-GPU."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _get_tp_size() -> int:
    """Get tensor parallel world size. Returns 1 for single-GPU."""
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


class ThawModelLoader(BaseModelLoader):
    """Load model weights from a .thaw snapshot file."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self.snapshot_path = extra.get("snapshot")
        if not self.snapshot_path:
            raise ValueError(
                'load_format="thaw" requires '
                'model_loader_extra_config={"snapshot": "/path/to/weights.thaw"}'
            )

    def download_model(self, model_config: ModelConfig) -> None:
        # Weights come from the snapshot, not HuggingFace.
        # We still need the model config/tokenizer to be downloadable,
        # but vLLM handles that separately.
        pass

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        from thaw_vllm.snapshot import (
            restore_model_from_ram,
            restore_model_pipelined,
            restore_model,
        )

        tp_rank = _get_tp_rank()
        tp_size = _get_tp_size()

        # For TP > 1, each worker loads from its per-rank snapshot file.
        # Per-rank URI is computed first (string manipulation), then
        # resolved — so s3://bucket/weights.thaw with TP=2 naturally becomes
        # s3://bucket/weights.rank1.thaw for rank 1, each downloaded per-worker.
        snapshot_path = _rank_snapshot_path(self.snapshot_path, tp_rank)
        snapshot_path = _resolve_snapshot_path(snapshot_path)

        if tp_size > 1 and not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"[thaw] Per-rank snapshot not found: {snapshot_path}\n"
                f"  With tensor_parallel_size={tp_size}, thaw expects per-rank files.\n"
                f"  Freeze with: thaw freeze --model MODEL --output {self.snapshot_path} "
                f"--tensor-parallel {tp_size}"
            )

        # Try RAM path first (pre-stage file into memory, then DMA — avoids
        # slow pread-to-pinned-memory kernel path, ~6x faster on most systems).
        # Fall back to file-based pipelined, then pure Python. Every fallback
        # is logged; THAW_STRICT=1 re-raises instead of degrading silently.
        try:
            stats = restore_model_from_ram(model, snapshot_path)
        except Exception as e_ram:
            _fallback_warning("ThawModelLoader.restore_model_from_ram", e_ram,
                              dst="restore_model_pipelined")
            if _strict_mode():
                raise
            try:
                stats = restore_model_pipelined(model, snapshot_path)
            except Exception as e_pipe:
                _fallback_warning("ThawModelLoader.restore_model_pipelined", e_pipe,
                                  dst="restore_model (pure python)")
                if _strict_mode():
                    raise
                stats = restore_model(model, snapshot_path)

        size_gb = stats['total_bytes'] / 1e9
        rank_info = f" (rank {tp_rank}/{tp_size})" if tp_size > 1 else ""
        logger.info(
            "Restored %d regions, %.2f GB in %.1fs (%.2f GB/s)%s",
            stats['num_regions'], size_gb, stats['elapsed_s'],
            stats['throughput_gb_s'], rank_info,
        )


_REGISTERED = False


def register() -> None:
    # Called by vLLM's plugin loader when thaw is installed as a
    # vllm.general_plugins entrypoint, and by thaw_vllm/__init__.py
    # for the `import thaw_vllm` path. Idempotent across calls in
    # the same process (vLLM plugin loader may fire it more than once).
    global _REGISTERED
    if _REGISTERED:
        return
    register_model_loader("thaw")(ThawModelLoader)
    _REGISTERED = True
