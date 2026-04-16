"""
thaw_sglang.loader — SGLang ModelLoader integration.

Enables loading models from .thaw snapshots via SGLang's class-passthrough
mechanism:

    import thaw_sglang
    engine = sglang.Engine(
        model_path="meta-llama/Meta-Llama-3-8B",
        load_format=thaw_sglang.ThawSGLangModelLoader,
        model_loader_extra_config={"snapshot": "/path/to/weights.thaw"},
    )

TP support: SGLang passes the loader class to each worker process. Each
worker creates its own loader instance, gets its own tp_rank, loads its
own rank-specific snapshot. No collective_rpc needed.

Multi-GPU / tensor parallel:
    With tp_size > 1, each worker loads from a per-rank snapshot file.
    If snapshot="/path/to/weights.thaw" and tp_size=4:
      rank 0 loads weights.thaw
      rank 1 loads weights.rank1.thaw
      rank 2 loads weights.rank2.thaw
      rank 3 loads weights.rank3.thaw
"""

import os
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import BaseModelLoader

from thaw_common.util import rank_snapshot_path
from thaw_common.snapshot import (
    restore_model_from_ram,
    restore_model_pipelined,
    restore_model,
)


def _get_tp_rank() -> int:
    """Get this worker's tensor parallel rank. Returns 0 for single-GPU."""
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


def _get_tp_size() -> int:
    """Get tensor parallel world size. Returns 1 for single-GPU."""
    try:
        from sglang.srt.distributed import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size()
    except Exception:
        return 1


class ThawSGLangModelLoader(BaseModelLoader):
    """Load model weights from a .thaw snapshot file.

    SGLang integration via class-passthrough: pass this class as
    load_format to sglang.Engine() and it will be instantiated directly
    by SGLang's get_model_loader().
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self.snapshot_path = extra.get("snapshot")
        if not self.snapshot_path:
            raise ValueError(
                'ThawSGLangModelLoader requires '
                'model_loader_extra_config={"snapshot": "/path/to/weights.thaw"}'
            )

    def download_model(self, model_config) -> None:
        # Weights come from the snapshot, not HuggingFace.
        pass

    def load_model(self, *, model_config, device_config, **kwargs) -> nn.Module:
        """Initialize model with empty weights, then restore from snapshot.

        SGLang's load_model() must RETURN the model (unlike vLLM's
        load_weights() which receives an already-initialized model).
        """
        from sglang.srt.model_loader.loader import _initialize_model

        # Step 1: Create model skeleton with uninitialized weights.
        model = _initialize_model(
            model_config=model_config,
            load_config=self.load_config,
            device_config=device_config,
            **kwargs,
        )

        # Step 2: Determine per-rank snapshot path for TP.
        tp_rank = _get_tp_rank()
        tp_size = _get_tp_size()
        snapshot_path = rank_snapshot_path(self.snapshot_path, tp_rank)

        if tp_size > 1 and not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"[thaw] Per-rank snapshot not found: {snapshot_path}\n"
                f"  With tp_size={tp_size}, thaw expects per-rank files.\n"
                f"  Freeze with: thaw freeze --model MODEL --output "
                f"{self.snapshot_path} --tensor-parallel {tp_size}"
            )

        # Step 3: Restore weights from snapshot via DMA.
        # Try RAM path first (fastest), fall back through strategies.
        try:
            stats = restore_model_from_ram(model, snapshot_path)
        except Exception:
            try:
                stats = restore_model_pipelined(model, snapshot_path)
            except Exception:
                stats = restore_model(model, snapshot_path)

        size_gb = stats['total_bytes'] / 1e9
        rank_info = f" (rank {tp_rank}/{tp_size})" if tp_size > 1 else ""
        print(f"[thaw] Restored {stats['num_regions']} regions, "
              f"{size_gb:.2f} GB in {stats['elapsed_s']:.1f}s "
              f"({stats['throughput_gb_s']:.2f} GB/s){rank_info}")

        return model
