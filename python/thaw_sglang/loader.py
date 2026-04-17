"""
thaw_sglang.loader — SGLang ModelLoader integration.

Two loaders:

ThawSGLangModelLoader (restore):
    Restores model weights from a .thaw snapshot at load time.

    engine = sglang.Engine(
        model_path="meta-llama/Meta-Llama-3-8B",
        load_format=ThawSGLangModelLoader,
        model_loader_extra_config={"snapshot": "/path/to/weights.thaw"},
    )

ThawSGLangFreezeLoader (freeze):
    Loads model weights normally from HuggingFace, then freezes them to
    a .thaw snapshot as a side effect. Used by `thaw freeze --engine sglang`.

    engine = sglang.Engine(
        model_path="meta-llama/Meta-Llama-3-8B",
        load_format=ThawSGLangFreezeLoader,
        model_loader_extra_config={"snapshot": "/path/to/output.thaw"},
    )

TP support: SGLang passes the loader class to each worker process. Each
worker creates its own loader instance, gets its own tp_rank, loads/saves
its own rank-specific snapshot. No collective_rpc needed.
"""

import logging
import os
import torch.nn as nn

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.model_loader.loader import BaseModelLoader, get_model_loader

from thaw_common.util import rank_snapshot_path
from thaw_common.cloud import resolve_snapshot_path, upload_snapshot, is_remote
from thaw_common.telemetry import fallback_warning, strict_mode
from thaw_common.snapshot import (
    freeze_model,
    freeze_model_pipelined,
    restore_model_from_ram,
    restore_model_pipelined,
    restore_model,
)

logger = logging.getLogger(__name__)


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


# GPUs where SGLang TP>1 hits a piecewise CUDA graph bug during warmup
# (validated 2026-04-17 on 2×A40). The bug is in SGLang's graph capture,
# not thaw — but users still blame us. Emit a clear warning so the cause
# is obvious when they hit it, and point at the workaround.
_SGLANG_TP_BAD_GPUS = ("A40", "A10", "A30", "A100-PCI")


def _a40_preflight(tp_size: int) -> None:
    if tp_size <= 1:
        return
    try:
        import torch
        if not torch.cuda.is_available():
            return
        name = torch.cuda.get_device_name(0)
    except Exception:
        return
    for bad in _SGLANG_TP_BAD_GPUS:
        if bad in name:
            logger.warning(
                "SGLang TP=%d on %s is known to hit a piecewise CUDA graph "
                "bug during warmup. If capture hangs or crashes, either pass "
                "enforce_eager=True (slower but works) or move to H100/L40S. "
                "This is an SGLang bug, not thaw.",
                tp_size, name,
            )
            return


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

    def load_model(self, *, model_config, **kwargs) -> nn.Module:
        """Initialize model with empty weights, then restore from snapshot.

        SGLang's load_model() must RETURN the model (unlike vLLM's
        load_weights() which receives an already-initialized model).
        """
        from sglang.srt.model_loader.loader import _initialize_model

        # Step 1: Create model skeleton with uninitialized weights.
        # _initialize_model creates parameters on meta device (no GPU memory).
        model = _initialize_model(
            model_config=model_config,
            load_config=self.load_config,
        )

        # Step 1b: Materialize parameters on CUDA so restore can write to them.
        # Use model_config.dtype (e.g. float16 from Engine(dtype="float16"))
        # rather than the meta tensor's dtype, which defaults to float32.
        import torch
        target_dtype = getattr(model_config, 'dtype', None) or torch.float16
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                param.data = torch.empty(
                    param.shape, dtype=target_dtype, device='cuda'
                )

        # Step 1c: Move buffers (e.g. rotary embedding cos/sin cache) to CUDA.
        # Use .to() to preserve computed values (not torch.empty which is garbage).
        for name, buf in model.named_buffers():
            if buf.device.type not in ('cuda',):
                model_parts = name.split('.')
                parent = model
                for part in model_parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, model_parts[-1], buf.to(device='cuda'))

        # Step 2: Determine per-rank snapshot path for TP.
        # Remote URIs (s3://...) are resolved after per-rank name computation,
        # so s3://bucket/weights.thaw + TP=2 → s3://bucket/weights.rank1.thaw
        # for rank 1, downloaded to local cache on each worker.
        tp_rank = _get_tp_rank()
        tp_size = _get_tp_size()
        _a40_preflight(tp_size)
        snapshot_path = rank_snapshot_path(self.snapshot_path, tp_rank)
        snapshot_path = resolve_snapshot_path(snapshot_path)

        if tp_size > 1 and not os.path.exists(snapshot_path):
            raise FileNotFoundError(
                f"[thaw] Per-rank snapshot not found: {snapshot_path}\n"
                f"  With tp_size={tp_size}, thaw expects per-rank files.\n"
                f"  Freeze with: thaw freeze --model MODEL --output "
                f"{self.snapshot_path} --tensor-parallel {tp_size}"
            )

        # Step 3: Restore weights from snapshot via DMA.
        # Try RAM path first (fastest), fall back through strategies.
        # All fallbacks are logged; THAW_STRICT=1 re-raises instead.
        try:
            stats = restore_model_from_ram(model, snapshot_path)
        except Exception as e_ram:
            fallback_warning("ThawSGLangModelLoader.restore_model_from_ram", e_ram,
                             dst="restore_model_pipelined")
            if strict_mode():
                raise
            try:
                stats = restore_model_pipelined(model, snapshot_path)
            except Exception as e_pipe:
                fallback_warning("ThawSGLangModelLoader.restore_model_pipelined", e_pipe,
                                 dst="restore_model (pure python)")
                if strict_mode():
                    raise
                stats = restore_model(model, snapshot_path)

        size_gb = stats['total_bytes'] / 1e9
        rank_info = f" (rank {tp_rank}/{tp_size})" if tp_size > 1 else ""
        logger.info(
            "Restored %d regions, %.2f GB in %.1fs (%.2f GB/s)%s",
            stats['num_regions'], size_gb, stats['elapsed_s'],
            stats['throughput_gb_s'], rank_info,
        )

        return model


class ThawSGLangFreezeLoader(BaseModelLoader):
    """Load model weights normally from HuggingFace, then freeze to .thaw.

    Used by `thaw freeze --engine sglang`. Piggybacks on SGLang's default
    model loading — weights are loaded from safetensors as usual, then
    snapshotted to a .thaw file as a side effect.

    TP works automatically: each worker runs this loader independently,
    each saves its own per-rank file.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config or {}
        self.snapshot_path = extra.get("snapshot")
        if not self.snapshot_path:
            raise ValueError(
                'ThawSGLangFreezeLoader requires '
                'model_loader_extra_config={"snapshot": "/path/to/output.thaw"}'
            )

    def download_model(self, model_config) -> None:
        default_loader = get_model_loader(LoadConfig(load_format="auto"))
        default_loader.download_model(model_config)

    def load_model(self, *, model_config, **kwargs) -> nn.Module:
        """Load weights from HuggingFace via default loader, then freeze."""
        # Step 1: Load model normally using SGLang's default loader.
        default_loader = get_model_loader(LoadConfig(load_format="auto"))
        model = default_loader.load_model(
            model_config=model_config,
            **kwargs,
        )

        # Step 2: Freeze weights to .thaw snapshot.
        # For remote URIs (s3://...) we freeze to a local temp file first,
        # then upload. Freeze is the slow path (~1.6 GB/s), so layering the
        # upload on top is fine for MVP.
        tp_rank = _get_tp_rank()
        tp_size = _get_tp_size()
        target_uri = rank_snapshot_path(self.snapshot_path, tp_rank)

        if is_remote(target_uri):
            import tempfile
            local_fd, local_path = tempfile.mkstemp(suffix=".thaw")
            os.close(local_fd)
        else:
            local_path = target_uri

        try:
            stats = freeze_model_pipelined(model, local_path)
        except Exception as e:
            fallback_warning("ThawSGLangFreezeLoader.freeze_model_pipelined", e,
                             dst="freeze_model (pure python)")
            if strict_mode():
                raise
            stats = freeze_model(model, local_path)

        if is_remote(target_uri):
            logger.info("Uploading %s -> %s", local_path, target_uri)
            upload_snapshot(local_path, target_uri)
            os.unlink(local_path)

        size_gb = stats['total_bytes'] / 1e9
        rank_info = f" (rank {tp_rank}/{tp_size})" if tp_size > 1 else ""
        logger.info(
            "Frozen %d regions, %.2f GB in %.1fs (%.2f GB/s)%s",
            stats['num_regions'], size_gb, stats['elapsed_s'],
            stats['throughput_gb_s'], rank_info,
        )

        return model
