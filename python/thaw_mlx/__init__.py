"""
thaw_mlx — MLX freeze/restore for Apple Silicon unified memory.

On Apple Silicon, GPU and CPU share the same physical RAM pool. There is
no PCIe, no D2H/H2D DMA, no pinned-host staging. Freeze is "serialize
mx.array contents to disk"; restore is "mmap + view".

Public API:
    freeze(model, path) — write a .thaw snapshot from an mlx-lm model
    restore(model, path) — load a .thaw snapshot into an mlx-lm model
"""

from thaw_mlx.snapshot import freeze, restore
from thaw_mlx.pool import MLXPool

__all__ = ["freeze", "restore", "MLXPool"]
