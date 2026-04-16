"""
thaw_common.util — shared utilities for thaw packages.
"""

import os


def rank_snapshot_path(base_path: str, rank: int) -> str:
    """Get per-rank snapshot path. Rank 0 uses base path (backward compat).

    Examples:
        rank_snapshot_path("weights.thaw", 0) -> "weights.thaw"
        rank_snapshot_path("weights.thaw", 1) -> "weights.rank1.thaw"
        rank_snapshot_path("weights.thaw", 3) -> "weights.rank3.thaw"
    """
    if rank == 0:
        return base_path
    stem, ext = os.path.splitext(base_path)
    return f"{stem}.rank{rank}{ext}"
