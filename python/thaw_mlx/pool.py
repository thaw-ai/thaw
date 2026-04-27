"""
thaw_mlx.pool — pre-warmed shell pool for sub-second MLX hot-swap.

The 540ms residue in mlx_lm.load on small models is architecture init +
tokenizer load, not weight I/O. Once the shell exists, thaw_mlx.restore
swaps weights at mx.load native speed (~37 GB/s on M5 Pro).

A pool keeps N shells resident keyed by their architecture template
(the HF model id used to instantiate them). Snapshots that share a
template share a shell — load_snapshot() becomes a weight DMA into the
existing shell, not a full mlx_lm.load.

Usage:
    pool = MLXPool()
    pool.warm("mlx-community/Llama-3.2-1B-Instruct-4bit")  # one-time
    model, tok = pool.load_snapshot(
        "/snapshots/finetune-v1.thaw",
        arch_template="mlx-community/Llama-3.2-1B-Instruct-4bit",
    )
    # subsequent load_snapshot calls reuse the shell — 20ms instead of 540ms
"""

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ShellEntry:
    model: object
    tokenizer: object
    arch_template: str
    current_snapshot: Optional[str]
    warmed_at: float


class MLXPool:
    def __init__(self):
        self._shells: dict[str, ShellEntry] = {}

    def warm(self, arch_template: str) -> ShellEntry:
        """Build a shell for the given architecture if not already cached.

        First call pays the full mlx_lm.load cost (architecture init +
        tokenizer + initial weights). Subsequent load_snapshot calls
        reuse the shell.
        """
        existing = self._shells.get(arch_template)
        if existing is not None:
            return existing

        from mlx_lm import load as mlx_load
        import mlx.core as mx

        t0 = time.perf_counter()
        model, tokenizer = mlx_load(arch_template)
        mx.eval(model.parameters())
        warm_s = time.perf_counter() - t0

        entry = ShellEntry(
            model=model,
            tokenizer=tokenizer,
            arch_template=arch_template,
            current_snapshot=arch_template,
            warmed_at=warm_s,
        )
        self._shells[arch_template] = entry
        return entry

    def load_snapshot(
        self,
        snapshot_path: str,
        arch_template: str,
    ) -> tuple[object, object]:
        """Restore weights from a thaw snapshot into the shell for arch_template.

        Returns (model, tokenizer). The model object is the same Python
        object across calls — its parameters are swapped in place.
        """
        from thaw_mlx.snapshot import restore

        entry = self.warm(arch_template)
        if entry.current_snapshot != snapshot_path:
            restore(entry.model, snapshot_path)
            entry.current_snapshot = snapshot_path
        return entry.model, entry.tokenizer

    def evict(self, arch_template: str) -> bool:
        return self._shells.pop(arch_template, None) is not None

    def stats(self) -> dict:
        return {
            "num_shells": len(self._shells),
            "shells": [
                {
                    "arch_template": e.arch_template,
                    "current_snapshot": e.current_snapshot,
                    "warmed_at_s": e.warmed_at,
                }
                for e in self._shells.values()
            ],
        }
