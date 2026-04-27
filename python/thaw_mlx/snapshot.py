"""
thaw_mlx.snapshot — MLX freeze/restore for Apple Silicon unified memory.

Uses safetensors as the on-disk format and delegates bulk array
construction to mx.load (one C++ call). The earlier .thaw-envelope
prototype lost to mlx_lm.load by 2× because of per-parameter Python
overhead in mx.array(np.frombuffer(...)). On Apple Silicon the cheapest
restore IS the one mlx-lm itself uses.

Public surface stays stable:
    freeze(model, path) -> stats dict
    restore(model, path) -> stats dict

What thaw_mlx still adds over a raw mx.save_safetensors / mx.load pair:
    1. Single-file blob (vs HF multi-shard) — avoids file-list iteration
    2. Stable parameter ordering metadata (name + shape + dtype index)
    3. Optional .thaw_meta.json sidecar for engine_commit, custom tags,
       cross-engine identification when used alongside CUDA snapshots
    4. A surface to add KV-cache / fork-primitive snapshots later

The CUDA backend (thaw_common.snapshot) keeps the .thaw region-table
format unchanged. MLX uses safetensors because mx.load is the fastest
path on this hardware; the formats serve different cost models.
"""

import json
import os
import time
from typing import Optional

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


_META_SUFFIX = ".thaw_meta.json"
_SAFETENSORS_SUFFIX = ".safetensors"


def _data_path(path: str) -> str:
    """mx.save_safetensors silently appends .safetensors if the suffix is
    missing. Mirror that here so freeze and restore agree on the file."""
    if path.endswith(_SAFETENSORS_SUFFIX):
        return path
    return path + _SAFETENSORS_SUFFIX


def _meta_path(path: str) -> str:
    return _data_path(path) + _META_SUFFIX


def freeze(model, path: str, *, engine_commit: Optional[str] = None) -> dict:
    """Freeze an mlx-lm model's parameters to a safetensors snapshot.

    Walks model.parameters() via tree_flatten, then hands the dict to
    mx.save_safetensors — the same path mlx-lm uses to persist weights.
    A small .thaw_meta.json sidecar records the parameter order and any
    engine_commit tag for provenance.
    """
    flat = tree_flatten(model.parameters())
    if not flat:
        raise ValueError("model has no parameters to freeze")

    arrays = {}
    descriptors = []
    total_bytes = 0
    for name, arr in flat:
        mx.eval(arr)
        arrays[name] = arr
        nbytes = arr.nbytes
        descriptors.append({
            "name": name,
            "dtype": str(arr.dtype).rsplit(".", 1)[-1],
            "shape": list(arr.shape),
            "nbytes": nbytes,
        })
        total_bytes += nbytes

    t0 = time.perf_counter()
    mx.save_safetensors(_data_path(path), arrays)
    elapsed = time.perf_counter() - t0

    meta = {
        "format": "thaw_mlx_safetensors_v1",
        "engine_commit": engine_commit,
        "params": descriptors,
    }
    with open(_meta_path(path), "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "num_regions": len(flat),
        "num_weight_regions": len(flat),
        "total_bytes": total_bytes,
        "elapsed_s": elapsed,
        "throughput_gb_s": (total_bytes / 1e9) / elapsed if elapsed > 0 else 0,
        "backend": "mlx_safetensors",
        "path": path,
    }


def restore(model, path: str) -> dict:
    """Restore an mlx-lm model's parameters from a thaw_mlx snapshot.

    Single mx.load call returns a dict of mx.arrays (mmap-backed
    safetensors view). tree_unflatten + model.update + mx.eval
    finishes the swap.
    """
    data = _data_path(path)
    t0 = time.perf_counter()
    arrays = mx.load(data, format="safetensors")
    rebuilt = list(arrays.items())
    params_tree = tree_unflatten(rebuilt)
    model.update(params_tree)
    mx.eval(model.parameters())
    elapsed = time.perf_counter() - t0

    file_size = os.path.getsize(data)

    return {
        "num_regions": len(rebuilt),
        "num_weight_regions": len(rebuilt),
        "total_bytes": file_size,
        "elapsed_s": elapsed,
        "throughput_gb_s": (file_size / 1e9) / elapsed if elapsed > 0 else 0,
        "backend": "mlx_safetensors",
        "path": path,
    }


def read_meta(path: str) -> Optional[dict]:
    """Read the .thaw_meta.json sidecar if present."""
    mp = _meta_path(path)
    if not os.path.exists(mp):
        return None
    with open(mp) as f:
        return json.load(f)
