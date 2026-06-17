#!/usr/bin/env python3
"""Generate a fake fork-handle tree and render it with the GPU-free verbs.

This writes handle directories that match exactly what ``fork()`` /
``ForkHandle.save()`` produce (handle.json + kv.thawkv.meta sidecar), then
runs ``thaw log`` and ``thaw diff`` over them. No GPU, no torch, no vLLM —
the whole point is that inspecting and diffing live agent sessions is pure
file I/O you can do on a laptop.

    python3 demos/agentfs_demo_tree.py

The data here is fabricated for the demo; on a real run the handles come
from forking an actual vLLM session. The rendering path is identical.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from thaw_vllm.agentfs import diff_handles, log_handles  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BLOCK_SIZE = 16
_KV_MAGIC = b"THAWKV\x00\x00"


def _write_handle(root, dirname, *, block_hashes, handle_id, parent_id, label,
                  age_seconds, preview, has_weights=False):
    state_dir = os.path.join(root, dirname)
    os.makedirs(state_dir, exist_ok=True)
    n_blocks = len(block_hashes)

    manifest = {
        "model_id": MODEL,
        "state_dir": state_dir,
        "kv_path": os.path.join(state_dir, "kv.thawkv"),
        "weights_path": os.path.join(state_dir, "weights.thaw") if has_weights else None,
        "prefix_tokens": n_blocks * BLOCK_SIZE,
        "prefix_preview": preview,
        "prefix_token_ids": block_hashes,
        "block_shape": [2, BLOCK_SIZE, 8, 128],
        "num_layers": 32,
        "num_kv_blocks": n_blocks,
        "handle_id": handle_id,
        "parent_id": parent_id,
        "label": label,
        "tensor_parallel_size": 1,
        "vllm_version": "0.22.1",
        "created_at": time.time() - age_seconds,
        "version": 1,
    }
    with open(os.path.join(state_dir, "handle.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # KV payload + sidecar (block hashes are what diff intersects on)
    with open(os.path.join(state_dir, "kv.thawkv"), "wb") as f:
        f.write(b"\0" * (n_blocks * 4096))
    meta = {
        "num_blocks": n_blocks,
        "block_ids": list(range(n_blocks)),
        "block_hashes": block_hashes,
        "num_layers": 32,
        "block_shape": [2, BLOCK_SIZE, 8, 128],
        "dtype": "torch.bfloat16",
        "block_size": BLOCK_SIZE,
    }
    blob = json.dumps(meta).encode()
    with open(os.path.join(state_dir, "kv.thawkv.meta"), "wb") as f:
        f.write(_KV_MAGIC + struct.pack("<I", len(blob)) + blob)

    if has_weights:
        with open(os.path.join(state_dir, "weights.thaw"), "wb") as f:
            f.write(b"\0" * 4096)
    return state_dir


def build_tree(root: str) -> dict:
    """A code-review fan-out: one main session, three reviewers branched off
    the shared prompt, and one reviewer re-forked for a second pass."""
    trunk = [f"ctx{i}" for i in range(40)]  # 40 shared context blocks

    _write_handle(
        root, "session-main", block_hashes=trunk, handle_id="f0aa11",
        parent_id=None, label="main", age_seconds=600, has_weights=True,
        preview="You are a senior code reviewer. Review the auth refactor diff.")

    paths = {}
    paths["security"] = _write_handle(
        root, "rev-security", block_hashes=trunk + ["sec1", "sec2", "sec3"],
        handle_id="a1b2c3", parent_id="f0aa11", label="reviewer/security",
        age_seconds=420,
        preview="You are a senior code reviewer... focus on auth, secrets, injection.")

    paths["style"] = _write_handle(
        root, "rev-style", block_hashes=trunk + ["sty1", "sty2"],
        handle_id="b2c3d4", parent_id="f0aa11", label="reviewer/style",
        age_seconds=410,
        preview="You are a senior code reviewer... focus on naming and readability.")

    paths["perf"] = _write_handle(
        root, "rev-perf", block_hashes=trunk + ["prf1", "prf2", "prf3", "prf4", "prf5"],
        handle_id="c3d4e5", parent_id="f0aa11", label="reviewer/perf",
        age_seconds=400,
        preview="You are a senior code reviewer... focus on allocations and hot paths.")

    _write_handle(
        root, "rev-security-v2",
        block_hashes=trunk + ["sec1", "sec2", "sec3", "sec4", "sec5", "sec6"],
        handle_id="d4e5f6", parent_id="a1b2c3", label="reviewer/security@retry",
        age_seconds=120,
        preview="You are a senior code reviewer... re-run with the CVE list attached.")
    return paths


def main() -> None:
    root = tempfile.mkdtemp(prefix="thaw_agentfs_demo_")
    paths = build_tree(root)

    print("$ thaw log ./reviews\n")
    print(log_handles(root))
    print("\n" + "=" * 78 + "\n")
    print("$ thaw diff rev-security rev-perf\n")
    print(diff_handles(paths["security"], paths["perf"]))
    print(f"\n(fixtures written under {root})")


if __name__ == "__main__":
    main()
