"""thaw_vllm.agentfs — laptop-side inspection of fork handles.

The ``ForkHandle`` directory written by ``fork()`` / ``ForkHandle.save()``
is just a manifest (``handle.json``) plus a KV payload and its ``.meta``
sidecar. None of that needs a GPU, torch, or vLLM to read — so ``thaw
inspect`` and ``thaw diff`` run anywhere Python does, including a Mac with
nothing installed.

This module is deliberately stdlib-only. Importing it must never pull in
the GPU stack (that is the whole point of the lazy ``thaw_vllm`` package
init). ``branch`` / ``checkout`` — the verbs that touch the GPU — live in
``fork.py`` and route through the engine; everything here is pure file I/O.

``diff`` measures shared context without any token IDs: vLLM's prefix-cache
block hashes are deterministic over token content and chained, so two
sessions that share a prefix share the identical *set* of leading block
hashes. The size of that intersection is the shared-prefix length, robust
to block-pool ordering.
"""

from __future__ import annotations

import base64
import json
import os
import struct
from datetime import datetime, timezone
from typing import Optional

from thaw_vllm.fork import HANDLE_FILENAME, KV_FILENAME, WEIGHTS_FILENAME

_KV_SIDECAR_MAGIC = b"THAWKV\x00\x00"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def _fmt_size(num_bytes: Optional[int]) -> str:
    if not num_bytes:
        return "0 B"
    n = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def _fmt_time(unix: float) -> str:
    if not unix:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(unix), tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    except (TypeError, ValueError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _resolve_state_dir(path: str) -> str:
    """Accept a handle directory or a path to its handle.json."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    if os.path.basename(path) == HANDLE_FILENAME and os.path.isfile(path):
        return os.path.dirname(path)
    raise FileNotFoundError(
        f"{path!r} is not a thaw handle directory "
        f"(expected a folder containing {HANDLE_FILENAME})."
    )


def _read_kv_meta(state_dir: str) -> Optional[dict]:
    """Read the KV sidecar ``kv.thawkv.meta`` if present. Returns None when
    there's no KV payload (e.g. an empty fork or weights-only handle)."""
    meta_path = os.path.join(state_dir, KV_FILENAME + ".meta")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "rb") as f:
        magic = f.read(8)
        if magic != _KV_SIDECAR_MAGIC:
            return None
        (meta_len,) = struct.unpack("<I", f.read(4))
        return json.loads(f.read(meta_len))


def summarize_handle(path: str) -> dict:
    """Parse a fork handle into a plain dict. GPU-free; stdlib only.

    Raises FileNotFoundError if there's no handle.json at ``path``.
    """
    state_dir = _resolve_state_dir(path)
    manifest_path = os.path.join(state_dir, HANDLE_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"No {HANDLE_FILENAME} in {state_dir}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    meta = _read_kv_meta(state_dir)
    kv_path = os.path.join(state_dir, KV_FILENAME)
    weights_path = os.path.join(state_dir, WEIGHTS_FILENAME)

    block_size = None
    dtype = None
    block_hashes = []
    if meta:
        block_size = meta.get("block_size")
        dtype = meta.get("dtype")
        block_hashes = list(meta.get("block_hashes", []))

    return {
        "name": os.path.basename(state_dir.rstrip("/")) or state_dir,
        "state_dir": state_dir,
        "model_id": manifest.get("model_id", "?"),
        "created_at": manifest.get("created_at", 0.0),
        "prefix_tokens": manifest.get("prefix_tokens", 0),
        "num_kv_blocks": manifest.get("num_kv_blocks", 0),
        "num_layers": manifest.get("num_layers", "?"),
        "block_shape": manifest.get("block_shape", []),
        "block_size": block_size,
        "dtype": dtype,
        "tensor_parallel_size": manifest.get("tensor_parallel_size", 1),
        "vllm_version": manifest.get("vllm_version"),
        "version": manifest.get("version", "?"),
        "has_weights": bool(manifest.get("weights_path")),
        "kv_bytes": os.path.getsize(kv_path) if os.path.isfile(kv_path) else 0,
        "weights_bytes": (
            os.path.getsize(weights_path) if os.path.isfile(weights_path) else 0
        ),
        "block_hashes": block_hashes,
    }


# ---------------------------------------------------------------------------
# render: inspect
# ---------------------------------------------------------------------------


def inspect_handle(path: str) -> str:
    s = summarize_handle(path)
    lines = [f"thaw session  {s['name']}"]

    def row(label, value):
        lines.append(f"  {label:<12} {value}")

    row("model", s["model_id"])
    row("created", _fmt_time(s["created_at"]))

    blocks = s["num_kv_blocks"]
    if blocks:
        bs = f" × {s['block_size']}" if s["block_size"] else ""
        row(
            "prefix",
            f"~{_fmt_int(s['prefix_tokens'])} tokens cached  "
            f"({_fmt_int(blocks)} blocks{bs})",
        )
        kv_bits = [f"{_fmt_int(blocks)} blocks", f"{s['num_layers']} layers"]
        if s["kv_bytes"]:
            kv_bits.append(f"{_fmt_size(s['kv_bytes'])} on disk")
        if s["dtype"]:
            kv_bits.append(str(s["dtype"]))
        row("kv cache", " · ".join(kv_bits))
        if s["block_shape"]:
            row("block shape", str(s["block_shape"]))
    else:
        row("prefix", "empty (no cached KV blocks)")

    if s["has_weights"]:
        row("weights", f"included ({_fmt_size(s['weights_bytes'])})")
    else:
        row("weights", "not included (hydrate into an engine with weights loaded)")

    row("tp", s["tensor_parallel_size"])
    if s["vllm_version"]:
        row("vllm", s["vllm_version"])
    row("handle", f"v{s['version']}")
    row("path", s["state_dir"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# render: diff
# ---------------------------------------------------------------------------


def diff_handles(path_a: str, path_b: str) -> str:
    a = summarize_handle(path_a)
    b = summarize_handle(path_b)

    lines = ["thaw diff", f"  A  {a['name']}", f"  B  {b['name']}", ""]

    def row(label, value):
        lines.append(f"  {label:<12} {value}")

    # model
    if a["model_id"] == b["model_id"]:
        row("model", f"same  ({a['model_id']})")
    else:
        row("model", f"DIFFER  A={a['model_id']}  B={b['model_id']}")

    # shared KV via block-hash set intersection
    ha, hb = set(a["block_hashes"]), set(b["block_hashes"])
    if ha and hb:
        shared = len(ha & hb)
        smaller = min(len(ha), len(hb))
        bsize = a["block_size"] or b["block_size"]
        tok = f"  (~{_fmt_int(shared * bsize)} tokens)" if bsize else ""
        row("shared kv", f"{_fmt_int(shared)}/{_fmt_int(smaller)} blocks identical{tok}")
        row(
            "unique",
            f"A +{_fmt_int(len(ha - hb))} blocks · B +{_fmt_int(len(hb - ha))} blocks",
        )
    else:
        row("shared kv", "unavailable (one or both handles carry no KV metadata)")

    # prefix sizes
    row(
        "prefix",
        f"A ~{_fmt_int(a['prefix_tokens'])} tok · B ~{_fmt_int(b['prefix_tokens'])} tok",
    )

    # block shape
    if a["block_shape"] == b["block_shape"]:
        row("block shape", "match")
    else:
        row("block shape", f"mismatch  A={a['block_shape']}  B={b['block_shape']}")

    # weights / created
    row(
        "weights",
        f"A {'included' if a['has_weights'] else 'no'} · "
        f"B {'included' if b['has_weights'] else 'no'}",
    )
    row("created", f"A {_fmt_time(a['created_at'])} · B {_fmt_time(b['created_at'])}")
    return "\n".join(lines)
