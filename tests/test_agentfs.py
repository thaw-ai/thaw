"""Unit tests for thaw_vllm.agentfs — the GPU-free handle inspect/diff path.

These run anywhere Python does: no torch, no vLLM, no GPU. They fabricate
fork-handle directories matching what fork()/ForkHandle.save() write
(handle.json + kv.thawkv.meta sidecar) and assert on the rendered output.
"""

import json
import os
import shutil
import struct
import tempfile
import unittest

from thaw_vllm.agentfs import (
    summarize_handle,
    inspect_handle,
    diff_handles,
    log_handles,
)
from thaw_vllm.fork import ForkHandle


def _write_meta(d, hashes, block_size=16):
    meta = {
        "num_blocks": len(hashes),
        "block_ids": list(range(len(hashes))),
        "block_hashes": hashes,
        "num_layers": 32,
        "block_shape": [2, 16, 8, 128],
        "dtype": "torch.float16",
        "block_size": block_size,
    }
    b = json.dumps(meta).encode()
    with open(os.path.join(d, "kv.thawkv.meta"), "wb") as f:
        f.write(b"THAWKV\x00\x00")
        f.write(struct.pack("<I", len(b)))
        f.write(b)


def _make(
    root,
    name,
    hashes,
    weights=False,
    created=1776200000.0,
    handle_id="",
    parent_id=None,
    label=None,
):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    nb = len(hashes)
    with open(os.path.join(d, "handle.json"), "w") as f:
        json.dump(
            {
                "model_id": "meta-llama/Llama-3.1-8B-Instruct",
                "state_dir": d,
                "kv_path": os.path.join(d, "kv.thawkv"),
                "weights_path": os.path.join(d, "weights.thaw") if weights else None,
                "prefix_tokens": nb * 16,
                "block_shape": [2, 16, 8, 128],
                "num_layers": 32,
                "max_block_id": 104,
                "num_kv_blocks": nb,
                "handle_id": handle_id,
                "parent_id": parent_id,
                "label": label,
                "tensor_parallel_size": 1,
                "vllm_version": "0.19.1",
                "created_at": created,
                "version": 1,
            },
            f,
        )
    with open(os.path.join(d, "kv.thawkv"), "wb") as f:
        f.write(b"\0" * (nb * 2048))
    if weights:
        with open(os.path.join(d, "weights.thaw"), "wb") as f:
            f.write(b"\0" * 4096)
    _write_meta(d, hashes)
    return d


class AgentfsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agentfs_test_")
        self.a = _make(
            self.tmp,
            "feat-a",
            ["h0", "h1", "h2", "h3", "h4"],
            weights=True,
            handle_id="a0a0a0a0a0a0",
        )
        self.b = _make(self.tmp, "rev-1", ["h0", "h1", "h2", "r3", "r4", "r5", "r6"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summarize_fields(self):
        s = summarize_handle(self.a)
        self.assertEqual(s["model_id"], "meta-llama/Llama-3.1-8B-Instruct")
        self.assertEqual(s["num_kv_blocks"], 5)
        self.assertEqual(s["block_size"], 16)
        self.assertEqual(s["prefix_tokens"], 80)
        self.assertTrue(s["has_weights"])
        self.assertEqual(len(s["block_hashes"]), 5)

    def test_inspect_renders(self):
        out = inspect_handle(self.a)
        self.assertIn("feat-a", out)
        self.assertIn("Llama-3.1-8B", out)
        self.assertIn("5 blocks", out)
        self.assertIn("included", out)

    def test_accepts_handle_json_path(self):
        # Passing handle.json directly resolves to its directory.
        out = inspect_handle(os.path.join(self.a, "handle.json"))
        self.assertIn("feat-a", out)

    def test_diff_shared_blocks(self):
        out = diff_handles(self.a, self.b)
        self.assertIn("3/5 blocks identical", out)  # 3 shared, min(5,7)=5
        self.assertIn("A +2 blocks", out)  # 5 - 3
        self.assertIn("B +4 blocks", out)  # 7 - 3
        self.assertIn("same", out)  # model match
        self.assertIn("~48 tokens", out)  # 3 shared blocks * block_size 16

    def test_missing_handle_raises(self):
        with self.assertRaises(FileNotFoundError):
            summarize_handle(os.path.join(self.tmp, "does-not-exist"))

    def test_weights_only_handle_no_kv(self):
        d = os.path.join(self.tmp, "weights-only")
        os.makedirs(d)
        with open(os.path.join(d, "handle.json"), "w") as f:
            json.dump(
                {
                    "model_id": "m",
                    "num_kv_blocks": 0,
                    "prefix_tokens": 0,
                    "block_shape": [],
                    "num_layers": 0,
                    "max_block_id": -1,
                    "tensor_parallel_size": 1,
                    "version": 1,
                    "created_at": 0.0,
                    "weights_path": None,
                },
                f,
            )
        out = inspect_handle(d)
        self.assertIn("empty", out)

    def test_branch_sets_lineage(self):
        # ForkHandle.branch() is pure file I/O — no GPU.
        parent = ForkHandle.load(self.a)
        self.assertEqual(parent.handle_id, "a0a0a0a0a0a0")
        child_dir = os.path.join(self.tmp, "child-1")
        child = parent.branch(child_dir, label="reviewer-x")
        self.assertNotEqual(child.handle_id, parent.handle_id)
        self.assertTrue(child.handle_id)  # a fresh id was stamped
        self.assertEqual(child.parent_id, parent.handle_id)
        self.assertEqual(child.label, "reviewer-x")
        # payload copied
        self.assertTrue(os.path.isfile(os.path.join(child_dir, "kv.thawkv")))
        self.assertTrue(os.path.isfile(os.path.join(child_dir, "kv.thawkv.meta")))
        # and it round-trips through the manifest
        reloaded = summarize_handle(child_dir)
        self.assertEqual(reloaded["parent_id"], parent.handle_id)
        self.assertEqual(reloaded["label"], "reviewer-x")

    def test_log_renders_lineage_tree(self):
        repo = os.path.join(self.tmp, "repo")
        _make(repo, "trunk", ["h0", "h1"], handle_id="T", created=1.0, label="trunk")
        _make(
            repo,
            "branch-a",
            ["h0", "h1", "a2"],
            handle_id="A",
            parent_id="T",
            created=2.0,
            label="branch-a",
        )
        out = log_handles(repo)
        lines = out.splitlines()
        trunk_line = next(line for line in lines if "trunk" in line)
        branch_line = next(line for line in lines if "branch-a" in line)
        # parent rendered before child
        self.assertLess(lines.index(trunk_line), lines.index(branch_line))
        # trunk is a root: node glyph at column 0, no connector
        self.assertEqual(trunk_line.index("●"), 0)
        # branch-a hangs under trunk: a connector is drawn and the glyph is indented
        self.assertTrue(("├" in branch_line) or ("└" in branch_line))
        self.assertGreater(branch_line.index("●"), trunk_line.index("●"))


if __name__ == "__main__":
    unittest.main()
