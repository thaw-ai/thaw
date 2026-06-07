"""Unit tests for thaw_vllm.rewind — the GPU-free rollout inspect/diff/pivot path.

Run anywhere Python does: no torch, no vLLM, no GPU. Fabricate rollout.json
files matching what capture_rollouts()/rewind.write_rollout() write and assert
on the rendered output and scoring.
"""

import math
import os
import shutil
import tempfile
import unittest

from thaw_vllm import rewind


def _tok(token_id, text, logprob, alts=None):
    """One token entry with an optional top-k (list of (id, text, logprob))."""
    topk = [{"token_id": token_id, "text": text, "logprob": logprob}]
    for tid, t, lp in alts or []:
        topk.append({"token_id": tid, "text": t, "logprob": lp})
    topk.sort(key=lambda x: x["logprob"], reverse=True)
    return {"token_id": token_id, "text": text, "logprob": logprob, "topk": topk}


def _write(root, label, prompt, tokens, parent_id=None):
    rec = rewind.build_rollout(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        prompt_token_ids=prompt,
        tokens=tokens,
        parent_id=parent_id,
        label=label,
        sampling={"temperature": 0.8, "top_p": 0.95, "seed": 42},
        created_at=1781234567.0,
    )
    return rewind.write_rollout(rec, os.path.join(root, label))


# A and B share trunk [1,2,3] and the first two generated tokens, then split:
#   A: " race" (-0.3)   B: " null" (-0.5)
def _branch_a(root, label="branch-a"):
    return _write(root, label, [1, 2, 3], [
        _tok(10, " the", -0.1, [(11, " a", -1.0)]),
        _tok(20, " root", -0.2, [(21, " cause", -1.5)]),
        _tok(30, " race", -0.3, [(31, " null", -1.8)]),
    ])


def _branch_b(root, label="branch-b"):
    return _write(root, label, [1, 2, 3], [
        _tok(10, " the", -0.15, [(11, " a", -1.1)]),
        _tok(20, " root", -0.25, [(21, " cause", -1.6)]),
        _tok(31, " null", -0.5, [(30, " race", -2.1)]),
    ])


class RewindTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rewind_test_")
        self.a = _branch_a(self.tmp)
        self.b = _branch_b(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_summarize_scores(self):
        s = rewind.summarize_rollout(self.a)
        self.assertEqual(s["n_tokens"], 3)
        self.assertEqual(s["n_scored"], 3)
        self.assertAlmostEqual(s["seq_logprob"], -0.6, places=6)
        self.assertAlmostEqual(s["mean_logprob"], -0.2, places=6)
        self.assertAlmostEqual(s["perplexity"], math.exp(0.2), places=6)
        self.assertEqual(s["model_id"], "meta-llama/Llama-3.1-8B-Instruct")

    def test_inspect_renders(self):
        out = rewind.inspect_rollout(self.a)
        self.assertIn("branch-a", out)
        self.assertIn("Llama-3.1-8B", out)
        self.assertIn("3 tokens", out)
        self.assertIn("seq logprob", out)
        self.assertIn("-0.60", out)

    def test_diff_finds_pivot(self):
        out = rewind.diff_rollouts(self.a, self.b)
        # pivot at generated token 2 (third token, 0-indexed)
        self.assertIn("generated token 2", out)
        self.assertIn("race", out)
        self.assertIn("null", out)
        # A scored higher overall (-0.6 vs -0.9)
        self.assertIn("A higher by 0.30", out)
        # counterfactual: each branch ranked the other's choice in its top-k
        self.assertIn("#2", out)
        # trunks match
        self.assertIn("same", out)

    def test_diff_identical(self):
        c = _branch_a(self.tmp, label="branch-a-copy")
        out = rewind.diff_rollouts(self.a, c)
        self.assertIn("identical token sequence", out)

    def test_pivot_across_n(self):
        # add two more branches; C agrees with A at the pivot, D goes elsewhere
        _write(self.tmp, "branch-c", [1, 2, 3], [
            _tok(10, " the", -0.10),
            _tok(20, " root", -0.15),
            _tok(30, " race", -0.20),  # same token as A, but higher-scoring (-0.45)
        ])
        _write(self.tmp, "branch-d", [1, 2, 3], [
            _tok(10, " the", -0.13),
            _tok(20, " root", -0.23),
            _tok(32, " lock", -0.9),
        ])
        out = rewind.pivot_rollouts(self.tmp)
        self.assertIn("4 rollouts", out)
        self.assertIn("first divergence at generated token 2", out)
        self.assertIn("race", out)
        self.assertIn("null", out)
        self.assertIn("lock", out)
        # best branch is branch-c (seq logprob -0.62, highest)
        self.assertIn("best branch: branch-c", out)

    def test_pivot_needs_two(self):
        solo = tempfile.mkdtemp(dir=self.tmp)
        _write(solo, "only", [1, 2], [_tok(10, " x", -0.1)])
        out = rewind.pivot_rollouts(solo)
        self.assertIn("at least 2 rollouts", out)

    def test_extract_token_logprobs_ducktyped(self):
        class FakeLP:
            def __init__(self, logprob, decoded_token):
                self.logprob = logprob
                self.decoded_token = decoded_token

        class FakeComp:
            token_ids = [10, 20]
            text = " the root"
            logprobs = [
                {10: FakeLP(-0.1, " the"), 11: FakeLP(-1.0, " a")},
                {20: FakeLP(-0.2, " root")},
            ]

        tokens = rewind.extract_token_logprobs(FakeComp(), max_topk=3)
        self.assertEqual(tokens[0]["token_id"], 10)
        self.assertAlmostEqual(tokens[0]["logprob"], -0.1, places=6)
        self.assertEqual(tokens[0]["text"], " the")
        # top-k sorted descending by logprob; chosen token ranks first
        self.assertEqual(tokens[0]["topk"][0]["token_id"], 10)
        self.assertEqual(len(tokens[0]["topk"]), 2)

    def test_missing_rollout_raises(self):
        with self.assertRaises(FileNotFoundError):
            rewind.summarize_rollout(os.path.join(self.tmp, "does-not-exist"))

    def test_no_logprobs_graceful(self):
        d = _write(self.tmp, "no-lp", [1, 2], [
            {"token_id": 10, "text": " x", "logprob": None, "topk": []},
            {"token_id": 11, "text": " y", "logprob": None, "topk": []},
        ])
        s = rewind.summarize_rollout(d)
        self.assertIsNone(s["seq_logprob"])
        self.assertIn("not captured", rewind.inspect_rollout(d))


if __name__ == "__main__":
    unittest.main()
