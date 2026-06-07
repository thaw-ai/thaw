# `thaw rewind` — GPU validation

End-to-end validation of the `rewind` capture path on real hardware: does
`capture_rollouts` turn a live, sampled (temperature > 0) generation into
`rollout.json` files with genuine per-token logprobs, and do `thaw rewind
inspect / diff / pivot` produce correct output on that real data?

## Real-model run — Qwen2.5-7B-Instruct, best-of-8

The headline validation: a realistic best-of-N RL step. One reasoning
problem, 8 sampled chains (temperature 0.6) from the shared trunk, each
captured with per-token logprobs, then diffed on a laptop.

- **GPU:** 1× A100-SXM4-80GB (RunPod) · vLLM 0.22.1 · ~$0.30
- **Model:** `Qwen/Qwen2.5-7B-Instruct`, bf16
- **Workload:** best-of-8, temperature 0.6, 400 max tokens

```
thaw rewind pivot  examples/rewind-bestof8
  8 rollouts · trunk 93 tokens · Qwen/Qwen2.5-7B-Instruct
  first divergence at generated token 0:
    "Let"  →  rollout-2, rollout-3, rollout-4, rollout-5   (seq logprob -38.62, -59.62, -36.75, -28.31)
    "To"   →  rollout-0, rollout-1, rollout-6, rollout-7   (seq logprob -38.32, -31.47, -38.63, -32.40)
  best branch: rollout-5  (seq logprob -28.31, perplexity 1.08)

thaw rewind diff  rollout-2 rollout-3
  seq logprob   A -38.62 · B -59.62   (A higher by 21.00)
  pivot         generated token 37  (after 37 identical tokens)
    shared   … Step 1: Determine the
  - A  " head"      logprob -0.38   (B ranked this #1 @ -0.38)
  + B  " distance"  logprob -1.51   (A ranked this #2 @ -1.51)
  decisiveness  A preferred its token by 1.13 logprob — a confident split
```

Two chains shared **37 tokens** of identical reasoning, forked on how to set
up Step 1 (compute the *head start* vs the *distance*), and the counterfactual
shows the model still rated A's token most-likely (#1) on the very branch that
sampled B's. The 8 rollouts are checked in at `examples/rewind-bestof8` and
re-readable on any laptop with no GPU; reproduce the capture with
`demos/rewind_bestof8.py`.

The smaller opt-125m run below was the first end-to-end smoke check.

## Environment (initial smoke — opt-125m)

- **GPU:** 1× NVIDIA A100-SXM4-80GB (RunPod, community cloud)
- **Stack:** vLLM 0.22.1, torch 2.11.0+cu130, transformers 5.10.2, Python 3.12
- **Model:** `facebook/opt-125m` (small, cheap; the capture API is engine-version stable)
- **Branch:** `feat/thaw-rewind`
- **Date:** 2026-06-07 · run cost ≈ $0.40

## What ran

```python
from vllm import LLM, SamplingParams
import thaw_vllm
from thaw_vllm import rewind

llm = LLM("facebook/opt-125m", enforce_eager=True, dtype="float16")
trunk = "The root cause of the production outage was"
sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=48, seed=1234)

paths = thaw_vllm.capture_rollouts(
    llm, trunk, sp, out_dir="/workspace/rollouts", n=4, logprobs=5,
    labels=["branch-a", "branch-b", "branch-c", "branch-d"],
)
```

## Results — all checks passed

**Capture produced real logprobs.** `rollout.json[0]` held 26 generated
tokens, all 26 with a float logprob and a populated top-5. Sample record:

```json
{"token_id": 5, "text": " the", "logprob": -1.856,
 "topk": [{"token_id": 5, "text": " the", "logprob": -1.856},
          {"token_id": 10, "text": " a", "logprob": -1.946},
          {"token_id": 45, "text": " not", "logprob": -2.383},
          {"token_id": 14, "text": " that", "logprob": -2.969},
          {"token_id": 41, "text": " an", "logprob": -3.293}]}
```

**`thaw rewind diff` found the divergence + counterfactual on real data:**

```
  seq logprob   A -84.30 · B -121.70   (A higher by 37.40)
  pivot         generated token 0  (after 0 identical tokens)
  - A  " the"   logprob -1.86   (B ranked this #1 @ -1.86)
  + B  " in"    logprob -5.18
```

The counterfactual is correct: A's sampled token (" the") was also B's
single most-likely token, B just sampled a low-probability alternative.

**`thaw rewind pivot` ranked the branches by sequence logprob:**

```
  4 rollouts · trunk 9 tokens · facebook/opt-125m
  first divergence at generated token 0:
    " a"       →  branch-d   (seq logprob -62.61)
    " the"     →  branch-a   (seq logprob -84.30)
    " in"      →  branch-b   (seq logprob -121.70)
    " closed"  →  branch-c   (seq logprob -124.51)
  best branch: branch-d  (seq logprob -62.61, perplexity 6.67)
```

(At temperature 0.9 the four branches diverge on the very first token, so
the pivot is token 0 — expected for an unconstrained high-temp fan-out.)

## Honest scope

Validated: the capture path (`capture_rollouts`) on a live sampled
generation, and `inspect` / `diff` / `pivot` on the resulting real
`rollout.json` files. The read/diff/pivot logic is additionally covered by
`tests/test_rewind.py` (9 GPU-free unit tests).

Not yet validated here: large models (opt-125m only); the
`checkpoint() → capture_rollouts(handle=…)` lineage round-trip (this run
used `handle=None`; `parent_id` lineage is unit-tested separately); SGLang.
The KV/weights snapshot path is unrelated to rollout capture and is covered
by `docs/AGENTFS_VALIDATION.md`.
