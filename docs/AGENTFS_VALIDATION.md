# agentfs validation report

GPU validation of the `agentfs` surface (`checkpoint` / `branch` / `checkout` / `inspect` / `diff` / `log`) and the underlying fork/KV-snapshot paths.

## Environment

| | |
|---|---|
| Date | 2026-06-07 |
| Commit | `24816e5` (branch `feat/agentfs-inspect-diff`) |
| GPU | NVIDIA RTX 4090 (RunPod, community) |
| Engine | vLLM 0.19.1 |
| torch | 2.10.0+cu128 |
| thaw-native | 0.3.2 |
| Model | `facebook/opt-125m` (ungated; correctness test, not a perf benchmark) |

Run via two separate processes to exercise the *cross-process* restore path: `val.py` (parent — produces handles, runs agentfs checks + adversarial cases) and `val_checkout.py` (a fresh process that hydrates the on-disk handle).

## Results — all green

**Parent side (18/18 PASS):**
- lazy `import thaw_vllm` works with torch present **and** registers the `thaw` load format; `checkpoint`/`checkout`/`branch` exposed.
- `checkpoint(prompt=…, include_weights=True)` → handle with >0 KV blocks, weights written, prefix preview + token IDs captured.
- `branch()` stamps a fresh `handle_id` + correct `parent_id`, copies the payload.
- `inspect` / `diff` / `log` correct on real handles.

**Cross-process checkout round-trip — bit-identical:**
A fresh process loaded the handle from disk, `checkout()` restored weights (148 regions, 2.1 GB/s) + 11 KV blocks, and the restored engine reproduced the parent's greedy continuation **exactly**:
```
parent: [1470, 16, 5, 812, 9, 1470, 4, 50118, 1864, 35, 653, 16, 5, 812, 9, 5]
child : [1470, 16, 5, 812, 9, 1470, 4, 50118, 1864, 35, 653, 16, 5, 812, 9, 5]
RESULT: PASS bit-identical continuation after cross-process checkout
```

**Adversarial / break-it cases (all PASS):**
- `diff` flags different models as `DIFFER`.
- `inspect` on a handle missing its `.thawkv.meta` degrades gracefully (no crash).
- `inspect`/`summarize` on a missing dir raises `FileNotFoundError`; on malformed `handle.json` raises (no silent garbage).
- `log` renders an orphan `parent_id` (parent not present) as a root — no crash, no infinite loop.
- `checkpoint` without a prompt → `prefix_preview` None, handle still valid.
- `checkpoint` with a long unicode prompt → token IDs capped at 2048, preview capped — no crash.
- `checkpoint(prompt_token_ids=…)` path stores IDs verbatim.
- `diff` of unrelated prompts splits early (token-level divergence works).

**Committed artifacts:**
- `tests/test_agentfs.py` — passes on the pod.
- Shipped `examples/pr-review-fanout/` reproduces the README `diff`/`log` output exactly.

## Scope / not yet validated (honest)

- **Tensor parallel (TP≥2 / multi-GPU fork).** Only TP=1 was tested. The TP>1 KV path has historically been thorny (NCCL re-init, V1 MP). Needs a 2-GPU run.
- **Large models.** Only `opt-125m`. Correctness of the paths is model-agnostic, but large-model behavior (memory, block counts) wasn't re-exercised here.
- **SGLang.** vLLM only this round.
- **ForkPool subprocess pool at scale.** The single-fork/checkpoint/checkout paths are covered; the N-worker ForkPool amortization path was validated previously (see the 0.88s receipt), not re-run here.

Cost of this validation: ~$0.10 (RTX 4090, ~9 min).
