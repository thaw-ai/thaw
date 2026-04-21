# H100 receipts runbook

Exact copy-paste to validate the full fork primitive (including
subprocess workers) on an H100 80 GB pod and emit receipts to post
publicly.

**Required:** H100 80 GB (or H200, A100 80 GB). 48 GB cards OOM on the
subprocess-worker path — parent engine + N worker engines can't co-reside.

---

## 1. Spin up the pod

RunPod template: **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`**,
H100 80 GB, 50 GB container disk.

> Do NOT use `vllm/vllm-openai:latest` — its entrypoint exits without
> `--model`. The pytorch image gives a persistent shell.

---

## 2. Install (single paste block)

```bash
# Upgrade pip, install vllm + thaw-native. This also upgrades torch 2.4 → 2.10.
pip install --quiet --upgrade pip
pip install vllm thaw-native huggingface_hub pytest

# Sanity
python -c "import thaw, vllm; print('thaw', thaw.__version__ if hasattr(thaw, '__version__') else 'ok'); print('vllm', vllm.__version__)"
```

Expected: `thaw ok` and `vllm 0.19.1` (or later).

---

## 3. Clone repo + editable install

```bash
cd /workspace
git clone https://github.com/thaw-ai/thaw.git
cd thaw
pip install -e . --no-deps
```

---

## 4. HuggingFace login (gated Llama models)

```bash
huggingface-cli login --token ${HF_TOKEN} --add-to-git-credential
```

Replace `${HF_TOKEN}` with a fresh HF token (rotate after the pod
session — treat every token that touches a pod as burnt).

---

## 5. Run all three demos with receipts

```bash
mkdir -p /workspace/receipts

# 5a. Smoke test — same-process, fastest, ~1 min
python demos/fork_smoke_test.py \
    --json-out /workspace/receipts/2026-XX-XX_h100_fork_smoke_test.json

# 5b. Parallel coding agents — 1 subprocess worker (80 GB fits parent + 1).
# For --workers ≥2 use ≥2× H100 (TP=2) or a B200.
python demos/parallel_agents.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --branches 8 --workers 1 \
    --trunk-tokens 8000 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.25 \
    --json-out /workspace/receipts/2026-XX-XX_h100_parallel_agents.json

# 5c. RL rollout simulator — same single-worker shape for single 80 GB.
python demos/rl_rollout_simulator.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --rollouts 16 --workers 1 \
    --trunk-tokens 8000 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.25 \
    --json-out /workspace/receipts/2026-XX-XX_h100_rl_rollout.json

# 5d. ForkPool amortization — this is the core claim. Boot one pool,
# run 5 rounds of fork+generate, show per-round cost stays ~flat after
# the one-time init_pool cost. Without ForkPool, every round pays
# ~340s cold-boot (receipts 2026-04-20). With ForkPool, only round 0
# pays the cost — rounds 1-4 should be seconds each.
python demos/fork_pool_rl.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --workers 1 --rounds 5 --branches-per-round 4 \
    --trunk-tokens 4000 --max-tokens 64 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.25 \
    --worker-gpu-memory 0.55 --worker-max-model-len 12288 \
    --json-out /workspace/receipts/2026-XX-XX_h100_fork_pool_rl.json
```

The worker subprocess's own memory budget is controlled by
`THAW_WORKER_GPU_MEMORY` (default 0.35). On 80 GB parent 0.25 + worker
0.55 fits; bump the env var if you see `Available KV cache memory: -X GiB`:

```bash
THAW_WORKER_GPU_MEMORY=0.55 THAW_WORKER_MAX_MODEL_LEN=12288 python demos/...
```

Replace `2026-XX-XX` with today's date.

Expected wall-clocks on H100 80 GB (ballpark):
- `fork_smoke_test`: ~30s after model download.
- `parallel_agents` with 8K trunk, 1 worker: ~5-7 min (workers=1 cold-boot dominates).
- `rl_rollout_simulator` with 8K trunk, 1 worker: ~5-7 min.
- `fork_pool_rl` with 5 rounds, 1 worker: init_pool ~5-6 min (one-time), then rounds are seconds each. Total ~7-8 min. **The key receipt:** compare round 0 vs round 4 elapsed_s — both should be under 10s. If both are ~340s, the pool isn't swapping weights (regression).

---

## 6. Copy receipts back to laptop

From laptop, with pod IP and port (from RunPod UI):

```bash
scp -P ${POD_PORT} -i ~/.ssh/id_ed25519 \
    root@${POD_IP}:/workspace/receipts/*.json \
    ~/Desktop/projects/thaw/site/receipts/
```

---

## 7. (Optional) Contribute your receipt

If you'd like to share your receipt, open a PR against
`site/receipts/index.json` with an entry pointing to your JSON file —
re-running the script is how we keep the numbers honest.

---

## 8. Rotate tokens

Revoke and re-issue both `HF_TOKEN` and `GH_PAT`. Every token that has
been on a pod should be considered burnt.

---

## Troubleshooting

**OOM during `parallel_agents.py`:** Each subprocess worker is a full
vLLM engine (8B fp16 ≈ 16 GB weights each + ~8-12 GB KV + CUDA
overhead). A single 80 GB H100 fits **parent + 1 worker** comfortably.
For `--workers` ≥ 2 you need ≥2× H100 (TP=2) or a B200/H200.

**`VLLM_ENABLE_V1_MULTIPROCESSING=0`:** Already set inside the demos.
Don't override.

**`weights.thaw.thaw-incomplete` stuck at 9 GB:** The freeze is
write-limited by pod NVMe. RunPod container disk is ~50-200 MB/s;
use a pod with NVMe-backed persistent volume for faster freezes (or
accept the ~60s one-time freeze cost).

**Worker subprocess crashes with `CUDA out of memory`:** Parent engine
still holds its full KV budget. Workaround: lower parent
`gpu_memory_utilization` in the demo (currently 0.90) to 0.60 via
patch, giving workers more headroom. Proper fix is a
`release_parent=True` flag on `fork_completions` — tracked as
post-validation work.
