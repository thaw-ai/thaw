# thaw receipts

Machine-readable proofs that the primitive works on real hardware. Each JSON
file in this directory is the sanitized output of a demo run — you can
re-run the same demo and compare your numbers to ours.

## Files

- `index.json` — list of receipts with one-line summaries.
- `YYYY-MM-DD_<gpu>_<demo>.json` — individual run receipts.

## Schema

Every receipt conforms to `receipt_version: 1`:

```json
{
  "receipt_version": 1,
  "timestamp_utc": "...",
  "demo": "fork_smoke_test | parallel_agents | rl_rollout_simulator",
  "model": "meta-llama/Meta-Llama-3-8B-Instruct",
  "trunk_tokens": 56,
  "timings_s": { "load_s": ..., "trunk_warm_s": ..., "fork_handle_s": ..., "fork_completions_s": ... },
  "modes": { "A_native_batch": {...}, "B_fork_subprocess": {...}, "C_cold_baseline": {...} },
  "checks": { "all_nonempty": true, "all_diverge": true, "prefix_cache_enabled": true },
  "samples": [{ "mode": "...", "branch": 0, "text_preview": "...200 chars..." }],
  "pod": { "gpu_name": "...", "gpu_memory_mib": 46068, "cuda_version": "..." },
  "software": { "python": "...", "vllm": "...", "torch": "...", "thaw_native": "..." },
  "git": { "short": "...", "commit": "...", "dirty": false },
  "extra": { ... demo-specific args + provenance ... }
}
```

## Reproduce

```bash
pip install thaw-native vllm
git clone https://github.com/thaw-ai/thaw && cd thaw
pip install -e .

# Cheapest: same-process fork, runs on any 16+ GB GPU in ~30s + model download
python demos/fork_smoke_test.py --json-out my_smoke.json

# Real subprocess-worker path: N workers = N+1 full vLLM engines resident.
# 8B fp16 ≈ 16 GB weights each, so an 80 GB H100 fits parent + 1 worker.
# For parent + 4 workers use ≥2× H100 (TP=2) or an H200/B200.
python demos/parallel_agents.py --trunk-tokens 8000 --workers 1 \
    --json-out my_parallel.json
python demos/rl_rollout_simulator.py --trunk-tokens 8000 --rollouts 16 \
    --workers 1 --json-out my_rl.json
```

## Sanitization

Receipts strip absolute paths (`/tmp/*`, `/workspace/*`, `/home/*`,
`/Users/*`) and common secret patterns (`hf_*`, `ghp_*`, `sk-*`,
`pypi-*`). Branch text previews are truncated to 200 characters.

See `demos/_receipt.py` for the exact sanitization rules.
