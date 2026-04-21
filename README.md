# thaw

[![PyPI](https://img.shields.io/pypi/v/thaw-vllm.svg)](https://pypi.org/project/thaw-vllm/)
[![Python](https://img.shields.io/pypi/pyversions/thaw-vllm.svg)](https://pypi.org/project/thaw-vllm/)
[![Tests](https://github.com/thaw-ai/thaw/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/thaw-ai/thaw/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/thaw-vllm)](https://pepy.tech/project/thaw-vllm)

**The fork primitive for LLM inference.**

Snapshot a running AI agent — weights, KV cache, scheduler state, and prefix-hash table — into a single durable file. Restore it. Fork it N times. Each child shares the parent's state at the fork point and diverges from there. `git branch` for live GPU inference.

```bash
pip install thaw-vllm
```

### The receipt — ForkPool, 2026-04-20

Pre-warmed subprocess pool holds the engine once; each `fork_completions()` call snapshots KV only.

**Llama-3.1-8B on H100 80 GB PCIe, 5 rounds × 4 branches × 64 tokens:**

| Stage | Time |
|---|---|
| `init_pool` (one-time — workers boot with real weights) | 22.3s |
| First fork round | 1.16s |
| **Median fork round** | **0.88s** |

Per-round cost: ~340s cold-boot → sub-second (≈400× amortized). All rounds 4/4 non-empty and divergent. Bit-identical at the fork boundary. The first sub-second fork amortization proof on real hardware.

Reproducer: [`demos/fork_pool_rl.py`](demos/fork_pool_rl.py) · Receipt JSON: [`site/receipts/2026-04-20_h100_fork_pool_rl.json`](site/receipts/2026-04-20_h100_fork_pool_rl.json)

### What you can build with it

- **Agent branching** — fork a conversation into N parallel hypotheses mid-reasoning, run them concurrently, pick the winner.
- **RL rollouts** — collapse `num_rollouts × prefill_time` to `num_rollouts × memcpy_time`. Real dollars on $100k+/month training budgets. HuggingFace's 2026 async-RL survey: *"no current async library supports [KV pivot resampling] out of the box."* This ships it.
- **Parallel coding agents** — turn "8 agents exploring 8 solutions" from an expensive re-prefill tax into a fast primitive.
- **Session migration** — move a live inference session between GPUs, pods, or data centers without losing state.

### Who this is for

**RL post-training teams.** PPO, DPO, tree-GRPO, and best-of-N loops that fork rollouts from a shared trunk pay for prefill on every branch. The receipt above takes a round from ~340s cold-boot to 0.88s warm-pool. A step with 16 rollouts: ~90 minutes → ~15 seconds. Multiply by steps × epochs. HuggingFace's 2026 async-RL survey documented the gap: *"no current async library supports [KV pivot resampling] out of the box."*

**Coding-agent teams.** Parallel-exploration products — Cursor-style N approaches, SWE-bench agents, test-driven coding loops — pay a prefill tax on every branch. ForkPool turns "explore 8 approaches" from 8× full prefill into an 8-branch fork against one warm KV state. More hypotheses per user request at the same GPU spend.

**Platform + framework teams.** `thaw.fork(llm)` returns a portable, serializable handle you can ship across processes and pods. Session migration, multi-model hot-swap, session replay — without rewriting your inference layer. Drop-in for LangGraph nodes, Modal functions, Ray workers.

**Not for you yet.** Single-prompt serving — one request, one response, no shared trunk, no repeated forking — vLLM / SGLang alone are fine. thaw earns its keep when you fork ≥2 children from shared state or hot-swap between sessions.

Works with vLLM and SGLang. Open source (MIT).

<p align="center">
  <a href="https://youtu.be/zPmuvSKWrSY">
    <img src="https://img.youtube.com/vi/zPmuvSKWrSY/maxresdefault.jpg" alt="Watch: thaw hot-swaps a Llama-3-8B in 0.29s (75-second demo)" width="720">
  </a>
  <br>
  <sub><b>▶ 75-second demo</b> — <a href="https://youtu.be/zPmuvSKWrSY">Hot-swap LLMs in 0.29s</a> · <a href="https://youtu.be/aLF3lIuBeBY">How it works (4m)</a> · <a href="https://youtu.be/Fzk8sVGgi1g">Fork a running agent (2m 20s)</a></sub>
</p>

## Inside a single fork

ForkPool amortizes setup cost across repeated forks. A single one-shot fork — full state restore from cold, end-to-end — looks like this:

**Clone a running AI session (Llama-3-8B-Instruct, H100 SXM):**

| Operation | Time | Notes |
|-----------|------|-------|
| **KV cache restore** | **0.135s** | 65 blocks, 136 MB — prefill eliminated |
| Weight restore (warm-cache, post-freeze) | 1.1s | 14.79 GB/s |
| Total restore (incl. vLLM init) | **7.3s** | vs 16s normal cold start |
| Fork 3 parallel completions | **1.6s avg** | All share the 872-token cached prefix |

Every other "fast model loading" tool restores weights only. thaw restores the full state of a live inference session — weights + KV blocks + prefix-hash table + scheduler state — and that's what makes fork work.

> The 14.79 GB/s weight restore is warm-cache (the freeze that ran 5s earlier left the 16 GB file in Linux's page cache). Cold-cache NVMe hits 14.12 GB/s (next table), so the agent-fork flow lands in the same range either way.
>
> Watch the 2m20s agent-fork demo: **[Fork a running LLM agent](https://youtu.be/Fzk8sVGgi1g)**. Bit-identical output verified against the parent's next-token distribution.

## Speed benchmarks (why fork is viable)

Fork a live inference session only makes sense if restore is fast enough to be a primitive, not a pause. Here's where thaw lands:

**Llama-3-70B-Instruct (141 GB fp16) on 2x A100 SXM 80GB — tensor parallel cold start:**

| Method | Time | Speedup |
|--------|------|---------|
| Normal vLLM cold start | 546.5s | 1x |
| **thaw restore (TP=2)** | **31.8s** | **17.2x** |
| Weight restore only | 10.5s | 6.74 GB/s per rank |

**Llama-3-8B-Instruct (16 GB fp16) — single GPU, H100 SXM:**

| Method | Time | Throughput | Speedup |
|--------|------|-----------|---------|
| Normal vLLM cold start | 24.8s | — | 1x |
| **thaw restore (cold-cache NVMe)** | **2.6s** | 14.12 GB/s | **9.7x** |
| thaw restore (warm-cache) | 2.5s | 13.99 GB/s | 9.9x |
| **thaw freeze (end-to-end, v0.2.1)** | **1.7s** | **9.57 GB/s** | 2.4x over v0.1.2 |
| Freeze (pure Rust, 16 GiB synthetic) | 0.82s | 19.62 GB/s | — |

**Hot model swap (`thaw serve`, H100 SXM, Llama-3-8B-Instruct, 16 GB fp16):**

| Reload # | Time | Throughput | Backend |
|----------|------|-----------|---------|
| 0 (cold, one-time pin) | 6.40s | — | `rust_pipelined_pinned_mmap` |
| 1 | **0.29s** | **55.0 GB/s** | `rust_pipelined_pinned_mmap` |
| 2 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |
| 3 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |
| 4 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |

> `thaw serve` pins the snapshot mmap once when a pool slot warms up (~6s for 16 GB — the one-time `cudaHostRegister` cost), then reuses that pinned buffer on every subsequent swap. Steady-state = pure PCIe Gen5 DMA at 86% of theoretical peak. Extrapolates to **~2.5s hot-swap for Llama-70B** (140 GB). Bench: [`bench_slot_warm.py`](bench_slot_warm.py), correctness: [`bench_slot_warm_correctness.py`](bench_slot_warm_correctness.py).
>
> Cold-cache measurement verified with `vmtouch -e` (0% resident pages before restore, checked via `mincore`). fio parallel read on the same file confirms the NVMe ceiling. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for methodology.
>
> **Freeze runs on the same pipelined path as restore** (v0.2.1, 2026-04-17). Double-buffered WC-pinned memory, two CUDA streams, O_DIRECT writes. End-to-end 9.57 GB/s on H100 SXM; pure-Rust pipeline on synthetic 16 GiB buffer hits 19.62 GB/s (78% of PCIe Gen5 line rate).
>
> A pre-staged RAM path (mmap + `cudaHostRegister`) is implemented but gated off by default (`THAW_ZEROCOPY_MMAP=1`). `cudaHostRegister` is O(pages) — pinning a 16 GB mmap costs ~7s, which dominates one-shot restore. The path exists for `thaw serve`, where registration is amortized.

All paths produce **bit-identical** inference output. KV cache restore preserves prefix cache across cold starts — new requests skip prefill entirely.

<details>
<summary>More GPUs and models</summary>

| GPU | Model | Normal | thaw | Speedup |
|-----|-------|--------|------|---------|
| 2x A100 SXM 80GB | Llama-3-70B (TP=2) | 546.5s | 31.8s | **17.2x** |
| H100 SXM 80GB | Llama-3-8B | 24.8s | 2.6s | **9.7x** |
| RTX PRO 6000 (Blackwell) | Llama-3-8B | 28.6s | 3.2s | **8.9x** |
| RTX A6000 | Llama-3-8B | 73.2s | 5.8s | **12.6x** |

Larger models show bigger speedups because weight loading dominates more of the total cold start time.

</details>

## How it works

**Fork** is a composition of four primitives: freeze weights, freeze KV cache, freeze scheduler state, restore all three into a fresh process. None of that was possible at GPU speeds before thaw.

```
A running vLLM engine:
  [weights 16 GB] + [KV cache N blocks] + [scheduler state + prefix-hash table]
                             │
                             ▼ thaw.freeze_model + thaw.freeze_kv_cache
                             │
                    one durable artifact on disk (or S3)
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
    child process 1    child process 2    child process N
    [same weights]     [same weights]     [same weights]
    [same KV state]    [same KV state]    [same KV state]
    diverges here →    diverges here →    diverges here →
```

**Freeze** captures the full engine state into two binary files: `.thaw` (weights) and `.thawkv` (KV blocks + prefix-hash table + scheduler metadata).

**Restore** initializes a fresh vLLM engine with dummy weights (fast — no disk I/O), overwrites them from the snapshot via double-buffered pipelined DMA through pinned host memory, then rebuilds the prefix-cache block table from the `.thawkv` sidecar. Two CUDA streams overlap PCIe transfers with disk reads. New requests matching the restored prefix skip prefill entirely.

Three restore modes:
- **Disk**: reads snapshot from NVMe with O_DIRECT, bypassing the kernel page cache. Throughput limited by NVMe bandwidth — on H100 SXM NVMe this hits 14 GB/s with the Rust pipelined path, saturating the drive.
- **Pre-staged RAM**: snapshot already in memory (tmpfs, shared memory, or mmapped with page cache warm). The full zero-copy path (mmap + `cudaHostRegister`) is implemented behind `THAW_ZEROCOPY_MMAP=1`, but the one-time registration cost makes it a win only when amortized across many restores.
- **Slot-warm hot-swap (`thaw serve`)**: when a pool slot warms up, `thaw serve` pins the snapshot mmap once (~6s `cudaHostRegister` for 16 GB) and persists the pinned handle on the slot. Every subsequent model swap into that slot reuses the pinned buffer and runs as pure PCIe DMA — 0.29s at 55 GB/s for an 8B model on H100 SXM.

**KV cache snapshots** are the hard part. vLLM's prefix-cache hash table maps token-hash → block-id, and the scheduler assumes those block assignments are live. thaw serializes the block contents, the hash table, and the scheduler's view of which blocks are cached. On restore, the block data is DMA'd back to GPU and the hash table is rebuilt — so a request whose prefix was cached in the parent immediately hits cache in the child. Nobody else does this.

## Architecture

```
thaw/
  crates/
    thaw-core/       Rust. File format, region tables, I/O. No CUDA dep.
    thaw-cuda-sys/   Rust. FFI bindings to CUDA runtime (cudaMallocHost,
                     cudaMemcpyAsync, streams). Built via build.rs.
    thaw-runtime/    Rust. Orchestration: freeze/restore pipelines, double-
                     buffered DMA, O_DIRECT, thread-local WC-buffer cache,
                     unified zero-copy/staging restore. MockCuda for Mac.
    thaw-py/         Rust. PyO3 bindings exposing pipelined freeze/restore
                     to Python. Builds a native .so via maturin.
    thaw-cli/        Rust. thaw-bench-freeze binary + internal tooling.
  python/
    thaw_common/     Engine-agnostic freeze/restore primitives (shared).
    thaw_vllm/       vLLM integration + engine pool + OpenAI server.
      snapshot.py    vLLM TP freeze/restore via collective_rpc.
      kv_snapshot.py KV cache freeze/restore (pipelined path, .meta sidecar).
      loader.py      vLLM ModelLoader: load_format="thaw".
      pool.py        Engine pool: pre-warmed slots, model hot-swap.
      server.py      OpenAI-compatible API server.
      cli.py         CLI: thaw freeze, thaw serve, thaw info.
    thaw_sglang/     SGLang integration (class-passthrough loader).
    vllm_demo.py     End-to-end benchmark: normal vs thaw cold start.
    kv_cache_demo.py KV cache snapshot/restore demo with correctness test.
  demos/
    agent_fork.py    Agent fork demo: clone session, fork parallel completions.
```

**Testing on Mac, shipping on GPU.** The `CudaBackend` trait abstracts all GPU operations. `MockCuda` (a HashMap-backed fake) lets 48 runtime tests run on any machine. The `cuda` feature flag activates real GPU paths only when needed.

## Quick start

```bash
pip install thaw-vllm[all]
```

This installs the Python package, FastAPI server, and pre-built Rust+CUDA native extension. No Rust toolchain needed.

### Fork a running AI agent

The core capability, in one call:

```python
import thaw_vllm
from vllm import LLM, SamplingParams

# Load and run an agent until you hit a pivot point
llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct",
          enable_prefix_caching=True)
llm.generate([reasoning_trunk], SamplingParams(max_tokens=200))

# Fan out from that pivot — 8 parallel approaches in subprocess workers,
# each hydrates from one shared snapshot, zero reprefill of the trunk.
results = thaw_vllm.fork_completions(
    llm,
    prompts=[trunk + hint for hint in branch_hints],
    sampling_params=SamplingParams(temperature=0.9, max_tokens=512),
    workers=4,
)
for r in results:
    print(r.worker_index, r.text[:200])
```

Prefer the primitive when you want to persist, move, or hand off the
handle yourself:

```python
with thaw_vllm.fork(llm, include_weights=True) as handle:
    handle.save("s3://my-bucket/session-abc123/")       # ship it anywhere
    stats = handle.hydrate(other_llm)                    # or restore in-place
```

For RL training loops — boot the engine pool once, fork repeatedly at sub-second cost:

```python
from thaw_vllm import ForkPool

pool = ForkPool()
pool.init_pool(                      # one-time, ~22s on H100 8B
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    workers=4,
    preload_weights=True,            # workers hold real weights; fork swaps only KV
)

for epoch in range(num_epochs):
    # Each call reuses the warm pool — ~0.88s median per round on H100 8B
    results = thaw_vllm.fork_completions(llm, prompts, sampling_params, pool=pool)
    rewards = score(results)
    ...                              # PPO / best-of-N / tree-GRPO step
```

Working demos ship in the repo:

- [`demos/rl_rollout_simulator.py`](demos/rl_rollout_simulator.py) — Tree-GRPO-style pivot resampling. Builds a reasoning trunk, forks 16 rollouts, scores each. The table it prints is the arithmetic HuggingFace's async-RL survey said no library ships: `num_rollouts × prefill → num_rollouts × memcpy`.
- [`demos/parallel_agents.py`](demos/parallel_agents.py) — 8 parallel coding approaches from one reasoning trunk, ranked by pytest pass rate. The Cursor/Cognition reframe.
- [`demos/agent_fork.py`](demos/agent_fork.py) — the original end-to-end session-clone demo used in the launch video.

### Server mode (OpenAI-compatible)

**Freeze a model, then serve it:**

```bash
# Llama models are gated — authenticate with HuggingFace first
huggingface-cli login

# Step 1: Freeze model weights to a snapshot
thaw freeze --model meta-llama/Llama-3.1-8B-Instruct --output weights.thaw

# Step 2: Serve with pre-warmed engine pool
thaw serve --model meta-llama/Llama-3.1-8B-Instruct --snapshot weights.thaw
```

That's it. You now have an OpenAI-compatible API at `http://localhost:8000/v1`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 64}'
```

### How `thaw serve` works

`thaw serve` is PgBouncer for GPU inference. It keeps vLLM engines pre-initialized with dummy weights, then DMA-swaps real model weights from a snapshot on demand. First swap into a slot pays the one-time `cudaHostRegister` pin cost (~6s for 16 GB); every subsequent swap runs at **55 GB/s (0.29s for 8B, ~2.5s for 70B)** — that's the pinned mmap reused through PCIe Gen5 DMA without ever leaving the slot.

- **OpenAI-compatible API** — `/v1/completions`, `/v1/chat/completions`, streaming via SSE
- **Model affinity** — requests for an already-loaded model have zero swap cost
- **Hot model registration** — register new snapshots at runtime via `/admin/snapshots`
- **Pool status** — monitor slots, loaded models, and utilization via `/admin/pool`

```bash
# Multi-model pool with 2 warm slots
thaw serve --model meta-llama/Llama-3.1-8B-Instruct \
  --snapshot base.thaw \
  --pool-size 2 \
  --register finetune-v2=/snapshots/v2.thaw

# The model field in each request selects which snapshot to serve
curl localhost:8000/v1/completions -d '{"model": "finetune-v2", "prompt": "..."}'
```

### Python API

```python
import thaw_vllm
from vllm import LLM, SamplingParams

# Freeze: save model weights to a snapshot
llm = LLM(model="meta-llama/Meta-Llama-3-8B", dtype="float16", enforce_eager=True)
thaw_vllm.freeze_model_pipelined(model, "/path/to/weights.thaw")

# Restore: two lines, 9.2x faster cold start
llm = thaw_vllm.load("meta-llama/Meta-Llama-3-8B", "/path/to/weights.thaw")
```

Or use `load_format="thaw"` directly with vLLM:

```python
import thaw_vllm  # registers the loader
llm = LLM(model="meta-llama/Meta-Llama-3-8B",
          load_format="thaw",
          model_loader_extra_config={"snapshot": "/path/to/weights.thaw"})
```

**Multi-GPU** — tensor parallel with per-rank snapshots:

```python
# Freeze: each GPU saves its shard
llm = LLM(model="meta-llama/Meta-Llama-3-70B-Instruct", tensor_parallel_size=2, ...)
thaw_vllm.freeze_model_tp(llm, "/path/to/weights.thaw")
# Creates: weights.thaw (rank 0), weights.rank1.thaw (rank 1)

# Restore: 17.2x faster than normal cold start
llm = thaw_vllm.load("meta-llama/Meta-Llama-3-70B-Instruct", "/path/to/weights.thaw",
                      tensor_parallel_size=2)
```

**Cloud storage (S3)** — load snapshots directly from S3 URIs (install with `pip install thaw-vllm[cloud]`):

```python
# Freeze once, upload to S3, restore anywhere
llm = thaw_vllm.load("meta-llama/Meta-Llama-3-8B",
                     "s3://my-bucket/llama-3-8b.thaw")
```

First call downloads to `~/.cache/thaw/snapshots/` (override with `THAW_CACHE_DIR`); subsequent calls hit the local cache. For TP, per-rank files live at `s3://bucket/weights.thaw` and `s3://bucket/weights.rank1.thaw` — thaw derives the per-rank URIs automatically. AWS credentials come from the standard boto3 chain (env vars, `~/.aws/credentials`, IAM role).

**SGLang** — same API, class-passthrough loader (install with `pip install thaw-vllm[sglang]`):

```python
import sglang
from thaw_sglang import ThawSGLangModelLoader

engine = sglang.Engine(
    model_path="meta-llama/Meta-Llama-3-8B",
    load_format=ThawSGLangModelLoader,
    model_loader_extra_config={"snapshot": "/path/to/weights.thaw"},
    dtype="float16",
)
```

TP works automatically — each SGLang worker loads its own rank-specific snapshot. Freeze via `thaw freeze --engine sglang ...` or `ThawSGLangFreezeLoader`. Note: vLLM and SGLang cannot coexist in one env (torch version conflict) — use separate pods.

**Agent fork demo** — clone a running AI session, fork parallel completions:

```bash
python demos/agent_fork.py --snapshot weights.thaw
python demos/agent_fork.py --snapshot weights.thaw --full-cycle  # destroy + restore
```

### CLI reference

```bash
thaw freeze --model meta-llama/Meta-Llama-3-8B --output weights.thaw
thaw serve  --model meta-llama/Meta-Llama-3-8B --snapshot weights.thaw [--pool-size N] [--register NAME=PATH]
thaw info   weights.thaw
```

<details>
<summary>Troubleshooting</summary>

**`hf-xet` download crash** — Some versions of `huggingface_hub` ship with an `hf-xet` backend that can crash during large model downloads. If you see `RuntimeError: Data processing error: File reconstruction error`, set:
```bash
export HF_HUB_DISABLE_XET=1
```

**Disk space** — `pip install thaw-vllm[all]` plus a 8B model snapshot needs ~50 GB. Use at least 100 GB container disk on cloud providers.

**Gated models** — Llama models require HuggingFace authentication. Run `huggingface-cli login` before freeze/serve.

</details>

<details>
<summary>Building from source (alternative to pre-built wheels)</summary>

If you need to build the Rust+CUDA backend yourself (e.g., custom CUDA version):

```bash
git clone https://github.com/thaw-ai/thaw.git && cd thaw
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
pip install "maturin[patchelf]" vllm
maturin build --release --features cuda -m crates/thaw-py/Cargo.toml -o /tmp/wheels
pip install /tmp/wheels/*.whl
pip install -e ".[serve]"
```

</details>

## Competitive landscape

Lots of work in adjacent spaces. None of them fork a live session at the GPU-state layer.

| Capability | thaw | fastsafetensors | NVIDIA Model Streamer | vLLM Sleep Mode | Modal Snapshots | LMCache / Dynamo KV | InferX |
|---|---|---|---|---|---|---|---|
| Weight snapshot + fast restore | ✅ (14 GB/s NVMe, 55 GB/s slot-warm) | ✅ (26 GB/s w/ GDS+RAID) | ✅ (~2 GB/s) | ✅ (RAM only) | ✅ (alpha) | — | claimed (no public code) |
| **KV cache snapshot + prefix-hash restore** | **✅** | — | — | RAM only, same process | — | partial (block-level, not engine) | claimed |
| **Fork a running session into N divergent children** | **✅** | — | — | — | — | — | — |
| Cross-process / cross-pod restore | ✅ | ✅ (reload) | ✅ (reload) | — (same process) | ✅ | partial | claimed |
| Works on commodity hardware (no GDS / RAID) | ✅ | — | ✅ | ✅ | ✅ | ✅ | — |
| Open source, pip-installable | ✅ (MIT) | ✅ (Apache) | ✅ | ✅ | — | ✅ | — |

**What thaw uniquely owns:**

1. **Fork as a primitive.** Nobody else snapshots the combined weights + KV cache + prefix-hash table + scheduler state of a live inference engine and restores it into a fresh process. This is what makes agent branching, RL rollout deduplication, and session migration actually work. Everything below exists to make this primitive fast enough to be useful.
2. **KV cache snapshot with prefix-hash reconstruction.** The moat under the moat. LMCache / Dynamo tier KV blocks for their own cache; they don't let you transport a cache between engines. thaw does.
3. **Saturates commodity hardware.** 14 GB/s on a single NVMe (no GDS, no RAID) and 55 GB/s slot-warm (no special drivers). The speed is table stakes for the fork primitive to be viable — but it's still faster than fastsafetensors' GDS+RAID ceiling on the setups most people actually have.
4. **Works with vLLM and SGLang.** Two engines, one `.thaw` file. `load_format="thaw"` for vLLM, class-passthrough loader for SGLang.

**How thaw is not LMCache / Tensormesh.** LMCache (and Tensormesh, which commercializes it) is a *server-side cache-tiering proxy* that sits in front of your engine, watches incoming requests, and serves prefix cache hits from GPU/RAM/NVMe tiers. It's passive: requests come in, matches happen or don't. thaw is an *imperative primitive* your code calls at a specific pivot — `fork(llm) → handle` returns an atomic, portable reference to that session (weights + KV + scheduler state + prefix-hash table) that any other process can hydrate. LMCache can't give you a handle you hand to an RL worker; it's not the API shape. HuggingFace's 2026 async-RL survey documented this gap explicitly: *"no current async library supports [KV pivot resampling] out of the box."* Different product, different buyer — their raise is validation, not overlap.

## Roadmap

- [x] Weight snapshot/restore (pure Python path)
- [x] Rust+CUDA pipelined freeze/restore (double-buffered DMA, O_DIRECT)
- [x] RAM-backed restore path (mmap + chunked pinned staging; zero-copy mmap variant gated behind `THAW_ZEROCOPY_MMAP` for `thaw serve`)
- [x] PyO3 bindings + vLLM integration shim
- [x] H100 / A6000 / Blackwell benchmarks
- [x] **KV cache snapshot/restore** — the moat (freeze/restore prefix-cached blocks, verified on Llama-3-8B)
- [x] `pip install thaw-vllm` + CLI (`thaw freeze`, `thaw serve`, `thaw info`)
- [x] `load_format="thaw"` — native vLLM ModelLoader integration
- [x] OpenAI-compatible API server (`thaw serve`)
- [x] Streaming support in API server (SSE, OpenAI-compatible)
- [x] **Agent fork demo** — clone a running AI session, fork parallel completions from shared KV cache (full-cycle: 14.79 GB/s restore, 0.135s KV restore on H100 SXM)
- [x] **Multi-GPU / tensor parallel** — 17.2x speedup on Llama-3-70B with 2x A100 (TP=2), bit-exact correctness verified
- [x] **Engine pool (`thaw serve`)** — pre-warmed vLLM engines with hot model swapping, OpenAI-compatible API, multi-model serving
- [x] **Pre-built native wheels** — `pip install thaw-vllm[all]`, no Rust toolchain needed
- [x] **SGLang integration** — class-passthrough loader, freeze + restore, validated on H100 TP=2 (5.0 GB/s)
- [x] **Slot-warm hot-swap** — persistent `cudaHostRegister` per pool slot, 0.29s / 55 GB/s model swap on H100 SXM (`thaw serve`)
- [x] **Cloud snapshot storage (S3)** — `thaw freeze --output s3://...` and `thaw serve --snapshot s3://...`, validated H100 SXM 2026-04-17 (15 GiB freeze+upload in 5.6s, 229s single-stream S3 download — ranged-GET crate is next)
- [x] **Pipelined-freeze parity with restore** — `freeze_pipelined_to_file` with chunked WC-pinned buffers + O_DIRECT lands in v0.2.1 (2026-04-17). End-to-end 9.57 GB/s on H100 SXM (vs 3.82 GB/s in v0.1.2); pure-Rust 19.62 GB/s on synthetic buffer.
- [x] **ForkPool (v0.3.2, 2026-04-20)** — pre-warmed subprocess pool: boot N vLLM engines once with real weights, each `fork_completions()` call snapshots KV only. 22.3s init → 0.88s median/round on H100 8B (4 branches × 64 tokens). First sub-second fork amortization on real hardware.
- [x] **Plain-pinned freeze fix (thaw-native v0.3.1, 2026-04-20)** — v0.3.0 wheel capped freeze at 50 MB/s because CPU reads of WC-pinned memory are ~100× slower than plain pinned. Fresh `pip install` now pulls the fast path by default.
- [ ] LangGraph integration — expose `fork()` as a graph-level primitive in the framework teams already use
- [ ] Framework-layer RL helpers — TRL / `accelerate` wrappers around `fork_completions()` for tree-GRPO / best-of-N
- [ ] Rust `thaw-cloud` crate — concurrent ranged GETs for S3 restore at NIC line-rate (restore gap, deprioritized behind fork-layer distribution)
- [ ] GPUDirect Storage support

## Design

Full technical architecture, file format spec, and rationale: [DESIGN.md](./DESIGN.md)

## Get in touch

thaw is built by a Madison, WI team — Nils Matteson (founder), Matt Yu, Karan Kapur.

- **Evaluating for a real workload?** Email [nils@thaw.sh](mailto:nils@thaw.sh) — include your rollout shape or fork pattern and we'll help you wire it up.
- **Training RL models or running parallel agents at scale?** DM on LinkedIn: [Nils Matteson](https://www.linkedin.com/in/nilsmatteson/) — happy to screen-share and profile your loop.
- **Bug, feature request, or question?** [GitHub issues](https://github.com/thaw-ai/thaw/issues).

⭐ [Star on GitHub](https://github.com/thaw-ai/thaw) if you're watching this space.

## License

MIT
