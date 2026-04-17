# thaw

**Fast snapshot/restore for LLM inference. Sub-second model hot-swap at 55 GB/s, 17x faster cold starts on 70B, multi-GPU tensor parallel, KV cache preservation.**

`thaw serve` hot-swaps a Llama-3-8B-class model in **0.29 seconds** at **55 GB/s** — PCIe Gen5-saturating, bit-identical output. Cold-start Llama-3-70B on 2x A100 drops from 546s to **31.8s** (17.2x). Rust+CUDA pipelined DMA, pinned-memory persistence, and KV cache snapshots that no other tool offers.

<p align="center">
  <img src="site/terminal.gif" alt="thaw demo — pip install, freeze, serve, chat completions on RunPod A40" width="800">
</p>

## Benchmarks

**Hot model swap (`thaw serve`, H100 SXM, Llama-3-8B-Instruct, 16 GB fp16):**

| Reload # | Time | Throughput | Backend |
|----------|------|-----------|---------|
| 0 (cold, one-time pin) | 6.40s | — | `rust_pipelined_pinned_mmap` |
| 1 | **0.29s** | **55.0 GB/s** | `rust_pipelined_pinned_mmap` |
| 2 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |
| 3 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |
| 4 | **0.29s** | **55.1 GB/s** | `rust_pipelined_pinned_mmap` |

> `thaw serve` pins the snapshot mmap once when a pool slot warms up (~6s for 16 GB — the one-time `cudaHostRegister` cost), then reuses that pinned buffer on every subsequent swap. Steady-state = pure PCIe Gen5 DMA at 86% of theoretical peak. Bit-identical output verified across reloads. Extrapolates to **~2.5s hot-swap for Llama-70B** (140 GB), directly comparable to InferX's "sub-2s" claim. Bench: [`bench_slot_warm.py`](bench_slot_warm.py), correctness: [`bench_slot_warm_correctness.py`](bench_slot_warm_correctness.py).

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
| **thaw (cold-cache NVMe)** | **2.6s** | 14.12 GB/s | **9.7x** |
| thaw (warm-cache) | 2.5s | 13.99 GB/s | 9.9x |

> Cold-cache measurement verified with `vmtouch -e` (0% resident pages before restore, checked via `mincore`). fio parallel read on the same file confirms the NVMe ceiling — thaw's Rust reader saturates it. Reproducible across back-to-back runs. See [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for methodology.
>
> A pre-staged RAM path (mmap + `cudaHostRegister`) is implemented but gated off by default (`THAW_ZEROCOPY_MMAP=1` to enable). `cudaHostRegister` is O(pages) — pinning a 16 GB mmap costs ~7s, which dominates one-shot restore. The path exists for `thaw serve`, where registration is amortized across many restores. Tracking proper amortization in a follow-up.

**Agent fork — clone a running AI session (Llama-3-8B-Instruct, H100 SXM):**

| Operation | Time | Notes |
|-----------|------|-------|
| Weight restore (warm-cache, post-freeze) | 1.1s | 14.79 GB/s — file was in page cache, see note below |
| **KV cache restore** | **0.135s** | 65 blocks, 136 MB — prefill eliminated |
| Total restore (incl. vLLM init) | **7.3s** | vs 16s normal cold start |
| Fork 3 parallel completions | **1.6s avg** | All share 872-token cached prefix |

> The 14.79 GB/s weight restore here is a warm-cache measurement (the freeze that ran 5s earlier left the 16 GB file in Linux's page cache). The agent-fork *flow* is still the differentiator — no other tool restores KV cache at all. The KV restore number, the fork completion numbers, and the "skip prefill" claim all stand on their own regardless of where the weights came from.

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

```
Normal vLLM cold start:
  Download weights → deserialize safetensors → copy to GPU → init KV cache → ready
  [======================================] 24.8s

thaw restore (cold-cache NVMe):
  Dummy init → parallel NVMe read → pipelined DMA to GPU
  [====] 2.6s
```

**Freeze** captures all GPU state into binary snapshots — model weights (`.thaw`) and KV cache blocks (`.thawkv`).

**Restore** initializes vLLM with dummy weights (fast — no disk I/O), then overwrites them from the snapshot using double-buffered pipelined DMA through pinned host memory. Two CUDA streams overlap PCIe transfers with disk reads. KV cache blocks are restored separately with their prefix cache hash mappings, so new requests immediately get cache hits.

Three restore modes:
- **Disk**: reads snapshot from NVMe with O_DIRECT, bypassing the kernel page cache. Throughput limited by NVMe bandwidth — on H100 SXM NVMe this hits 14 GB/s with the Rust pipelined path, saturating the drive.
- **Pre-staged RAM**: snapshot already in memory (tmpfs, shared memory, or mmapped with page cache warm). The full zero-copy path (mmap + `cudaHostRegister`) is implemented behind `THAW_ZEROCOPY_MMAP=1`, but the one-time registration cost makes it a win only when amortized across many restores.
- **Slot-warm hot-swap (`thaw serve`)**: when a pool slot warms up, `thaw serve` pins the snapshot mmap once (~6s `cudaHostRegister` for 16 GB) and persists the pinned handle on the slot. Every subsequent model swap into that slot reuses the pinned buffer and runs as pure PCIe DMA — 0.29s at 55 GB/s for an 8B model on H100 SXM, saturating Gen5 PCIe.

**KV cache snapshots** capture the prefix-cached blocks that vLLM retains after generation. On restore, block data is DMA'd back to GPU and the prefix cache hash table is reconstructed. Requests with matching prefixes skip prefill — the most expensive part of inference.

## Architecture

```
thaw/
  crates/
    thaw-core/       Rust. File format, region tables, I/O. No CUDA dep.
    thaw-cuda-sys/   Rust. FFI bindings to CUDA runtime (cudaMallocHost,
                     cudaMemcpyAsync, streams). Built via build.rs.
    thaw-runtime/    Rust. Orchestration: freeze/restore pipelines, double-
                     buffered DMA, O_DIRECT, MockCuda for Mac testing.
    thaw-py/         Rust. PyO3 bindings exposing pipelined freeze/restore
                     to Python. Builds a native .so via maturin.
    thaw-cli/        Rust. GPU benchmark binary.
  python/
    thaw_vllm/       Python package (pip install thaw-vllm).
      snapshot.py    Freeze/restore weights, Rust backend fallback.
      kv_snapshot.py KV cache freeze/restore.
      loader.py      vLLM ModelLoader: load_format="thaw".
      pool.py        Engine pool: pre-warmed slots, model hot-swap, OpenAI API.
      server.py      Single-engine OpenAI-compatible API server.
      cli.py         CLI: thaw freeze, thaw serve, thaw info.
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

The model loading space is active. Here's how thaw compares:

| Project | Approach | Throughput | Limitations |
|---------|----------|-----------|-------------|
| **thaw** | Pipelined DMA, pinned memory, O_DIRECT + KV cache snapshot | 6.7-14.8 GB/s per GPU | — |
| fastsafetensors (IBM) | GDS + 4x NVMe RAID0 | 26.4 GB/s | Requires GDS setup + RAID hardware |
| NVIDIA Model Streamer | Multi-threaded concurrent streaming | ~2 GB/s (single SSD) | NVIDIA-maintained, less flexible |
| CoreWeave Tensorizer | HTTP/S3 streaming + deserialization | ~4.6 GB/s local | Tied to CoreWeave ecosystem |
| vLLM Sleep Mode | Offload to CPU RAM, reload | 0.26-3s | Not a cold start — requires prior warm load |
| Modal GPU Snapshots | CUDA checkpoint/restore API | ~10x reduction | Alpha. Doesn't help with large model weight loading |
| InferX | GPU runtime snapshotting | Claims 2s for 70B | No public code or benchmarks |

**thaw's differentiation:**
1. **KV cache snapshot/restore** — nobody else does this. Preserves prefix cache across cold starts, eliminates prefill. Enables agent forking, session migration, warm handoff.
2. **Single NVMe performance** — most deployments don't have RAID0. thaw already matches or beats multi-threaded alternatives on one drive.
3. **No special hardware** — no GDS, no RAID, no driver patches. Works on any CUDA 12+ GPU.

See [docs/LANDSCAPE.md](./docs/LANDSCAPE.md) for detailed analysis.

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
- [ ] Cloud snapshot storage (S3/GCS)
- [ ] GPUDirect Storage support

## Design

Full technical architecture, file format spec, and rationale: [DESIGN.md](./DESIGN.md)

## License

MIT
