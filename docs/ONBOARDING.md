# Onboarding: What thaw is, where we are, what's next

Welcome to thaw. This doc gets you up to speed fast.

## What thaw does

LLM inference servers (vLLM, SGLang) take 20-500+ seconds to cold start because they load model weights from disk (safetensors format) through a slow deserialization path. thaw snapshots the entire GPU state to a flat binary file and restores it via pipelined DMA, bypassing all deserialization. Think `fork()` for GPU processes.

**The core insight:** safetensors is a good serialization format but a terrible I/O format. We don't need to deserialize — we snapshot the raw GPU memory layout and blast it back via DMA.

## How it works (30-second version)

```
Freeze (once):
  GPU weights → pinned host buffer (async D2H DMA) → .thaw file on disk/tmpfs

Restore (every cold start):
  .thaw file → mmap (zero-copy on /dev/shm) → pinned buffer → GPU (async H2D DMA)
```

The .thaw file format is simple: 4096-byte header + region table + raw payload bytes. No compression, no checksums, no metadata beyond what's needed to map regions back to GPU addresses. See `crates/thaw-core/` for the format spec.

The performance comes from:
1. **Pinned memory** — `cudaMallocHost` pages that enable true async DMA (pageable memory silently falls back to synchronous copies)
2. **Pipelined double-buffering** — while one buffer DMAs to GPU, the next chunk reads from disk/RAM into the other buffer
3. **mmap on tmpfs** — `/dev/shm` is RAM-backed; mmap maps the same physical pages into userspace with zero copy
4. **O_DIRECT** — bypasses kernel page cache for NVMe reads (avoids double-buffering through page cache)

## Current benchmark results (2026-04-14, 2x H100 SXM 80GB)

8/8 models pass with bit-identical correctness.

| Model | Size | GPUs | Weight Load (normal) | DMA (thaw) | Weight Speedup | Throughput |
|-------|------|:----:|---------------------:|-----------:|:--------------:|-----------:|
| phi-3-mini | 7.6 GB | 1 | 10.0s | 1.1s | **8.8x** | 6.8 GB/s |
| mistral-7b | 14 GB | 1 | 11.2s | 2.1s | **5.3x** | 6.9 GB/s |
| llama-3.1-8b | 16 GB | 1 | 10.9s | 2.4s | **4.5x** | 6.6 GB/s |
| qwen-7b | 15 GB | 1 | 9.7s | 2.2s | **4.4x** | 7.0 GB/s |
| gemma-2-9b | 18 GB | 1 | 7.6s | 3.4s | **2.3x** | 5.5 GB/s |
| mixtral-8x7b (MoE) | 87 GB | 2 | 62.1s | 7.4s | **8.4x** | 14.4 GB/s |
| llama-3.1-70b | 141 GB | 2 | 86.7s | 11.2s | **7.7x** | 14.3 GB/s |
| qwen-72b | 145 GB | 2 | 92.6s | 13.0s | **7.1x** | 13.2 GB/s |

"Weight Speedup" strips out vLLM init overhead (~9-20s) which is identical in both paths.
End-to-end speedup including vLLM init: 1.3-3.4x.

Multi-GPU runs at 13-14 GB/s (near PCIe Gen4 per-GPU limit). Single-GPU is 5.5-7 GB/s — there's room to improve here.

Raw data: `benchmarks/h100_stress_test_2026-04-14.json`

## Codebase architecture

```
thaw/
  crates/                        # Rust (the performance engine)
    thaw-core/                   #   File format, headers, region tables
    thaw-cuda-sys/               #   Raw CUDA FFI (cudaMalloc, cudaMemcpy, etc.)
    thaw-runtime/                #   Orchestration: backends, pipelining, mock GPU
      src/backend.rs             #     CudaBackend + PipelinedBackend traits
      src/mock.rs                #     MockCuda — run tests on Mac, no GPU needed
      src/real.rs                #     RealCuda — actual CUDA calls
      src/pipeline.rs            #     Double-buffered pipelined restore
      src/direct_io.rs           #     O_DIRECT file abstraction
    thaw-py/                     #   PyO3 bindings (thin wrappers, no logic)
    thaw-cli/                    #   CLI benchmarking tool
  python/thaw_vllm/              # Python (vLLM integration layer)
    snapshot.py                  #   freeze/restore functions, TP support
    loader.py                    #   vLLM ModelLoader (load_format="thaw")
    kv_snapshot.py               #   KV cache freeze/restore
    server.py                    #   Streaming inference server
  demos/                         # Stress test, diagnostics, demos
  site/                          # Landing page
```

**Design principle:** Rust owns the hot path (DMA, pipelining, file I/O). Python owns the integration layer (vLLM internals, model introspection, tensor mapping). The boundary is `crates/thaw-py/` which is intentionally thin.

## Key concepts you need to know

### Pinned vs pageable memory
The single most important CUDA concept for this project. `cudaMemcpyAsync` with pageable (normal) memory silently falls back to synchronous copying at ~3 GB/s. With pinned memory (`cudaMallocHost`), it's truly async and hits ~12-14 GB/s. Every performance bug we've hit traces back to accidentally using pageable memory.

### vLLM model access patterns
vLLM doesn't expose the `nn.Module` directly. You navigate through:
```python
llm.llm_engine.engine_core.model_executor.driver_worker.model_runner.model
```
With `tensor_parallel_size > 1`, vLLM spawns worker processes via `multiprocessing.spawn`. These are fresh Python processes — they don't inherit imports. Use `collective_rpc` to dispatch functions to workers.

### The region map
thaw's file format stores "regions" — contiguous byte ranges with a kind tag (weights, KV cache, metadata) and a logical ID. On restore, Python tells Rust "region 0 maps to GPU address 0x7f..., region 1 maps to 0x7f..." and Rust blasts bytes to those addresses. The mapping is built by iterating `model.named_parameters()`.

### MockCuda
All Rust tests run on Mac without a GPU. `MockCuda` implements the `CudaBackend` trait using plain `Vec<u8>` allocations. This means you can develop and test the pipeline logic, file format parsing, and error handling without touching a GPU. Only the final integration tests need real hardware.

## Where we are now (what shipped)

- [x] Binary file format (.thaw) — Rust + Python, byte-compatible
- [x] Pipelined freeze/restore — Rust, double-buffered async DMA
- [x] vLLM `load_format="thaw"` integration — ModelLoader plugin
- [x] Multi-GPU tensor parallel (TP=2 tested, arbitrary TP supported)
- [x] KV cache snapshots — freeze/restore prefix cache across cold starts
- [x] Agent fork demo — clone running AI sessions via weight + KV restore
- [x] mmap zero-copy restore from /dev/shm
- [x] Streaming inference server (SSE)
- [x] Stress test suite — 8 models, 5 architectures, bit-identical correctness

## What's next (where you can contribute)

### High-impact performance work (OS/systems knowledge needed)

1. **Single-GPU throughput: 7 GB/s → 12+ GB/s**
   - Current bottleneck: mmap page faults during Rust pipelined read
   - Investigate: `madvise(MADV_SEQUENTIAL)`, `MAP_POPULATE`, `readahead()`
   - Or: dedicated prefetch thread that touches pages ahead of the DMA thread
   - Files: `crates/thaw-runtime/src/pipeline.rs`, `crates/thaw-runtime/src/direct_io.rs`

2. **Multi-GPU freeze: 1.4 GB/s → 10+ GB/s**
   - TP>1 freeze currently uses the pure Python path (not Rust pipelined)
   - Need: `freeze_model_tp` to use Rust pipelined freeze via `collective_rpc`
   - Same pattern as `restore_model_tp` in `python/thaw_vllm/snapshot.py`

3. **Eliminate vLLM init overhead (the big one)**
   - vLLM dummy init takes 9-20s (model graph construction, KV cache alloc, warmup)
   - This is the same whether you load from disk or thaw — it's the floor
   - Approach: snapshot the entire vLLM engine state, not just weights
   - Or: pre-warmed engine pool that reuses initialized engines
   - This would turn 3.4x end-to-end into 8-10x

4. **Freeze throughput optimization**
   - D2H (GPU→host) is limited by PCIe in the reverse direction
   - Current: 2 GB/s. Should be 8-12 GB/s with proper pipelining
   - The Rust pipelined freeze exists but isn't wired through for TP>1

### Feature work

5. **SGLang support** — second most popular inference framework after vLLM
6. **Cloud storage streaming** — restore from S3/GCS directly, no local staging
7. **LoRA hot-swap** — freeze/restore adapter weights for fast model switching
8. **Pre-built wheels** — CI/CD to publish `thaw-native` for common platforms

## Development setup

```bash
# Clone
git clone https://github.com/thaw-ai/thaw.git && cd thaw

# Python package (no GPU needed)
pip install -e .

# Rust (tests run on Mac, no GPU)
cargo test -p thaw-core
cargo test -p thaw-cuda-sys
cargo test -p thaw-runtime

# Rust with CUDA (needs GPU box)
cargo test --release -p thaw-runtime --features cuda

# Build native extension (needs GPU box + CUDA toolkit)
pip install "maturin[patchelf]"
export CUDA_PATH=/usr/local/cuda
maturin build --release -m crates/thaw-py/Cargo.toml --features cuda --skip-auditwheel
pip install target/wheels/thaw_native-*.whl --no-deps --force-reinstall
```

## Key files to read first

1. `crates/thaw-runtime/src/backend.rs` — the CudaBackend trait (the abstraction everything else builds on)
2. `crates/thaw-runtime/src/pipeline.rs` — the pipelined restore hot path
3. `python/thaw_vllm/snapshot.py` — freeze/restore Python layer + TP support
4. `DESIGN.md` — architecture decisions and rationale
5. `demos/stress_test.py` — how all the pieces fit together
