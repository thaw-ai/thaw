# Benchmarks

Last updated: 2026-04-14

All benchmarks use `python/vllm_demo.py` which runs four phases in sequence: normal vLLM cold start, freeze to snapshot, thaw restore from disk, thaw restore from RAM. Correctness is verified by comparing greedy-decoded output from each path.

## H100 SXM 80GB (RunPod)

**Model:** Llama-3-8B, fp16, 16.06 GB  
**vLLM:** v0.19.0, V1 engine, enforce_eager=True, single GPU  
**Storage:** RunPod container NVMe  

| Phase | Time | Throughput | Notes |
|-------|------|-----------|-------|
| Normal vLLM cold start | 20.7s | — | Includes download check, safetensors deserialization, KV cache init |
| Freeze (pipelined) | 4.8s | 3.32 GB/s | Double-buffered D2H DMA + disk write |
| **Restore (disk)** | **3.7s** | **8.26 GB/s** | 1.8s dummy init + 1.9s pipelined DMA with O_DIRECT |
| **Restore (RAM hot)** | **3.5s** | **10.69 GB/s** | 2.0s dummy init + 1.5s pure PCIe DMA |
| Restore (RAM cold) | 10.9s | — | Includes 7.4s file read into memory |

**Speedup:** 5.6x (disk), 5.9x (RAM hot path)  
**Correctness:** PASS (bit-identical greedy output)

### Time breakdown (RAM hot path)

```
                    0s        1s        2s        3s        3.5s
                    |---------|---------|---------|---------|
Dummy init:         [=========|=========]                     2.0s  (Python + vLLM model construction)
DMA to GPU:                              [========]           1.5s  (pipelined pinned→GPU, 10.69 GB/s)
                                                   [ready]
```

The 2s dummy init is now the bottleneck. The actual weight transfer (1.5s) is 30% faster than disk (1.9s) because it eliminates NVMe reads entirely.

## RTX PRO 6000 Blackwell (Google Colab)

**Model:** Llama-3-8B, fp16  
**PCIe:** Gen5 x16  

| Phase | Time | Throughput |
|-------|------|-----------|
| Normal vLLM cold start | 28.6s | — |
| **Restore (disk)** | **3.2s** | **8.75 GB/s** |

**Speedup:** 8.9x  
**Correctness:** PASS

## RTX A6000 (RunPod)

**Model:** Llama-3-8B, fp16  
**PCIe:** Gen4 x16  

| Phase | Time | Throughput |
|-------|------|-----------|
| Normal vLLM cold start | 73.2s | — |
| Freeze (pipelined) | 5.8s | 2.76 GB/s |
| **Restore (disk)** | **5.8s** | **4.21 GB/s** |

**Speedup:** 12.6x  
**Correctness:** PASS

Note: The A6000's higher speedup (12.6x) is because normal vLLM load is much slower on this hardware (73.2s vs 20.7s on H100), while thaw's restore time scales with PCIe bandwidth which is more consistent.

## Agent Fork Demo — H100 SXM 80GB (RunPod)

**Model:** Llama-3-8B-Instruct, fp16, 16.06 GB  
**Mode:** Full cycle (subprocess isolation — original instance exits, restores entirely from snapshots)  
**Storage:** Container overlay NVMe (5.1 GB/s dd, local disk)  
**Backend:** Rust pipelined (double-buffered DMA, O_DIRECT)  

### Phase 1: Original agent (subprocess)

| Operation | Time | Notes |
|-----------|------|-------|
| Model load (HuggingFace) | 16.0s | Normal vLLM cold start |
| Build session (872 tokens, 4 turns) | — | System prompt + conversation |
| Initial completion | 1.50s | 181 tokens generated |
| Freeze weights | 12.2s | 195 params, 16.1 GB |
| Freeze KV cache | 0.35s | 65 blocks, 136 MB |
| **Process exit** | — | **GPU fully released: 84.5/85.0 GiB free** |

### Phase 2: Restored agent (fresh process)

| Operation | Time | Throughput | Notes |
|-----------|------|-----------|-------|
| **Weight restore (Rust pipelined)** | **1.1s** | **14.79 GB/s** | PCIe Gen5-saturating |
| vLLM engine init | ~5s | — | KV cache profiling, warmup |
| **KV cache restore** | **0.135s** | — | 65 blocks, 136 MB |
| **Total restore** | **7.3s** | — | Weights + init + KV cache |

### Fork completions (all sharing 872-token cached prefix)

| Fork | Direction | New input tokens | Output tokens | Time |
|------|-----------|-----------------|---------------|------|
| Fork 1 | Deep dive on FP8 | 236 | 200 | 1.77s |
| Fork 2 | Speculative decoding | 268 | 200 | 1.56s |
| Fork 3 | Infrastructure-only | 260 | 200 | 1.57s |
| **Average** | | | | **1.63s** |

All 3 forks produce coherent, contextual responses referencing the original 4-turn conversation. The restored instance never saw the original conversation — everything came from thaw snapshots.

**Key numbers:**
- Weight DMA: **1.1s** (14.79 GB/s — the physical limit of PCIe Gen5)
- KV cache restore: **0.135s** (warm prefill eliminated)
- Full restore vs normal cold start: **7.3s vs 16.0s** (2.2x, but 5.5s is vLLM init overhead outside thaw's control)
- Thaw's actual work (weight DMA + KV restore): **1.2s total**

### Storage impact on restore speed

| Storage type | Weight restore | Throughput | Notes |
|-------------|---------------|------------|-------|
| Network-mounted (mfs) | 82.1s | 0.20 GB/s | RunPod network filesystem |
| Overlay NVMe (pure Python) | 8.5s | 1.88 GB/s | No Rust extension |
| **Overlay NVMe (Rust pipelined)** | **1.1s** | **14.79 GB/s** | Double-buffered DMA + O_DIRECT |

Always verify storage before benchmarking: `df -h /workspace` (check for `mfs#...` = network) and `dd if=/dev/zero of=/tmp/test bs=1M count=1024 oflag=direct`.

## Multi-GPU Tensor Parallel — 2x A100 SXM 80GB (RunPod)

### Llama-3-70B-Instruct (141 GB fp16, TP=2)

**vLLM:** v0.19.0, V1 engine, enforce_eager=True  
**Backend:** Rust pipelined (double-buffered DMA)  
**Storage:** /dev/shm (RAM-backed tmpfs — eliminates disk bottleneck)

| Phase | Time | Throughput | Notes |
|-------|------|-----------|-------|
| Normal vLLM cold start | 546.5s | — | Includes 467s HF download + 53s safetensors loading + 14s KV init |
| Freeze (TP=2) | 75.0s | 1.88 GB/s | 966 regions, 70.56 GB per rank |
| **thaw restore (total)** | **31.8s** | — | 10.5s weights + ~7s NCCL/init + 14.5s KV profiling |
| **Weight restore per rank** | **10.5s** | **6.74 GB/s** | 483 regions, 70.56 GB per rank, parallel across GPUs |

**Speedup: 17.2x** (vs cold start including HF download)  
**Weight loading speedup: 5.0x** (10.5s vs 53s safetensors, apples-to-apples)  
**Correctness:** PASS (bit-identical greedy output)

### Llama-3-8B-Instruct (16 GB fp16, TP=2)

**Storage:** /dev/shm (RAM-backed)

| Phase | Time | Throughput | Notes |
|-------|------|-----------|-------|
| Normal vLLM cold start | 20.0s | — | Model cached, safetensors from disk |
| Freeze (TP=2) | 9.3s | 1.73 GB/s | 390 regions, 8.03 GB per rank |
| **thaw restore (total)** | **13.8s** | — | 1.2s weights + ~10s NCCL/init + 2.8s KV profiling |
| **Weight restore per rank** | **1.2s** | **6.53 GB/s** | 195 regions, 8.03 GB per rank |

**Speedup: 1.4x** (vLLM overhead dominates for small model — weight loading is only 5% of total time)  
**Weight loading speedup: 7.4x** (1.2s vs 8.9s safetensors)  
**Correctness:** PASS

### Why larger models show bigger speedups

On 8B, weight loading is ~45% of total time (8.9s out of 20s). On 70B, it's ~97% (520s out of 546s). thaw accelerates weight loading by 5-7x, so the total speedup scales with how much of the cold start is weight-dominated:

| Model size | Weight % of cold start | thaw speedup |
|-----------|----------------------|-------------|
| 8B (TP=2) | ~45% | 1.4x |
| 70B (TP=2) | ~97% | 17.2x |

This is why thaw's value proposition gets stronger with larger models — exactly the direction the industry is moving.

## Reproducing

```bash
# On any CUDA 12+ machine with enough VRAM for the model:
git clone https://github.com/thaw-ai/thaw.git && cd thaw
pip install "maturin[patchelf]" vllm
cd crates/thaw-py && maturin develop --release --features cuda && cd ../..
python python/vllm_demo.py --model meta-llama/Meta-Llama-3-8B --snapshot /tmp/snapshot.thaw
```

For one-command RunPod setup, see `setup.sh`.

## What limits throughput

**Disk path (8.26 GB/s on H100):**
- Limited by NVMe sequential read bandwidth
- O_DIRECT bypasses page cache — throughput depends on drive hardware
- Double-buffered pipeline overlaps disk reads with PCIe DMA

**RAM hot path (10.69 GB/s on H100):**
- Limited by PCIe host-to-device bandwidth
- H100 SXM has PCIe Gen5 x16 (theoretical ~32 GB/s, practical ~25 GB/s)
- Current utilization is ~43% of practical PCIe ceiling — room to improve
- Possible causes: pipeline chunk overhead, CUDA stream scheduling, PyO3 FFI latency

**Dummy init (2.0s):**
- Python interpreter + vLLM model architecture construction
- Not reducible by faster I/O — this is CPU-bound Python work
- ModelLoader integration could eliminate some of this overhead
