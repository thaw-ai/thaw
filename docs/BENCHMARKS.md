# Benchmarks

Last updated: 2026-04-17

All benchmarks use `python/vllm_demo.py`. Correctness is verified by comparing greedy-decoded output from each path.

## Methodology: cold-cache vs warm-cache

A `thaw restore` benchmark is only honest if the source file is NOT already in the Linux page cache. Immediately after a freeze, the file IS in cache, so a naive "disk restore" benchmark actually measures RAM throughput. We explicitly flush the page cache between freeze and restore using `posix_fadvise(POSIX_FADV_DONTNEED)` (works in containers without root).

We report three scenarios separately:

| Scenario | What it measures | When it matches production |
|----------|------------------|----------------------------|
| **Cold-cache NVMe** (headline) | Read `.thaw` from NVMe via pipelined DMA | Fresh pod, container just started, snapshot on local disk |
| **Warm-cache** | File is already in page cache from prior read | Repeated restores of the same snapshot; tells you the ceiling |
| **Pre-staged RAM** | Snapshot resident in process memory before restore starts | `thaw serve` daemon with pre-warmed slots |

Older versions of this doc reported the **warm-cache** number as the "disk restore" headline. Those numbers assumed the file was left in cache by the preceding freeze — which is not a real cold start. Entries marked "(post-freeze, warm cache)" below are those older numbers, kept for reference. Entries marked **"cold-cache"** are produced by the updated harness with the cache explicitly dropped.

Cache eviction: the benchmark calls `vmtouch -e` (verified via `mincore`) when available, else falls back to `posix_fadvise(POSIX_FADV_DONTNEED)`. On hosts with abundant free RAM, fadvise is often ignored by the kernel — install `vmtouch` (`apt-get install vmtouch`) for a guaranteed hard eviction.

Re-run the harness on your hardware with:
```bash
python python/vllm_demo.py --model meta-llama/Meta-Llama-3-8B --snapshot /tmp/snap.thaw
```
`--skip-cache-drop` disables eviction (produces the old, dishonest warm-cache number; for debugging only).

## H100 SXM 80GB (RunPod) — cold-cache NVMe, verified 2026-04-17

**Model:** Llama-3-8B, fp16, 16.06 GB  
**vLLM:** v0.19.0, V1 engine, enforce_eager=True, single GPU  
**Storage:** RunPod container NVMe (MD RAID10 over 6 drives)  
**Harness:** `vllm_demo.py --skip-warm`, `vmtouch -e` between freeze and restore (mincore-verified 0% resident)  
**Runs:** 3 back-to-back; numbers below are per-run, averaged in the headline  

| Phase | Time | Throughput | Notes |
|-------|------|-----------|-------|
| Normal vLLM cold start | 25.0s | — | Includes download check, safetensors deserialization, KV cache init |
| Freeze (v0.1.2, pre-rewrite) | 4.2s | 3.82 GB/s | Old BufWriter path — kept for comparison |
| **Freeze (v0.2.1, pipelined)** | **1.7s** | **9.57 GB/s** | New `freeze_pipelined_to_file` — WC-pinned double-buffer + O_DIRECT |
| **Restore (cold-cache NVMe)** | **1.2s** | **13.0 GB/s** | vmtouch-verified 0% resident; Rust pipelined reader saturates RAID10 parallel reads |
| Dummy init (model skeleton) | 1.5s | — | vLLM `load_format="dummy"` |
| **Total thaw cold start** | **2.7s** | — | 1.5s init + 1.2s cold restore |

**Headline speedup (cold-cache NVMe): 9.2×** (25.0s → 2.7s, averaged over 3 runs: 9.1× / 9.2× / 9.4×)  
**Freeze speedup (v0.1.2 → v0.2.1): 2.4×** (3.82 → 9.57 GB/s, end-to-end through `thaw freeze --model`)  
**Correctness:** PASS (bit-identical greedy output, all 3 runs)

### Pure-Rust freeze ceiling (synthetic 16 GiB buffer)

The `thaw-bench-freeze` binary exercises `freeze_pipelined_to_file` with synthetic data to measure the Rust pipeline without Python-side overhead (`named_parameters()` iteration, per-region PyO3 boundary crossings):

| Path | Time | Throughput | Notes |
|------|------|-----------|-------|
| Old `freeze_pipelined` + BufWriter | 84.5s | 0.19 GB/s | v0.1.2 baseline |
| **New `freeze_pipelined_to_file` + O_DIRECT** | **0.82s** | **19.62 GB/s** | 78% of PCIe Gen5 x16 practical peak, 104× over old |

The ~2× gap between the synthetic 19.62 GB/s and the end-to-end 9.57 GB/s is Python overhead; batching the region enumeration into Rust is a future sprint. Repro: `bash scripts/pod-bench-freeze.sh`.

Why 13.0 GB/s beats single-threaded `dd iflag=direct` (≈7.2 GB/s on the same file): the Rust pipelined reader issues parallel I/Os across the 6-disk RAID10 stripe; single-threaded `dd` can't extract that parallelism and underestimates the true NVMe ceiling.

fio confirms the parallel ceiling on this pod:

```
# After vmtouch -e /workspace/snap.thaw:
fio --name=read --filename=/workspace/snap.thaw --direct=1 --rw=read \
    --bs=1M --numjobs=8 --iodepth=32 --size=16G --group_reporting --runtime=30
# → READ: bw=11.1GiB/s (11.9GB/s)
```

The 13.0 GB/s Rust reader is at/slightly above the fio-measured ceiling — thaw's pipelined reader is I/O-saturating this storage.

### Historical (prior harness — warm-cache + pre-staged RAM)

Kept for reference; produced on a different H100 SXM pod (storage + hugepage behavior differ):

| Phase | Time | Throughput | Notes |
|-------|------|-----------|-------|
| Normal vLLM cold start | 20.7s | — | |
| Restore (disk, post-freeze, warm cache) | 3.7s | 8.26 GB/s | Was old headline — reading from page cache, not NVMe |
| Restore (RAM hot, pre-staged) | 3.5s | 10.69 GB/s | 2.0s dummy init + 1.5s pure PCIe DMA |

> **Known regression (2026-04-17):** On the pod used for the cold-cache verification above, the pre-staged-RAM path (`rust_pipelined_mmap`) reproducibly hits ~1.9 GB/s instead of 10.69 GB/s, with `madvise(MADV_HUGEPAGE)` returning `errno=12/22` (ENOMEM/EINVAL). Likely a transparent-hugepage / container-privilege mismatch on this pod, not a thaw code regression. Cold-cache NVMe path is unaffected. Tracking.

### Time breakdown (pre-staged RAM scenario)

```
                    0s        1s        2s        3s        3.5s
                    |---------|---------|---------|---------|
Dummy init:         [=========|=========]                     2.0s  (Python + vLLM model construction)
DMA to GPU:                              [========]           1.5s  (pipelined pinned→GPU, 10.69 GB/s)
                                                   [ready]
```

In the pre-staged RAM scenario the 2s dummy init dominates. The actual weight DMA (1.5s) is faster than NVMe because RAM has more bandwidth than the SSD. This scenario represents `thaw serve` with pre-warmed slots — not a true cold start.

## RTX PRO 6000 Blackwell (Google Colab)

**Model:** Llama-3-8B, fp16  
**PCIe:** Gen5 x16  

| Phase | Time | Throughput |
|-------|------|-----------|
| Normal vLLM cold start | 28.6s | — |
| Restore (post-freeze, warm cache) | 3.2s | 8.75 GB/s |

**Speedup:** 8.9x (warm-cache — **not** a cold start)  
**Correctness:** PASS

> Cold-cache NVMe re-run pending. Colab's disk varies; expect 4–7 GB/s cold.

## RTX A6000 (RunPod)

**Model:** Llama-3-8B, fp16  
**PCIe:** Gen4 x16  

| Phase | Time | Throughput |
|-------|------|-----------|
| Normal vLLM cold start | 73.2s | — |
| Freeze (pipelined) | 5.8s | 2.76 GB/s |
| Restore (post-freeze, warm cache) | 5.8s | 4.21 GB/s |

**Speedup:** 12.6x (warm-cache headline; A6000's huge speedup comes from the 73s baseline, not thaw magic)  
**Correctness:** PASS

Note: The A6000's high "speedup" multiplier is mostly because normal vLLM load is slow on this hardware (73.2s vs 20.7s on H100). Cold-cache NVMe re-run pending.

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
| Weight restore (post-freeze, warm cache) | 1.1s | 14.79 GB/s | PCIe Gen5-saturating — but file was left warm in cache by the freeze |
| vLLM engine init | ~5s | — | KV cache profiling, warmup |
| **KV cache restore** | **0.135s** | — | 65 blocks, 136 MB |
| **Total restore** | **7.3s** | — | Weights + init + KV cache |

> The 14.79 GB/s weight restore is a **warm-cache** measurement: the freeze that ran 5s earlier left the 16 GB file in the Linux page cache, so this is effectively a RAM-to-GPU copy, not NVMe-to-GPU. A true cold-cache NVMe cold start on this container would likely land around 5–7 GB/s. The KV restore and fork math stand on their own; KV blocks are small enough they weren't the throughput bottleneck either way.

### Fork completions (all sharing 872-token cached prefix)

| Fork | Direction | New input tokens | Output tokens | Time |
|------|-----------|-----------------|---------------|------|
| Fork 1 | Deep dive on FP8 | 236 | 200 | 1.77s |
| Fork 2 | Speculative decoding | 268 | 200 | 1.56s |
| Fork 3 | Infrastructure-only | 260 | 200 | 1.57s |
| **Average** | | | | **1.63s** |

All 3 forks produce coherent, contextual responses referencing the original 4-turn conversation. The restored instance never saw the original conversation — everything came from thaw snapshots.

**Key numbers:**
- Weight DMA: 1.1s (14.79 GB/s — warm-cache, post-freeze)
- KV cache restore: **0.135s** (warm prefill eliminated — this number is honest regardless of page cache)
- Full restore vs normal cold start: **7.3s vs 16.0s** (2.2x, with 5.5s of that being vLLM init overhead outside thaw's control)
- Thaw's actual work (weight DMA + KV restore): 1.2s total (warm-cache)

### Storage impact on restore speed

| Storage type | Weight restore | Throughput | Notes |
|-------------|---------------|------------|-------|
| Network-mounted (mfs) | 82.1s | 0.20 GB/s | RunPod network filesystem — cold or warm, this is the floor |
| Overlay NVMe (pure Python) | 8.5s | 1.88 GB/s | No Rust extension |
| **Overlay NVMe (Rust pipelined, warm cache)** | **1.1s** | **14.79 GB/s** | Double-buffered DMA — **warm-cache**, not a true cold start |

The 14.79 GB/s number above is RAM-to-GPU throughput for a file that was left in the page cache by the preceding freeze. For a true cold cache (fresh pod, file on disk not in RAM), NVMe becomes the bottleneck and you should expect 5–7 GB/s on container overlay storage — still a large win over safetensors, but not PCIe-saturating.

Always verify storage before benchmarking: `df -h /workspace` (check for `mfs#...` = network) and `dd if=/dev/zero of=/tmp/test bs=1M count=1024 oflag=direct`.

## Multi-GPU Tensor Parallel

**Prior 2×A100 "17.2× / 546.5s → 31.8s" number has been removed pending re-measurement.** The value came from a single single-pod run with a full HF download included in the "normal cold start" baseline; a later H100 TP=2 run on current code showed 74.2s → 33.1s (2.24×) with the restore cascade mis-ordered. Cascade fix is in the commit log; re-validation on 2×H100 is on the Tier-1 benchmark matrix. New receipts will land in `site/receipts/` when produced.

Validated today (bit-identical output):

- 2× A40 TP=2 (Llama-3-8B): V1 MP default path, freeze+restore round-trip PASS (`project_issue3_complete` memory)
- 2× H100 TP=2 (Llama-3-70B, 2026-04-19): restore 8.55 GB/s per rank, 6.06 GB/s freeze, 74.2s → 33.1s e2e — see `project_70b_h100_bench`

For why whole-flow speedup numbers depend heavily on whether HF download is in the baseline, see "Apples-to-apples" notes in `docs/STRATEGY.md`.

## SGLang — H100 SXM 80GB (RunPod)

**Model:** Llama-3-8B-Instruct, fp16, 16.06 GB
**SGLang:** v0.5.10
**Backend:** Rust pipelined (fallback chain: RAM → pipelined → pure Python)

### Single GPU

| Phase | Throughput | Notes |
|-------|-----------|-------|
| Freeze | ~1.6 GB/s | Via `ThawSGLangFreezeLoader` (piggybacks on SGLang default loader) |
| **Restore** | **5.01 GB/s** | Via `ThawSGLangModelLoader` (class-passthrough, meta → CUDA materialization) |

**Correctness:** PASS — coherent generation verified after restore

### Tensor Parallel (TP=2)

| Phase | Throughput | Notes |
|-------|-----------|-------|
| Freeze (per rank) | ~1.6 GB/s | Each SGLang worker freezes its own rank-specific `.thaw` file |
| **Restore (per rank)** | **4.88 GB/s** | Each worker loads its own per-rank snapshot, no `collective_rpc` needed |

**Correctness:** PASS — coherent generation on TP=2

### Notes on SGLang integration

- SGLang's `get_model_loader()` supports class-passthrough: pass `load_format=ThawSGLangModelLoader` directly, no monkey-patching needed
- TP works automatically because SGLang spawns separate worker processes; each calls `get_tensor_model_parallel_rank()` independently and loads its own rank-specific snapshot
- Meta-tensor materialization must use `model_config.dtype` (e.g. float16), not the meta-tensor default (float32), or you get a buffer-size mismatch
- Buffers (rotary embedding cos/sin cache) must be moved with `.to('cuda')` to preserve computed values, not `torch.empty()`
- **A40 TP=2 crashes during SGLang's piecewise CUDA graph compilation** — this is an SGLang bug, not thaw. Use H100 or L40S for multi-GPU SGLang

## Reproducing

```bash
# On any CUDA 12+ machine with enough VRAM for the model:
git clone https://github.com/matteso1/thaw.git && cd thaw
pip install "maturin[patchelf]" vllm
cd crates/thaw-py && maturin develop --release --features cuda && cd ../..

# Cold-cache NVMe (honest headline):
python python/vllm_demo.py --model meta-llama/Meta-Llama-3-8B --snapshot /tmp/snapshot.thaw

# Debug: reproduce the old (dishonest) warm-cache number:
python python/vllm_demo.py --snapshot /tmp/snapshot.thaw --skip-cache-drop
```

The harness prints the active cache state for every phase (look for the `[cache]` line). `posix_fadvise(POSIX_FADV_DONTNEED)` is called between Phase 2 (freeze) and Phase 3 (restore) to guarantee Phase 3 is a true cold read.

For one-command RunPod setup, see `setup.sh`.

## What limits throughput

**Cold-cache NVMe (5–7 GB/s expected):**
- Limited by NVMe sequential read bandwidth (container overlay storage)
- `posix_fadvise(POSIX_FADV_DONTNEED)` ensures we measure disk, not page cache
- To push past this: GPUDirect Storage (NVMe → GPU, bypass CPU), faster NVMe (local PCIe Gen5 SSDs hit 14+ GB/s)

**Warm-cache / post-freeze (8–10 GB/s on H100):**
- File is in Linux page cache, so reads go RAM → pinned → GPU
- This is effectively the "pre-staged RAM" path's ceiling, minus the overhead of the kernel copy into pinned memory

**Pre-staged RAM (10.69 GB/s on H100):**
- Limited by PCIe host-to-device bandwidth
- H100 SXM has PCIe Gen5 x16 (theoretical ~32 GB/s, practical ~25 GB/s)
- Current utilization is ~43% of practical PCIe ceiling — room to improve
- Possible causes: pipeline chunk overhead, CUDA stream scheduling, PyO3 FFI latency

**Dummy init (2.0s):**
- Python interpreter + vLLM model architecture construction
- Not reducible by faster I/O — this is CPU-bound Python work
- ModelLoader integration could eliminate some of this overhead
