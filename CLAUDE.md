# thaw — Project State & Roadmap

*Updated 2026-04-18 (launch day — videos live, site redesigned, first investor inbound). This file is for Claude Code sessions — start here before suggesting work.*

## What thaw is

Snapshot/restore for LLM inference state. Freeze a fully-loaded model (weights + KV cache) to disk, restore it via pipelined CUDA DMA in seconds instead of minutes. Open source, pip-installable, works on any CUDA 12+ GPU. The only tool that snapshots KV cache — enables agent forking and session cloning.

**Verified benchmarks (bit-identical output):**
- Llama-3-70B on 2× A100 TP=2: 546.5s → 31.8s (**17.2×**)
- Llama-3-8B on H100 SXM: 20.7s → 3.5s (**5.9×**, 10.69 GB/s RAM hot path)
- Llama-3-8B on RTX A6000: 73.2s → 5.8s (**12.6×**)
- Agent fork restore (weights): 1.1s at **14.79 GB/s** — PCIe Gen5-saturating

## Architecture

```
crates/                         # Rust core — 9,424 LOC across 5 crates
  thaw-core                     # format constants, CRC
  thaw-cuda-sys                 # CUDA FFI (CudaBackend trait → MockCuda for Mac tests)
  thaw-runtime                  # pipelined freeze/restore, pinned memory, O_DIRECT
  thaw-py                       # PyO3 bindings
  thaw-cli                      # `thaw freeze|restore|serve` binary

python/
  thaw_common/    (537 LOC)     # engine-agnostic freeze/restore primitives
  thaw_vllm/      (2,150 LOC)   # vLLM integration + engine pool + server
    loader.py                   # load_format="thaw" ModelLoader
    snapshot.py                 # freeze_model_tp, restore_model_tp (TP via collective_rpc)
    kv_snapshot.py  (540 LOC)   # KV cache freeze/restore — the moat
    pool.py         (651 LOC)   # EnginePool: pre-warmed slots + hot model swap
    server.py                   # OpenAI-compatible streaming server
    cli.py                      # thaw freeze / thaw serve CLI
  thaw_sglang/    (327 LOC)     # SGLang integration — validated on H100 2026-04-17
    loader.py                   # ThawSGLangModelLoader + ThawSGLangFreezeLoader
```

## What's shipped

1. **vLLM `load_format="thaw"`** — native ModelLoader passthrough, one-liner `thaw_vllm.load()`
2. **SGLang integration** — class-passthrough loader, freeze + restore, validated on H100 TP=2 (5.0 GB/s). A40 TP=2 hits an SGLang CUDA-graph bug, not ours.
3. **Multi-GPU tensor parallel** — 17.2× on Llama-70B TP=2, arbitrary TP via `collective_rpc`
4. **KV cache snapshot/restore** — 540 LOC, prefix-cache-hash reconstruction, nobody else has this
5. **Agent fork demo** — clone running session, skip prefill, parallel completions from shared context
6. **`thaw serve` daemon** — EnginePool with pre-warmed slots and hot model swap (651 LOC, 33 tests)
7. **OpenAI-compatible server** — streaming chat completions on top of EnginePool
8. **Pre-built wheels** on PyPI (`pip install thaw-vllm`)
9. **Stress-tested** — 8/8 models across 5 architectures, bit-identical on 2× H100 SXM
10. **V1 MP default support** — weights freeze/restore/load/pool/TP all work under vLLM's default multiprocessing mode. No more `VLLM_ENABLE_V1_MULTIPROCESSING=0` env-var hack for users (validated 2×A40 2026-04-17). Only KV cache path still needs V1 MP=0 — that's auto-set internally when `--kv-output`/`kv_snapshot` is used.
11. **S3 end-to-end** — `thaw freeze --output s3://...` and `thaw serve --snapshot s3://...` both work. Freeze+upload 2.88 GB/s on H100 SXM (validated 2026-04-17). Download path is still boto3 single-stream — slow but functional. TP round-trip validated: `weights.thaw` + `weights.rank1.thaw` derived automatically.
12. **Pipelined freeze rewrite (v0.2.1)** — `freeze_pipelined_to_file` matches restore architecture (WC-pinned double buffers, O_DIRECT, two CUDA streams). 9.57 GB/s end-to-end on H100 SXM (2.4× over v0.1.2), 19.62 GB/s pure Rust on synthetic buffer (104× over old BufWriter path).
13. **Slot-warm hot-swap validated** — 2026-04-17 H100 SXM: 55 GB/s sustained, 0.29s per 8B reload after one-time pin. Extrapolates to ~2.5s for 70B (140 GB).
14. **Launch assets (2026-04-18)** — Three production YouTube videos live (Public): Video 1 hero (75s), Video 2 how-it-works (4m), Video 3 agent fork (2m20s). Site redesigned with Team section (Nils founder, Matt + Karan co-founders), embedded videos, SGLang + S3 features surfaced. First investor inbound: Caroline McManus, Hyde Park Venture Partners.

## What's NOT built (the gaps)

| Gap | Why it matters | Owner |
|-----|----------------|-------|
| **Fast S3/GCS restore (ranged-GET pinned ring)** | `s3://` URIs work end-to-end (shipped 2026-04-17; freeze+upload 2.88 GB/s). But boto3 single-stream caps restore at ~67 MB/s → 229s ready on 8B. Rust `thaw-cloud` crate with parallel ranged GETs into pinned ring is the unlock. | Karan |
| **GPUDirect Storage (GDS)** | NVMe → GPU without CPU bounce. Required to beat fastsafetensors' 26 GB/s. | Matt |
| **vLLM upstream PR for `load_format="thaw"`** | Distribution + credibility. Even unmerged shows intent for YC. | Nils |
| **Multi-GPU freeze throughput** | Currently ~1.4 GB/s (restore is 10+ GB/s). Wire Rust pipelined freeze through `collective_rpc`. | — |
| **Single-GPU restore 7 → 12+ GB/s** | `MADV_SEQUENTIAL`, `MAP_POPULATE`, prefetch thread. Pure OS work. | Matt |
| **EnginePool multi-model demo at scale** | 10+ models, <2s hot-swap. Matches InferX's "30 models on 2 GPUs" claim. | Nils |
| **vLLM init overhead floor (9–20s)** | Pre-warmed engines help, but init itself is the remaining floor. Would turn 3.4× e2e into 10–15×. | — |
| **LoRA hot-swap** | Freeze/restore adapter weights only (<1s). Serve 100 fine-tunes from one base. | — |
| **Global snapshot CDN** | Pre-stage popular models at edge. `thaw restore cdn://llama-3.1-70b`. | — |
| **Cross-node RDMA restore** | Restore from another machine's GPU via InfiniBand/RoCE. Sub-second cluster-wide. | — |

## What to build next (ranked by leverage)

**Tier 1 — before YC deadline 2026-05-04:**

1. **Rust `thaw-cloud` ranged-GET crate** — S3 freeze+upload ships at 2.88 GB/s (validated H100 2026-04-17). Restore still single-stream boto3: ~67 MB/s, 229s ready on 8B. Parallel ranged GETs into a pinned ring should saturate the NIC and close the "managed service demo is slow" gap. Karan owns.
2. **vLLM upstream PR** — Even if not merged by 2026-05-04, the PR link in the YC app proves ecosystem traction. Low effort, high signal.
3. **EnginePool 10-model demo + video** — Directly matches InferX's claim. 30 seconds of screen recording = the YC money shot.
4. **Agent fork demo polish + video** — Already works; needs a narrative recording. This is the differentiator nobody can match.

**Tier 2 — 1–2 months post-YC:**

5. **GPUDirect Storage** — Pushes single-GPU restore toward fastsafetensors territory. Matt.
6. **Eliminate vLLM init floor** — The last unlock to 10×+ e2e. Harder problem (touches vLLM internals), high payoff.
7. **LoRA hot-swap** — Cheap to build, huge multiplier for fine-tuning customers.

**Tier 3 — the moat (3–6 months):**

8. **Global snapshot CDN** — Own the data layer between storage and compute.
9. **Cross-node RDMA restore** — Sub-second restore for any model anywhere in the cluster.
10. **Full engine state serialization** — True `fork()` for GPU processes. Zero cold-start endgame.

## Business progression

Open-source library (shipped) → `thaw serve` daemon (shipped) → `s3://` round-trip (shipped) → **fast S3 restore + managed CDN (tier 1 gap)** → universal GPU state layer (endgame). Databricks started with Spark → Delta Lake → platform. Same playbook.

## Critical constraints

- **vLLM and SGLang cannot coexist in one env** — torch version conflict (vLLM 0.19 needs torch 2.10; SGLang 0.5.x needs 2.8/2.9). Test in separate pods.
- **SGLang TP=2 requires H100 or L40S** — A40 hits SGLang's piecewise CUDA graph bug during warmup. Not a thaw issue; don't chase it.
- **Gated HF models need `huggingface-cli login`** on every fresh pod before `thaw freeze/serve`.
- **Public repo = release quality only.** Never push to `thaw-ai/thaw` main without GPU validation. Use `sync-public.sh` with a branch.
- **`VLLM_ALLOW_INSECURE_SERIALIZATION=1` is set automatically** by `thaw_vllm` on import. vLLM 0.19+ defaults to msgspec for EngineCore IPC, which rejects function objects; `collective_rpc(fn, ...)` needs cloudpickle fallback. Setdefault means users can override.
- **KV cache path still requires V1 MP=0** — scheduler state only reachable in V1-inproc/V0. `cmd_freeze` (with `--kv-output`) and `load()` (with `kv_snapshot`) set this internally. Weights-only paths run under V1 MP default.
- **Building `thaw-native` with `--features cuda` on a pod**: must pass `--auditwheel skip` to maturin. auditwheel-repair can't resolve torch's hash-suffixed `libcudart-<hash>.so.12` and will abort the build.

## Team

- **Nils** — product, Python, integration, orchestration
- **Matt** — Rust throughput (OS/memory, GDS)
- **Karan** — Rust cloud storage (S3/GCS streaming)

## Key files for new sessions

- `README.md` — public-facing pitch + benchmarks
- `docs/STRATEGY.md` — full business plan, pricing, YC narrative
- `docs/LANDSCAPE.md` — competitive analysis
- `plans/2026-04-18_launch-readiness.md` — current source of truth for YC countdown (supersedes 2026-04-15 sprint plan)
- `.claude/projects/-Users-nils-Desktop-projects-thaw/memory/` — persistent memory across sessions
