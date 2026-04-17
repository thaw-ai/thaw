# thaw — Project State & Roadmap

*Updated 2026-04-17. This file is for Claude Code sessions — start here before suggesting work.*

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

## What's NOT built (the gaps)

| Gap | Why it matters | Owner |
|-----|----------------|-------|
| **Cloud snapshot storage (S3/GCS streaming)** | The actual managed-service hook. `thaw restore s3://...` is the Databricks-playbook moment. | Karan |
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

1. **Cloud snapshot storage** — `thaw freeze s3://bucket/model.thaw`, `thaw restore s3://...`. New crate `thaw-cloud`. This is the wedge from "free tool" to "managed service." Without it, there is no business model to demo; with it, the YC pitch writes itself ("Databricks for GPU state"). Karan owns.
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

Open-source library (shipped) → `thaw serve` daemon (shipped) → **thaw Cloud CDN (tier 1 gap)** → universal GPU state layer (endgame). Databricks started with Spark → Delta Lake → platform. Same playbook.

## Critical constraints

- **vLLM and SGLang cannot coexist in one env** — torch version conflict (vLLM 0.19 needs torch 2.10; SGLang 0.5.x needs 2.8/2.9). Test in separate pods.
- **SGLang TP=2 requires H100 or L40S** — A40 hits SGLang's piecewise CUDA graph bug during warmup. Not a thaw issue; don't chase it.
- **Gated HF models need `huggingface-cli login`** on every fresh pod before `thaw freeze/serve`.
- **Public repo = release quality only.** Never push to `thaw-ai/thaw` main without GPU validation. Use `sync-public.sh` with a branch.

## Team

- **Nils** — product, Python, integration, orchestration
- **Matt** — Rust throughput (OS/memory, GDS)
- **Karan** — Rust cloud storage (S3/GCS streaming)

## Key files for new sessions

- `README.md` — public-facing pitch + benchmarks
- `docs/STRATEGY.md` — full business plan, pricing, YC narrative
- `docs/LANDSCAPE.md` — competitive analysis
- `plans/2026-04-15_sprint-plan.md` — week-by-week YC countdown
- `.claude/projects/-Users-nils-Desktop-projects-thaw/memory/` — persistent memory across sessions
