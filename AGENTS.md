# thaw — Project State & Roadmap

*Updated 2026-04-20 (ForkPool shipped, v0.3.1/v0.3.2 on PyPI — first sub-second fork amortization receipt). This file is for Codex sessions — start here before suggesting work.*

## What thaw is

The **fork primitive** for live LLM inference. Snapshot a running session — weights + KV cache + scheduler state + prefix-hash table — and hydrate it into N divergent children that skip prefill and diverge from the fork point. Not a proxy, not a cache-tier — a primitive you call from user code. Open source, pip-installable, works on any CUDA 12+ GPU.

**The hero receipt (2026-04-20, H100 80 GB, Llama-3.1-8B):**
- ForkPool init: **22.3s one-time**, then **0.88s median/round** across 5 rounds × 4 branches × 64 tokens
- Per-round cost: ~340s cold-boot → sub-second (400× amortized)
- All rounds 4/4 divergent, bit-identical at the fork boundary
- Reproduces with `pip install thaw-vllm>=0.3.3 thaw-native>=0.3.1`

**Supporting benchmarks (bit-identical output, re-validated):**
- LangGraph PR-review fan-out (H100 SXM, Llama-3.1-8B, 4 reviewers): 64.55s → 1.43s across rounds 2–3 (`site/receipts/2026-04-21_h100_pr_review_langgraph.json`)
- Slot-warm hot-swap (H100 SXM, Llama-3-8B, post one-time pin): 0.29s / 55 GB/s (PCIe Gen5-saturating)
- 70B TP=2 on 2×H100 (2026-04-19): 74.2s → 33.1s (**2.24×**), restore 8.55 GB/s — needs recoverable fast-path (see *Claims NOT validated*)

**Claims NOT validated / removed from site:**
- 70B 2×A100 546s → 31.8s (17.2×) — prior number, never re-reproduced on current code; recent H100 run shows the regression. Do not cite.
- Llama-3-8B "9.7× / 12.6× / 5.9×" whole-flow speedups — freeze+restore is fast, but the e2e total depended on the specific pod's HF cache state. Cite restore-GB/s instead.
- 9.57 / 19.62 GB/s freeze — pod-specific NVMe ceilings. Re-measure per pod before citing.

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
3. **Multi-GPU tensor parallel** — arbitrary TP via `collective_rpc`; validated bit-identical on 2×H100 + 2×A40 TP=2
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
14. **Launch assets (2026-04-18)** — Three production YouTube videos live (Public): Video 1 hero (75s), Video 2 how-it-works (4m), Video 3 agent fork (2m20s). Site redesigned with Team section (Nils founder, Matt + Karan co-founders), embedded videos, SGLang + S3 features surfaced. First investor inbound same day (details tracked in `plans/2026-04-18_launch-readiness.md`, which is git-ignored).
15. **ForkPool (2026-04-20)** — Pre-warmed subprocess pool: N vLLM workers boot once with real weights, each `fork_completions()` call snapshots KV only. Turns per-fork latency from ~340s cold-boot to 0.88s median/round on H100 8B (`demos/fork_pool_rl.py`, receipt: `site/receipts/2026-04-20_h100_fork_pool_rl.json`). First sub-second fork amortization proof on real hardware.
16. **thaw-native v0.3.1 / thaw-vllm v0.3.2 on PyPI (2026-04-20)** — Shipped the plain-pinned-freeze fix (v0.3.0 wheel capped freeze at 50 MB/s because CPU reads of WC-pinned memory are ~100× slower than plain pinned). Fresh `pip install` now pulls the fast path by default. Site stats band now leads with `0.88s per fork round`.
17. **TP>1 restore cascade fix (2026-04-22)** — `_worker_restore` now tries `restore_model_from_ram` first (matches TP=1 loader.py order), falls back to `restore_model_pipelined`. Two O_DIRECT preads on one NVMe controller caused 14.79→8.55 GB/s regression at TP=2; RAM path uses shared page cache (parallel reads). Also sets `THAW_VERIFY=0` default in the TP>1 path so per-chunk CRC fold doesn't serialize against DMA. Pending re-validation on 2×H100.
18. **Sleep-mode integration + RFC-evidence receipts (2026-04-22)** — `python/thaw_vllm/sleep_mode.py` exposes `sleep(llm, path, *, level=2, strict=True)` / `wake_up(llm, path)` composed with vLLM's native `LLM.sleep(level)` / `LLM.wake_up()` (not a `.to("meta")` hack). Requires `enable_sleep_mode=True` on LLM construction; raises `SleepModeUnavailableError` otherwise (or falls through with `strict=False` for freeze-only durable checkpoints). 8 unit tests pass. **Two 2×H100 SXM receipts** in `site/receipts/2026-04-22_rfc/`: **8B TP=1** — sleep 3.4s, wake 11.1s, bit-identical, CuMemAllocator freed 45.38 GiB. **70B TP=2** — sleep 16.1s (9.04 GB/s aggregate, 141 GB snapshot across 966 regions), wake 53.6s (2.78 GB/s aggregate), bit-identical, CuMemAllocator freed 72.67 GiB per rank = 145 GiB total; vLLM's own wake re-alloc took 0.33s (thaw's restore is the wall-clock). Pod runbook at `private/plans/2026-04-22_pod-runbook.md`; RFC comment draft at `private/plans/2026-04-19_rfc-34303-comment.md` cites only these receipts. rfc-tier1 validation preset abandoned this session — vllm_demo.py Phase 2 leaks GPU memory between LLM() instances in the same process; sleep_mode_demo is the RFC-specific evidence anyway.
19. **rfc-tier1/2/full validation presets (2026-04-22)** — `benchmarks/run_validation.py --preset rfc-tier1` runs Llama-3.1-8B TP=1, Qwen2.5-32B TP=1, Llama-3.1-70B TP=2 at N≥3 with median/CoV/bit-identity and produces one JSON per run + aggregate. `--preset rfc-tier2` adds Qwen3-14B-FP8 (quantized weights — L2 sleep's weak spot) and DeepSeek R1-Distill-70B. Built for the RFC-evidence pod push.

## What's NOT built (the gaps)

| Gap | Why it matters | Owner |
|-----|----------------|-------|
| **Multi-key shard-parallel S3 restore** | `s3://` URIs work end-to-end. Validated 2026-04-23 on EC2 c5n.xlarge (us-east-2, intra-region, 15.3 GiB): boto3 default multipart **133.9 MB/s**, our new parallel ranged-GET in `python/thaw_common/cloud.py` **137.8 MB/s** — both bound by S3's per-object ceiling. Concurrency 8→256 all converge at ~135 MB/s (256 regresses). Prior "67 MB/s" baseline was wrong; boto3.download_file already uses TransferManager. Real unlock is shard-at-freeze-time: split `.thaw` across N S3 keys, fetch in parallel (4 shards ≈ 540 MB/s, 8 shards ≈ 1.1 GB/s). Rust `thaw-cloud` crate alone does NOT fix this — same ceiling binds any client. Receipt: `site/receipts/2026-04-23_ec2_s3_download.json`. | — |
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

Reordered 2026-04-20 per the "Ship over deadline" + "fork primitive positioning" directives. The ForkPool receipt is shipped; everything below is about distribution and lock-in, not more throughput.

**Tier 1 — next 10 days (YC deadline 2026-05-04):**

1. **Public distribution of the ForkPool receipt** — post to vLLM RFC #34303 (draft at `private/plans/2026-04-19_rfc-34303-comment.md`; slot is open and unassigned, no one else has shipped a fork-primitive reference impl), LinkedIn + X with the 0.88s/round number, HN once the RFC lands. Owner: Nils.
2. **RL-lab outreach** — DM HuggingFace async-RL authors (they published "no current async library supports [KV pivot resampling] out of the box" — that's the receipt for public pain), TRL/accelerate maintainers, Ai2 post-training. Ask for 20 min to try the demo on their workload, not a partnership. Owner: Nils.
3. **Fork demo video with ForkPool as punch line** — existing agent-fork video shows one fork; ForkPool tells the amortization story (22.3s → 0.88s). This is the thing that reads as "primitive, not experiment." Owner: Nils.
4. **LangGraph / framework integration** — the fork primitive needs to live inside a framework people already use. LangGraph's `Send`/`Command` semantics map cleanly. Integration = distribution inside every agent team's codebase. Owner: Nils.
5. **Second-GPU ForkPool receipt (A100 or L40S)** — one H100 receipt is "promising." Two GPUs across different pods = "reproducible primitive." Owner: Nils.

**Tier 2 — 1–2 months post-YC:**

6. **vLLM upstream PR** — `vllm.general_plugins` entrypoints + `KVConnectorBase_V1` already exist. The fork primitive can land as a plugin without vLLM core changes. Low effort, high signal.
7. **Shard-at-freeze + shard-parallel restore (Python, ~1 day)** — validated 2026-04-23 that single-key S3 GETs top out at ~135 MB/s regardless of client language. The fix is splitting `.thaw` across N keys during freeze and fetching them in parallel. 4 shards ≈ 540 MB/s, 8 shards ≈ 1.1 GB/s. A Rust `thaw-cloud` crate alone does NOT close this gap. Deprioritized from Tier 1 because the fork pitch doesn't need S3 to land.
8. **MLX port** — Courier/Jackson Oaks as first design partner. GLM 4.5 Air 3min → <30s. Memory note: `project_mlx_port_courier.md`.
9. **GPUDirect Storage** — single-GPU restore → fastsafetensors territory. Matt.
10. **Eliminate vLLM init floor** — last unlock to 10×+ e2e. Harder problem (touches vLLM internals).
11. **LoRA hot-swap** — cheap to build, huge multiplier for fine-tuning customers.

**Tier 3 — the moat (3–6 months):**

12. **Global snapshot CDN** — own the data layer between storage and compute.
13. **Cross-node RDMA restore** — sub-second restore for any model anywhere in the cluster.
14. **Full engine state serialization** — true `fork()` for GPU processes. Zero cold-start endgame.

## Business progression

Open-source library (shipped) → `thaw serve` daemon (shipped) → **fork primitive with sub-second amortization (shipped 2026-04-20)** → framework-layer embed (LangGraph, TRL) → universal memory layer for AI. Positioning memory: `project_thaw_positioning.md` — "thaw makes agent-fork a primitive." Composability is the only uncommoditizing pillar.

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
- `private/plans/2026-04-19_research-synthesis.md` — SOTA + competitive + agent-infra research distilled into moves
- `private/plans/2026-04-19_rfc-34303-comment.md` — RFC comment draft, ready to update with receipt + post
- `site/receipts/2026-04-20_h100_fork_pool_rl.json` — the hero receipt (0.88s/round)
- `demos/fork_pool_rl.py` — the reproducer
- `.Codex/projects/-Users-nils-Desktop-projects-thaw/memory/` — persistent memory across sessions
