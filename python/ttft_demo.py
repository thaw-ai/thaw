"""
ttft_demo.py — Time-to-first-token benchmark: thaw KV cache vs cold prefill.

The killer demo: a long system prompt (~2-4K tokens) that would normally
require expensive prefill computation on every cold start. With thaw's
KV cache snapshot, that prefill drops to zero.

Measures:
  - TTFT without KV cache (full prefill required)
  - TTFT with KV cache restored (prefill skipped)
  - Isolates the KV cache benefit from the weight restore benefit

Usage:
    python ttft_demo.py [--model MODEL] [--snapshot PATH] [--kv-snapshot PATH]
"""

import argparse
import gc
import os
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch

# Realistic long system prompt — this is what production deployments look like.
# RAG context, API docs, few-shot examples, safety guidelines, output format.
# Targets ~2000-3000 tokens with Llama-3 tokenizer.
SYSTEM_PROMPT = """\
You are an expert infrastructure assistant for Nebula Cloud Platform. Your role \
is to help engineers diagnose issues, optimize deployments, and manage GPU \
clusters. You have deep knowledge of Kubernetes, CUDA, networking, and \
distributed systems.

## Platform Architecture

Nebula Cloud runs on a heterogeneous GPU fleet:
- **Tier 1 (Training):** 512x NVIDIA H100 SXM 80GB nodes, 8 GPUs per node, \
connected via NVLink 4.0 (900 GB/s bidirectional) and NVSwitch. InfiniBand \
NDR (400 Gbps) for inter-node communication. Each node has 2x Intel Xeon \
8480+ (56 cores), 2TB DDR5, 8x 3.84TB NVMe Gen4 SSDs in RAID-0.
- **Tier 2 (Inference):** 2048x NVIDIA L40S 48GB nodes, 4 GPUs per node, \
PCIe Gen4 x16 interconnect. Each node has 1x AMD EPYC 9654 (96 cores), \
768GB DDR5, 4x 7.68TB NVMe Gen5 SSDs. These nodes run our model serving \
infrastructure with autoscaling from 0 to 2048 nodes.
- **Tier 3 (Development):** 256x NVIDIA A100 80GB nodes, 2 GPUs per node. \
Used for fine-tuning, experimentation, and CI/CD model validation.

## Networking

All nodes are connected via a leaf-spine Clos topology:
- Leaf switches: Arista 7060X5 (400GbE)
- Spine switches: Arista 7800R3 (800GbE)
- Inter-pod links: 4x 400GbE ECMP bundles
- Storage network: Dedicated 100GbE fabric to Ceph and Lustre clusters

GPUDirect RDMA is enabled on Tier 1 nodes for NCCL all-reduce operations. \
We use Weka.io for high-performance shared storage (aggregate 2.4 TB/s \
read throughput) and Ceph for cold object storage.

## Model Serving Stack

Our inference platform uses the following stack:
- **Orchestration:** Kubernetes 1.29 with custom GPU-aware scheduler
- **Serving framework:** vLLM v0.19 with V1 engine, prefix caching enabled
- **Load balancer:** Envoy proxy with custom prefix-aware routing
- **Autoscaler:** Custom controller that scales based on queue depth, GPU \
utilization, and SLA targets (P99 TTFT < 500ms for cached prefixes)
- **Model registry:** MLflow + custom artifact store backed by Weka
- **Monitoring:** Prometheus + Grafana with custom GPU metrics exporter

### Deployment Patterns

Models are deployed as Kubernetes Deployments with the following configuration:
- Each pod runs one vLLM instance with tensor parallelism matching GPU count
- Horizontal scaling via HPA based on custom metrics (request queue depth)
- Cold start target: < 5 seconds for 8B parameter models
- Warm cache hit rate target: > 85% for system prompt prefixes

### Cold Start Optimization

Cold start latency is critical for our autoscaling infrastructure. When a \
new pod starts, it must:
1. Pull the model snapshot from Weka (or local NVMe cache)
2. Initialize the vLLM engine with dummy weights
3. Restore model weights from the thaw snapshot via pipelined DMA
4. Restore KV cache for common system prompts
5. Begin serving requests

Current measured cold start times on Tier 2 (L40S):
- Model weight restore: 5.8s (4.21 GB/s via PCIe Gen4)
- KV cache restore: < 100ms for 4K-token system prompts
- Total pod ready time: < 8s including container startup

## Supported Models

| Model | Parameters | VRAM (fp16) | TP Size | Max Batch | Throughput |
|-------|-----------|-------------|---------|-----------|------------|
| Llama-3-8B | 8B | 16 GB | 1 | 256 | 4,200 tok/s |
| Llama-3-70B | 70B | 140 GB | 4 | 64 | 1,800 tok/s |
| Mixtral-8x7B | 47B | 94 GB | 2 | 128 | 3,100 tok/s |
| Qwen2-72B | 72B | 144 GB | 4 | 64 | 1,650 tok/s |
| Command-R+ | 104B | 208 GB | 8 | 32 | 1,200 tok/s |

## API Response Format

Always respond in the following JSON structure when providing infrastructure \
recommendations:

```json
{
  "diagnosis": "Brief summary of the issue",
  "severity": "critical|high|medium|low",
  "affected_components": ["list", "of", "components"],
  "recommended_actions": [
    {
      "action": "Description of what to do",
      "command": "kubectl command or script if applicable",
      "risk_level": "safe|caution|dangerous",
      "estimated_impact": "Description of expected outcome"
    }
  ],
  "monitoring_queries": [
    "Prometheus PromQL queries to validate the fix"
  ],
  "escalation": "When to escalate and to whom"
}
```

## Safety Guidelines

- NEVER suggest deleting PersistentVolumeClaims without explicit confirmation
- NEVER recommend scaling to zero if there are active serving endpoints
- Always warn before suggesting node cordoning or draining
- GPU memory errors should always be investigated before recommending restarts
- When in doubt about blast radius, recommend a canary deployment first
- All kubectl commands that modify state should include --dry-run=client first

## Few-Shot Examples

### Example 1: GPU Memory Pressure

User: "Pods on node gpu-tier2-0042 are getting OOMKilled"

```json
{
  "diagnosis": "GPU memory exhaustion on inference node, likely due to batch \
size exceeding VRAM capacity or memory fragmentation from long-running sessions",
  "severity": "high",
  "affected_components": ["gpu-tier2-0042", "vllm-serving", "model-llama-3-8b"],
  "recommended_actions": [
    {
      "action": "Check current GPU memory allocation per pod",
      "command": "kubectl exec -n serving $(kubectl get pods -n serving \
-l node=gpu-tier2-0042 -o name | head -1) -- nvidia-smi",
      "risk_level": "safe",
      "estimated_impact": "Diagnostic only, no state change"
    },
    {
      "action": "Reduce max_num_batched_tokens if memory is fragmented",
      "command": "kubectl set env deployment/llama-3-8b-serving \
VLLM_MAX_NUM_BATCHED_TOKENS=8192 --dry-run=client -o yaml",
      "risk_level": "caution",
      "estimated_impact": "May reduce throughput by 10-15% but stabilize memory"
    }
  ],
  "monitoring_queries": [
    "gpu_memory_used_bytes{node='gpu-tier2-0042'} / gpu_memory_total_bytes",
    "rate(container_oom_events_total{node='gpu-tier2-0042'}[5m])"
  ],
  "escalation": "If OOMs persist after batch size reduction, escalate to \
GPU Platform team (#gpu-platform Slack)"
}
```

### Example 2: High Tail Latency

User: "P99 TTFT spiked to 3 seconds on the Llama-3-8B endpoint"

```json
{
  "diagnosis": "TTFT spike indicates prefix cache miss storm, possibly due to \
recent pod scaling event that lost cached KV blocks",
  "severity": "medium",
  "affected_components": ["vllm-serving", "prefix-cache", "autoscaler"],
  "recommended_actions": [
    {
      "action": "Check recent scaling events and cache hit rates",
      "command": "kubectl get events -n serving --sort-by=.lastTimestamp | \
grep -i scale | tail -20",
      "risk_level": "safe",
      "estimated_impact": "Diagnostic only"
    },
    {
      "action": "Verify KV cache snapshots are being restored on new pods",
      "command": "kubectl logs -n serving -l app=llama-3-8b --since=10m | \
grep -i 'kv.*restore\\|prefix.*cache'",
      "risk_level": "safe",
      "estimated_impact": "Confirms thaw KV restore is functioning"
    }
  ],
  "monitoring_queries": [
    "histogram_quantile(0.99, rate(vllm_ttft_seconds_bucket[5m]))",
    "vllm_prefix_cache_hit_rate{model='llama-3-8b'}"
  ],
  "escalation": "If cache hit rate < 50% for > 15 minutes, escalate to \
ML Infrastructure team"
}
```

Remember: you are assisting trained infrastructure engineers. Be precise, \
include specific commands, and always consider the blast radius of any action.\
"""

QUERY = "Three of our L40S inference pods just got OOMKilled simultaneously. \
The autoscaler is trying to replace them but new pods are taking 45 seconds \
to become ready. What should we investigate first?"


def find_model(llm):
    """Extract the nn.Module from a vLLM LLM instance."""
    engine = llm.llm_engine
    try:
        return engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    try:
        core = engine.engine_core
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        return core.model_executor.driver_worker.model_runner.model
    except AttributeError:
        pass
    raise RuntimeError("Could not locate nn.Module in vLLM LLM instance.")


def get_engine_core(llm):
    ec = llm.llm_engine.engine_core
    if hasattr(ec, 'engine_core'):
        ec = ec.engine_core
    return ec


def count_cached_blocks(llm):
    ec = get_engine_core(llm)
    block_pool = ec.scheduler.kv_cache_manager.block_pool
    count = 0
    for block in block_pool.blocks:
        if block._block_hash is not None and not block.is_null:
            count += 1
    return count


def measure_ttft(llm, prompt, sampling):
    """Measure time-to-first-token by generating max_tokens=1."""
    t0 = time.perf_counter()
    out = llm.generate([prompt], sampling)
    elapsed = time.perf_counter() - t0
    return elapsed, out[0].outputs[0].text


def main():
    parser = argparse.ArgumentParser(description="thaw TTFT benchmark")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--snapshot", default="/tmp/ttft_weights.thaw")
    parser.add_argument("--kv-snapshot", default="/tmp/ttft_kv.thawkv")
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined
    from thaw_vllm.kv_snapshot import freeze_kv_cache, restore_kv_cache

    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {QUERY}\nAssistant:"

    # Count tokens for display
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    system_tokens = len(tokenizer.encode(SYSTEM_PROMPT))
    full_tokens = len(tokenizer.encode(full_prompt))
    print(f"System prompt: {system_tokens} tokens")
    print(f"Full prompt:   {full_tokens} tokens")

    # Sampling for TTFT measurement (1 token) and full generation
    ttft_sampling = SamplingParams(temperature=0, max_tokens=1)
    full_sampling = SamplingParams(temperature=0, max_tokens=200)

    # =================================================================
    # Phase 1: Normal cold start + TTFT measurement
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 1: Normal vLLM cold start")
    print("=" * 60)

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.25,
    )
    load_time = time.perf_counter() - t0
    print(f"Model load: {load_time:.1f}s")

    # TTFT = time to generate first token (includes prefill)
    ttft_cold, _ = measure_ttft(llm, full_prompt, ttft_sampling)
    print(f"TTFT (cold, no cache): {ttft_cold:.3f}s")

    # Full generation for reference output
    _, ref_text = measure_ttft(llm, full_prompt, full_sampling)
    print(f"Output: {ref_text[:100]}...")

    # Second call hits prefix cache — measure cached TTFT on warm instance
    ttft_warm, _ = measure_ttft(llm, full_prompt, ttft_sampling)
    print(f"TTFT (warm, cache hit): {ttft_warm:.3f}s")

    cached_blocks = count_cached_blocks(llm)
    print(f"Cached blocks: {cached_blocks}")

    # =================================================================
    # Phase 2: Freeze weights + KV cache
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 2: Freeze weights + KV cache")
    print("=" * 60)

    model = find_model(llm)
    wstats = freeze_model_pipelined(model, args.snapshot)
    kv_stats = freeze_kv_cache(llm, args.kv_snapshot)
    kv_mb = kv_stats['total_bytes'] / 1e6
    print(f"Weights: {wstats['total_bytes'] / 1e9:.2f} GB in {wstats['elapsed_s']:.1f}s")
    print(f"KV cache: {kv_stats['num_blocks']} blocks, {kv_mb:.1f} MB in {kv_stats['elapsed_s']:.3f}s")

    del llm, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    # Phase 3: Thaw cold start WITHOUT KV cache (weights only)
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 3: Thaw cold start — weights only (no KV cache)")
    print("=" * 60)

    t0 = time.perf_counter()
    llm_no_kv = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.25,
        load_format="dummy",
    )
    init_time_no_kv = time.perf_counter() - t0

    model = find_model(llm_no_kv)
    t0 = time.perf_counter()
    restore_model_pipelined(model, args.snapshot)
    weight_time_no_kv = time.perf_counter() - t0

    cold_start_no_kv = init_time_no_kv + weight_time_no_kv
    print(f"Cold start: {cold_start_no_kv:.1f}s (init {init_time_no_kv:.1f}s + weights {weight_time_no_kv:.1f}s)")

    # TTFT still requires full prefill (no KV cache)
    ttft_no_kv, _ = measure_ttft(llm_no_kv, full_prompt, ttft_sampling)
    print(f"TTFT (no KV cache): {ttft_no_kv:.3f}s")

    # Full generation for correctness check
    _, text_no_kv = measure_ttft(llm_no_kv, full_prompt, full_sampling)

    del llm_no_kv, model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    # =================================================================
    # Phase 4: Thaw cold start WITH KV cache
    # =================================================================
    print("\n" + "=" * 60)
    print("Phase 4: Thaw cold start — weights + KV cache")
    print("=" * 60)

    t0 = time.perf_counter()
    llm_with_kv = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        gpu_memory_utilization=0.25,
        load_format="dummy",
    )
    init_time_kv = time.perf_counter() - t0

    model = find_model(llm_with_kv)
    t0 = time.perf_counter()
    restore_model_pipelined(model, args.snapshot)
    weight_time_kv = time.perf_counter() - t0

    t0 = time.perf_counter()
    kv_rstats = restore_kv_cache(llm_with_kv, args.kv_snapshot)
    kv_time = time.perf_counter() - t0

    cold_start_kv = init_time_kv + weight_time_kv + kv_time
    print(f"Cold start: {cold_start_kv:.1f}s (init {init_time_kv:.1f}s + weights {weight_time_kv:.1f}s + KV {kv_time:.3f}s)")
    print(f"KV restored: {kv_rstats['num_blocks']} blocks")
    print(f"Cached blocks: {count_cached_blocks(llm_with_kv)}")

    # TTFT should be faster — prefix cache hit skips prefill
    ttft_with_kv, _ = measure_ttft(llm_with_kv, full_prompt, ttft_sampling)
    print(f"TTFT (with KV cache): {ttft_with_kv:.3f}s")

    # Full generation for correctness
    _, text_with_kv = measure_ttft(llm_with_kv, full_prompt, full_sampling)

    # =================================================================
    # Results
    # =================================================================
    print("\n" + "=" * 60)
    print("RESULTS — Time to First Token")
    print("=" * 60)
    print(f"  System prompt length:     {system_tokens} tokens")
    print(f"  KV cache size:            {kv_mb:.1f} MB ({kv_stats['num_blocks']} blocks)")
    print()
    print(f"  Normal cold start:        {load_time:.1f}s")
    print(f"  Thaw (weights only):      {cold_start_no_kv:.1f}s")
    print(f"  Thaw (weights + KV):      {cold_start_kv:.1f}s")
    print()
    print(f"  TTFT after normal load:   {ttft_cold:.3f}s  (cold, full prefill)")
    print(f"  TTFT after warm cache:    {ttft_warm:.3f}s  (warm, prefix hit)")
    print(f"  TTFT after thaw (no KV):  {ttft_no_kv:.3f}s  (full prefill)")
    print(f"  TTFT after thaw (w/ KV):  {ttft_with_kv:.3f}s  (prefix hit!)")
    print()

    # End-to-end: cold start + TTFT
    e2e_normal = load_time + ttft_cold
    e2e_thaw_no_kv = cold_start_no_kv + ttft_no_kv
    e2e_thaw_kv = cold_start_kv + ttft_with_kv

    print(f"  End-to-end (load + TTFT):")
    print(f"    Normal vLLM:            {e2e_normal:.1f}s")
    print(f"    Thaw (weights only):    {e2e_thaw_no_kv:.1f}s  ({e2e_normal / e2e_thaw_no_kv:.1f}x faster)")
    print(f"    Thaw (weights + KV):    {e2e_thaw_kv:.1f}s  ({e2e_normal / e2e_thaw_kv:.1f}x faster)")
    print()

    prefill_saved = ttft_no_kv - ttft_with_kv
    print(f"  Prefill time saved:       {prefill_saved:.3f}s ({prefill_saved/ttft_no_kv*100:.0f}% of prefill eliminated)")
    print()

    match_no_kv = ref_text == text_no_kv
    match_kv = ref_text == text_with_kv
    print(f"  Output match (no KV):     {'PASS' if match_no_kv else 'FAIL'}")
    print(f"  Output match (with KV):   {'PASS' if match_kv else 'FAIL'}")

    if not match_no_kv:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got:      {text_no_kv[:100]}")
    if not match_kv:
        print(f"    Expected: {ref_text[:100]}")
        print(f"    Got:      {text_with_kv[:100]}")


if __name__ == "__main__":
    main()
