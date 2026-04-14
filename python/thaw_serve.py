"""
thaw_serve.py — OpenAI-compatible API server with thaw-powered cold start.

Starts a vLLM server with dummy weights, restores real weights + KV cache
from thaw snapshots, then serves requests with prefix cache warm.

Usage:
    python thaw_serve.py --model meta-llama/Meta-Llama-3-8B \
        --snapshot weights.thaw \
        --kv-snapshot kv.thawkv \
        --port 8000

    # Then hit the OpenAI-compatible API:
    curl http://localhost:8000/v1/completions -d '{
      "model": "meta-llama/Meta-Llama-3-8B",
      "prompt": "The future of AI is",
      "max_tokens": 50
    }'
"""

import argparse
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


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


def main():
    parser = argparse.ArgumentParser(
        description="thaw-powered vLLM OpenAI-compatible API server"
    )
    parser.add_argument("--model", required=True, help="HuggingFace model name")
    parser.add_argument("--snapshot", required=True, help="Path to .thaw weight snapshot")
    parser.add_argument("--kv-snapshot", default=None, help="Path to .thawkv KV cache snapshot")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.snapshot):
        print(f"Error: snapshot not found: {args.snapshot}")
        sys.exit(1)
    if args.kv_snapshot and not os.path.exists(args.kv_snapshot):
        print(f"Error: KV snapshot not found: {args.kv_snapshot}")
        sys.exit(1)

    import torch
    from vllm import LLM, SamplingParams
    from thaw_vllm import restore_model_pipelined
    from thaw_vllm.kv_snapshot import restore_kv_cache

    total_t0 = time.perf_counter()

    # ── Step 1: Fast init with dummy weights ────────────────────────
    print(f"[thaw] Initializing {args.model} with dummy weights...")
    t0 = time.perf_counter()

    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        load_format="dummy",
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    init_time = time.perf_counter() - t0
    print(f"[thaw] Dummy init: {init_time:.1f}s")

    # ── Step 2: Restore weights from snapshot ───────────────────────
    print(f"[thaw] Restoring weights from {args.snapshot}...")
    t0 = time.perf_counter()
    model = find_model(llm)
    rstats = restore_model_pipelined(model, args.snapshot)
    weight_time = time.perf_counter() - t0
    size_gb = rstats['total_bytes'] / 1e9
    print(f"[thaw] Weight restore: {weight_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s, {size_gb:.1f} GB)")

    # ── Step 3: Restore KV cache (optional) ─────────────────────────
    kv_time = 0
    if args.kv_snapshot:
        print(f"[thaw] Restoring KV cache from {args.kv_snapshot}...")
        t0 = time.perf_counter()
        kv_stats = restore_kv_cache(llm, args.kv_snapshot)
        kv_time = time.perf_counter() - t0
        print(f"[thaw] KV restore: {kv_stats['num_blocks']} blocks, "
              f"{kv_stats['total_bytes'] / 1e6:.1f} MB in {kv_time:.3f}s")

    total_time = time.perf_counter() - total_t0
    print(f"[thaw] Ready in {total_time:.1f}s "
          f"(init {init_time:.1f}s + weights {weight_time:.1f}s + KV {kv_time:.3f}s)")

    # ── Step 4: Serve via OpenAI-compatible API ─────────────────────
    print(f"[thaw] Starting API server on {args.host}:{args.port}")
    print(f"[thaw] OpenAI base URL: http://{args.host}:{args.port}/v1")

    from thaw_vllm.server import create_app
    import uvicorn

    app = create_app(llm, args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
