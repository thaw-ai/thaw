"""
thaw CLI — freeze, restore, and serve LLM inference snapshots.

Usage:
    thaw freeze  --model MODEL --output snapshot.thaw [--kv-output kv.thawkv]
    thaw serve   --model MODEL --snapshot snapshot.thaw [--kv-snapshot kv.thawkv]
    thaw info    --snapshot snapshot.thaw
"""

import argparse
import sys


def cmd_freeze(args):
    """Freeze model weights (and optionally KV cache) to snapshot files."""
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import time
    from vllm import LLM, SamplingParams
    from thaw_vllm import freeze_model_pipelined, freeze_model_tp
    from thaw_vllm.kv_snapshot import freeze_kv_cache, freeze_kv_cache_tp

    tp = getattr(args, 'tensor_parallel', 1)

    print(f"[thaw] Loading {args.model}" +
          (f" (tensor_parallel={tp})" if tp > 1 else "") + "...")
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tp,
    )
    load_time = time.perf_counter() - t0
    print(f"[thaw] Model loaded in {load_time:.1f}s")

    # If KV snapshot requested, generate with a warmup prompt to populate prefix cache
    if args.kv_output and args.kv_warmup_prompt:
        print(f"[thaw] Running warmup prompt to populate KV cache...")
        sampling = SamplingParams(temperature=0, max_tokens=1)
        llm.generate([args.kv_warmup_prompt], sampling)

    # Freeze weights
    print(f"[thaw] Freezing weights to {args.output}...")
    if tp > 1:
        wstats = freeze_model_tp(llm, args.output)
        size_gb = wstats['total_bytes'] / 1e9
        print(f"[thaw] Weights: {wstats['num_regions']} regions across {tp} GPUs, "
              f"{size_gb:.2f} GB in {wstats['elapsed_s']:.1f}s "
              f"({wstats['throughput_gb_s']:.2f} GB/s)")
    else:
        engine = llm.llm_engine
        try:
            model = engine.model_executor.driver_worker.model_runner.model
        except AttributeError:
            core = engine.engine_core
            if hasattr(core, 'engine_core'):
                core = core.engine_core
            model = core.model_executor.driver_worker.model_runner.model

        wstats = freeze_model_pipelined(model, args.output)
        size_gb = wstats['total_bytes'] / 1e9
        print(f"[thaw] Weights: {wstats['num_regions']} regions, {size_gb:.2f} GB "
              f"in {wstats['elapsed_s']:.1f}s ({wstats['throughput_gb_s']:.2f} GB/s)")

    # Freeze KV cache
    if args.kv_output:
        print(f"[thaw] Freezing KV cache to {args.kv_output}...")
        if tp > 1:
            kv_stats = freeze_kv_cache_tp(llm, args.kv_output)
        else:
            kv_stats = freeze_kv_cache(llm, args.kv_output)
        print(f"[thaw] KV cache: {kv_stats['num_blocks']} blocks, "
              f"{kv_stats['total_bytes'] / 1e6:.1f} MB in {kv_stats['elapsed_s']:.3f}s")

    print("[thaw] Done.")


def cmd_serve(args):
    """Start an OpenAI-compatible API server with thaw-powered cold start."""
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import time
    import torch
    from vllm import LLM, SamplingParams
    from thaw_vllm import restore_model_pipelined
    from thaw_vllm.kv_snapshot import restore_kv_cache

    if not os.path.exists(args.snapshot):
        print(f"[thaw] Error: snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    total_t0 = time.perf_counter()

    # Step 1: Fast init with dummy weights
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

    # Step 2: Restore weights
    print(f"[thaw] Restoring weights from {args.snapshot}...")
    t0 = time.perf_counter()
    engine = llm.llm_engine
    try:
        model = engine.model_executor.driver_worker.model_runner.model
    except AttributeError:
        core = engine.engine_core
        if hasattr(core, 'engine_core'):
            core = core.engine_core
        model = core.model_executor.driver_worker.model_runner.model

    rstats = restore_model_pipelined(model, args.snapshot)
    weight_time = time.perf_counter() - t0
    print(f"[thaw] Weights: {weight_time:.1f}s ({rstats['throughput_gb_s']:.2f} GB/s)")

    # Step 3: Restore KV cache
    kv_time = 0
    if args.kv_snapshot and os.path.exists(args.kv_snapshot):
        print(f"[thaw] Restoring KV cache from {args.kv_snapshot}...")
        t0 = time.perf_counter()
        kv_stats = restore_kv_cache(llm, args.kv_snapshot)
        kv_time = time.perf_counter() - t0
        print(f"[thaw] KV cache: {kv_stats['num_blocks']} blocks in {kv_time:.3f}s")

    total_time = time.perf_counter() - total_t0
    print(f"[thaw] Ready in {total_time:.1f}s")

    # Step 4: Serve
    from thaw_vllm.server import create_app
    import uvicorn

    app = create_app(llm, args.model)
    print(f"[thaw] Serving on http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_info(args):
    """Print info about a thaw snapshot file."""
    import struct

    with open(args.snapshot, 'rb') as f:
        header = f.read(4096)

    magic = header[0:4]
    if magic == b"THAW":
        version = struct.unpack_from("<I", header, 4)[0]
        num_regions = struct.unpack_from("<Q", header, 8)[0]

        # Read region table
        total_bytes = 0
        with open(args.snapshot, 'rb') as f:
            f.seek(4096)
            for i in range(num_regions):
                entry = f.read(32)
                kind = struct.unpack_from("<I", entry, 0)[0]
                size = struct.unpack_from("<Q", entry, 8)[0]
                total_bytes += size

        import os
        file_size = os.path.getsize(args.snapshot)

        print(f"File:      {args.snapshot}")
        print(f"Format:    THAW v{version}")
        print(f"Regions:   {num_regions}")
        print(f"Data size: {total_bytes / 1e9:.2f} GB")
        print(f"File size: {file_size / 1e9:.2f} GB")

    elif header[0:8] == b"THAWKV\x00\x00":
        import json
        meta_len = struct.unpack_from("<I", header, 8)[0]
        with open(args.snapshot, 'rb') as f:
            f.seek(12)
            metadata = json.loads(f.read(meta_len))

        import os
        file_size = os.path.getsize(args.snapshot)

        print(f"File:        {args.snapshot}")
        print(f"Format:      THAWKV")
        print(f"Blocks:      {metadata.get('num_blocks', 0)}")
        print(f"Layers:      {metadata.get('num_layers', '?')}")
        print(f"Block size:  {metadata.get('block_size', '?')}")
        print(f"Dtype:       {metadata.get('dtype', '?')}")
        print(f"File size:   {file_size / 1e6:.1f} MB")
    else:
        print(f"Unknown format: {magic!r}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="thaw",
        description="Fast snapshot/restore for LLM inference",
    )
    sub = parser.add_subparsers(dest="command")

    # thaw freeze
    p_freeze = sub.add_parser("freeze", help="Freeze model weights + KV cache to snapshot")
    p_freeze.add_argument("--model", required=True)
    p_freeze.add_argument("--output", "-o", required=True, help="Output .thaw file")
    p_freeze.add_argument("--kv-output", default=None, help="Output .thawkv file for KV cache")
    p_freeze.add_argument("--kv-warmup-prompt", default=None, help="Prompt to populate KV cache before freezing")
    p_freeze.add_argument("--dtype", default="float16")
    p_freeze.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p_freeze.add_argument("--tensor-parallel", "-tp", type=int, default=1,
                          help="Tensor parallel size (number of GPUs)")

    # thaw serve
    p_serve = sub.add_parser("serve", help="Start OpenAI-compatible API server")
    p_serve.add_argument("--model", required=True)
    p_serve.add_argument("--snapshot", required=True, help=".thaw weight snapshot")
    p_serve.add_argument("--kv-snapshot", default=None, help=".thawkv KV cache snapshot")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--dtype", default="float16")
    p_serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p_serve.add_argument("--max-model-len", type=int, default=None)
    p_serve.add_argument("--tensor-parallel", "-tp", type=int, default=1,
                         help="Tensor parallel size (number of GPUs)")

    # thaw info
    p_info = sub.add_parser("info", help="Print snapshot file info")
    p_info.add_argument("snapshot", help=".thaw or .thawkv file")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "freeze":
        cmd_freeze(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
