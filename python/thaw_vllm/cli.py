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
    """Start a pre-warmed engine pool server with hot model swapping."""
    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    import logging
    import time

    logging.basicConfig(
        level=logging.INFO,
        format="[thaw] %(message)s",
    )
    logger = logging.getLogger("thaw.pool")

    if not os.path.exists(args.snapshot):
        print(f"[thaw] Error: snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    from thaw_vllm.pool import EnginePool, create_pool_app

    pool = EnginePool()
    total_t0 = time.perf_counter()

    # Step 1: Initialize engine pool with dummy weights
    tp = getattr(args, "tensor_parallel", 1)
    pool_size = getattr(args, "pool_size", 1)

    print(f"[thaw] Initializing {pool_size} engine(s) for {args.model}"
          + (f" (TP={tp})" if tp > 1 else "") + "...")

    llm_kwargs = dict(
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    pool.init_pool(
        args.model,
        pool_size=pool_size,
        tensor_parallel_size=tp,
        **llm_kwargs,
    )

    # Step 2: Register and pre-load the initial model
    model_name = args.model
    pool.register(model_name, args.snapshot)

    # Register any additional snapshots passed via --register
    for reg in getattr(args, "register", None) or []:
        name, path = reg.split("=", 1)
        pool.register(name, path)

    print(f"[thaw] Pre-loading '{model_name}' from {args.snapshot}...")
    pool.preload(model_name, slot_id=0)

    # Step 3: Restore KV cache if provided
    if args.kv_snapshot and os.path.exists(args.kv_snapshot):
        from thaw_vllm.kv_snapshot import restore_kv_cache
        print(f"[thaw] Restoring KV cache from {args.kv_snapshot}...")
        t0 = time.perf_counter()
        kv_stats = restore_kv_cache(pool.slots[0].llm, args.kv_snapshot)
        print(f"[thaw] KV cache: {kv_stats['num_blocks']} blocks in "
              f"{time.perf_counter() - t0:.3f}s")

    total_time = time.perf_counter() - total_t0
    print(f"[thaw] Ready in {total_time:.1f}s "
          f"({pool_size} slot(s), {len(pool.snapshots)} model(s))")

    # Step 4: Serve
    import uvicorn

    app = create_pool_app(pool)
    print(f"[thaw] Serving on http://{args.host}:{args.port}/v1")
    print(f"[thaw] Admin API at http://{args.host}:{args.port}/admin/pool")
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
    p_serve = sub.add_parser("serve", help="Start pre-warmed engine pool server")
    p_serve.add_argument("--model", required=True,
                         help="Base model architecture (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    p_serve.add_argument("--snapshot", required=True, help=".thaw weight snapshot for initial model")
    p_serve.add_argument("--kv-snapshot", default=None, help=".thawkv KV cache snapshot")
    p_serve.add_argument("--pool-size", type=int, default=1,
                         help="Number of pre-warmed engine slots (default: 1)")
    p_serve.add_argument("--register", action="append", metavar="NAME=PATH",
                         help="Register additional model snapshots (e.g. --register finetune-v2=/snapshots/v2.thaw)")
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
