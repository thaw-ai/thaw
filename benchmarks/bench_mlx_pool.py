"""
bench_mlx_pool.py — measure the EnginePool hot-swap path on MLX.

Compares two paths for "model X is requested again after being unloaded":

  baseline: mlx_lm.load(model)
            full path — architecture init + tokenizer + weight load every time

  thaw_mlx via MLXPool:
            one-time warm() pays the architecture cost; subsequent
            load_snapshot() calls only swap weights into the resident shell

This is the model that maps onto any auto-unload runtime: when a stale
model gets re-requested, you don't want to re-instantiate the architecture
and re-load the tokenizer from disk.

Usage:
    python benchmarks/bench_mlx_pool.py
    python benchmarks/bench_mlx_pool.py --model mlx-community/Devstral-24B-4bit
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "THAW_MLX_BENCH_MODEL",
            "mlx-community/Llama-3.2-1B-Instruct-4bit",
        ),
    )
    p.add_argument("--prompt", default="Write a haiku about Apple Silicon.")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--rounds", type=int, default=4,
                   help="how many hot-swap rounds to time")
    p.add_argument("--snapshot", default=None)
    p.add_argument("--receipt", default=None)
    args = p.parse_args()

    print(f"[pool-bench] model = {args.model}")

    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        import mlx.core as mx
    except ImportError as e:
        print(f"[pool-bench] mlx_lm not installed: {e}")
        sys.exit(2)

    import thaw_mlx
    from thaw_mlx.pool import MLXPool

    snapshot_path = (
        args.snapshot
        or f"/tmp/{args.model.replace('/', '__')}.thaw"
    )

    # --- one-time freeze so we have a snapshot to load ------------------
    print("[pool-bench] freezing baseline weights to snapshot...")
    seed_model, seed_tok = mlx_load(args.model)
    mx.eval(seed_model.parameters())
    baseline_text = mlx_generate(
        seed_model, seed_tok, prompt=args.prompt,
        max_tokens=args.max_tokens, verbose=False,
    )
    freeze_stats = thaw_mlx.freeze(seed_model, snapshot_path)
    print(f"[pool-bench]   freeze: {freeze_stats['elapsed_s']:.2f}s "
          f"({freeze_stats['throughput_gb_s']:.2f} GB/s)")
    del seed_model, seed_tok

    # --- baseline: cold mlx_lm.load each round --------------------------
    print(f"[pool-bench] baseline mlx_lm.load × {args.rounds} ...")
    baseline_rounds = []
    for r in range(args.rounds):
        t0 = time.perf_counter()
        m, t = mlx_load(args.model)
        mx.eval(m.parameters())
        elapsed = time.perf_counter() - t0
        baseline_rounds.append(elapsed)
        print(f"[pool-bench]   round {r+1}: {elapsed*1000:.0f} ms")
        del m, t

    # --- thaw_mlx pool: one warm, then N hot-swaps ----------------------
    print(f"[pool-bench] MLXPool warm + {args.rounds} hot-swaps ...")
    pool = MLXPool()
    t0 = time.perf_counter()
    pool.warm(args.model)
    warm_s = time.perf_counter() - t0
    print(f"[pool-bench]   warm: {warm_s*1000:.0f} ms (one-time)")

    pool_rounds = []
    pool_text = None
    for r in range(args.rounds):
        # Force a re-load even when the shell already holds this snapshot:
        # bypass the no-op shortcut in load_snapshot for honest timing.
        entry = pool._shells[args.model]
        entry.current_snapshot = None

        t0 = time.perf_counter()
        m, t = pool.load_snapshot(snapshot_path, arch_template=args.model)
        elapsed = time.perf_counter() - t0
        pool_rounds.append(elapsed)
        print(f"[pool-bench]   round {r+1}: {elapsed*1000:.0f} ms")
        if r == 0:
            pool_text = mlx_generate(
                m, t, prompt=args.prompt,
                max_tokens=args.max_tokens, verbose=False,
            )

    bit_identical = pool_text == baseline_text
    print(f"[pool-bench] generation parity: {'PASS' if bit_identical else 'FAIL'}")
    if not bit_identical:
        print(f"[pool-bench]   baseline: {baseline_text!r}")
        print(f"[pool-bench]   pool:     {pool_text!r}")

    # --- summary --------------------------------------------------------
    baseline_median = sorted(baseline_rounds)[len(baseline_rounds) // 2]
    pool_median = sorted(pool_rounds)[len(pool_rounds) // 2]
    speedup = baseline_median / pool_median if pool_median > 0 else float("inf")

    print()
    print("=" * 60)
    print(f"  baseline mlx_lm.load (median)    : {baseline_median*1000:.0f} ms")
    print(f"  MLXPool warm (one-time)          : {warm_s*1000:.0f} ms")
    print(f"  MLXPool hot-swap (median)        : {pool_median*1000:.0f} ms")
    print(f"  hot-swap speedup vs baseline     : {speedup:.1f}×")
    print(f"  amortization break-even at round : "
          f"{int((warm_s + pool_median) / max(baseline_median, 1e-6)) + 1}")
    print("=" * 60)

    receipt = {
        "timestamp_utc": _now_iso(),
        "platform": {
            "uname": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else str(os.uname()),
            "python": sys.version.split()[0],
        },
        "model": args.model,
        "snapshot_path": snapshot_path,
        "rounds": args.rounds,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "freeze_stats": freeze_stats,
        "baseline_mlx_lm_load_s": baseline_rounds,
        "baseline_median_s": baseline_median,
        "pool_warm_s": warm_s,
        "pool_hot_swap_s": pool_rounds,
        "pool_median_s": pool_median,
        "hot_swap_speedup_x": speedup,
        "bit_identical_generation": bit_identical,
        "baseline_text": baseline_text,
        "pool_text": pool_text,
    }

    receipt_path = args.receipt or str(
        REPO_ROOT / "site" / "receipts"
        / f"{datetime.now().strftime('%Y-%m-%d')}_mlx_pool_"
          f"{args.model.replace('/', '__')}.json"
    )
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[pool-bench] receipt -> {receipt_path}")

    sys.exit(0 if bit_identical else 1)


if __name__ == "__main__":
    main()
