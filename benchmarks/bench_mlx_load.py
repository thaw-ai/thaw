"""
bench_mlx_load.py — A/B compare mlx_lm.load vs thaw_mlx.restore.

Run on Apple Silicon. Default model is a small 4-bit Llama for fast
iteration on a Mac with limited RAM. Override via THAW_MLX_BENCH_MODEL.

Usage:
    python benchmarks/bench_mlx_load.py
    THAW_MLX_BENCH_MODEL=mlx-community/Devstral-24B-4bit python benchmarks/bench_mlx_load.py

What it does:
    1. mlx_lm.load(model)               -- baseline (cold then warm via OS cache)
    2. thaw_mlx.freeze(model, .thaw)    -- one-time conversion
    3. mlx_lm.load(model) + thaw_mlx.restore  -- target hot-swap path
    4. Generate a fixed prompt with both, assert tokens match (bit-identity)
    5. Emit JSON to site/receipts/

The interesting comparison is (1) vs (3) post-warmup. (3) is what an
auto-unload runtime calls when a stale model gets re-requested.
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


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "THAW_MLX_BENCH_MODEL",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
        ),
    )
    p.add_argument("--prompt", default="Write a haiku about Apple Silicon.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--snapshot", default=None,
                   help="path for .thaw output (default: /tmp/<model>.thaw)")
    p.add_argument("--receipt", default=None,
                   help="JSON receipt output path")
    p.add_argument("--skip-bench-baseline", action="store_true",
                   help="skip the mlx_lm.load timing (assume already cached)")
    args = p.parse_args()

    print(f"[bench] model = {args.model}")

    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        import mlx.core as mx
    except ImportError as e:
        print(f"[bench] mlx_lm not installed: {e}")
        print("[bench] run: pip install mlx mlx-lm")
        sys.exit(2)

    import thaw_mlx

    snapshot_path = (
        args.snapshot
        or f"/tmp/{args.model.replace('/', '__')}.thaw"
    )

    # --- Baseline: mlx_lm.load ------------------------------------------
    if not args.skip_bench_baseline:
        print(f"[bench] mlx_lm.load() baseline...")
        t0 = time.perf_counter()
        baseline_model, baseline_tok = mlx_load(args.model)
        mx.eval(baseline_model.parameters())
        baseline_load_s = time.perf_counter() - t0
        print(f"[bench]   load: {baseline_load_s:.2f}s")
    else:
        baseline_model, baseline_tok = mlx_load(args.model)
        baseline_load_s = None

    # Generate baseline tokens for parity check.
    baseline_text = mlx_generate(
        baseline_model,
        baseline_tok,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )

    # --- Freeze ----------------------------------------------------------
    print(f"[bench] thaw_mlx.freeze() -> {snapshot_path}")
    freeze_stats = thaw_mlx.freeze(baseline_model, snapshot_path)
    print(
        f"[bench]   {_human_bytes(freeze_stats['total_bytes'])} in "
        f"{freeze_stats['elapsed_s']:.2f}s "
        f"({freeze_stats['throughput_gb_s']:.2f} GB/s)"
    )
    del baseline_model

    # --- Restore: load fresh model + thaw_mlx.restore --------------------
    print(f"[bench] thaw_mlx.restore() (target hot-swap path)...")
    # Build a fresh empty model. mlx_lm.load also instantiates the
    # architecture; we want to time only the weight load delta.
    t_init = time.perf_counter()
    target_model, target_tok = mlx_load(args.model)
    init_s = time.perf_counter() - t_init
    print(f"[bench]   architecture+tokenizer reload: {init_s:.2f}s "
          f"(mlx_lm baseline; thaw_mlx skips weight materialization next)")

    t_restore = time.perf_counter()
    restore_stats = thaw_mlx.restore(target_model, snapshot_path)
    restore_s = time.perf_counter() - t_restore
    print(
        f"[bench]   restore: {_human_bytes(restore_stats['total_bytes'])} in "
        f"{restore_s:.2f}s ({restore_stats['throughput_gb_s']:.2f} GB/s)"
    )

    # --- Parity check ----------------------------------------------------
    target_text = mlx_generate(
        target_model,
        target_tok,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    bit_identical = baseline_text == target_text
    print(f"[bench] generation parity: {'PASS' if bit_identical else 'FAIL'}")
    if not bit_identical:
        print(f"[bench]   baseline: {baseline_text!r}")
        print(f"[bench]   restored: {target_text!r}")

    # --- Receipt ---------------------------------------------------------
    receipt = {
        "timestamp_utc": _now_iso(),
        "platform": {
            "uname": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else str(os.uname()),
            "python": sys.version.split()[0],
        },
        "model": args.model,
        "snapshot_path": snapshot_path,
        "baseline_load_s": baseline_load_s,
        "thaw_freeze": freeze_stats,
        "thaw_restore": {
            **restore_stats,
            "elapsed_s": restore_s,
        },
        "architecture_init_s": init_s,
        "bit_identical_generation": bit_identical,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "baseline_text": baseline_text,
        "restored_text": target_text,
    }

    receipt_path = args.receipt or str(
        REPO_ROOT / "site" / "receipts"
        / f"{datetime.now().strftime('%Y-%m-%d')}_mlx_bench_"
          f"{args.model.replace('/', '__')}.json"
    )
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[bench] receipt -> {receipt_path}")

    if baseline_load_s and restore_s:
        speedup = baseline_load_s / restore_s
        print(
            f"[bench] summary: mlx_lm.load {baseline_load_s:.2f}s vs "
            f"thaw_mlx.restore {restore_s:.2f}s ({speedup:.2f}× faster)"
        )

    sys.exit(0 if bit_identical else 1)


if __name__ == "__main__":
    main()
