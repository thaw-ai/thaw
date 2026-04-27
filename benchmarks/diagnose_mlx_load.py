"""
diagnose_mlx_load.py — phase-by-phase timing of mlx_lm.load.

Run on Apple Silicon production hardware to decompose mlx_lm.load into
its underlying phases. Tells you whether your slow load is dominated by
weight I/O, architecture init, tokenizer load, or first-token warmup.

Phases timed:
    1. HF resolve / file presence check
    2. Architecture instantiation (model class + layer construction)
    3. Weight load via mx.load on the cached safetensors file
    4. Tokenizer load
    5. (Optional) first-token warmup latency

Usage:
    pip install mlx mlx-lm
    python diagnose_mlx_load.py --model mlx-community/Devstral-24B-4bit
    python diagnose_mlx_load.py --model mlx-community/GLM-4.5-Air-4bit --no-warmup

Output: a JSON receipt + a one-screen summary.
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _find_safetensors(model_id: str):
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    pattern = f"{hub}/**/*.safetensors"
    hits = glob.glob(pattern, recursive=True)
    canon = model_id.replace("/", "--")
    matches = [p for p in hits if canon in p]
    return sorted(matches)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id (mlx-community/... preferred)")
    p.add_argument("--prompt", default="Hello, world.")
    p.add_argument("--no-warmup", action="store_true",
                   help="skip first-token warmup")
    p.add_argument("--receipt", default=None)
    args = p.parse_args()

    print(f"[diagnose] model = {args.model}")

    try:
        import mlx.core as mx
        from mlx_lm import load as mlx_load, generate as mlx_generate
    except ImportError as e:
        print(f"[diagnose] mlx_lm not installed: {e}")
        sys.exit(2)

    receipt = {
        "timestamp_utc": _now_iso(),
        "platform": str(os.uname()),
        "python": sys.version.split()[0],
        "mlx_version": getattr(mx, "__version__", "unknown"),
        "model": args.model,
    }

    # --- Phase 0: file presence ----------------------------------------
    # If files aren't cached, the first mlx_lm.load triggers an HF
    # download. Time that separately.
    files_before = _find_safetensors(args.model)
    if files_before:
        total_size = sum(os.path.getsize(f) for f in files_before)
        print(f"[diagnose] cached files: {len(files_before)} "
              f"({_human_bytes(total_size)})")
        receipt["files_cached_before"] = True
        receipt["safetensors_files"] = len(files_before)
        receipt["total_safetensors_bytes"] = total_size
    else:
        print("[diagnose] no cached files — first run includes HF download")
        receipt["files_cached_before"] = False

    # --- Phase 1: full mlx_lm.load (the user-visible cost) -------------
    print("[diagnose] phase 1: full mlx_lm.load ...")
    t0 = time.perf_counter()
    model, tokenizer = mlx_load(args.model)
    mx.eval(model.parameters())
    full_load_s = time.perf_counter() - t0
    print(f"[diagnose]   full_load: {full_load_s:.3f}s")
    receipt["full_load_s"] = full_load_s

    # --- Phase 2: re-time individual phases on warm cache --------------
    # mlx_lm.load mostly does: load_config, get_model_class, build model,
    # mx.load weights, set tokenizer. We can decompose:
    files_now = _find_safetensors(args.model)
    if files_now:
        total_size = sum(os.path.getsize(f) for f in files_now)
        receipt["safetensors_files"] = len(files_now)
        receipt["total_safetensors_bytes"] = total_size
        print(f"[diagnose] safetensors now: {len(files_now)} files, "
              f"{_human_bytes(total_size)}")

        # Phase 2a: time raw mx.load on the safetensors file(s).
        print("[diagnose] phase 2a: raw mx.load(safetensors) ...")
        load_times = []
        for run in range(3):
            t0 = time.perf_counter()
            for f in files_now:
                arrays = mx.load(f, format="safetensors")
                mx.eval(list(arrays.values()))
                del arrays
            load_times.append(time.perf_counter() - t0)
        print(f"[diagnose]   warm runs: {[f'{t*1000:.1f}ms' for t in load_times]}")
        receipt["mx_load_warm_s"] = load_times
        receipt["mx_load_throughput_gb_s"] = (
            (total_size / 1e9) / min(load_times) if min(load_times) > 0 else 0
        )

    # Phase 2b: re-instantiate architecture only (skip weight load).
    # We can't easily do this without monkeypatching mlx_lm; instead,
    # call mlx_lm.load again and time the delta vs phase 2a.
    print("[diagnose] phase 2b: warm mlx_lm.load (full) ...")
    del model, tokenizer
    warm_loads = []
    for run in range(2):
        t0 = time.perf_counter()
        model, tokenizer = mlx_load(args.model)
        mx.eval(model.parameters())
        warm_loads.append(time.perf_counter() - t0)
        print(f"[diagnose]   run {run+1}: {warm_loads[-1]:.3f}s")
        if run == 0:
            del model, tokenizer
    receipt["mlx_lm_load_warm_s"] = warm_loads

    # --- Phase 3: warmup token latency ---------------------------------
    if not args.no_warmup:
        print("[diagnose] phase 3: first-token warmup ...")
        t0 = time.perf_counter()
        out = mlx_generate(
            model, tokenizer, prompt=args.prompt, max_tokens=1, verbose=False
        )
        warmup_s = time.perf_counter() - t0
        print(f"[diagnose]   first token: {warmup_s:.3f}s")
        receipt["first_token_s"] = warmup_s
        receipt["first_token_output"] = out

    # --- Summary -------------------------------------------------------
    print()
    print("=" * 60)
    print(f"  full mlx_lm.load (cold)   : {receipt['full_load_s']:.2f}s")
    if "mlx_lm_load_warm_s" in receipt and receipt["mlx_lm_load_warm_s"]:
        print(f"  full mlx_lm.load (warm)   : {min(receipt['mlx_lm_load_warm_s']):.2f}s")
    if "mx_load_warm_s" in receipt:
        mx_warm = min(receipt["mx_load_warm_s"])
        print(f"  raw mx.load (warm)        : {mx_warm*1000:.0f}ms")
        if "mlx_lm_load_warm_s" in receipt:
            arch_overhead = min(receipt["mlx_lm_load_warm_s"]) - mx_warm
            print(f"  architecture+tokenizer    : {arch_overhead*1000:.0f}ms (mlx_lm.load - mx.load)")
    if "first_token_s" in receipt:
        print(f"  first token after load    : {receipt['first_token_s']:.2f}s")
    print("=" * 60)
    print()
    print("Where the time goes is now visible. If `architecture+tokenizer`")
    print("dominates, the moat is NOT faster weight loading.")

    receipt_path = args.receipt or (
        f"diagnose_mlx_{args.model.replace('/', '__')}.json"
    )
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[diagnose] receipt -> {receipt_path}")


if __name__ == "__main__":
    main()
