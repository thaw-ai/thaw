"""
validate_hotswap.py — multi-size slot-warm hot-swap validator.

`bench_slot_warm.py` proves the slot-warm fast path works for one model on
one pod: after a one-time cudaHostRegister pin, subsequent loads hit PCIe
throughput. This script extends the measurement across model sizes and
emits a structured JSON report so `run_validation.py` can aggregate.

EnginePool's dummy-init slot is tied to one base architecture, so we spin
up one pool per model rather than swapping across architectures (which
would corrupt weights). Within each pool we cycle the registered snapshot
through the slot N times: load 0 is the pin cost (discarded), loads 1..N-1
are the steady-state numbers we report.

Why this matters for the YC claim
---------------------------------
A single 55 GB/s measurement on one 8B checkpoint on one H100 is a data
point. What an investor needs to believe "LLMs hot-swap in under a second"
is scales up: that the same pipeline sustains near-PCIe throughput at 13B,
34B, 70B — not just the size that fits easily in slot memory.

Inputs
------
    --models       list of "<label>:<snapshot_path>" (label is a short id;
                   snapshot is a path or s3:// URI).
    --base-model   the vLLM model id for dummy init (must match the
                   architecture the snapshots were frozen from).
    --iterations   total loads per model including warmup (default 5;
                   load 0 is warmup and excluded from stats).
    --json-out     path to write the structured report.
    --tp           tensor_parallel_size (default 1).

Example
-------
    python benchmarks/validate_hotswap.py \
        --base-model meta-llama/Meta-Llama-3-8B \
        --models \
            finetuneA:/snaps/ft_a.thaw \
            finetuneB:/snaps/ft_b.thaw \
            finetuneC:/snaps/ft_c.thaw \
        --iterations 6 \
        --json-out /tmp/hotswap.json

Outputs median / min / max / stdev / CoV per model over loads 1..N-1, plus
whole-run aggregate statistics. If CoV on swap time exceeds 10% the report
is flagged so the number doesn't get cited as stable.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def parse_model_spec(spec: str) -> tuple[str, str]:
    """Parse a 'label:path' spec. Label may not contain colons; path may
    contain them (e.g. 's3://bucket/key'). Splits on the first colon."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"model spec '{spec}' must be 'label:path' (e.g. 'ft_a:/tmp/a.thaw')"
        )
    label, path = spec.split(":", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError(f"empty label or path in '{spec}'")
    return label, path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base-model", required=True,
                   help="vLLM model id used to dummy-init the pool engine.")
    p.add_argument("--models", type=parse_model_spec, nargs="+", required=True,
                   help="'label:snapshot_path' entries — all same arch as base-model.")
    p.add_argument("--iterations", type=int, default=5,
                   help="Loads per model (first is warmup, excluded from stats).")
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size.")
    p.add_argument("--gpu-mem-util", type=float, default=0.25)
    p.add_argument("--json-out", default=None,
                   help="Path to write structured JSON report.")
    p.add_argument("--cov-threshold", type=float, default=0.10)
    return p.parse_args()


def stats_of(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    n = len(values)
    median = statistics.median(values)
    out = {"n": n, "median": median, "min": min(values),
           "max": max(values), "mean": statistics.fmean(values)}
    if n >= 2:
        stdev = statistics.stdev(values)
        out["stdev"] = stdev
        out["cov"] = stdev / median if median else None
    return out


def main():
    args = parse_args()

    # Import lazily so --help works without torch installed.
    from thaw_vllm.pool import EnginePool
    try:
        from vllm_demo import hardware_fingerprint  # reach into python/ if on path
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
        try:
            from vllm_demo import hardware_fingerprint
        except ImportError:
            def hardware_fingerprint():
                return {"error": "vllm_demo.hardware_fingerprint unavailable"}

    results = {
        "schema_version": 1,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": hardware_fingerprint(),
        "config": {
            "base_model": args.base_model,
            "models": [{"label": l, "path": p} for l, p in args.models],
            "iterations": args.iterations,
            "tp": args.tp,
            "gpu_mem_util": args.gpu_mem_util,
            "cov_threshold": args.cov_threshold,
        },
        "per_model": {},
        "status": "started",
    }

    def _dump():
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))

    try:
        print(f"[init] EnginePool base_model={args.base_model} tp={args.tp}", flush=True)
        t0 = time.perf_counter()
        pool = EnginePool()
        pool.init_pool(
            args.base_model,
            pool_size=1,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem_util,
            enforce_eager=True,
            dtype="float16",
        )
        init_s = time.perf_counter() - t0
        print(f"[init] pool ready in {init_s:.1f}s", flush=True)
        results["pool_init_s"] = init_s
        _dump()

        for label, path in args.models:
            pool.register(label, path)

        overall_swap_times: list[float] = []
        overall_throughputs: list[float] = []

        for label, path in args.models:
            print(f"\n=== {label} ({path}) ===", flush=True)
            per_iter: list[dict] = []
            for i in range(args.iterations):
                t0 = time.perf_counter()
                stats = pool.preload(label, slot_id=0)
                elapsed = time.perf_counter() - t0
                backend = stats.get("backend", "?")
                thr = stats.get("throughput_gb_s", 0.0)
                total_bytes = stats.get("total_bytes", 0)
                is_warmup = (i == 0)
                tag = " (warmup — excluded)" if is_warmup else ""
                print(f"  load {i}: {elapsed:6.2f}s  "
                      f"{thr:5.1f} GB/s  "
                      f"backend={backend}{tag}",
                      flush=True)
                per_iter.append({
                    "iteration": i,
                    "warmup": is_warmup,
                    "elapsed_s": elapsed,
                    "throughput_gb_s": thr,
                    "total_bytes": total_bytes,
                    "backend": backend,
                })
                _dump()

            steady = [r for r in per_iter if not r["warmup"]]
            times = [r["elapsed_s"] for r in steady]
            thrs = [r["throughput_gb_s"] for r in steady]
            overall_swap_times.extend(times)
            overall_throughputs.extend(thrs)

            summary = {
                "path": path,
                "iterations": per_iter,
                "warmup_elapsed_s": per_iter[0]["elapsed_s"] if per_iter else None,
                "steady_elapsed_s": stats_of(times),
                "steady_throughput_gb_s": stats_of(thrs),
                "flags": [],
            }
            cov = summary["steady_elapsed_s"].get("cov")
            if cov is not None and cov > args.cov_threshold:
                summary["flags"].append(
                    f"elapsed CoV={cov:.1%} exceeds threshold {args.cov_threshold:.0%}"
                )
            results["per_model"][label] = summary
            _dump()

        results["overall"] = {
            "steady_elapsed_s": stats_of(overall_swap_times),
            "steady_throughput_gb_s": stats_of(overall_throughputs),
        }
        results["status"] = "ok"
        _dump()

        print("\n" + "=" * 60)
        print("HOT-SWAP SUMMARY")
        print("=" * 60)
        for label, data in results["per_model"].items():
            s = data["steady_elapsed_s"]
            t = data["steady_throughput_gb_s"]
            flags = " ".join(f"[!] {f}" for f in data["flags"])
            print(f"  {label:20s}  "
                  f"median {s.get('median', 0):5.2f}s  "
                  f"{t.get('median', 0):5.1f} GB/s  "
                  f"cov={s.get('cov')}  {flags}")

    except BaseException as e:
        results["status"] = "error"
        results["error"] = f"{type(e).__name__}: {e}"
        _dump()
        raise


if __name__ == "__main__":
    main()
