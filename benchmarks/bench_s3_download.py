#!/usr/bin/env python3
"""
bench_s3_download.py — measure thaw S3 download throughput.

Runs `resolve_snapshot_path` against a real s3:// URI and reports wall
time + MB/s. Optionally compares the parallel ranged-GET path (default)
against the legacy boto3.download_file() path for a before/after receipt.

Example:
  # basic: one run with current env settings
  python benchmarks/bench_s3_download.py s3://my-bucket/weights.thaw

  # sweep concurrency
  python benchmarks/bench_s3_download.py s3://my-bucket/weights.thaw \
      --concurrency 8 16 32 64 --part-size-mb 16

  # head-to-head vs legacy boto3.download_file
  python benchmarks/bench_s3_download.py s3://my-bucket/weights.thaw --compare-legacy

  # write a JSON receipt for site/receipts/
  python benchmarks/bench_s3_download.py s3://my-bucket/weights.thaw \
      --json-out site/receipts/2026-04-23_s3_download.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def _legacy_download(uri: str, dest: str) -> None:
    """boto3.download_file — single-stream reference path."""
    import boto3
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    bucket, key = parsed.netloc, parsed.path.lstrip("/")
    boto3.client("s3").download_file(bucket, key, dest)


def _human_rate(bytes_: int, seconds: float) -> str:
    if seconds <= 0:
        return "∞"
    mb = bytes_ / 1024 / 1024
    return f"{mb / seconds:.1f} MB/s"


def _progress_printer():
    last = {"t": time.monotonic(), "bytes": 0}
    start = time.monotonic()

    def cb(done: int, total: int):
        now = time.monotonic()
        if now - last["t"] < 1.0 and done < total:
            return
        delta_b = done - last["bytes"]
        delta_t = now - last["t"]
        inst = _human_rate(delta_b, delta_t) if delta_t > 0 else "—"
        pct = 100.0 * done / total if total else 100.0
        elapsed = now - start
        avg = _human_rate(done, elapsed)
        sys.stderr.write(
            f"\r  [{pct:5.1f}%] {done / 1024 / 1024:7.0f}/{total / 1024 / 1024:.0f} MiB  "
            f"inst {inst}  avg {avg}  elapsed {elapsed:5.1f}s"
        )
        sys.stderr.flush()
        last["t"] = now
        last["bytes"] = done
        if done >= total and total > 0:
            sys.stderr.write("\n")

    return cb


def run_once(
    uri: str,
    *,
    concurrency: Optional[int] = None,
    part_size_mb: Optional[int] = None,
    legacy: bool = False,
    show_progress: bool = True,
) -> dict:
    if concurrency is not None:
        os.environ["THAW_S3_CONCURRENCY"] = str(concurrency)
    if part_size_mb is not None:
        os.environ["THAW_S3_PART_SIZE_MB"] = str(part_size_mb)

    # Load thaw_common.cloud directly (bypass the package __init__.py, which
    # imports torch — not needed for pure S3 benchmarking on a CPU-only box).
    # Re-exec each call so env-var defaults rebind.
    import importlib.util
    cloud_path = os.path.join(
        os.path.dirname(__file__), "..", "python", "thaw_common", "cloud.py"
    )
    spec = importlib.util.spec_from_file_location("_thaw_cloud_bench", cloud_path)
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)

    with tempfile.TemporaryDirectory() as cache_dir:
        t0 = time.monotonic()
        if legacy:
            dest = os.path.join(cache_dir, "legacy.thaw")
            _legacy_download(uri, dest)
            local = dest
        else:
            progress = _progress_printer() if show_progress else None
            local = cloud.resolve_snapshot_path(
                uri, cache_dir=cache_dir, progress=progress
            )
        elapsed = time.monotonic() - t0
        size = os.path.getsize(local)

    return {
        "uri": uri,
        "path": "legacy_download_file" if legacy else "ranged_get_parallel",
        "concurrency": concurrency,
        "part_size_mb": part_size_mb,
        "bytes": size,
        "elapsed_s": round(elapsed, 3),
        "throughput_mb_s": round(size / 1024 / 1024 / elapsed, 2) if elapsed > 0 else 0,
        "throughput_gbps": round(size * 8 / 1e9 / elapsed, 3) if elapsed > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="thaw S3 download benchmark")
    ap.add_argument("uri", help="s3://bucket/key.thaw to download")
    ap.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[None],
        help="Sweep THAW_S3_CONCURRENCY values (default: current env)",
    )
    ap.add_argument(
        "--part-size-mb",
        type=int,
        default=None,
        help="THAW_S3_PART_SIZE_MB (default: current env / 16)",
    )
    ap.add_argument(
        "--compare-legacy",
        action="store_true",
        help="Also run boto3.download_file for a before/after receipt",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repetitions per config (medians reported)",
    )
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write results JSON to this path",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-chunk progress bar",
    )
    args = ap.parse_args()

    results = []

    def _do(label: str, **kwargs):
        print(f"\n▶ {label}")
        runs = []
        for i in range(args.repeats):
            if args.repeats > 1:
                print(f"  run {i + 1}/{args.repeats}")
            r = run_once(args.uri, show_progress=not args.quiet, **kwargs)
            runs.append(r)
            print(
                f"    {r['elapsed_s']:.2f}s · {r['throughput_mb_s']:.1f} MB/s "
                f"· {r['throughput_gbps']:.2f} Gbps · {r['bytes'] / 1024 / 1024:.1f} MiB"
            )
        # median by throughput. Copy to avoid a circular reference when
        # the median entry itself ends up in all_runs.
        runs.sort(key=lambda x: x["throughput_mb_s"])
        median = dict(runs[len(runs) // 2])
        median["label"] = label
        median["all_runs"] = runs
        results.append(median)

    if args.compare_legacy:
        _do("legacy (boto3.download_file)", legacy=True)

    for c in args.concurrency:
        label = (
            f"ranged-GET  concurrency={c or 'env'}  "
            f"part={args.part_size_mb or 'env'}MiB"
        )
        _do(label, concurrency=c, part_size_mb=args.part_size_mb, legacy=False)

    print("\n── Summary ──")
    for r in results:
        print(
            f"  {r['label']:55s}  {r['elapsed_s']:6.2f}s  {r['throughput_mb_s']:7.1f} MB/s"
        )
    if args.compare_legacy and len(results) >= 2:
        legacy = next(r for r in results if r["path"] == "legacy_download_file")
        fastest = max(
            (r for r in results if r["path"] == "ranged_get_parallel"),
            key=lambda x: x["throughput_mb_s"],
        )
        speedup = fastest["throughput_mb_s"] / legacy["throughput_mb_s"] if legacy["throughput_mb_s"] else float("inf")
        print(
            f"\n  Speedup vs legacy boto3.download_file: {speedup:.1f}×  "
            f"({legacy['throughput_mb_s']:.1f} → {fastest['throughput_mb_s']:.1f} MB/s)"
        )

    if args.json_out:
        out = {
            "schema": "thaw.bench.s3_download.v1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "uri": args.uri,
            "repeats": args.repeats,
            "results": results,
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  receipt → {args.json_out}")


if __name__ == "__main__":
    main()
