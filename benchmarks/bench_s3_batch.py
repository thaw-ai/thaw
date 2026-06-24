#!/usr/bin/env python3
"""
bench_s3_batch.py — measure the two thaw S3 optimizations WITHOUT real AWS.

This benchmark is deliberately network-free and repeatable. It isolates the
two changes in thaw_common.cloud so reviewers can reproduce the wins on a
laptop in seconds:

  Part A — shared client reuse
    The old code built a fresh boto3 client on every download and every
    upload. Building a client resolves credentials (an IMDS round-trip on a
    pod role), loads the S3 service model, and — the part that bites on a
    real network — starts with an EMPTY connection pool, so the first request
    pays a fresh TCP+TLS handshake. We measure the construction overhead saved
    per operation. The connection-reuse half of the win only shows up against
    real S3 (no sockets under moto); see --help for the real-S3 recipe.

  Part B — bounded-concurrency batch transfers
    The old code had no many-object API; a corpus was resolved one file at a
    time. resolve_snapshots() fans out across files with a bounded pool. We
    inject a synthetic per-object latency (a sleeping stand-in for S3 request
    latency) and compare a sequential loop against resolve_snapshots() at
    several file-concurrency settings, reporting wall time and speedup.

Examples:
  # both parts, default sizes, write a receipt
  python benchmarks/bench_s3_batch.py --json-out site/receipts/2026-06-24_s3_batch.json

  # sweep corpus size and file concurrency for Part B
  python benchmarks/bench_s3_batch.py --objects 16 64 256 --file-concurrency 1 8 16 32 --latency-ms 40

Real-S3 batch benchmark (requires credentials + a bucket of N objects):
  Use resolve_snapshots() directly against s3:// URIs and time it; compare
  against a Python loop over resolve_snapshot_path(). The synthetic latency
  here models the request-latency floor that dominates many-small-object
  corpora — the regime where file concurrency, not per-file ranging, wins.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Hermetic credentials so client construction never blocks on the IMDS
# credential probe. These are never used to talk to AWS in this benchmark.
os.environ.setdefault("AWS_ACCESS_KEY_ID", "bench")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "bench")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_EC2_METADATA_DISABLED", "true")


def _load_cloud():
    """Load thaw_common.cloud as a standalone module (no torch import)."""
    cloud_path = os.path.join(
        os.path.dirname(__file__), "..", "python", "thaw_common", "cloud.py"
    )
    spec = importlib.util.spec_from_file_location("_thaw_cloud_batchbench", cloud_path)
    cloud = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cloud)
    return cloud


def bench_client_reuse(cloud, ops: int, repeats: int) -> dict:
    """Part A: per-op client construction overhead, build-per-op vs cached."""
    # Warm import + service-model cache once so we measure steady-state cost.
    cloud.reset_s3_client_cache()
    cloud._build_s3_client(36)

    build_times = []
    for _ in range(repeats):
        cloud.reset_s3_client_cache()
        t0 = time.perf_counter()
        for _ in range(ops):
            # Old behavior: a brand-new client per operation.
            cloud._build_s3_client(36)
        build_times.append(time.perf_counter() - t0)

    cached_times = []
    for _ in range(repeats):
        cloud.reset_s3_client_cache()
        t0 = time.perf_counter()
        for _ in range(ops):
            # New behavior: cached client, built once, reused thereafter.
            cloud._s3_client()
        cached_times.append(time.perf_counter() - t0)

    build_total = statistics.median(build_times)
    cached_total = statistics.median(cached_times)
    return {
        "ops": ops,
        "build_per_op_ms": round(build_total / ops * 1000, 4),
        "cached_per_op_ms": round(cached_total / ops * 1000, 4),
        "build_total_ms": round(build_total * 1000, 2),
        "cached_total_ms": round(cached_total * 1000, 2),
        "construction_ms_saved_total": round((build_total - cached_total) * 1000, 2),
        "note": (
            "Construction overhead only. On real S3 each rebuilt client also "
            "pays a fresh TCP+TLS handshake on its first request; reuse keeps "
            "keep-alive connections warm across operations."
        ),
    }


def bench_batch_concurrency(
    cloud, n_objects: int, file_concurrencies: List[int], latency_ms: float, repeats: int
) -> dict:
    """Part B: sequential vs resolve_snapshots() with synthetic per-object latency."""
    latency = latency_ms / 1000.0
    uris = [f"s3://bench-bucket/obj-{i:05d}.thaw" for i in range(n_objects)]

    # Replace the single-file resolver with a sleeping stand-in for one S3
    # GET's request latency. resolve_snapshots()'s REAL executor/ordering
    # logic is exercised; only the network is simulated.
    orig = cloud.resolve_snapshot_path

    def fake_resolve(uri, cache_dir=None, force=False):
        time.sleep(latency)
        return "/cache/" + uri.rsplit("/", 1)[1]

    cloud.resolve_snapshot_path = fake_resolve
    try:
        # Baseline: the old one-at-a-time pattern.
        seq = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = [fake_resolve(u) for u in uris]
            seq.append(time.perf_counter() - t0)
        seq_med = statistics.median(seq)

        rows = []
        for fc in file_concurrencies:
            runs = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                out = cloud.resolve_snapshots(uris, max_files=fc)
                runs.append(time.perf_counter() - t0)
                assert len(out) == n_objects  # order/return-shape sanity
            med = statistics.median(runs)
            rows.append({
                "file_concurrency": fc,
                "wall_s": round(med, 4),
                "speedup_vs_sequential": round(seq_med / med, 2) if med > 0 else 0,
            })
    finally:
        cloud.resolve_snapshot_path = orig

    return {
        "n_objects": n_objects,
        "latency_ms": latency_ms,
        "sequential_wall_s": round(seq_med, 4),
        "batch": rows,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objects", type=int, nargs="+", default=[64],
                    help="Corpus sizes for Part B (default: 64)")
    ap.add_argument("--file-concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32],
                    help="File-concurrency settings for Part B")
    ap.add_argument("--latency-ms", type=float, default=40.0,
                    help="Synthetic per-object S3 request latency (default: 40ms)")
    ap.add_argument("--reuse-ops", type=int, default=100,
                    help="Operations for the Part A client-reuse measurement")
    ap.add_argument("--repeats", type=int, default=5, help="Repetitions; medians reported")
    ap.add_argument("--json-out", type=str, default=None, help="Write a JSON receipt")
    args = ap.parse_args()

    cloud = _load_cloud()

    print("── Part A: shared client reuse ──")
    a = bench_client_reuse(cloud, args.reuse_ops, args.repeats)
    print(f"  build-per-op : {a['build_per_op_ms']:.3f} ms/op  "
          f"({a['build_total_ms']:.1f} ms for {a['ops']} ops)")
    print(f"  cached       : {a['cached_per_op_ms']:.4f} ms/op  "
          f"({a['cached_total_ms']:.1f} ms for {a['ops']} ops)")
    print(f"  construction saved over {a['ops']} ops: {a['construction_ms_saved_total']:.1f} ms "
          f"(+ one fewer TCP/TLS handshake per op on real S3)")

    print("\n── Part B: bounded-concurrency batch ──")
    part_b = []
    for n in args.objects:
        b = bench_batch_concurrency(cloud, n, args.file_concurrency, args.latency_ms, args.repeats)
        part_b.append(b)
        print(f"  {n} objects @ {args.latency_ms:.0f}ms each — sequential {b['sequential_wall_s']:.3f}s")
        for row in b["batch"]:
            print(f"     file_concurrency={row['file_concurrency']:>3}  "
                  f"{row['wall_s']:.3f}s  ({row['speedup_vs_sequential']:.1f}× vs sequential)")

    if args.json_out:
        out = {
            "schema": "thaw.bench.s3_batch.v1",
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "python": sys.version.split()[0],
            "part_a_client_reuse": a,
            "part_b_batch_concurrency": part_b,
            "params": vars(args),
        }
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  receipt → {args.json_out}")


if __name__ == "__main__":
    main()
