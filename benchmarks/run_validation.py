"""
run_validation.py — multi-model validation orchestrator for thaw.

Drives `python/vllm_demo.py` as a subprocess over a matrix of models, captures
the per-run JSON report each demo emits, and aggregates them into a single
validation report with median / stdev / min / max / CoV for every timing.

Rationale
---------
A single end-to-end measurement is a data point; a benchmark is a distribution.
YC investors and engineers both know the difference. Publishing a median over
N ≥ 3 runs with the coefficient of variation attached is what separates
"we ran it once" from "this is a repeatable number." If the CoV on restore
time exceeds `--cov-threshold` (default 10%), we flag the result as unstable
so we don't cite it in marketing copy.

Per-run isolation: each run is its own subprocess, so the Python interpreter,
vLLM engine, CUDA context, and page cache state are all fresh — no leaked
tensors or accidental caching across iterations. Between runs we explicitly
drop the snapshot path so Phase 3 of every run is a real NVMe cold restore.

Output layout
-------------
    <out-dir>/
      aggregate.json           # schema_version, host fingerprint, matrix, stats
      runs/
        <model>-<i>.json       # raw per-run JSON from vllm_demo --json-out
        <model>-<i>.log        # stdout/stderr for postmortem
      report.md                # human-readable table

Usage
-----
    python benchmarks/run_validation.py \
        --models meta-llama/Meta-Llama-3-8B mistralai/Mistral-7B-v0.3 \
        --runs 3 --out-dir ./val_$(date +%Y%m%d_%H%M)

The default matrix targets the "three sizes on one H100" plan.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mistral-7B-v0.3",
    "Qwen/Qwen2.5-14B",
]


# Named presets for the vLLM RFC #34303 evidence push.
#
# Each preset is an ordered list of (model_id, tp, gpu_mem_util) tuples. A
# preset overrides --models/--tp/--gpu-mem-util on the command line; everything
# else (--runs, --out-dir, --snapshot) still applies uniformly.
#
# rfc-tier1 — the minimal set the RFC cares about: a TP=1 small, a TP=1
# mid (matching @fergusfinn's H200 table spirit), and a TP=2 large to
# prove multi-GPU. Enough for a credible "three sizes, three configs"
# claim. Target: 2× H100 80 GB pod, ~3 hours of runtime at N=5.
#
# rfc-tier2 — adds FP8 (Qwen3-14B-FP8) and a dense 70B distill to stress
# the quantized weight-swizzling edge cases that L2 sleep mode skips.
# The fp8/fp4-correctness-on-restore gap is exactly what the RFC
# comment advertises, so we need the receipt.
#
# rfc-full — full matrix; ~8 hours at N=5 on 2×H100. Run once right
# before the RFC reply; don't run during development.
PRESETS: dict[str, list[tuple[str, int, float | None]]] = {
    "rfc-tier1": [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 1, None),
        ("Qwen/Qwen2.5-32B-Instruct",             1, 0.90),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", 2, 0.90),
    ],
    "rfc-tier2": [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 1, None),
        ("Qwen/Qwen2.5-32B-Instruct",             1, 0.90),
        ("Qwen/Qwen3-14B-FP8",                    1, None),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", 2, 0.90),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 2, 0.90),
    ],
    "rfc-full": [
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 1, None),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", 2, None),
        ("Qwen/Qwen2.5-32B-Instruct",             1, 0.90),
        ("Qwen/Qwen2.5-32B-Instruct",             2, 0.90),
        ("Qwen/Qwen3-14B-FP8",                    1, None),
        ("mistralai/Mistral-7B-v0.3",             1, None),
        ("meta-llama/Meta-Llama-3.1-70B-Instruct", 2, 0.90),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", 2, 0.90),
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=None,
                   help="Hugging Face model IDs to validate. Ignored if "
                        "--preset is set.")
    p.add_argument("--preset", choices=sorted(PRESETS.keys()), default=None,
                   help="Named preset of (model, tp, gpu_mem_util) tuples. "
                        "Overrides --models/--tp/--gpu-mem-util. Use "
                        "rfc-tier1 for the minimal RFC evidence push, "
                        "rfc-tier2 for FP8 + dense-70B-distill coverage, "
                        "rfc-full for the exhaustive matrix.")
    p.add_argument("--runs", type=int, default=3,
                   help="Runs per model. N=3 is the minimum for a median; "
                        "bump to 5 when a result is CoV-flagged.")
    p.add_argument("--out-dir", required=True,
                   help="Directory to write aggregate.json, report.md, and per-run files.")
    p.add_argument("--snapshot", default="/tmp/thaw_val.thaw",
                   help="Temporary snapshot path reused across runs (flushed between).")
    p.add_argument("--demo-path", default=None,
                   help="Path to vllm_demo.py. Defaults to <repo>/python/vllm_demo.py.")
    p.add_argument("--cov-threshold", type=float, default=0.10,
                   help="Flag timings whose coefficient of variation exceeds this ratio.")
    p.add_argument("--skip-warm", action="store_true",
                   help="Forward --skip-warm to vllm_demo (skips Phase 3b).")
    p.add_argument("--tp", type=int, default=1,
                   help="tensor_parallel_size forwarded to vllm_demo. Use 2 for "
                        "multi-GPU revalidation (70B TP=2, etc).")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="gpu_memory_utilization forwarded to vllm_demo. "
                        "Raise to 0.85+ for large models at TP=2.")
    p.add_argument("--extra-arg", action="append", default=[],
                   help="Extra argument passed through to vllm_demo, e.g. --extra-arg=--profile-init.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned subprocess invocations and exit.")
    return p.parse_args()


def find_demo_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    candidate = here.parent / "python" / "vllm_demo.py"
    if not candidate.exists():
        raise SystemExit(f"vllm_demo.py not found at {candidate}; pass --demo-path")
    return candidate


def model_slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def _rank_sibling_paths(base: str) -> list[str]:
    """Sibling files a TP>1 freeze writes alongside base (`weights.rank1.thaw`,
    etc). Returned as a glob-match against the same directory."""
    import glob
    root, ext = os.path.splitext(base)
    return sorted(glob.glob(f"{root}.rank*{ext}"))


def stats_of(values: list[float]) -> dict:
    """Median-first stats. Median is robust to a single bad outlier;
    mean can be dragged around by a lone slow run (kernel paging, NVMe
    hiccup, noisy neighbor). We still report mean for completeness."""
    if not values:
        return {"n": 0}
    n = len(values)
    median = statistics.median(values)
    out = {
        "n": n,
        "median": median,
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }
    if n >= 2:
        stdev = statistics.stdev(values)
        out["stdev"] = stdev
        out["cov"] = stdev / median if median else None
    return out


# Keys we want median/stdev for. Every entry is (phase, field, human_label).
METRIC_KEYS: list[tuple[str, str, str]] = [
    ("normal_cold_start", "elapsed_s",                 "normal_cold_start_s"),
    ("freeze",            "elapsed_s",                 "freeze_s"),
    ("freeze",            "throughput_gb_s",           "freeze_gb_s"),
    ("thaw_cold_nvme",    "total_s",                   "thaw_cold_total_s"),
    ("thaw_cold_nvme",    "restore_s",                 "thaw_cold_restore_s"),
    ("thaw_cold_nvme",    "restore_throughput_gb_s",   "thaw_cold_restore_gb_s"),
    ("thaw_warm_cache",   "total_s",                   "thaw_warm_total_s"),
    ("thaw_warm_cache",   "restore_throughput_gb_s",   "thaw_warm_restore_gb_s"),
    ("thaw_prestaged_ram","total_s",                   "thaw_ram_total_s"),
    ("thaw_prestaged_ram","dma_throughput_gb_s",       "thaw_ram_dma_gb_s"),
]


def extract_metric(report: dict, phase: str, field: str) -> float | None:
    phases = report.get("phases", {})
    node = phases.get(phase)
    if not isinstance(node, dict):
        return None
    val = node.get(field)
    return float(val) if isinstance(val, (int, float)) else None


def summarize_model(runs: list[dict], cov_threshold: float) -> dict:
    summary: dict = {"num_runs": len(runs), "metrics": {}, "flags": []}

    # Bit-identity rollup: every run's summary.all_outputs_match_ref must be True.
    all_match = [bool(r.get("summary", {}).get("all_outputs_match_ref", False)) for r in runs]
    summary["all_outputs_bit_identical"] = all(all_match) if all_match else False
    if not summary["all_outputs_bit_identical"]:
        summary["flags"].append("bit-identity failed on at least one run")

    # Speedup derived from cold NVMe total — the headline number.
    speedups = [r.get("summary", {}).get("speedup_cold_nvme") for r in runs]
    speedups = [float(s) for s in speedups if isinstance(s, (int, float))]
    summary["metrics"]["speedup_cold_nvme"] = stats_of(speedups)

    for phase, field, label in METRIC_KEYS:
        vals = [extract_metric(r, phase, field) for r in runs]
        vals = [v for v in vals if v is not None]
        s = stats_of(vals)
        summary["metrics"][label] = s
        cov = s.get("cov")
        if cov is not None and cov > cov_threshold:
            summary["flags"].append(
                f"{label}: CoV={cov:.1%} exceeds threshold {cov_threshold:.0%} "
                f"(median={s['median']:.3f}, stdev={s['stdev']:.3f})"
            )
    return summary


def render_report(agg: dict) -> str:
    lines: list[str] = []
    host = agg.get("host", {})
    lines.append(f"# thaw validation report — {agg['timestamp']}\n")
    lines.append("## Host")
    lines.append(f"- GPU: {host.get('gpu','?')} x{host.get('gpu_count','?')} "
                 f"({host.get('gpu_mem_gb','?')} GB, cap {host.get('gpu_capability','?')})")
    lines.append(f"- Driver: {host.get('nvidia_driver','?')}, CUDA: {host.get('cuda','?')}, "
                 f"torch: {host.get('torch','?')}")
    lines.append(f"- OS: {host.get('os','?')} ({host.get('arch','?')}), "
                 f"Python {host.get('python','?')}, RAM {host.get('ram_gb','?')} GB")
    lines.append(f"- thaw git: {host.get('thaw_git_sha','?')[:12]} "
                 f"native={host.get('thaw_native_version','?')}")
    lines.append("")

    for model, model_data in agg["models"].items():
        s = model_data["summary"]
        lines.append(f"## {model}  ({s['num_runs']} runs)")
        if s["flags"]:
            lines.append("**Flags:**")
            for f in s["flags"]:
                lines.append(f"- :warning: {f}")
        else:
            lines.append("_all metrics within CoV threshold_")
        lines.append("")
        lines.append(f"- bit-identical outputs across runs: **{s['all_outputs_bit_identical']}**")
        lines.append("")
        lines.append("| metric | median | min | max | stdev | CoV |")
        lines.append("|---|---|---|---|---|---|")
        for label in [m[2] for m in METRIC_KEYS] + ["speedup_cold_nvme"]:
            m = s["metrics"].get(label) or {}
            if not m or m.get("n", 0) == 0:
                continue
            med = m.get("median")
            mn = m.get("min")
            mx = m.get("max")
            sd = m.get("stdev")
            cov = m.get("cov")
            lines.append(
                f"| `{label}` | {med:.3f} | {mn:.3f} | {mx:.3f} | "
                f"{'—' if sd is None else f'{sd:.3f}'} | "
                f"{'—' if cov is None else f'{cov:.1%}'} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def run_once(demo: Path, model: str, snapshot: str, json_out: Path,
             log_path: Path, skip_warm: bool, tp: int,
             gpu_mem_util: float | None, extra: list[str]) -> int:
    cmd = [sys.executable, str(demo),
           "--model", model,
           "--snapshot", snapshot,
           "--json-out", str(json_out),
           "--tp", str(tp)]
    if gpu_mem_util is not None:
        cmd += ["--gpu-mem-util", str(gpu_mem_util)]
    if skip_warm:
        cmd.append("--skip-warm")
    cmd.extend(extra)
    print(f"[run] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    with log_path.open("w") as log_fh:
        proc = subprocess.run(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc.returncode


def _expand_matrix(args) -> list[tuple[str, int, float | None]]:
    """Resolve (model, tp, gpu_mem_util) matrix from preset or CLI flags."""
    if args.preset:
        return list(PRESETS[args.preset])
    models = args.models or DEFAULT_MODELS
    return [(m, args.tp, args.gpu_mem_util) for m in models]


def main():
    args = parse_args()
    demo = find_demo_path(args.demo_path)

    out_dir = Path(args.out_dir).resolve()
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    matrix = _expand_matrix(args)

    # Pre-resolve host fingerprint by stealing vllm_demo.hardware_fingerprint.
    # Import lazily so --dry-run works without torch installed.
    host_fp: dict = {}
    if not args.dry_run:
        sys.path.insert(0, str(demo.parent))
        try:
            from vllm_demo import hardware_fingerprint  # type: ignore
            host_fp = hardware_fingerprint()
        except Exception as e:
            host_fp = {"probe_error": repr(e)}

    agg = {
        "schema_version": 2,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "host": host_fp,
        "matrix": {
            "preset": args.preset,
            "entries": [
                {"model": m, "tp": t, "gpu_mem_util": g}
                for (m, t, g) in matrix
            ],
            "runs_per_entry": args.runs,
            "snapshot_path": args.snapshot,
            "skip_warm": args.skip_warm,
            "extra_args": args.extra_arg,
            "cov_threshold": args.cov_threshold,
        },
        "models": {},
    }

    if args.dry_run:
        for model, tp, gmu in matrix:
            for i in range(args.runs):
                slug = f"{model_slug(model)}__tp{tp}"
                jp = runs_dir / f"{slug}-{i+1}.json"
                gmu_s = "-" if gmu is None else f"{gmu:.2f}"
                print(f"[dry-run] {model} tp={tp} gmu={gmu_s} "
                      f"run {i+1}/{args.runs} -> {jp}")
        return

    for model, tp, gmu in matrix:
        # Key the aggregate by "model @ tp=N" so a preset that includes the
        # same model at multiple TP degrees (rfc-full) produces two rows.
        key = f"{model} @ tp={tp}"
        slug = f"{model_slug(model)}__tp{tp}"
        reports: list[dict] = []
        failures: list[dict] = []

        for i in range(args.runs):
            # Flush prior snapshot so freeze re-measures honestly and cold
            # restore sees no pre-existing file. TP>1 also writes rank-suffixed
            # siblings (weights.rank1.thaw, ...) — remove them too so a stale
            # rank file from the previous run can't shadow a fresh freeze.
            for p in [args.snapshot, *_rank_sibling_paths(args.snapshot)]:
                if os.path.exists(p):
                    os.remove(p)

            jp = runs_dir / f"{slug}-{i+1}.json"
            lp = runs_dir / f"{slug}-{i+1}.log"
            rc = run_once(demo, model, args.snapshot, jp, lp,
                          args.skip_warm, tp, gmu,
                          args.extra_arg)
            if rc != 0 or not jp.exists():
                failures.append({
                    "run": i + 1,
                    "returncode": rc,
                    "log": str(lp),
                    "json_written": jp.exists(),
                })
                print(f"[warn] {key} run {i+1} FAILED rc={rc}. See {lp}",
                      flush=True)
                continue
            try:
                with jp.open() as fh:
                    reports.append(json.load(fh))
            except json.JSONDecodeError as e:
                failures.append({"run": i + 1, "returncode": rc,
                                 "parse_error": str(e), "json_path": str(jp)})

        summary = summarize_model(reports, args.cov_threshold) if reports else {
            "num_runs": 0, "metrics": {}, "flags": ["no successful runs"],
            "all_outputs_bit_identical": False,
        }
        agg["models"][key] = {
            "model": model,
            "tp": tp,
            "gpu_mem_util": gmu,
            "summary": summary,
            "failures": failures,
            "runs": [{"json_path": str(runs_dir / f"{slug}-{i+1}.json")}
                     for i in range(args.runs)],
        }

    with (out_dir / "aggregate.json").open("w") as fh:
        json.dump(agg, fh, indent=2, default=str)
    (out_dir / "report.md").write_text(render_report(agg))
    print(f"\n[done] aggregate: {out_dir / 'aggregate.json'}")
    print(f"[done] report:    {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
