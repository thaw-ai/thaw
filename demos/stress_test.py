#!/usr/bin/env python3
"""
stress_test.py — comprehensive thaw benchmark across models and configs.

Runs freeze/restore cycles for multiple models, architectures, sizes,
and GPU configs. Outputs structured JSON results + human-readable summary.

Usage:
    # Full suite on 2x A100/H100:
    python demos/stress_test.py

    # Single GPU only:
    python demos/stress_test.py --max-gpus 1

    # Specific models only:
    python demos/stress_test.py --models "mistral-7b,qwen-7b"

    # Quick smoke test (smallest models only):
    python demos/stress_test.py --quick

Designed for RunPod / Lambda / bare metal with 2+ GPUs.
All results go to /workspace/thaw_results/ (or --output-dir).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Model registry ──────────────────────────────────────────

MODELS = {
    # --- Single GPU (fp16, fits in 80GB) ---
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "tp": 1,
        "prompt": "The future of artificial intelligence is",
        "size_gb": 16,
        "arch": "llama",
        "tags": ["quick", "single-gpu"],
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "tp": 1,
        "prompt": "Explain how a CPU works in simple terms:",
        "size_gb": 14,
        "arch": "mistral",
        "tags": ["quick", "single-gpu"],
    },
    "qwen-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "tp": 1,
        "prompt": "What is the difference between TCP and UDP?",
        "size_gb": 15,
        "arch": "qwen2",
        "tags": ["quick", "single-gpu"],
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "tp": 1,
        "prompt": "Write a Python function to check if a number is prime:",
        "size_gb": 7.6,
        "arch": "phi3",
        "tags": ["quick", "single-gpu"],
    },
    "gemma-2-9b": {
        "hf_id": "google/gemma-2-9b-it",
        "tp": 1,
        "prompt": "Describe the process of photosynthesis:",
        "size_gb": 18,
        "arch": "gemma2",
        "tags": ["single-gpu"],
    },

    # --- Multi-GPU (TP=2, needs 2x 80GB) ---
    "llama-3.1-70b": {
        "hf_id": "meta-llama/Llama-3.1-70B-Instruct",
        "tp": 2,
        "prompt": "Explain quantum computing to a five year old:",
        "size_gb": 141,
        "arch": "llama",
        "tags": ["multi-gpu"],
    },
    "qwen-72b": {
        "hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "tp": 2,
        "prompt": "What are the three laws of thermodynamics?",
        "size_gb": 144,
        "arch": "qwen2",
        "tags": ["multi-gpu"],
    },
    "mixtral-8x7b": {
        "hf_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "tp": 2,
        "prompt": "Explain the concept of mixture of experts in neural networks:",
        "size_gb": 87,
        "arch": "mixtral-moe",
        "tags": ["multi-gpu", "moe"],
    },
}


def get_gpu_info():
    """Get GPU count and names."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    gpus = [line.strip() for line in r.stdout.strip().split("\n") if line.strip()]
    return gpus


def get_storage_info(path):
    """Check storage type and speed."""
    parent = str(Path(path).parent)
    r = subprocess.run(["df", "-h", parent], capture_output=True, text=True)
    return r.stdout.strip()


def run_single_model_test(model_key, model_cfg, snapshot_dir, timeout=1800):
    """Run freeze/restore test for a single model in a subprocess.

    Returns dict with results or error info.
    """
    hf_id = model_cfg["hf_id"]
    tp = model_cfg["tp"]
    prompt = model_cfg["prompt"]
    snapshot_path = os.path.join(snapshot_dir, f"{model_key}.thaw")
    result_path = os.path.join(snapshot_dir, f"{model_key}_result.json")

    # This script runs inside the subprocess
    test_script = f'''
import gc, json, os, time, traceback
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

result = {{
    "model": "{model_key}",
    "hf_id": "{hf_id}",
    "tp": {tp},
    "arch": "{model_cfg['arch']}",
}}

try:
    import torch
    from vllm import LLM, SamplingParams
    import thaw_vllm
    from thaw_vllm import freeze_model_pipelined, restore_model_pipelined

    sampling = SamplingParams(temperature=0, max_tokens=50)

    def find_model(llm):
        engine = llm.llm_engine
        for path_fn in [
            lambda: engine.model_executor.driver_worker.model_runner.model,
            lambda: engine.engine_core.model_runner.model,
            lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
            lambda: engine.engine_core.model_executor.model_runner.model,
        ]:
            try:
                return path_fn()
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("Could not find nn.Module in vLLM")

    # ── Phase 1: Normal cold start ──
    print(f"  [{{'{model_key}'}}] Phase 1: Normal cold start...")
    t0 = time.perf_counter()
    llm = LLM(
        model="{hf_id}",
        dtype="float16",
        enforce_eager=True,
        tensor_parallel_size={tp},
        gpu_memory_utilization=0.40,
    )
    normal_time = time.perf_counter() - t0
    result["normal_load_s"] = round(normal_time, 2)
    print(f"  [{{'{model_key}'}}] Normal load: {{normal_time:.1f}}s")

    out = llm.generate(["""{prompt}"""], sampling)
    ref_text = out[0].outputs[0].text.strip()
    result["ref_text_prefix"] = ref_text[:100]
    print(f"  [{{'{model_key}'}}] Reference: {{ref_text[:60]}}...")

    # ── Phase 2: Freeze ──
    print(f"  [{{'{model_key}'}}] Phase 2: Freeze...")
    if {tp} > 1:
        fstats = thaw_vllm.freeze_model_tp(llm, "{snapshot_path}")
    else:
        model = find_model(llm)
        fstats = freeze_model_pipelined(model, "{snapshot_path}")

    result["freeze_time_s"] = round(fstats["elapsed_s"], 2)
    result["freeze_throughput_gbs"] = round(fstats["throughput_gb_s"], 2)
    result["freeze_regions"] = fstats["num_regions"]
    result["freeze_bytes"] = fstats["total_bytes"]
    result["freeze_backend"] = fstats.get("backend", "python")
    print(f"  [{{'{model_key}'}}] Frozen: {{fstats['elapsed_s']:.1f}}s "
          f"({{fstats['throughput_gb_s']:.2f}} GB/s)")

    # Teardown
    del llm
    if {tp} == 1:
        del model
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    # ── Phase 3: Restore ──
    print(f"  [{{'{model_key}'}}] Phase 3: Restore...")
    if {tp} > 1:
        t0 = time.perf_counter()
        llm2 = thaw_vllm.load(
            "{hf_id}", "{snapshot_path}",
            tensor_parallel_size={tp},
        )
        restore_total = time.perf_counter() - t0
        result["restore_total_s"] = round(restore_total, 2)
        result["restore_backend"] = "thaw_loader"
    else:
        t0 = time.perf_counter()
        llm2 = LLM(
            model="{hf_id}",
            dtype="float16",
            enforce_eager=True,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.40,
            load_format="dummy",
        )
        init_time = time.perf_counter() - t0
        result["dummy_init_s"] = round(init_time, 2)

        t0 = time.perf_counter()
        model2 = find_model(llm2)
        rstats = restore_model_pipelined(model2, "{snapshot_path}")
        restore_time = time.perf_counter() - t0
        restore_total = init_time + restore_time

        result["restore_dma_s"] = round(restore_time, 2)
        result["restore_throughput_gbs"] = round(rstats["throughput_gb_s"], 2)
        result["restore_total_s"] = round(restore_total, 2)
        result["restore_backend"] = rstats.get("backend", "python")

    print(f"  [{{'{model_key}'}}] Restored: {{restore_total:.1f}}s")

    # ── Phase 4: Verify ──
    out2 = llm2.generate(["""{prompt}"""], sampling)
    restored_text = out2[0].outputs[0].text.strip()
    result["restored_text_prefix"] = restored_text[:100]

    match = ref_text == restored_text
    result["correctness"] = "PASS" if match else "FAIL"
    result["speedup"] = round(normal_time / restore_total, 1) if restore_total > 0 else 0
    result["status"] = "success"

    if not match:
        print(f"  [{{'{model_key}'}}] CORRECTNESS FAIL!")
        print(f"    ref:      {{ref_text[:80]}}")
        print(f"    restored: {{restored_text[:80]}}")
    else:
        print(f"  [{{'{model_key}'}}] PASS — {{result['speedup']}}x speedup")

except Exception as e:
    result["status"] = "error"
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()
    print(f"  [{{'{model_key}'}}] ERROR: {{e}}")

with open("{result_path}", "w") as f:
    json.dump(result, f, indent=2)
'''

    print(f"\n{'='*60}")
    print(f"  Testing: {model_key} ({hf_id})")
    print(f"  TP={tp} | ~{model_cfg['size_gb']} GB | arch={model_cfg['arch']}")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "-c", test_script],
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
        timeout=timeout,
    )
    wall_time = time.perf_counter() - t0

    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
        result["wall_time_s"] = round(wall_time, 2)
    else:
        result = {
            "model": model_key,
            "hf_id": hf_id,
            "status": "crash",
            "exit_code": r.returncode,
            "wall_time_s": round(wall_time, 2),
        }

    return result


def print_summary(results, output_dir):
    """Print human-readable summary table and save to file."""
    print("\n\n")
    print("=" * 80)
    print("  STRESS TEST RESULTS")
    print("=" * 80)

    # Header
    fmt = "  {:<22} {:>5} {:>8} {:>8} {:>8} {:>8} {:>8}"
    print(fmt.format("Model", "TP", "Normal", "Restore", "Speedup", "GB/s", "Status"))
    print("  " + "-" * 76)

    pass_count = 0
    fail_count = 0
    error_count = 0

    for r in results:
        model = r.get("model", "?")
        tp = r.get("tp", "?")
        status = r.get("status", "?")

        if status == "success":
            normal = f"{r.get('normal_load_s', 0):.1f}s"
            restore = f"{r.get('restore_total_s', 0):.1f}s"
            speedup = f"{r.get('speedup', 0):.1f}x"
            throughput = f"{r.get('restore_throughput_gbs', r.get('freeze_throughput_gbs', 0)):.1f}"
            correctness = r.get("correctness", "?")
            if correctness == "PASS":
                pass_count += 1
                tag = "PASS"
            else:
                fail_count += 1
                tag = "FAIL"
        elif status == "error":
            normal = restore = speedup = throughput = "-"
            tag = "ERROR"
            error_count += 1
        else:
            normal = restore = speedup = throughput = "-"
            tag = "CRASH"
            error_count += 1

        print(fmt.format(model, str(tp), normal, restore, speedup, throughput, tag))

    print()
    print(f"  Passed: {pass_count}  |  Failed: {fail_count}  |  Errors: {error_count}")
    print(f"  Results saved to: {output_dir}")
    print()

    # Save summary as markdown too
    md_path = os.path.join(output_dir, "RESULTS.md")
    with open(md_path, "w") as f:
        f.write(f"# Stress Test Results\n\n")
        f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n")

        # GPU info
        gpus = get_gpu_info()
        f.write(f"**GPUs:** {len(gpus)}x\n")
        for g in gpus:
            f.write(f"- {g}\n")
        f.write("\n")

        f.write("| Model | TP | Normal | Restore | Speedup | Correctness |\n")
        f.write("|-------|:--:|-------:|--------:|--------:|:-----------:|\n")

        for r in results:
            if r.get("status") == "success":
                f.write(f"| {r['hf_id']} | {r['tp']} | "
                        f"{r.get('normal_load_s', 0):.1f}s | "
                        f"{r.get('restore_total_s', 0):.1f}s | "
                        f"**{r.get('speedup', 0):.1f}x** | "
                        f"{r.get('correctness', '?')} |\n")
            else:
                f.write(f"| {r.get('hf_id', '?')} | {r.get('tp', '?')} | "
                        f"- | - | - | {r.get('status', '?').upper()} |\n")

        f.write(f"\n**Summary:** {pass_count} passed, {fail_count} failed, "
                f"{error_count} errors\n")

    print(f"  Markdown summary: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="thaw stress test suite")
    parser.add_argument("--models", type=str, default=None,
                        help="Comma-separated model keys (e.g. 'mistral-7b,qwen-7b')")
    parser.add_argument("--max-gpus", type=int, default=None,
                        help="Max GPUs available (auto-detected if not set)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only run models tagged 'quick'")
    parser.add_argument("--output-dir", type=str, default="/workspace/thaw_results",
                        help="Directory for results and snapshots")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-model timeout in seconds (default 1800)")
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Snapshot directory (default: /dev/shm for RAM-backed)")
    args = parser.parse_args()

    # Setup
    gpus = get_gpu_info()
    gpu_count = len(gpus)
    max_gpus = args.max_gpus or gpu_count

    print("=" * 60)
    print("  thaw stress test")
    print("=" * 60)
    print(f"  GPUs: {gpu_count}")
    for g in gpus:
        print(f"    {g}")
    print(f"  Max GPUs for tests: {max_gpus}")

    # Pick snapshot dir: /dev/shm if available (RAM-backed), else output dir
    snapshot_dir = args.snapshot_dir
    if snapshot_dir is None:
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            snapshot_dir = "/dev/shm/thaw_stress"
        else:
            snapshot_dir = os.path.join(args.output_dir, "snapshots")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    print(f"  Snapshots: {snapshot_dir}")
    print(f"  Results:   {args.output_dir}")

    # Storage info
    print(f"\n  Storage info:")
    print(f"  {get_storage_info(snapshot_dir)}")

    # Select models
    if args.models:
        selected = args.models.split(",")
        model_list = [(k, MODELS[k]) for k in selected if k in MODELS]
        unknown = [k for k in selected if k not in MODELS]
        if unknown:
            print(f"\n  WARNING: Unknown models: {unknown}")
            print(f"  Available: {list(MODELS.keys())}")
    elif args.quick:
        model_list = [(k, v) for k, v in MODELS.items()
                      if "quick" in v["tags"] and v["tp"] <= max_gpus]
    else:
        model_list = [(k, v) for k, v in MODELS.items() if v["tp"] <= max_gpus]

    print(f"\n  Models to test ({len(model_list)}):")
    for k, v in model_list:
        print(f"    {k}: {v['hf_id']} (TP={v['tp']}, ~{v['size_gb']} GB)")

    # Pre-download all models
    print(f"\n  Pre-downloading models...")
    for k, v in model_list:
        print(f"    Downloading {v['hf_id']}...")
        subprocess.run(
            ["huggingface-cli", "download", v["hf_id"], "--exclude", "original/*"],
            env={**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"},
            capture_output=True,
        )

    # Run tests
    results = []
    start_time = time.perf_counter()

    for k, v in model_list:
        try:
            result = run_single_model_test(k, v, snapshot_dir, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            result = {
                "model": k,
                "hf_id": v["hf_id"],
                "status": "timeout",
                "timeout_s": args.timeout,
            }
            print(f"  [{k}] TIMEOUT after {args.timeout}s")
        except Exception as e:
            result = {
                "model": k,
                "hf_id": v["hf_id"],
                "status": "error",
                "error": str(e),
            }

        results.append(result)

        # Clean up snapshot files between models to save disk/RAM
        for f in Path(snapshot_dir).glob(f"{k}*"):
            try:
                f.unlink()
            except OSError:
                pass

    total_time = time.perf_counter() - start_time

    # Save raw results
    raw_path = os.path.join(args.output_dir, "results.json")
    with open(raw_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpus": gpus,
            "total_time_s": round(total_time, 1),
            "results": results,
        }, f, indent=2)

    # Print summary
    print_summary(results, args.output_dir)
    print(f"  Total time: {total_time / 60:.1f} minutes")
    print(f"  Raw JSON:   {raw_path}")


if __name__ == "__main__":
    main()
