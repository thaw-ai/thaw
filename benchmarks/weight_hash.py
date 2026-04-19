"""
weight_hash.py — bit-identical proof for thaw freeze/restore.

Greedy-output match across freeze/restore is a *necessary* but *insufficient*
correctness check: numerical drift small enough to survive argmax through
every logit can still pass. This script closes the gap by hashing every
parameter tensor's raw byte buffer before freeze and after restore, and
reporting any mismatched tensor (with shape / dtype / byte count) so a
failure is diagnosable, not just "outputs differ."

Methodology
-----------
1. Load model with vLLM (normal cold start, real weights).
2. Hash every parameter in `state_dict()` — CPU copy, contiguous view,
   SHA256 over raw bytes. Record `(name, shape, dtype, nbytes, sha256)`.
3. Freeze to `--snapshot` via `thaw_vllm.freeze_model_pipelined`.
4. Tear down the engine; rebuild with `load_format="dummy"`.
5. Restore from snapshot via `thaw_vllm.restore_model_pipelined`.
6. Re-hash every parameter in the restored state_dict.
7. Diff: per-tensor match/mismatch; global pass iff every tensor matches.

Output
------
- `--json-out` (optional): structured report consumable by run_validation.
  Always contains a `mismatches` list so a pass/fail is machine-checkable.
- stdout: short PASS/FAIL with count of mismatches and first few examples.

Usage
-----
    python benchmarks/weight_hash.py \
        --model meta-llama/Meta-Llama-3-8B \
        --snapshot /tmp/weight_hash.thaw \
        --json-out /tmp/weight_hash.json

Requires a GPU with enough VRAM for the model. Hashing is on CPU and runs
at ~1 GB/s SHA256 throughput — expect ~30s per pass for a 16 GB model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Match vllm_demo's in-process engine mode so we can reach the nn.Module.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch


def hash_tensor(t: torch.Tensor) -> tuple[str, int]:
    """SHA256 of a tensor's raw byte buffer. Returns (hex_digest, nbytes).

    We hash raw bytes (not semantic value) because that is exactly what
    freeze/restore moves — any re-layout, stride change, or dtype narrowing
    would surface here even if numerically equal. Use untyped_storage on a
    cpu contiguous view: works for any dtype including bfloat16 where
    numpy()/view(uint8) hit dtype compatibility edges."""
    cpu = t.detach().to("cpu").contiguous()
    nbytes = cpu.element_size() * cpu.numel()
    storage = cpu.untyped_storage()
    buf = bytes(storage)[:nbytes]
    h = hashlib.sha256(buf).hexdigest()
    return h, nbytes


def snapshot_state(model: torch.nn.Module) -> dict[str, dict]:
    """Hash every parameter tensor in a model's state_dict."""
    out: dict[str, dict] = {}
    sd = model.state_dict()
    for name, tensor in sd.items():
        try:
            h, nbytes = hash_tensor(tensor)
        except Exception as e:
            out[name] = {"error": repr(e), "shape": list(tensor.shape),
                         "dtype": str(tensor.dtype)}
            continue
        out[name] = {
            "sha256": h,
            "nbytes": nbytes,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
    return out


def _worker_hash(self):
    """collective_rpc callable: hash this worker's local shard of the model.
    Runs on every TP rank; each rank returns a dict keyed by parameter name.
    Must be top-level (picklable); the vLLM worker injects `self` as the
    worker instance."""
    import hashlib as _hashlib
    import torch as _torch

    model = self.model_runner.model
    rank = getattr(self, "rank", 0)
    out: dict = {"rank": rank, "params": {}}
    for name, tensor in model.state_dict().items():
        try:
            cpu = tensor.detach().to("cpu").contiguous()
            nbytes = cpu.element_size() * cpu.numel()
            buf = bytes(cpu.untyped_storage())[:nbytes]
            h = _hashlib.sha256(buf).hexdigest()
            out["params"][name] = {
                "sha256": h, "nbytes": nbytes,
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            }
        except Exception as e:
            out["params"][name] = {"error": repr(e)}
    return out


def snapshot_state_tp(llm) -> dict[str, dict]:
    """Hash every worker's local state_dict via collective_rpc. Keys are
    namespaced `rank{N}:{param_name}` so ranks don't collide — two ranks
    both hold tensors named `model.layers.0.mlp.gate_proj.weight` but they
    are DIFFERENT half-shards and must be compared separately."""
    per_rank = llm.collective_rpc(_worker_hash)
    out: dict[str, dict] = {}
    for entry in per_rank:
        rank = entry.get("rank", "?")
        for name, info in entry.get("params", {}).items():
            out[f"rank{rank}:{name}"] = info
    return out


def find_model(llm):
    """Reach into vLLM to extract the nn.Module (same probing as vllm_demo)."""
    engine = llm.llm_engine
    for getter in (
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_executor.model_runner.model,
        lambda: engine.get_model(),
    ):
        try:
            return getter()
        except AttributeError:
            continue
    raise RuntimeError("could not locate model in vLLM engine")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--snapshot", default="/tmp/weight_hash.thaw")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor_parallel_size. TP>1 hashes every rank's shard "
                         "via collective_rpc and diffs them separately.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.25,
                    help="gpu_memory_utilization. Raise for larger models at TP=2.")
    ap.add_argument("--max-mismatch-print", type=int, default=10,
                    help="How many mismatched tensors to print in stdout summary.")
    args = ap.parse_args()

    results = {
        "schema_version": 1,
        "model": args.model,
        "snapshot": args.snapshot,
        "status": "started",
    }

    def _dump():
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(results, indent=2, default=str))

    try:
        from vllm import LLM
        from thaw_vllm import freeze_model_pipelined, restore_model_pipelined
        from thaw_vllm.snapshot import freeze_model_tp, restore_model_tp

        tp = args.tp
        tp_mode = tp > 1
        results["config"] = {"tp": tp, "gpu_mem_util": args.gpu_mem_util}

        # ------- Phase A: load + hash the ground truth -------
        print(f"[A] loading {args.model} tp={tp} (real weights) ...", flush=True)
        t0 = time.perf_counter()
        llm = LLM(model=args.model, dtype="float16", enforce_eager=True,
                  tensor_parallel_size=tp,
                  gpu_memory_utilization=args.gpu_mem_util)
        print(f"    loaded in {time.perf_counter()-t0:.1f}s. Hashing state_dict ...", flush=True)
        t0 = time.perf_counter()
        before = snapshot_state_tp(llm) if tp_mode else snapshot_state(find_model(llm))
        print(f"    hashed {len(before)} tensors in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------- Phase B: freeze -------
        print(f"[B] freezing to {args.snapshot} ...", flush=True)
        if tp_mode:
            fstats = freeze_model_tp(llm, args.snapshot)
        else:
            fstats = freeze_model_pipelined(find_model(llm), args.snapshot)
        print(f"    freeze: {fstats['elapsed_s']:.1f}s "
              f"({fstats['throughput_gb_s']:.2f} GB/s)", flush=True)
        results["freeze"] = {
            "elapsed_s": fstats["elapsed_s"],
            "throughput_gb_s": fstats["throughput_gb_s"],
            "total_bytes": fstats["total_bytes"],
            "num_regions": fstats["num_regions"],
        }

        # Tear down before restoring so we really reload from disk.
        del llm
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()

        # ------- Phase C: dummy init + restore -------
        print(f"[C] dummy-init + restore ...", flush=True)
        t0 = time.perf_counter()
        llm2 = LLM(model=args.model, dtype="float16", enforce_eager=True,
                   tensor_parallel_size=tp,
                   gpu_memory_utilization=args.gpu_mem_util,
                   load_format="dummy")
        print(f"    dummy init: {time.perf_counter()-t0:.1f}s", flush=True)
        if tp_mode:
            rstats = restore_model_tp(llm2, args.snapshot)
        else:
            rstats = restore_model_pipelined(find_model(llm2), args.snapshot)
        print(f"    restore: {rstats['elapsed_s']:.1f}s "
              f"({rstats['throughput_gb_s']:.2f} GB/s)", flush=True)
        results["restore"] = {
            "elapsed_s": rstats["elapsed_s"],
            "throughput_gb_s": rstats["throughput_gb_s"],
        }

        # ------- Phase D: hash again -------
        print(f"[D] hashing restored state_dict ...", flush=True)
        t0 = time.perf_counter()
        after = snapshot_state_tp(llm2) if tp_mode else snapshot_state(find_model(llm2))
        print(f"    hashed {len(after)} tensors in {time.perf_counter()-t0:.1f}s", flush=True)

        # ------- Phase E: diff -------
        all_names = sorted(set(before) | set(after))
        mismatches: list[dict] = []
        only_before: list[str] = []
        only_after: list[str] = []
        total_bytes = 0
        for name in all_names:
            b = before.get(name)
            a = after.get(name)
            if b is None:
                only_after.append(name)
                continue
            if a is None:
                only_before.append(name)
                continue
            total_bytes += b.get("nbytes", 0)
            if b.get("sha256") != a.get("sha256"):
                mismatches.append({
                    "name": name,
                    "shape": b.get("shape"),
                    "dtype": b.get("dtype"),
                    "nbytes": b.get("nbytes"),
                    "sha256_before": b.get("sha256"),
                    "sha256_after": a.get("sha256"),
                })

        passed = not mismatches and not only_before and not only_after
        results["diff"] = {
            "num_tensors": len(all_names),
            "total_bytes_hashed": total_bytes,
            "num_mismatches": len(mismatches),
            "num_only_before": len(only_before),
            "num_only_after": len(only_after),
            "mismatches": mismatches,
            "only_before": only_before,
            "only_after": only_after,
            "passed": passed,
        }
        results["status"] = "ok"
        _dump()

        print()
        print("=" * 60)
        if passed:
            print(f"PASS — {len(all_names)} tensors, "
                  f"{total_bytes/1e9:.2f} GB bit-identical across freeze/restore")
            sys.exit(0)
        print(f"FAIL — {len(mismatches)} mismatched tensor(s), "
              f"{len(only_before)} missing-after, {len(only_after)} extra-after")
        for m in mismatches[:args.max_mismatch_print]:
            print(f"  [mismatch] {m['name']}  shape={m['shape']} dtype={m['dtype']}")
            print(f"             before={m['sha256_before'][:16]}.. "
                  f"after={m['sha256_after'][:16]}..")
        if len(mismatches) > args.max_mismatch_print:
            print(f"  ... and {len(mismatches) - args.max_mismatch_print} more "
                  f"(see {args.json_out or 'json-out'} for full list)")
        sys.exit(1)

    except BaseException as e:
        results["status"] = "error"
        results["error"] = f"{type(e).__name__}: {e}"
        _dump()
        raise


if __name__ == "__main__":
    main()
