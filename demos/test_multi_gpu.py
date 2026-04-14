#!/usr/bin/env python3
"""Quick multi-GPU test for thaw tensor parallel support.

Uses subprocess isolation between freeze and restore phases
(vLLM doesn't release GPU memory within a single process).

IMPORTANT: All code must be inside main() / if __name__ == '__main__'
because vLLM with TP>1 uses spawn multiprocessing, which re-imports
this module in child workers.
"""

import json
import os
import subprocess
import sys
import time

MODEL = os.environ.get("MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
SNAPSHOT = os.environ.get("SNAPSHOT", "/dev/shm/weights_tp2.thaw")
HANDOFF = os.environ.get("HANDOFF", "/dev/shm/thaw_tp_test.json")
TP = int(os.environ.get("TP", "2"))


def main():
    short_model = MODEL.split("/")[-1]
    print("=" * 60)
    print(f"  thaw multi-GPU test — {short_model} (TP={TP})")
    print("=" * 60)

    result = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True)
    print(result.stdout.strip())
    gpu_count = result.stdout.strip().count("\n") + 1
    if gpu_count < TP:
        print(f"\nERROR: Need {TP} GPUs, found {gpu_count}")
        sys.exit(1)
    print(f"\nFound {gpu_count} GPUs. Using TP={TP}.\n")

    # ── Phase 1: Freeze (subprocess) ──────────────────────────

    print("[1/3] Freezing model with TP=2 (subprocess)...")

    freeze_script = f'''
import os, json, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

from vllm import LLM, SamplingParams
import thaw_vllm

t0 = time.perf_counter()
llm = LLM(
    model="{MODEL}",
    dtype="float16",
    enforce_eager=True,
    tensor_parallel_size={TP},
    enable_prefix_caching=True,
)
load_time = time.perf_counter() - t0
print(f"      Model loaded in {{load_time:.1f}}s")

out = llm.generate(["Explain quantum computing in one sentence."],
                    SamplingParams(temperature=0, max_tokens=50))
ref_text = out[0].outputs[0].text.strip()
print(f"      Reference: \\"{{ref_text[:80]}}...\\"")

print(f"      Freezing weights to {SNAPSHOT}...")
wstats = thaw_vllm.freeze_model_tp(llm, "{SNAPSHOT}")
print(f"      Done: {{wstats['num_regions']}} regions, "
      f"{{wstats['total_bytes'] / 1e9:.2f}} GB in {{wstats['elapsed_s']:.1f}}s")

from thaw_vllm.loader import _rank_snapshot_path
for rank in range({TP}):
    p = _rank_snapshot_path("{SNAPSHOT}", rank)
    size = os.path.getsize(p) / 1e9
    print(f"      {{p}}: {{size:.2f}} GB")

with open("{HANDOFF}", "w") as f:
    json.dump({{"ref_text": ref_text, "load_time": load_time,
                "num_regions": wstats["num_regions"],
                "total_bytes": wstats["total_bytes"],
                "freeze_time": wstats["elapsed_s"]}}, f)

print("      Phase 1 done. Exiting to release GPU memory.")
'''

    r = subprocess.run(
        [sys.executable, "-c", freeze_script],
        env={**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
    )
    if r.returncode != 0:
        print(f"\n  ERROR: Freeze subprocess failed (exit code {r.returncode})")
        sys.exit(1)

    with open(HANDOFF) as f:
        handoff = json.load(f)

    # ── Phase 2: Restore (this process) ───────────────────────

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Use nvidia-smi instead of torch.cuda.mem_get_info to avoid
    # initializing CUDA — which would force vLLM to use 'spawn'
    # multiprocessing, and spawned workers won't have the thaw
    # loader registered.
    mem_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    for i, line in enumerate(mem_result.stdout.strip().split("\n")):
        print(f"\n      GPU {i}: {int(line.strip()) / 1024:.1f} GiB free")

    print(f"\n[2/3] Restoring model from thaw snapshots (TP=2)...")

    import thaw_vllm
    from vllm import SamplingParams

    t0 = time.perf_counter()
    llm2 = thaw_vllm.load(MODEL, SNAPSHOT, tensor_parallel_size=TP, enable_prefix_caching=True)
    restore_time = time.perf_counter() - t0
    print(f"      Restored in {restore_time:.1f}s")

    # ── Phase 3: Verify ───────────────────────────────────────

    print(f"\n[3/3] Verifying correctness...")
    out2 = llm2.generate(["Explain quantum computing in one sentence."],
                          SamplingParams(temperature=0, max_tokens=50))
    restored_text = out2[0].outputs[0].text.strip()
    print(f"      Restored: \"{restored_text[:80]}...\"")

    ref_text = handoff["ref_text"]
    match = ref_text == restored_text

    load_time = handoff["load_time"]
    print(f"""
{'=' * 50}
  Results
{'=' * 50}
  Model:           {MODEL}
  Tensor parallel:  {TP}
  Normal load:      {load_time:.1f}s
  thaw restore:     {restore_time:.1f}s
  Speedup:          {load_time / restore_time:.1f}x
  Correctness:      {'PASS' if match else 'FAIL'}
  Freeze:           {handoff['num_regions']} regions, {handoff['total_bytes'] / 1e9:.2f} GB in {handoff['freeze_time']:.1f}s
""")

    if not match:
        print(f"  Reference: {ref_text[:100]}")
        print(f"  Restored:  {restored_text[:100]}")


if __name__ == "__main__":
    main()
