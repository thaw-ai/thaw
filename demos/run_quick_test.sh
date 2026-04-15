#!/bin/bash
# Quick validation test for thaw optimizations (~5 min on any CUDA GPU).
#
# Tests:
#   1. Rust thaw-bench: raw DMA throughput (tests WC pinned memory)
#   2. Single model (phi-3-mini, 7.6 GB): freeze/restore + MAP_POPULATE
#
# Works on any GPU with >=16 GB VRAM: L4, A10, T4, A40, 4090, etc.
#
# Usage on a fresh RunPod/Lambda pod:
#   export HF_TOKEN=hf_xxxxx
#   export GITHUB_PAT=ghp_xxxxx
#   curl -sSL https://raw.githubusercontent.com/matteso1/thaw/main/demos/run_quick_test.sh | bash -s -- --setup

set -euo pipefail

SETUP=false
if [[ "${1:-}" == "--setup" ]]; then
    SETUP=true
fi

echo "=== thaw quick validation test ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# --- Setup (only on fresh pod) ---
if $SETUP; then
    echo "=== Setup ==="

    if [ -z "${HF_TOKEN:-}" ]; then
        echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=hf_xxxxx"
        exit 1
    fi

    export HF_HOME="${HF_HOME:-/workspace/hf_cache}"

    # Clone repo
    if [ ! -d /workspace/thaw ]; then
        if [ -n "${GITHUB_PAT:-}" ]; then
            git clone "https://${GITHUB_PAT}@github.com/matteso1/thaw.git" /workspace/thaw
        else
            git clone https://github.com/thaw-ai/thaw.git /workspace/thaw
        fi
    fi
    cd /workspace/thaw
    git pull --ff-only || true

    # Python deps
    pip install vllm -q 2>/dev/null
    pip install -e . --no-deps -q
    pip install fastapi uvicorn -q 2>/dev/null

    # Rust toolchain
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    pip install "maturin[patchelf]" -q 2>/dev/null

    # Build Rust extension
    export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
    echo "Building Rust extension..."
    maturin build --release -m crates/thaw-py/Cargo.toml --features cuda --skip-auditwheel 2>&1 | tail -3
    pip install target/wheels/thaw_native-*.whl --no-deps --force-reinstall -q

    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true

    echo "Setup done."
    echo ""
fi

# Make sure we're in the repo dir
if [ -f /workspace/thaw/Cargo.toml ]; then
    cd /workspace/thaw
fi

export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH:-}"
[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"

# --- Test 1: Rust raw DMA ---
echo "========================================"
echo "  Test 1: thaw-bench (2 GB raw DMA)"
echo "  Tests: WC pinned memory, pipelined restore"
echo "========================================"
echo ""

cargo run --release -p thaw-cli --features cuda -- 2048

echo ""

# --- Test 2: phi-3-mini end-to-end ---
# Uses the same proven pattern as stress_test.py:
#   Phase 1: Normal load → freeze (subprocess, frees GPU on exit)
#   Phase 2: load_format="dummy" → manual restore from mmap (subprocess)
# This avoids vLLM's GPU memory leak and tests the actual DMA path.
echo "========================================"
echo "  Test 2: phi-3-mini (7.6 GB, TP=1)"
echo "  Tests: MAP_POPULATE, MADV_HUGEPAGE, full vLLM integration"
echo "========================================"
echo ""

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
SNAPSHOT="/tmp/thaw_quick_test_phi3.thaw"
RESULT_DIR="/tmp/thaw_quick_test"
mkdir -p "$RESULT_DIR"

# --- Phase 1: Normal load + freeze (separate process so GPU memory is freed) ---
echo "Phase 1: Normal cold start + freeze"
python3 << 'PYEOF'
import json, os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Write a Python function to check if a number is prime:"
SNAPSHOT = "/tmp/thaw_quick_test_phi3.thaw"

sampling = SamplingParams(temperature=0, max_tokens=30)

def find_model(llm):
    engine = llm.llm_engine
    for path_fn in [
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_executor.model_runner.model,
    ]:
        try: return path_fn()
        except (AttributeError, TypeError): continue
    raise RuntimeError("Cannot find model in vLLM engine")

t0 = time.perf_counter()
llm = LLM(MODEL, gpu_memory_utilization=0.60, max_model_len=4096, dtype="float16")
normal_time = time.perf_counter() - t0
print(f"  Normal load: {normal_time:.1f}s")

out = llm.generate([PROMPT], sampling)[0].outputs[0].text.strip()
print(f"  Output: {out[:80]}...")

model = find_model(llm)
from thaw_vllm import freeze_model_pipelined
stats = freeze_model_pipelined(model, SNAPSHOT)
print(f"  Freeze: {stats['elapsed_s']:.2f}s, {stats['throughput_gb_s']:.1f} GB/s")

json.dump({
    "normal_time": normal_time,
    "output": out,
    "model_bytes": stats["total_bytes"],
    "freeze_throughput": stats["throughput_gb_s"],
}, open("/tmp/thaw_quick_test/phase1.json", "w"))
PYEOF

if [ $? -ne 0 ]; then echo "Phase 1 FAILED"; exit 1; fi
echo ""

# --- Phase 2: Dummy init + manual restore + mmap from /dev/shm ---
# Same pattern as stress_test.py: load_format="dummy" creates the model
# skeleton with empty weights, then we manually restore via thaw.
# This tests the actual DMA path and gives separate init/DMA timing.
echo "Phase 2: Thaw restore (dummy init + manual DMA restore)"
python3 << 'PYEOF'
import json, mmap as _mmap, os, shutil, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Write a Python function to check if a number is prime:"
SNAPSHOT = "/tmp/thaw_quick_test_phi3.thaw"

sampling = SamplingParams(temperature=0, max_tokens=30)
phase1 = json.load(open("/tmp/thaw_quick_test/phase1.json"))

def find_model(llm):
    engine = llm.llm_engine
    for path_fn in [
        lambda: engine.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_runner.model,
        lambda: engine.engine_core.model_executor.driver_worker.model_runner.model,
        lambda: engine.engine_core.model_executor.model_runner.model,
    ]:
        try: return path_fn()
        except (AttributeError, TypeError): continue
    raise RuntimeError("Cannot find model in vLLM engine")

# Step 1: Init vLLM with dummy weights (no disk load, just model skeleton)
print("  Initializing vLLM with dummy weights...")
t0 = time.perf_counter()
llm = LLM(MODEL, load_format="dummy", enforce_eager=True,
          gpu_memory_utilization=0.60, max_model_len=4096, dtype="float16")
init_time = time.perf_counter() - t0
print(f"  Dummy init: {init_time:.1f}s")

# Step 2: Restore weights via mmap (tests MAP_POPULATE + MADV_HUGEPAGE)
model = find_model(llm)

# Build the tensor mapping (same as stress_test.py)
params = [(name, param) for name, param in model.named_parameters() if param.is_cuda]
mapping = [("weights", i, p.data.data_ptr(), p.data.nbytes) for i, (_, p) in enumerate(params)]

# Stage to /dev/shm if available (tests MAP_POPULATE + MADV_HUGEPAGE on tmpfs)
restore_path = SNAPSHOT
mmap_source = "disk"
if os.path.exists("/dev/shm"):
    shm_path = "/dev/shm/thaw_quick_test_phi3.thaw"
    shutil.copy2(SNAPSHOT, shm_path)
    restore_path = shm_path
    mmap_source = "/dev/shm"

# Use Rust pipelined mmap restore (same path as stress_test.py)
try:
    import thaw as _thaw
    t_read = time.perf_counter()
    fd = os.open(restore_path, os.O_RDONLY)
    fsize = os.fstat(fd).st_size
    snapshot_bytes = _mmap.mmap(fd, fsize, access=_mmap.ACCESS_READ)
    os.close(fd)
    read_time = time.perf_counter() - t_read

    t0 = time.perf_counter()
    rresult = _thaw.restore_from_bytes_pipelined(snapshot_bytes, mapping, chunk_size_mb=64)
    dma_time = time.perf_counter() - t0
    snapshot_bytes.close()
    total_bytes = rresult["bytes_copied"]
    throughput = (total_bytes / 1e9) / dma_time if dma_time > 0 else 0
    backend = "rust_pipelined_mmap"
    print(f"  DMA restore: {dma_time:.2f}s, {throughput:.1f} GB/s ({backend} from {mmap_source})")
except (ImportError, AttributeError) as e:
    # Fallback to Python path
    from thaw_vllm.snapshot import restore_model_from_ram
    t0 = time.perf_counter()
    rstats = restore_model_from_ram(model, restore_path)
    dma_time = rstats["elapsed_s"]
    throughput = rstats["throughput_gb_s"]
    total_bytes = rstats["total_bytes"]
    backend = rstats.get("backend", "python")
    print(f"  DMA restore: {dma_time:.2f}s, {throughput:.1f} GB/s ({backend} from {mmap_source})")

# Cleanup staged file
if mmap_source == "/dev/shm" and os.path.exists(shm_path):
    os.unlink(shm_path)

# Step 3: Verify correctness — generate with restored weights
restore_total = init_time + dma_time
print(f"  Total restore: {restore_total:.1f}s (init {init_time:.1f}s + DMA {dma_time:.2f}s)")

out = llm.generate([PROMPT], sampling)[0].outputs[0].text.strip()
print(f"  Output: {out[:80]}...")

match = phase1["output"] == out
print(f"  Correctness: {'PASS - outputs match' if match else 'FAIL - outputs differ'}")
if not match:
    print(f"    Normal: {phase1['output']}")
    print(f"    Thaw:   {out}")

# Also test restore_model_from_ram directly (the MAP_POPULATE/MADV_HUGEPAGE path)
mmap_throughput = None
if os.path.exists("/dev/shm"):
    print("\n  Testing restore_model_from_ram (MAP_POPULATE + MADV_HUGEPAGE)...")
    shutil.copy2(SNAPSHOT, "/dev/shm/thaw_quick_test_phi3.thaw")
    from thaw_vllm.snapshot import restore_model_from_ram
    mstats = restore_model_from_ram(model, "/dev/shm/thaw_quick_test_phi3.thaw")
    mmap_throughput = mstats["throughput_gb_s"]
    print(f"  restore_model_from_ram: {mstats['elapsed_s']:.2f}s, {mmap_throughput:.1f} GB/s")
    os.unlink("/dev/shm/thaw_quick_test_phi3.thaw")

json.dump({
    "init_time": init_time,
    "dma_time": dma_time,
    "restore_total": restore_total,
    "throughput": throughput,
    "mmap_throughput": mmap_throughput,
    "backend": backend,
    "match": match,
    "normal_time": phase1["normal_time"],
    "model_bytes": phase1["model_bytes"],
    "freeze_throughput": phase1["freeze_throughput"],
}, open("/tmp/thaw_quick_test/phase2.json", "w"))

if not match:
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then echo "Phase 2 FAILED"; exit 1; fi
echo ""

# --- Summary ---
python3 << 'PYEOF'
import json
r = json.load(open("/tmp/thaw_quick_test/phase2.json"))

print("========================================")
print("  Summary")
print("========================================")
print(f"  Model:            phi-3-mini ({r['model_bytes']/1e9:.1f} GB)")
print(f"  Normal load:      {r['normal_time']:.1f}s (safetensors)")
print(f"  Thaw restore:     {r['restore_total']:.1f}s (init {r['init_time']:.1f}s + DMA {r['dma_time']:.2f}s)")
normal_weight_time = r['normal_time'] - r['init_time']  # strip shared vLLM init
if normal_weight_time > 0 and r['dma_time'] > 0:
    print(f"  Weight speedup:   {normal_weight_time/r['dma_time']:.1f}x ({normal_weight_time:.1f}s -> {r['dma_time']:.2f}s)")
print(f"  End-to-end:       {r['normal_time']/r['restore_total']:.1f}x ({r['normal_time']:.1f}s -> {r['restore_total']:.1f}s)")
print(f"  DMA throughput:   {r['throughput']:.1f} GB/s ({r['backend']})")
print(f"  Freeze:           {r['freeze_throughput']:.1f} GB/s")
if r.get("mmap_throughput"):
    print(f"  mmap restore:     {r['mmap_throughput']:.1f} GB/s (MAP_POPULATE + MADV_HUGEPAGE)")
print(f"  Correctness:      {'PASS' if r['match'] else 'FAIL'}")
print()
print("All tests passed." if r['match'] else "TESTS FAILED.")
PYEOF

echo ""

# --- Test 3: TP=2 multi-GPU freeze/restore (only if 2+ GPUs available) ---
# TP>1 requires vLLM multiprocessing, undo the single-GPU override
unset VLLM_ENABLE_V1_MULTIPROCESSING
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -ge 2 ]; then
echo "========================================"
echo "  Test 3: phi-3-mini TP=2 (multi-GPU pipelined freeze)"
echo "  Tests: collective_rpc freeze + restore across $NUM_GPUS GPUs"
echo "========================================"
echo ""

# Force NCCL to use shared memory transport instead of P2P
# (L4, consumer GPUs, and some cloud configs lack P2P/NVLink)
export NCCL_P2P_DISABLE=1

# Phase 1: Normal TP=2 load + pipelined freeze
# VLLM_ENABLE_V1_MULTIPROCESSING=0 keeps EngineCore in main process so we can
# access model_executor.collective_rpc. TP workers still spawn as subprocesses.
# enforce_eager=True avoids CUDA graph + custom_all_reduce crashes on non-H100 GPUs.
echo "Phase 1: Normal TP=2 load + pipelined freeze"
python3 << 'PYEOF'
import json, os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams
import thaw_vllm
from thaw_vllm import freeze_model_tp

MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Write a Python function to check if a number is prime:"
SNAPSHOT = "/tmp/thaw_quick_test_phi3_tp2.thaw"

sampling = SamplingParams(temperature=0, max_tokens=30)

t0 = time.perf_counter()
llm = LLM(MODEL, tensor_parallel_size=2, gpu_memory_utilization=0.60,
          max_model_len=4096, dtype="float16", enforce_eager=True)
normal_time = time.perf_counter() - t0
print(f"  Normal TP=2 load: {normal_time:.1f}s")

out = llm.generate([PROMPT], sampling)[0].outputs[0].text.strip()
print(f"  Output: {out[:80]}...")

t0 = time.perf_counter()
stats = freeze_model_tp(llm, SNAPSHOT)
freeze_time = time.perf_counter() - t0
print(f"  Pipelined freeze: {freeze_time:.2f}s, {stats['throughput_gb_s']:.1f} GB/s")
rank_summary = [(r.get('rank',0), round(r['throughput_gb_s'], 1)) for r in stats.get('per_rank', [])]
print(f"    per_rank: {rank_summary}")

json.dump({
    "normal_time": normal_time,
    "output": out,
    "model_bytes": stats["total_bytes"],
    "freeze_time": freeze_time,
    "freeze_throughput": stats["throughput_gb_s"],
    "per_rank_freeze": stats.get("per_rank", []),
}, open("/tmp/thaw_quick_test/tp2_phase1.json", "w"))
PYEOF

if [ $? -ne 0 ]; then echo "TP=2 Phase 1 FAILED"; else

# Phase 2: TP=2 dummy init + restore
echo ""
echo "Phase 2: TP=2 dummy init + pipelined restore"
python3 << 'PYEOF'
import json, os, sys, time
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams
import thaw_vllm
from thaw_vllm import restore_model_tp

MODEL = "microsoft/Phi-3-mini-4k-instruct"
PROMPT = "Write a Python function to check if a number is prime:"
SNAPSHOT = "/tmp/thaw_quick_test_phi3_tp2.thaw"

sampling = SamplingParams(temperature=0, max_tokens=30)
phase1 = json.load(open("/tmp/thaw_quick_test/tp2_phase1.json"))

t0 = time.perf_counter()
llm = LLM(MODEL, tensor_parallel_size=2, load_format="dummy", enforce_eager=True,
          gpu_memory_utilization=0.60, max_model_len=4096, dtype="float16")
init_time = time.perf_counter() - t0
print(f"  Dummy TP=2 init: {init_time:.1f}s")

t0 = time.perf_counter()
stats = restore_model_tp(llm, SNAPSHOT)
dma_time = time.perf_counter() - t0
print(f"  Pipelined restore: {dma_time:.2f}s, {stats['throughput_gb_s']:.1f} GB/s")
rank_summary = [(r.get('rank',0), round(r['throughput_gb_s'], 1)) for r in stats.get('per_rank', [])]
print(f"    per_rank: {rank_summary}")

out = llm.generate([PROMPT], sampling)[0].outputs[0].text.strip()
print(f"  Output: {out[:80]}...")

match = phase1["output"] == out
print(f"  Correctness: {'PASS' if match else 'FAIL'}")
if not match:
    print(f"    Normal: {phase1['output']}")
    print(f"    Thaw:   {out}")

restore_total = init_time + dma_time
print(f"\n  === TP=2 Summary ===")
print(f"  Normal TP=2 load:   {phase1['normal_time']:.1f}s")
print(f"  Thaw TP=2 restore:  {restore_total:.1f}s (init {init_time:.1f}s + DMA {dma_time:.2f}s)")
print(f"  Freeze throughput:  {phase1['freeze_throughput']:.1f} GB/s")
print(f"  Restore throughput: {stats['throughput_gb_s']:.1f} GB/s")
print(f"  Correctness:        {'PASS' if match else 'FAIL'}")

if not match:
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then echo "TP=2 Phase 2 FAILED"; fi
fi

echo ""
else
    echo "Skipping Test 3 (TP=2): only $NUM_GPUS GPU(s) detected, need 2+"
    echo ""
fi

echo "=== Quick validation complete ==="
