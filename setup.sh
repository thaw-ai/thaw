#!/bin/bash
# thaw RunPod setup — one command to benchmark
# Usage: bash setup.sh
#
# Set HF_TOKEN as a RunPod environment variable in the pod config UI,
# or export it before running this script:
#   export HF_TOKEN=hf_xxxxx

set -euo pipefail

# ── Check tokens ──────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN as an environment variable."
    echo "  export HF_TOKEN=hf_xxxxx"
    exit 1
fi

MODEL="meta-llama/Meta-Llama-3-8B"
SNAPSHOT="/tmp/llama3-8b.thaw"

echo "=== [1/7] GPU info ==="
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader

echo "=== [2/7] Install Rust ==="
if ! command -v rustc &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable 2>&1 | tail -3
    source "$HOME/.cargo/env"
fi
rustc --version

echo "=== [3/7] Clone/update repo ==="
if [ -d /workspace/thaw ]; then
    cd /workspace/thaw && git pull
else
    git clone https://github.com/thaw-ai/thaw.git /workspace/thaw
fi
cd /workspace/thaw

echo "=== [4/7] Install Python deps ==="
pip install uv -q 2>&1 | tail -1
uv pip install "maturin[patchelf]" vllm hf_transfer --system -q 2>&1 | tail -5

echo "=== [5/7] Build thaw (Rust+CUDA) ==="
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
maturin build --release -m crates/thaw-py/Cargo.toml --features cuda --skip-auditwheel 2>&1 | tail -3
pip install target/wheels/*.whl --force-reinstall -q

echo "=== [6/7] Download model ==="
huggingface-cli login --token "$HF_TOKEN"
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL" --exclude "original/*"

echo "=== [7/7] Run benchmark ==="
export VLLM_ENABLE_V1_MULTIPROCESSING=0
cd /workspace/thaw
python python/vllm_demo.py --model "$MODEL" --snapshot "$SNAPSHOT"

echo ""
echo "Done. Stop the pod when finished to save money."
