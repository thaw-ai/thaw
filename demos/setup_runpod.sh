#!/bin/bash
# thaw RunPod setup — one-shot install + test
# Usage: curl -sL <raw_github_url> | bash
#   or:  bash setup_runpod.sh [MODEL]
#
# Default model: meta-llama/Meta-Llama-3-8B-Instruct
# For 70B:       bash setup_runpod.sh meta-llama/Meta-Llama-3-70B-Instruct

set -e

MODEL="${1:-meta-llama/Meta-Llama-3-8B-Instruct}"
echo "========================================"
echo "  thaw RunPod setup"
echo "  Model: $MODEL"
echo "========================================"

# 1. Install vLLM + thaw
echo "[1/5] Installing vLLM..."
pip install vllm

echo "[2/5] Cloning thaw..."
cd /root
[ -d thaw ] && cd thaw && git pull || (git clone https://github.com/thaw-ai/thaw.git && cd thaw)
cd /root/thaw
pip install -e .

echo "[3/5] HuggingFace login..."
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "Set HF_TOKEN env var or run: huggingface-cli login"
    huggingface-cli login
fi

# 2. Build Rust backend
echo "[4/5] Building Rust pipelined backend..."
if ! command -v cargo &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
pip install maturin patchelf
cd /root/thaw/crates/thaw-py
maturin build --release --features cuda
pip install /root/thaw/target/wheels/thaw_py-*.whl --force-reinstall

# 3. Run test
echo "[5/5] Running multi-GPU test..."
cd /root/thaw
MODEL="$MODEL" python demos/test_multi_gpu.py

echo "Done!"
