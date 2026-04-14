#!/bin/bash
# Run thaw stress test on a RunPod / Lambda / bare metal GPU box.
# Usage: bash demos/run_stress_test.sh [--quick]
#
# Prerequisites:
#   - 2x A100 or 2x H100 (80GB each)
#   - HF_TOKEN set (for gated models like Llama)
#   - CUDA 12+
#
# Quick mode (3 small models, ~15 min):
#   bash demos/run_stress_test.sh --quick
#
# Full suite (8 models including 70B+, ~2-3 hours):
#   bash demos/run_stress_test.sh

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first"
    echo "  export HF_TOKEN=hf_xxxxx"
    exit 1
fi

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Storage Info ==="
df -h /dev/shm /workspace /tmp 2>/dev/null || true
echo ""

# Install deps
echo "=== Installing deps ==="
pip install uv -q 2>&1 | tail -1
uv pip install vllm hf_transfer --system -q 2>&1 | tail -3

# Clone/update repo
echo "=== Setting up thaw ==="
cd /workspace
if [ -d thaw ]; then
    cd thaw && git pull
else
    git clone https://github.com/thaw-ai/thaw.git && cd thaw
fi

# Install thaw Python package
pip install -e ".[serve]" -q 2>&1 | tail -1

# Build Rust extension (optional but much faster)
echo "=== Building Rust extension ==="
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable 2>&1 | tail -3
    source "$HOME/.cargo/env"
fi

pip install "maturin[patchelf]" -q
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
maturin build --release -m crates/thaw-py/Cargo.toml --features cuda --skip-auditwheel 2>&1 | tail -3
pip install target/wheels/thaw_py-*.whl --force-reinstall -q

echo "=== HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN"

echo ""
echo "=== Running stress test ==="
mkdir -p /workspace/thaw_results

python demos/stress_test.py "$@" \
    --output-dir /workspace/thaw_results \
    --snapshot-dir /dev/shm/thaw_stress

echo ""
echo "=== Done ==="
echo "Results: /workspace/thaw_results/"
echo "  results.json  — raw data"
echo "  RESULTS.md    — markdown summary"
echo ""
echo "To copy results locally:"
echo "  scp root@<pod-ip>:/workspace/thaw_results/results.json ."
