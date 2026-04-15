#!/bin/bash
# thaw RunPod setup — one command to get started
# Usage: bash setup.sh
#
# Set HF_TOKEN for gated models (e.g., Llama):
#   export HF_TOKEN=hf_xxxxx

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
SNAPSHOT="${SNAPSHOT:-/workspace/weights.thaw}"

echo "=== [1/4] GPU info ==="
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader

echo "=== [2/4] Install thaw ==="
pip install thaw-vllm[all] -q 2>&1 | tail -3

echo "=== [3/4] Freeze model ==="
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null
fi

if [ ! -f "$SNAPSHOT" ]; then
    echo "Freezing $MODEL to $SNAPSHOT..."
    thaw freeze --model "$MODEL" --output "$SNAPSHOT"
else
    echo "Snapshot already exists at $SNAPSHOT"
fi

echo "=== [4/4] Serve ==="
echo "Starting thaw serve on port 8000..."
echo ""
echo "  curl http://localhost:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
thaw serve --model "$MODEL" --snapshot "$SNAPSHOT" --port 8000
