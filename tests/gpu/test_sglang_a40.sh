#!/bin/bash
# GPU validation: SGLang freeze + restore on 2x A40
# Fully self-contained — only needs SGLang, no vLLM.
# Usage: bash tests/gpu/test_sglang_a40.sh
set -euo pipefail

echo "=== System deps ==="
apt-get update && apt-get install -y libnuma-dev

echo "=== Install SGLang + thaw ==="
pip install "sglang[all]"
pip install -e .

echo "=== HuggingFace login ==="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=your_token"
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

echo "=== Single-GPU: freeze with SGLang ==="
thaw freeze --model meta-llama/Meta-Llama-3-8B --output /tmp/llama8b.thaw --engine sglang

echo "=== Single-GPU: restore with SGLang ==="
python3 -c "
import thaw_sglang
engine = thaw_sglang.load('meta-llama/Meta-Llama-3-8B', '/tmp/llama8b.thaw')
out = engine.generate('The capital of France is')
print('OUTPUT:', out)
engine.shutdown()
"

echo "=== TP=2: freeze with SGLang ==="
NCCL_P2P_DISABLE=1 thaw freeze --model meta-llama/Meta-Llama-3-8B \
    --output /tmp/llama8b-tp2.thaw --tensor-parallel 2 --engine sglang

echo "=== TP=2: restore with SGLang ==="
NCCL_P2P_DISABLE=1 python3 -c "
import thaw_sglang
engine = thaw_sglang.load('meta-llama/Meta-Llama-3-8B', '/tmp/llama8b-tp2.thaw', tp_size=2)
out = engine.generate('The capital of France is')
print('OUTPUT:', out)
engine.shutdown()
"

echo ""
echo "=== ALL PASSED: SGLang freeze + restore ==="
