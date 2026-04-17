#!/bin/bash
# GPU validation: vLLM freeze + restore on 2x A40
# Fully self-contained — only needs vLLM, no SGLang.
# Usage: bash tests/gpu/test_vllm_a40.sh
set -euo pipefail

echo "=== System deps ==="
apt-get update && apt-get install -y libnuma-dev

echo "=== Install thaw + vLLM ==="
pip install -e ".[vllm,native]"

echo "=== HuggingFace login ==="
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first: export HF_TOKEN=your_token"
    exit 1
fi
huggingface-cli login --token "$HF_TOKEN"

echo "=== Single-GPU: freeze ==="
thaw freeze --model meta-llama/Meta-Llama-3-8B --output /tmp/llama8b.thaw

echo "=== Single-GPU: restore + generate ==="
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM('meta-llama/Meta-Llama-3-8B', load_format='thaw',
           model_loader_extra_config={'snapshot': '/tmp/llama8b.thaw'},
           enforce_eager=True, dtype='float16')
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=20))
print('OUTPUT:', out[0].outputs[0].text)
del llm
"

echo "=== TP=2: freeze ==="
NCCL_P2P_DISABLE=1 thaw freeze --model meta-llama/Meta-Llama-3-8B \
    --output /tmp/llama8b-tp2.thaw --tensor-parallel 2

echo "=== TP=2: restore + generate ==="
NCCL_P2P_DISABLE=1 python3 -c "
from vllm import LLM, SamplingParams
llm = LLM('meta-llama/Meta-Llama-3-8B', load_format='thaw',
           model_loader_extra_config={'snapshot': '/tmp/llama8b-tp2.thaw'},
           enforce_eager=True, dtype='float16', tensor_parallel_size=2)
out = llm.generate(['The capital of France is'], SamplingParams(max_tokens=20))
print('OUTPUT:', out[0].outputs[0].text)
del llm
"

echo ""
echo "=== ALL PASSED: vLLM freeze + restore ==="
