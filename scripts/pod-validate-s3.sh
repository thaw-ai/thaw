#!/bin/bash
# pod-validate-s3.sh — end-to-end S3 validation for thaw.
#
# Runs on the `vllm/vllm-openai:latest` RunPod template (vLLM + torch +
# CUDA prebaked). No Rust build, no --force-reinstall, no torch
# perturbation — just install thaw-native from PyPI (prebuilt manylinux
# wheel) and overlay our git checkout for the Python code.
#
# Required env:
#   GH_PAT                  GitHub PAT with repo scope
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_DEFAULT_REGION      e.g. us-east-2
#   THAW_BUCKET             e.g. thaw-test-nils-2026
#
# Optional:
#   HF_TOKEN                HF token for gated Llama weights
#   MODEL                   default meta-llama/Meta-Llama-3-8B
#
# Usage (pod template: vllm-latest → vllm/vllm-openai:latest):
#   export GH_PAT=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-2 THAW_BUCKET=thaw-test-nils-2026
#   git clone https://$GH_PAT@github.com/matteso1/thaw.git /workspace/thaw
#   bash /workspace/thaw/scripts/pod-validate-s3.sh
#
# The script fetches + hard-resets origin/main itself, so to re-run with
# the latest code just invoke the same bash command again.

set -euo pipefail

: "${GH_PAT:?set GH_PAT}"
: "${AWS_ACCESS_KEY_ID:?set AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?set AWS_SECRET_ACCESS_KEY}"
: "${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION}"
: "${THAW_BUCKET:?set THAW_BUCKET}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
MODEL_KEY="$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]').thaw"
SNAPSHOT="s3://$THAW_BUCKET/$MODEL_KEY"
THAW_DIR="/workspace/thaw"
SERVE_LOG="/tmp/thaw-serve.log"

banner() { echo ""; echo "=== $* ==="; }

banner "thaw S3 validation (vllm-openai image)"
echo "Model:    $MODEL"
echo "Snapshot: $SNAPSHOT"
echo "Bucket:   s3://$THAW_BUCKET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version
python3 -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm ', vllm.__version__)"

# ── Pip deps (no torch perturbation) ─────────────────────────────────
banner "[1/3] Pip deps"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "thaw-native>=0.1.2" boto3 awscli moto

aws --version
aws s3 ls "s3://$THAW_BUCKET" > /dev/null
echo "OK: can list s3://$THAW_BUCKET"

if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p ~/.cache/huggingface
  echo "$HF_TOKEN" > ~/.cache/huggingface/token
  echo "OK: HF token written"
fi

# ── Clone thaw, install Python packages --no-deps ───────────────────
banner "[2/3] Clone + install thaw (Python, --no-deps)"
mkdir -p "$(dirname "$THAW_DIR")"
if [ ! -d "$THAW_DIR/.git" ]; then
  git clone "https://$GH_PAT@github.com/matteso1/thaw.git" "$THAW_DIR"
else
  cd "$THAW_DIR"
  git fetch --quiet origin main
  git reset --hard --quiet origin/main
fi
cd "$THAW_DIR"
git log -1 --oneline

# --no-deps: we already have torch+vllm from the image, and thaw-native
# from PyPI. Don't let pip touch them.
python3 -m pip install --quiet -e . --no-deps

# Fastapi/uvicorn: needed for `thaw serve`. Cheap, no torch touch.
python3 -m pip install --quiet fastapi uvicorn

python3 -c "
import thaw, thaw_common, thaw_vllm, torch, vllm
from thaw_common.cloud import is_remote
assert is_remote('s3://b/k')
print('OK thaw       ', thaw.__file__)
print('OK thaw_common', thaw_common.__file__)
print('OK thaw_vllm  ', thaw_vllm.__file__)
print('OK torch      ', torch.__version__, 'cuda', torch.version.cuda)
print('OK vllm       ', vllm.__version__)
"

# ── Tests ────────────────────────────────────────────────────────────
banner "[3/3] Tests"

# Test A: freeze + upload
banner "Test A — thaw freeze → $SNAPSHOT"
if aws s3 ls "$SNAPSHOT" > /dev/null 2>&1; then
  echo "Snapshot already on S3 — skipping freeze"
  aws s3 ls "$SNAPSHOT" --human-readable
else
  t0=$(date +%s)
  thaw freeze --model "$MODEL" --output "$SNAPSHOT"
  echo "Test A elapsed: $(($(date +%s) - t0))s"
  aws s3 ls "$SNAPSHOT" --human-readable
fi

# Test B: serve ← S3
banner "Test B — thaw serve ← $SNAPSHOT"
rm -rf ~/.cache/thaw/snapshots
thaw serve --model "$MODEL" --snapshot "$SNAPSHOT" --host 0.0.0.0 --port 8000 \
  > "$SERVE_LOG" 2>&1 &
SERVE_PID=$!
trap "kill $SERVE_PID 2>/dev/null; wait $SERVE_PID 2>/dev/null; true" EXIT

t0=$(date +%s)
READY=0
for _ in $(seq 1 600); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    READY_TIME=$(($(date +%s) - t0))
    echo "Ready after ${READY_TIME}s"
    READY=1
    break
  fi
  if ! kill -0 $SERVE_PID 2>/dev/null; then
    echo "ERROR: thaw serve exited"
    break
  fi
  sleep 1
done

if [ "$READY" != "1" ]; then
  banner "serve log (FAILED)"
  cat "$SERVE_LOG"
  exit 1
fi

# Test C: admin register + chat
banner "Test C — admin register + chat"
curl -s -X POST http://localhost:8000/admin/snapshots \
  -H 'Content-Type: application/json' \
  -d "{\"name\":\"llama-8b-alt\",\"snapshot\":\"$SNAPSHOT\"}"
echo ""
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":40}"
echo ""

banner "serve log (tail)"
tail -40 "$SERVE_LOG"

banner "SUMMARY"
echo "Model:       $MODEL"
echo "Snapshot:    $SNAPSHOT"
echo "Serve ready: ${READY_TIME}s"
grep -E 'Slot 0: loaded|GB/s|Ready in' "$SERVE_LOG" || true
banner "Validation complete"
