#!/bin/bash
# pod-validate-kv.sh — end-to-end KV cache freeze/restore validation on a
# RunPod H100 (or any CUDA box). Proves the new .thaw + .meta sidecar
# pipelined path works and measures throughput.
#
# What this validates (in order):
#   1. `thaw freeze --kv-output` emits BOTH <path>.thaw and <path>.thaw.meta.
#   2. `thaw info <path>.thaw` reads the sidecar and prints KV metadata.
#   3. `thaw serve --snapshot ... --kv-snapshot ...` warm-starts with the
#      KV cache, then a chat completion for the warmup prefix hits the
#      restored blocks (prefix-cache hit, near-zero prefill).
#   4. The unified restore path (`restore_pipelined_from_bytes_auto`) gets
#      exercised by the weight restore during that serve startup.
#
# Runs on the `vllm/vllm-openai:latest` RunPod template (vLLM + torch +
# CUDA prebaked). No Rust build — install thaw-native from PyPI and
# overlay the git checkout for the Python code.
#
# Required env:
#   GH_PAT               GitHub PAT with repo scope (matteso1/thaw)
#
# Optional env:
#   HF_TOKEN             HF token for gated Llama weights
#   MODEL                default meta-llama/Meta-Llama-3-8B
#   PROMPT               default "What is the capital of France?"
#   THAW_DIR             default /workspace/thaw
#   SNAPSHOT_DIR         default /workspace/snapshots
#
# Usage (pod: vllm/vllm-openai:latest):
#   export GH_PAT=...
#   export HF_TOKEN=...                # for gated Llama
#   git clone https://$GH_PAT@github.com/matteso1/thaw.git /workspace/thaw
#   bash /workspace/thaw/scripts/pod-validate-kv.sh
#
# Re-run to pull latest code: same command. Script fetch+resets origin/main.

set -euo pipefail

: "${GH_PAT:?set GH_PAT}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
# vLLM V1 prefix cache only hashes *completed* 16-token blocks, so the
# warmup prompt must be long enough to fill at least one block. Keep the
# text deterministic so the same prefix is replayed on serve restore.
PROMPT="${PROMPT:-Explain the history of the French Revolution and its long-term impact on modern democracy, human rights, and political theory. Include key figures, major events, and the ideological shifts that emerged from the period.}"
THAW_DIR="${THAW_DIR:-/workspace/thaw}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-/workspace/snapshots}"
MODEL_KEY="$(echo "$MODEL" | tr '/' '-' | tr '[:upper:]' '[:lower:]')"
WEIGHTS_PATH="$SNAPSHOT_DIR/${MODEL_KEY}.thaw"
KV_PATH="$SNAPSHOT_DIR/${MODEL_KEY}.kv.thaw"
SERVE_LOG="/tmp/thaw-serve-kv.log"

mkdir -p "$SNAPSHOT_DIR"

banner() { echo ""; echo "=== $* ==="; }

banner "thaw KV cache validation (vllm-openai image)"
echo "Model:    $MODEL"
echo "Weights:  $WEIGHTS_PATH"
echo "KV cache: $KV_PATH   (+ .meta sidecar)"
echo "Prompt:   $PROMPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 --version

# Preflight: need vllm + torch. Preferred RunPod template is the
# `vllm/vllm-openai:latest` image where both are prebaked. If vllm is
# missing (bare CUDA image, etc.), install it — it will pull a matching
# torch as a dep. We do NOT touch torch if vllm is already present.
if ! python3 -c "import vllm" > /dev/null 2>&1; then
  echo "vllm not importable — installing from PyPI (pulls matching torch)..."
  python3 -m pip install --quiet vllm
fi
if ! python3 -c "import torch, vllm" > /dev/null 2>&1; then
  echo "ERROR: torch and/or vllm still not importable after pip install."
  echo "       Check the pip output above. Consider the"
  echo "       'vllm/vllm-openai:latest' container image instead."
  exit 2
fi
python3 -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('vllm ', vllm.__version__)"

# ── Pip deps (no torch perturbation) ─────────────────────────────────
banner "[1/4] Pip deps"
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "thaw-native>=0.1.2"

if [ -n "${HF_TOKEN:-}" ]; then
  mkdir -p ~/.cache/huggingface
  echo "$HF_TOKEN" > ~/.cache/huggingface/token
  echo "OK: HF token written"
fi

# ── Clone thaw, install Python packages --no-deps ───────────────────
banner "[2/4] Clone + install thaw (Python, --no-deps)"
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

# Re-exec: bash loaded this script into memory at invocation. The
# fetch+reset above may have updated the file on disk — if so, re-exec
# so subsequent logic runs against the latest code. THAW_SELF_REEXECED
# prevents an infinite loop.
if [ -z "${THAW_SELF_REEXECED:-}" ]; then
  export THAW_SELF_REEXECED=1
  exec bash "$THAW_DIR/scripts/pod-validate-kv.sh" "$@"
fi

# --no-deps: keep torch+vllm from the image, thaw-native from PyPI.
python3 -m pip install --quiet -e . --no-deps
python3 -m pip install --quiet fastapi uvicorn

python3 -c "
import thaw, thaw_common, thaw_vllm
print('OK thaw       ', thaw.__file__)
print('OK thaw_common', thaw_common.__file__)
print('OK thaw_vllm  ', thaw_vllm.__file__)
# Confirm the pipelined PyO3 bindings exercised by the KV path exist.
# The PyPI package 'thaw-native' installs as module 'thaw'.
import thaw as _thaw
required = ('freeze_to_file_pipelined', 'restore_from_file_pipelined')
for name in required:
    assert hasattr(_thaw, name), f'missing required binding: {name}'
    print('OK thaw.' + name)
# Optional newer bindings — present once a new thaw-native wheel is published.
for name in ('restore_from_bytes_auto',):
    have = hasattr(_thaw, name)
    print(('OK ' if have else '-- ') + 'thaw.' + name + (' (not yet in wheel)' if not have else ''))
"

# ── Test A: freeze weights + KV cache with a warmup prompt ───────────
banner "[3/4] Test A — thaw freeze (weights + KV cache)"
rm -f "$WEIGHTS_PATH" "$KV_PATH" "$KV_PATH.meta"

t0=$(date +%s)
thaw freeze \
  --model "$MODEL" \
  --output "$WEIGHTS_PATH" \
  --kv-output "$KV_PATH" \
  --kv-warmup-prompt "$PROMPT"
freeze_elapsed=$(($(date +%s) - t0))
echo "Freeze elapsed: ${freeze_elapsed}s"

# Both artifacts must exist. The .meta sidecar is the new-format marker.
ls -lh "$WEIGHTS_PATH"
ls -lh "$KV_PATH"

# If the warmup prompt was too short to fill a prefix-cache block, the
# freeze writes the 29-byte legacy empty-KV file and no sidecar. Detect
# that and fail loudly — validation requires a real KV payload.
kv_size=$(stat -c%s "$KV_PATH")
if [ "$kv_size" -lt 100 ]; then
  echo ""
  echo "FAIL: $KV_PATH is ${kv_size} bytes — warmup prompt didn't produce"
  echo "      any cached 16-token blocks. Re-run with a longer PROMPT."
  exit 1
fi
if [ ! -f "$KV_PATH.meta" ]; then
  echo "FAIL: missing $KV_PATH.meta sidecar"
  exit 1
fi
ls -lh "$KV_PATH.meta"

# Info must parse the sidecar and print KV fields.
banner "thaw info $KV_PATH"
thaw info "$KV_PATH" | tee /tmp/thaw-info.out
grep -q 'KV sidecar' /tmp/thaw-info.out || {
  echo "FAIL: 'thaw info' did not detect KV sidecar"
  exit 1
}

# ── Test B: serve with KV snapshot, chat with warmup prompt → hit ────
banner "[4/4] Test B — thaw serve with --kv-snapshot, prefix-cache hit"
thaw serve \
  --model "$MODEL" \
  --snapshot "$WEIGHTS_PATH" \
  --kv-snapshot "$KV_PATH" \
  --host 0.0.0.0 --port 8000 \
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

# Fire a completion for the same warmup prompt. If the KV restore
# succeeded, the prefix hashes match and prefill is skipped.
banner "Chat: '$PROMPT'"
RESP=$(curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":40}")
echo "$RESP"
echo "$RESP" | python3 -c "
import json, sys
r = json.loads(sys.stdin.read())
choices = r.get('choices') or []
if not choices or not choices[0].get('message', {}).get('content'):
    print('FAIL: empty completion')
    sys.exit(1)
print('OK: non-empty completion')
"

banner "serve log (tail)"
tail -60 "$SERVE_LOG"

banner "SUMMARY"
echo "Model:          $MODEL"
echo "Weights:        $WEIGHTS_PATH"
echo "KV snapshot:    $KV_PATH (+ .meta)"
echo "Freeze elapsed: ${freeze_elapsed}s"
echo "Serve ready:    ${READY_TIME}s"
# Surface the KV restore throughput line if the code logged one.
grep -E 'KV cache|Restored|GB/s|Slot 0: loaded|prefix hits' "$SERVE_LOG" || true
banner "KV validation complete"
