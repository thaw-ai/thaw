#!/bin/bash
# bench_cold_cache.sh — one-shot cold-cache benchmark run for a fresh RunPod.
#
# Validates the cold-cache harness (posix_fadvise flush between freeze and
# restore) and captures the --profile-init waterfall. Use to replace the
# "pending re-run" placeholder in README.md + docs/BENCHMARKS.md.
#
# Usage (fresh pod, one line):
#   export GH_PAT=ghp_xxx HF_TOKEN=hf_xxx
#   curl -sSL -H "Authorization: token $GH_PAT" \
#     https://raw.githubusercontent.com/matteso1/thaw/main/demos/bench_cold_cache.sh \
#     | GH_PAT=$GH_PAT HF_TOKEN=$HF_TOKEN bash
#
# Or after the repo is cloned:
#   cd thaw && GH_PAT=ghp_xxx HF_TOKEN=hf_xxx bash demos/bench_cold_cache.sh
#
# Env vars:
#   GH_PAT     (required) GitHub personal access token with repo:read
#   HF_TOKEN   (required) HuggingFace token with access to Meta-Llama-3-8B
#   MODEL      (optional) default: meta-llama/Meta-Llama-3-8B
#   SNAPSHOT   (optional) default: /workspace/snap.thaw
#   WORKDIR    (optional) default: /workspace
#   SKIP_SETUP (optional) set to 1 to skip clone/install/build (for reruns)

set -euo pipefail

: "${GH_PAT:?Need GH_PAT — GitHub PAT for private repo clone}"
: "${HF_TOKEN:?Need HF_TOKEN — HuggingFace token for gated Llama models}"

MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B}"
SNAPSHOT="${SNAPSHOT:-/workspace/snap.thaw}"
WORKDIR="${WORKDIR:-/workspace}"
SKIP_SETUP="${SKIP_SETUP:-0}"

LOG_DIR="$WORKDIR/thaw_bench_logs"
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_${STAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

log "===================================================="
log "thaw cold-cache benchmark run"
log "  model:     $MODEL"
log "  snapshot:  $SNAPSHOT"
log "  workdir:   $WORKDIR"
log "  log:       $MAIN_LOG"
log "===================================================="

log "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader | tee -a "$MAIN_LOG"

log "=== Storage info ==="
df -h "$WORKDIR" 2>&1 | tee -a "$MAIN_LOG"
log "=== Disk write speed (reference) ==="
dd if=/dev/zero of="$WORKDIR/.ddtest" bs=1M count=2048 oflag=direct 2>&1 | tail -1 | tee -a "$MAIN_LOG"
rm -f "$WORKDIR/.ddtest"

if [ "$SKIP_SETUP" != "1" ]; then
    log "=== Installing system deps ==="
    apt-get update -qq >/dev/null 2>&1 || true
    apt-get install -y -qq git build-essential >/dev/null 2>&1 || true

    log "=== Installing Rust (if missing) ==="
    if ! command -v cargo >/dev/null 2>&1; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y >/dev/null
        source "$HOME/.cargo/env"
    else
        source "$HOME/.cargo/env" 2>/dev/null || true
    fi

    log "=== Cloning thaw (private) ==="
    cd "$WORKDIR"
    if [ -d thaw ]; then
        log "  thaw/ already exists — pulling latest"
        cd thaw && git pull
    else
        git clone "https://${GH_PAT}@github.com/matteso1/thaw.git"
        cd thaw
    fi
    # Scrub PAT from remote so it doesn't linger in git config on the pod
    git remote set-url origin "https://github.com/matteso1/thaw.git"

    log "=== Installing vLLM + maturin ==="
    pip install -q vllm "maturin[patchelf]" huggingface_hub 2>&1 | tail -3 | tee -a "$MAIN_LOG"

    log "=== Installing thaw Python packages (editable) ==="
    pip install -q -e . 2>&1 | tail -3 | tee -a "$MAIN_LOG"

    log "=== Building Rust pipelined backend (CUDA) ==="
    # `maturin develop` requires a virtualenv; RunPod uses system Python.
    # Build a wheel and pip install it directly to avoid the venv requirement.
    cd "$WORKDIR/thaw/crates/thaw-py"
    maturin build --release --features cuda 2>&1 | tail -5 | tee -a "$MAIN_LOG"
    pip install --quiet --force-reinstall --no-deps \
        "$WORKDIR/thaw/target/wheels/"thaw_native-*.whl 2>&1 | tee -a "$MAIN_LOG"
    cd "$WORKDIR/thaw"
else
    log "=== SKIP_SETUP=1 — using existing install ==="
    cd "$WORKDIR/thaw"
fi

log "=== HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -2 | tee -a "$MAIN_LOG"

log "=== Sanity: Rust extension importable ==="
python3 -c "import thaw; print('thaw (Rust ext) OK:', thaw.__file__)" 2>&1 | tee -a "$MAIN_LOG"
python3 -c "import thaw_vllm; print('thaw_vllm OK:', thaw_vllm.__file__)" 2>&1 | tee -a "$MAIN_LOG"

log "===================================================="
log "Running vllm_demo.py with --profile-init"
log "  This runs 4 phases:"
log "    1  normal vLLM cold start (baseline)"
log "    2  freeze to .thaw"
log "    --> drop_file_from_cache (posix_fadvise) <--"
log "    3  HONEST cold-cache NVMe restore   [headline]"
log "    3b warm-cache restore               [old headline, for comparison]"
log "    4  pre-staged RAM restore           [upper bound]"
log "===================================================="

BENCH_LOG="$LOG_DIR/vllm_demo_${STAMP}.log"
set +e
python3 python/vllm_demo.py \
    --model "$MODEL" \
    --snapshot "$SNAPSHOT" \
    --profile-init 2>&1 | tee "$BENCH_LOG"
BENCH_RC=$?
set -e

log ""
log "===================================================="
log "Benchmark complete (exit=$BENCH_RC)"
log "  full log:     $BENCH_LOG"
log "  session log:  $MAIN_LOG"
log "===================================================="

log ""
log "=== Quick grep for headline numbers ==="
grep -E "Normal vLLM|cold|warm|Pre-staged|HEADLINE|Speedup|cache\]" "$BENCH_LOG" | tee -a "$MAIN_LOG" || true

log ""
log "=== Next steps ==="
log "  1. Paste $BENCH_LOG contents back to Claude"
log "  2. Claude replaces 'pending re-run' placeholders in README/BENCHMARKS"
log "  3. Sync to public repo (branch + PR)"

exit $BENCH_RC
