#!/bin/bash
# pod-bench-freeze.sh — A/B bench the new pipelined-freeze path on a
# RunPod H100 (or any CUDA box). Compares three freeze paths at Rust
# level (no Python / vLLM / model loading):
#
#   A   old freeze_pipelined (BufWriter<File>)        — pre-rewrite baseline (~0.19 GB/s)
#   B'  new freeze_pipelined_to_file, buffered pwrite  — isolates pipeline wins
#   B   new freeze_pipelined_to_file, O_DIRECT pwrite  — full new path (~19.62 GB/s H100)
#
# All three paths write identical bytes (verified inline). The bench
# allocates `SIZE_MB` of device memory, seeds it with a deterministic
# pattern, and measures wall-clock time for each path.
#
# Required env:
#   GH_PAT      GitHub PAT with repo access to matteso1/thaw
#
# Optional env:
#   SIZE_MB     default 16384  (16 GiB — approx Llama-3-8B weights)
#   CHUNK_MB    default 64
#   THAW_DIR    default /workspace/thaw
#   TMPDIR      default /workspace/tmp   (must have ~3×SIZE_MB free)
#
# Usage on vllm/vllm-openai:latest pod:
#   export GH_PAT=...
#   git clone https://$GH_PAT@github.com/matteso1/thaw.git /workspace/thaw
#   bash /workspace/thaw/scripts/pod-bench-freeze.sh
#
# If the checkout already exists, the script will `git fetch + reset --hard`
# to origin/main itself — so to pull the latest just re-run:
#   bash /workspace/thaw/scripts/pod-bench-freeze.sh

set -euo pipefail

: "${GH_PAT:?set GH_PAT}"
SIZE_MB="${SIZE_MB:-16384}"
CHUNK_MB="${CHUNK_MB:-64}"
THAW_DIR="${THAW_DIR:-/workspace/thaw}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$TMPDIR"

banner() { echo ""; echo "=== $* ==="; }

banner "thaw pipelined-freeze A/B bench"
echo "SIZE_MB:  $SIZE_MB"
echo "CHUNK_MB: $CHUNK_MB"
echo "THAW_DIR: $THAW_DIR"
echo "TMPDIR:   $TMPDIR"
echo ""
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
  --format=csv,noheader || true
df -h "$TMPDIR" | tail -1

# Rough disk check: need ~3× SIZE_MB free in TMPDIR.
need_mb=$(( SIZE_MB * 3 ))
avail_mb=$(df --output=avail -BM "$TMPDIR" | tail -1 | tr -dc '0-9')
if [ -n "$avail_mb" ] && [ "$avail_mb" -lt "$need_mb" ]; then
  echo ""
  echo "WARNING: TMPDIR has ${avail_mb} MB free but bench needs ~${need_mb} MB."
  echo "         Reduce SIZE_MB or change TMPDIR to a bigger volume."
fi

# ── Clone / update thaw (private) ────────────────────────────────────
banner "[1/3] Clone + update thaw"
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

# ── Rust toolchain (idempotent) ──────────────────────────────────────
banner "[2/3] Rust toolchain"
if [ -f "$HOME/.cargo/env" ]; then
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  . "$HOME/.cargo/env"
fi
rustc --version
cargo --version

# ── Build + run bench ────────────────────────────────────────────────
banner "[3/3] cargo build --release -p thaw-cli --features cuda"
cargo build --release --quiet \
  -p thaw-cli --features cuda --bin thaw-bench-freeze

BIN="$THAW_DIR/target/release/thaw-bench-freeze"
[ -x "$BIN" ] || { echo "build did not produce $BIN"; exit 1; }

banner "Run: thaw-bench-freeze $SIZE_MB $CHUNK_MB"
"$BIN" "$SIZE_MB" "$CHUNK_MB"

banner "Done."
