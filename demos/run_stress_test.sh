#!/bin/bash
# Run thaw stress test on a RunPod / Lambda / bare metal GPU box.
#
# Usage:
#   export HF_TOKEN=hf_xxxxx
#   export GITHUB_PAT=ghp_xxxxx   # only needed for private repo
#   bash demos/run_stress_test.sh [--quick]
#
# Quick mode (~15 min, small models only):
#   bash demos/run_stress_test.sh --quick
#
# Full suite (~2-3 hours, all models including 70B+):
#   bash demos/run_stress_test.sh

set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: Set HF_TOKEN first"
    echo "  export HF_TOKEN=hf_xxxxx"
    exit 1
fi

# Use /workspace for HF cache (large volume, not small root disk)
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
echo "  HF_HOME=$HF_HOME"

echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Storage Info ==="
df -h /dev/shm /workspace /tmp 2>/dev/null || true
echo ""

# ── Step 1: Check what's already installed ────────────────────
echo "=== Checking environment ==="
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "none")
VLLM_VER=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none")
echo "  torch: $TORCH_VER"
echo "  vllm:  $VLLM_VER"

# ── Step 2: Install vllm if needed (respects pre-installed torch) ─
if [ "$VLLM_VER" = "none" ]; then
    echo "=== Installing vllm ==="
    pip install vllm -q 2>&1 | tail -3
else
    # Verify vllm actually works (not just installed but ABI-broken)
    if ! python -c "from vllm import LLM" 2>/dev/null; then
        echo "=== vllm broken, reinstalling ==="
        pip install vllm --force-reinstall --no-cache-dir -q 2>&1 | tail -3
    fi
fi

# Verify vllm works
python -c "from vllm import LLM; print('  vllm: OK')"

# ── Step 3: Clone/update repo ────────────────────────────────
echo "=== Setting up thaw ==="
cd /workspace
if [ -d thaw ]; then
    cd thaw && git pull
else
    if [ -n "${GITHUB_PAT:-}" ]; then
        git clone "https://${GITHUB_PAT}@github.com/matteso1/thaw.git"
    else
        git clone https://github.com/thaw-ai/thaw.git
    fi
    cd thaw
fi

# ── Step 4: Install thaw Python package (no-deps to avoid touching torch) ─
echo "=== Installing thaw ==="
pip install -e . --no-deps -q 2>&1 | tail -1
pip install fastapi uvicorn -q 2>&1 | tail -1

# Verify thaw_vllm works
python -c "import thaw_vllm; print('  thaw_vllm: OK')"

# ── Step 5: Build Rust extension ─────────────────────────────
echo "=== Building Rust extension ==="
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable 2>&1 | tail -3
    source "$HOME/.cargo/env"
fi

pip install "maturin[patchelf]" -q 2>&1 | tail -1
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
maturin build --release -m crates/thaw-py/Cargo.toml --features cuda --skip-auditwheel 2>&1 | tail -3

# Install the Rust native extension (.so).
# The wheel is now named thaw-native (separate from thaw-vllm Python package)
# so pip won't uninstall one when installing the other.
pip install target/wheels/thaw_native-*.whl --no-deps --force-reinstall -q 2>&1 | tail -1

# ── Step 6: Final verification ───────────────────────────────
echo "=== Verifying all imports ==="
python -c "
import torch
from vllm import LLM
import thaw_vllm
from thaw_vllm import freeze_model_pipelined, restore_model_pipelined
try:
    import thaw
    has_freeze = hasattr(thaw, 'freeze_to_file_pipelined')
    has_restore = hasattr(thaw, 'restore_from_file_pipelined')
    if has_freeze and has_restore:
        rust = 'OK (Rust pipelined freeze + restore)'
    else:
        rust = f'PARTIAL (freeze={has_freeze}, restore={has_restore})'
except ImportError:
    rust = 'NOT LOADED — will use Python fallback (slower)'
print(f'  torch:     {torch.__version__}')
print(f'  vllm:      OK')
print(f'  thaw_vllm: OK')
print(f'  thaw rust: {rust}')
print()
if 'NOT LOADED' in rust:
    print('  WARNING: Rust backend not available. Throughput will be ~4 GB/s instead of ~14 GB/s.')
    print('  Check the maturin build output above for errors.')
else:
    print('  All imports OK. Rust backend active. Ready to test.')
"

# ── Step 7: HuggingFace login ────────────────────────────────
echo "=== HuggingFace login ==="
huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

# ── Step 8: Run stress test ──────────────────────────────────
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
