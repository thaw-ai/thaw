"""
demo.py -- freeze/restore a GPT-2 model to prove the thaw pipeline works.

Usage:
    python demo.py [--model MODEL_NAME] [--output PATH]

Requires: torch, transformers. Does NOT require vLLM -- this exercises the
thaw binary format against a vanilla HuggingFace model on CUDA.

What it does:
  1. Load a model onto GPU the normal way (transformers AutoModel).
  2. Freeze all weights to a .thaw file via thaw_vllm.freeze_model().
  3. Create a fresh (random-initialized) copy of the same model on GPU.
  4. Restore weights from the .thaw file via thaw_vllm.restore_model().
  5. Verify every parameter matches the original byte-for-byte.
  6. Print timing and throughput.
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoConfig

from thaw_vllm import freeze_model, restore_model


def fmt_size(nbytes: int) -> str:
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.2f} GB"
    return f"{nbytes / 1e6:.1f} MB"


def fmt_throughput(nbytes: int, elapsed: float) -> str:
    if elapsed <= 0:
        return "inf"
    gb_s = (nbytes / 1e9) / elapsed
    return f"{gb_s:.2f} GB/s"


def main():
    parser = argparse.ArgumentParser(description="thaw freeze/restore demo")
    parser.add_argument(
        "--model",
        default="gpt2",
        help="HuggingFace model name (default: gpt2, ~500 MB)",
    )
    parser.add_argument(
        "--output",
        default="snapshot.thaw",
        help="path for the .thaw file (default: snapshot.thaw)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. This demo requires a GPU.")
        sys.exit(1)

    device = torch.device("cuda:0")
    thaw_path = args.output

    # -- Step 1: load model normally --
    print(f"[1/5] Loading {args.model} onto GPU...")
    t0 = time.perf_counter()
    model_a = AutoModel.from_pretrained(args.model).to(device).eval()
    load_time = time.perf_counter() - t0

    num_params = sum(p.numel() for p in model_a.parameters())
    total_bytes = sum(p.data.nbytes for p in model_a.parameters() if p.is_cuda)
    print(f"       {num_params:,} parameters, {fmt_size(total_bytes)} on GPU")
    print(f"       normal load: {load_time:.3f}s")

    # -- Step 2: freeze --
    print(f"\n[2/5] Freezing to {thaw_path}...")
    stats = freeze_model(model_a, thaw_path)
    print(f"       {stats['num_regions']} regions, {fmt_size(stats['total_bytes'])}")
    print(f"       freeze: {stats['elapsed_s']:.3f}s ({fmt_throughput(stats['total_bytes'], stats['elapsed_s'])})")

    file_size = Path(thaw_path).stat().st_size
    print(f"       file size: {fmt_size(file_size)}")

    # -- Step 3: create a fresh model (random weights) --
    print(f"\n[3/5] Creating fresh {args.model} on GPU (random weights)...")
    config = AutoConfig.from_pretrained(args.model)
    model_b = AutoModel.from_config(config).to(device).eval()

    # Sanity: the fresh model should NOT match the original.
    mismatch_before = 0
    for (na, pa), (nb, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        assert na == nb, f"parameter order mismatch: {na} vs {nb}"
        if not torch.equal(pa.data, pb.data):
            mismatch_before += 1
    print(f"       {mismatch_before}/{num_params} parameters differ (expected)")

    # -- Step 4: restore --
    print(f"\n[4/5] Restoring from {thaw_path}...")
    rstats = restore_model(model_b, thaw_path)
    print(f"       restore: {rstats['elapsed_s']:.3f}s ({fmt_throughput(rstats['total_bytes'], rstats['elapsed_s'])})")

    # -- Step 5: verify --
    print("\n[5/5] Verifying byte-exact match...")
    mismatches = []
    for (na, pa), (nb, pb) in zip(
        model_a.named_parameters(), model_b.named_parameters()
    ):
        if not torch.equal(pa.data, pb.data):
            mismatches.append(na)

    if mismatches:
        print(f"  FAIL: {len(mismatches)} parameters differ after restore:")
        for name in mismatches[:10]:
            print(f"    - {name}")
        sys.exit(1)
    else:
        print("  PASS: all parameters match byte-for-byte")

    # -- Summary --
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    print(f"  model:          {args.model}")
    print(f"  parameters:     {num_params:,}")
    print(f"  weight size:    {fmt_size(total_bytes)}")
    print(f"  normal load:    {load_time:.3f}s")
    print(f"  thaw freeze:    {stats['elapsed_s']:.3f}s ({fmt_throughput(stats['total_bytes'], stats['elapsed_s'])})")
    print(f"  thaw restore:   {rstats['elapsed_s']:.3f}s ({fmt_throughput(rstats['total_bytes'], rstats['elapsed_s'])})")
    speedup = load_time / rstats['elapsed_s'] if rstats['elapsed_s'] > 0 else float('inf')
    print(f"  speedup:        {speedup:.1f}x faster than normal load")
    print(f"  file:           {thaw_path} ({fmt_size(file_size)})")


if __name__ == "__main__":
    main()
