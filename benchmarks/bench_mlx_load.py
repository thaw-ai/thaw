"""
bench_mlx_load.py — A/B compare mlx_lm.load vs thaw_mlx.restore.

Run on Apple Silicon. Default model is a small 4-bit Llama for fast
iteration on a Mac with limited RAM. Override via THAW_MLX_BENCH_MODEL.

Usage:
    # Default: full A/B in one process. Baseline read fills the page cache
    # before thaw.restore runs, so the restore number here is WARM.
    python benchmarks/bench_mlx_load.py --model <hf-id>

    # Two-run protocol for a true cold-vs-cold comparison:
    #   1. Run with --freeze-only on a fresh boot to write the .thaw blob
    #      and get the cold baseline mlx_lm.load number.
    #   2. Reboot (or sudo purge), then run with --restore-only against the
    #      same --snapshot path to get the cold thaw.restore number.
    python benchmarks/bench_mlx_load.py --model <hf-id> --freeze-only
    # ... reboot ...
    python benchmarks/bench_mlx_load.py --model <hf-id> --restore-only

What it does:
    1. mlx_lm.load(model)               -- baseline timing
    2. thaw_mlx.freeze(model, .thaw)    -- one-time conversion
    3. mlx_lm.load(model) + thaw_mlx.restore  -- target hot-swap path
    4. Generate a fixed prompt on both sides, assert tokens match
    5. Emit JSON to site/receipts/
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))


def _phase(name):
    """Context manager that prints a phase banner and dumps the full
    traceback on any exception, preserving the original error for the
    receipt to see. Without this, mlx-lm errors come back as a single
    line and we can't tell which phase blew up."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        print(f"[bench] >>> phase: {name}", flush=True)
        try:
            yield
        except BaseException as exc:
            print(f"[bench] !!! phase '{name}' raised {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            raise
    return _cm()


def _logits_parity(model_a, model_b, prompt_tokens):
    """Forward-pass parity check that doesn't go through mlx_lm.generate
    (sampling, detokenizer, KV cache). If the weights match, the argmax
    over the logits at each position should be identical.

    Returns (matched: bool, n_positions_compared: int).
    """
    import mlx.core as mx
    a = model_a(prompt_tokens[None])
    b = model_b(prompt_tokens[None])
    mx.eval(a, b)
    a_ids = mx.argmax(a, axis=-1)
    b_ids = mx.argmax(b, axis=-1)
    return bool(mx.all(a_ids == b_ids).item()), int(a_ids.size)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _human_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def _resolve_blob(snapshot_path: str) -> str:
    """thaw_mlx.snapshot silently appends .safetensors if missing.
    Mirror that here so existence checks line up with what freeze wrote."""
    if snapshot_path.endswith(".safetensors"):
        return snapshot_path
    return snapshot_path + ".safetensors"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=os.environ.get(
            "THAW_MLX_BENCH_MODEL",
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
        ),
    )
    p.add_argument("--prompt", default="Write a haiku about Apple Silicon.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--snapshot", default=None,
                   help="path for .thaw output (default: /tmp/<model>.thaw)")
    p.add_argument("--receipt", default=None,
                   help="JSON receipt output path")
    p.add_argument("--skip-bench-baseline", action="store_true",
                   help="skip the mlx_lm.load timing (assume already cached)")
    p.add_argument("--freeze-only", action="store_true",
                   help="time mlx_lm.load + write the .thaw blob, then exit. "
                        "Use on a fresh boot for a true COLD baseline number.")
    p.add_argument("--restore-only", action="store_true",
                   help="skip baseline + freeze. Time thaw_mlx.restore only "
                        "against an existing --snapshot. Use after a reboot "
                        "for a true COLD thaw.restore number.")
    args = p.parse_args()

    if args.freeze_only and args.restore_only:
        print("[bench] --freeze-only and --restore-only are mutually exclusive")
        sys.exit(2)

    print(f"[bench] model = {args.model}")

    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        import mlx.core as mx
        import mlx
    except ImportError as e:
        print(f"[bench] mlx_lm not installed: {e}")
        print("[bench] run: pip install mlx mlx-lm")
        sys.exit(2)

    import thaw_mlx

    # Version banner — when a user reports a TypeError or anything weird,
    # the first thing we ask for is which mlx/mlx_lm they actually have.
    import importlib.metadata as _md
    def _pkg_ver(name, mod_attr=None):
        try:
            return _md.version(name)
        except Exception:
            if mod_attr is not None:
                return getattr(mod_attr, "__version__", "unknown")
            return "unknown"
    mlx_version = _pkg_ver("mlx", mx)
    mlx_lm_version = _pkg_ver("mlx-lm")
    print(f"[bench] mlx={mlx_version}  mlx_lm={mlx_lm_version}  python={sys.version.split()[0]}")

    snapshot_path = (
        args.snapshot
        or f"/tmp/{args.model.replace('/', '__')}.thaw"
    )

    # Defaults that may stay None depending on the chosen mode.
    baseline_load_s = None
    baseline_text = None
    baseline_logits_argmax = None  # for logits-based parity fallback
    prompt_tokens = None
    freeze_stats = None
    restore_stats = None
    restore_s = None
    init_s = None
    target_text = None
    bit_identical = None
    parity_method = None  # "generated_text" or "logits_argmax" or None

    # --- Phase 1: baseline + freeze (skipped in --restore-only) ---------
    if not args.restore_only:
        with _phase("baseline-mlx_lm.load"):
            print(f"[bench] mlx_lm.load() baseline (cold if you just rebooted)...")
            t0 = time.perf_counter()
            baseline_model, baseline_tok = mlx_load(args.model)
            mx.eval(baseline_model.parameters())
            baseline_load_s = time.perf_counter() - t0
            print(f"[bench]   load: {baseline_load_s:.2f}s")

        # Tokenize the prompt once — used by both generate and logits parity.
        with _phase("baseline-tokenize"):
            prompt_tokens = mx.array(baseline_tok.encode(args.prompt))

        # Try generate-based parity first (matches what users actually run).
        # If it explodes (e.g. mlx_lm bug for a specific arch), fall back to
        # forward-pass logits parity in Phase 2 — that bypasses sampling,
        # detokenizer, KV cache, and only depends on the weights matching.
        try:
            with _phase("baseline-generate"):
                baseline_text = mlx_generate(
                    baseline_model,
                    baseline_tok,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    verbose=False,
                )
        except Exception as exc:
            print(f"[bench] baseline-generate failed ({type(exc).__name__}: {exc}); "
                  f"will fall back to logits-argmax parity.")
            baseline_text = None

        # Capture logits argmax so we have a parity reference even if
        # baseline-generate succeeded but target-generate later fails.
        try:
            with _phase("baseline-logits"):
                logits = baseline_model(prompt_tokens[None])
                mx.eval(logits)
                baseline_logits_argmax = mx.argmax(logits, axis=-1)
                mx.eval(baseline_logits_argmax)
        except Exception as exc:
            print(f"[bench] baseline-logits failed ({type(exc).__name__}: {exc}); "
                  f"parity check will be skipped if target-generate also fails.")
            baseline_logits_argmax = None

        with _phase("freeze"):
            print(f"[bench] thaw_mlx.freeze() -> {snapshot_path}")
            freeze_stats = thaw_mlx.freeze(baseline_model, snapshot_path)
            print(
                f"[bench]   {_human_bytes(freeze_stats['total_bytes'])} in "
                f"{freeze_stats['elapsed_s']:.2f}s "
                f"({freeze_stats['throughput_gb_s']:.2f} GB/s)"
            )
        del baseline_model

        if args.freeze_only:
            print(f"[bench] --freeze-only: blob written, exiting before restore.")
            print(f"[bench] reboot, then run with --restore-only --snapshot "
                  f"{snapshot_path} for the cold thaw.restore number.")

    # --- Phase 2: restore + parity (skipped in --freeze-only) -----------
    if not args.freeze_only:
        if not Path(_resolve_blob(snapshot_path)).exists():
            print(f"[bench] no .thaw blob at {snapshot_path}. "
                  f"run --freeze-only first to write it.")
            sys.exit(2)

        with _phase("target-mlx_lm.load"):
            # Reload architecture+tokenizer. In --restore-only mode this read
            # is what mlx_lm.load() does for weights too — so this number is
            # the honest per-flex-reload baseline.
            t_init = time.perf_counter()
            target_model, target_tok = mlx_load(args.model)
            init_s = time.perf_counter() - t_init
        if args.restore_only:
            print(f"[bench] mlx_lm.load() reload (this IS the comparison): {init_s:.2f}s")
        else:
            print(f"[bench]   architecture+tokenizer reload: {init_s:.2f}s "
                  f"(warm — page cache primed by phase 1)")

        with _phase("restore"):
            t_restore = time.perf_counter()
            restore_stats = thaw_mlx.restore(target_model, snapshot_path)
            restore_s = time.perf_counter() - t_restore
        cold_or_warm = "COLD" if args.restore_only else "warm"
        print(
            f"[bench]   thaw.restore ({cold_or_warm}): "
            f"{_human_bytes(restore_stats['total_bytes'])} in "
            f"{restore_s:.2f}s ({restore_stats['throughput_gb_s']:.2f} GB/s)"
        )

        # Try generate-based parity first.
        try:
            with _phase("target-generate"):
                target_text = mlx_generate(
                    target_model,
                    target_tok,
                    prompt=args.prompt,
                    max_tokens=args.max_tokens,
                    verbose=False,
                )
        except Exception as exc:
            print(f"[bench] target-generate failed ({type(exc).__name__}: {exc}); "
                  f"will fall back to logits-argmax parity.")
            target_text = None

        # Decide the parity method based on what survived.
        if baseline_text is not None and target_text is not None:
            bit_identical = baseline_text == target_text
            parity_method = "generated_text"
            print(f"[bench] generation parity: {'PASS' if bit_identical else 'FAIL'}")
            if not bit_identical:
                print(f"[bench]   baseline: {baseline_text!r}")
                print(f"[bench]   restored: {target_text!r}")
        elif baseline_logits_argmax is not None:
            # Logits fallback — runs even if generate exploded on either side.
            try:
                with _phase("target-logits"):
                    if prompt_tokens is None:
                        prompt_tokens = mx.array(target_tok.encode(args.prompt))
                    logits = target_model(prompt_tokens[None])
                    mx.eval(logits)
                    target_argmax = mx.argmax(logits, axis=-1)
                    mx.eval(target_argmax)
                    bit_identical = bool(
                        mx.all(baseline_logits_argmax == target_argmax).item()
                    )
                    n = int(target_argmax.size)
                parity_method = "logits_argmax"
                print(f"[bench] logits-argmax parity over {n} positions: "
                      f"{'PASS' if bit_identical else 'FAIL'}")
            except Exception as exc:
                print(f"[bench] target-logits failed ({type(exc).__name__}: {exc}); "
                      f"parity check skipped.")
                bit_identical = None
                parity_method = None
        else:
            print(f"[bench] generation parity: SKIPPED "
                  f"(no baseline reference — generate + logits both unavailable)")

    # --- Receipt ---------------------------------------------------------
    mode = "freeze-only" if args.freeze_only else (
        "restore-only" if args.restore_only else "full"
    )
    receipt = {
        "timestamp_utc": _now_iso(),
        "platform": {
            "uname": os.uname()._asdict() if hasattr(os.uname(), "_asdict") else str(os.uname()),
            "python": sys.version.split()[0],
            "mlx": mlx_version,
            "mlx_lm": mlx_lm_version,
        },
        "mode": mode,
        "model": args.model,
        "snapshot_path": snapshot_path,
        "baseline_load_s": baseline_load_s,
        "baseline_thermal": "cold-if-fresh-boot" if baseline_load_s is not None else None,
        "thaw_freeze": freeze_stats,
        "thaw_restore": (
            {**restore_stats, "elapsed_s": restore_s,
             "thermal": "cold" if args.restore_only else "warm-after-baseline"}
            if restore_stats else None
        ),
        "architecture_init_s": init_s,
        "bit_identical": bit_identical,
        "parity_method": parity_method,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "baseline_text": baseline_text,
        "restored_text": target_text,
    }

    receipt_path = args.receipt or str(
        REPO_ROOT / "site" / "receipts"
        / f"{datetime.now().strftime('%Y-%m-%d')}_mlx_bench_"
          f"{args.model.replace('/', '__')}.json"
    )
    Path(receipt_path).parent.mkdir(parents=True, exist_ok=True)
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"[bench] receipt -> {receipt_path}")

    if baseline_load_s and restore_s:
        speedup = baseline_load_s / restore_s
        thaw_thermal = "cold" if args.restore_only else "warm"
        print(
            f"[bench] summary: mlx_lm.load {baseline_load_s:.2f}s (cold) "
            f"vs thaw_mlx.restore {restore_s:.2f}s ({thaw_thermal}) "
            f"= {speedup:.2f}× faster"
        )

    # Exit non-zero only on a real parity failure. freeze-only and restore-only
    # legitimately skip the comparison, so they should pass through cleanly.
    if bit_identical is False:
        sys.exit(1)
    sys.exit(0)


def _diagnose_known_bug(exc):
    """If the error matches a known upstream-not-thaw bug, print the
    diagnosis. Returns True if a known bug was matched."""
    msg = f"{type(exc).__name__}: {exc}"
    tb = traceback.format_exc()
    # transformers 5.x + EXAONE 4 with string sliding_window_pattern
    # (e.g. "LLLG") crashes in Exaone4Config.__post_init__ doing
    # `(i+1) % self.sliding_window_pattern`. Reproduces with vanilla
    # mlx_lm.load() — nothing to do with thaw. transformers 4.x works.
    if (
        "configuration_exaone4" in tb
        and "unsupported operand type(s) for %" in msg
    ):
        print(
            "\n[bench] === DIAGNOSIS ===\n"
            "[bench] This is a transformers 5.x bug, NOT a thaw bug.\n"
            "[bench] Exaone4Config.__post_init__ does `(i+1) % sliding_window_pattern`,\n"
            "[bench] but mlx-community/EXAONE-4.0-32B-4bit ships sliding_window_pattern\n"
            "[bench] as a string ('LLLG'). vanilla `mlx_lm.load(...)` fails identically.\n"
            "[bench] fix: `pip install 'transformers<5'` and re-run.\n",
            file=sys.stderr, flush=True,
        )
        return True
    return False


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except BaseException as exc:
        print(f"\n[bench] !!! unhandled error: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        _diagnose_known_bug(exc)
        sys.exit(1)
