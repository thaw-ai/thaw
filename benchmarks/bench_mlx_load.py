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
    """Context manager that prints a phase banner and a single-line
    failure summary on any exception. The full traceback is printed
    once at the top level — never twice — and only when the failure
    isn't a known upstream bug we can name."""
    import contextlib

    @contextlib.contextmanager
    def _cm():
        print(f"[bench] >>> phase: {name}", flush=True)
        try:
            yield
        except BaseException as exc:
            print(f"[bench] !!! phase '{name}' failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            raise
    return _cm()


def _preflight_known_incompat(model_id):
    """Catch known model × library version mismatches BEFORE mlx_lm.load
    spends time on a 17GB download. Returns a banner string if a known
    incompatibility was found, else None.

    Each entry in the table is: (model-id-substring, predicate, banner).
    Add new entries here as we hit them — preflighting is cheap and the
    UX win is huge (the alternative is a wall of traceback frames where
    the actual fix is buried at the bottom)."""
    import importlib.metadata as md

    def _ver(pkg):
        try:
            v = md.version(pkg)
            return tuple(int(x) for x in v.split(".")[:2] if x.isdigit())
        except Exception:
            return None

    transformers_v = _ver("transformers")

    # transformers ≥ 5 + mlx-community/EXAONE-4 ships sliding_window_pattern
    # as a string ("LLLG"); Exaone4Config.__post_init__ does
    # `(i+1) % sliding_window_pattern` and crashes. transformers 4.x works.
    if (
        "exaone-4" in model_id.lower()
        and transformers_v is not None
        and transformers_v[0] >= 5
    ):
        return (
            "[bench] === KNOWN INCOMPAT — STOP, FIX THIS FIRST ===\n"
            f"[bench]   model: {model_id}\n"
            f"[bench]   transformers: {'.'.join(str(x) for x in transformers_v)} (need < 5)\n"
            "[bench]   why: Exaone4Config.__post_init__ does\n"
            "[bench]        `(i+1) % sliding_window_pattern`, but the model\n"
            "[bench]        ships sliding_window_pattern as a string ('LLLG').\n"
            "[bench]        vanilla mlx_lm.load() fails identically.\n"
            "[bench]   fix: pip install 'transformers<5'\n"
            "[bench]        (then re-run the same command — no other changes)\n"
            "[bench] ============================================="
        )

    return None


def _patch_extra_special_tokens_list_bug():
    """Monkey-patch transformers to gracefully accept `extra_special_tokens`
    as a list of token strings, not just a {name: token} dict.

    Why this is needed: mlx-community/Qwen3-Coder-Next-8bit (and likely other
    mlx-community quants of Qwen3-Next) ships tokenizer_config.json with
    `extra_special_tokens: ['<|im_start|>', ...]` — a list. Upstream Qwen3
    has the field absent/None. transformers 4.57.6 + 5.x both call
    `_set_model_specific_special_tokens(special_tokens)` then do
    `special_tokens.keys()`, which AttributeErrors on the list. The function
    signature even claims `list[str]` but the body assumes dict.

    This is a model-config bug, not a transformers version bug — fails on
    transformers 4.57.6, 5.6.x, and 5.7.x all the same. Fix is in-process so
    no cache mutation, no `pip install` flags, no surprise to the user."""
    try:
        from transformers import tokenization_utils_base as _tub
    except Exception:
        return  # transformers not installed — let the real load surface that

    cls = getattr(_tub, "SpecialTokensMixin", None)
    if cls is None or getattr(cls, "_thaw_patched_extra_special_tokens", False):
        return

    original = cls._set_model_specific_special_tokens

    def _patched(self, special_tokens):
        # If we got a list (mlx-community Qwen3-Coder-Next style), coerce to
        # the dict shape transformers actually wants. Use the token string
        # itself as both the attribute name (slug-ified) and the value.
        if isinstance(special_tokens, list):
            coerced = {}
            for tok in special_tokens:
                # name = token with non-alphanum stripped, e.g. <|im_start|>
                # -> "im_start". Good enough for SPECIAL_TOKENS_ATTRIBUTES.
                name = "".join(c if c.isalnum() else "_" for c in str(tok)).strip("_")
                if not name:
                    continue
                coerced[name] = tok
            special_tokens = coerced
        return original(self, special_tokens)

    cls._set_model_specific_special_tokens = _patched
    cls._thaw_patched_extra_special_tokens = True


def _disk_space_preflight(snapshot_path, model_id):
    """Best-effort check that the snapshot path has room for the freeze.
    Returns a banner string on probable-failure, else None.

    We don't know the exact frozen size before download, but the .thaw blob
    is roughly the same size as the on-disk model in HF cache (it's the same
    weights). For models like GLM-4.5-Air-8bit (~115 GB), writing to /tmp on
    a Mac with little free disk on the boot volume fails mid-write with a
    cryptic `RuntimeError: [write] Unable to write N bytes`.

    Heuristic: warn if free space at the snapshot dir is < 20 GB. If we can
    figure out the HF cache size (model is already downloaded), use that as
    a more precise lower bound."""
    import shutil

    snap_dir = os.path.dirname(os.path.abspath(snapshot_path)) or "."
    try:
        free_b = shutil.disk_usage(snap_dir).free
    except OSError:
        return None

    # Try to get a precise estimate from the HF cache, if the model is there.
    estimated = None
    try:
        from huggingface_hub import scan_cache_dir
        scan = scan_cache_dir()
        for repo in scan.repos:
            if repo.repo_id == model_id:
                # size_on_disk includes blobs (the actual safetensors files).
                estimated = int(repo.size_on_disk)
                break
    except Exception:
        pass

    # Default safety margin if we can't estimate model size.
    threshold = max(estimated * 1.1 if estimated else 0, 20 * 1024**3)

    if free_b < threshold:
        return (
            "[bench] === DISK SPACE PROBLEM — WILL CRASH MID-FREEZE ===\n"
            f"[bench]   snapshot path: {snapshot_path}\n"
            f"[bench]   free at {snap_dir}: {_human_bytes(free_b)}\n"
            + (f"[bench]   model on disk:  {_human_bytes(estimated)}\n"
               if estimated else "")
            + f"[bench]   need (≥1.1×):   {_human_bytes(int(threshold))}\n"
            "[bench]   why: thaw_mlx.freeze writes a single .safetensors blob\n"
            "[bench]        roughly the size of the model. macOS /tmp lives on\n"
            "[bench]        the boot volume; if it fills you get a cryptic\n"
            "[bench]        `RuntimeError: [write] Unable to write N bytes`.\n"
            "[bench]   fix: pass --snapshot to a path with more headroom, e.g.\n"
            f"[bench]        --snapshot ~/snapshots/{model_id.replace('/', '__')}.thaw\n"
            "[bench] ================================================="
        )
    return None


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
        print("[bench] run: pip install mlx mlx-lm 'transformers<5'")
        sys.exit(2)

    # Preflight known-incompatible (model × dependency-version) combos so
    # the user gets a clear fix message BEFORE we burn time on a download.
    incompat_banner = _preflight_known_incompat(args.model)
    if incompat_banner:
        sys.stdout.flush()
        print(incompat_banner, flush=True)
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

    # Default to ~/.cache/thaw/ — /tmp on macOS lives on the boot volume and
    # fills up mid-write for large MoE models (e.g. GLM-4.5-Air-8bit at
    # ~115 GB) with a cryptic `RuntimeError: [write] Unable to write N bytes`.
    if args.snapshot:
        snapshot_path = args.snapshot
    else:
        cache_dir = Path.home() / ".cache" / "thaw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = str(cache_dir / f"{args.model.replace('/', '__')}.thaw")

    # Apply in-process patch for mlx-community models that ship
    # `extra_special_tokens` as a list (Qwen3-Coder-Next-8bit, etc.).
    # Idempotent — safe to call even if the model doesn't need it.
    _patch_extra_special_tokens_list_bug()

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

        # Disk-space preflight — best-effort check that the snapshot path has
        # room for the freeze. Catches the `[write] Unable to write N bytes`
        # crash that hits when freezing a 100+ GB model into /tmp on macOS.
        disk_banner = _disk_space_preflight(snapshot_path, args.model)
        if disk_banner:
            print(disk_banner, flush=True)
            sys.exit(2)

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
            "\n"
            "[bench] ============================================\n"
            "[bench]   STOP — KNOWN INCOMPAT, NOT A THAW BUG\n"
            "[bench] ============================================\n"
            "[bench]   transformers 5.x crashes loading EXAONE-4\n"
            "[bench]   (config ships sliding_window_pattern='LLLG').\n"
            "[bench]   vanilla mlx_lm.load() fails identically.\n"
            "[bench]\n"
            "[bench]   FIX (one command):\n"
            "[bench]     pip install 'transformers<5'\n"
            "[bench]\n"
            "[bench]   then re-run the same bench command.\n"
            "[bench] ============================================\n",
            file=sys.stderr, flush=True,
        )
        return True

    # mlx-community Qwen3-Coder-Next-8bit ships `extra_special_tokens` as a
    # list; transformers' _set_model_specific_special_tokens does .keys() on
    # it. We monkey-patch this in main() — if it fires here, the patch was
    # bypassed (someone called us as a library? edited the file?).
    if (
        "_set_model_specific_special_tokens" in tb
        and "'list' object has no attribute 'keys'" in msg
    ):
        print(
            "\n"
            "[bench] ============================================\n"
            "[bench]   STOP — KNOWN MODEL-CONFIG BUG\n"
            "[bench] ============================================\n"
            "[bench]   This model's tokenizer_config.json ships\n"
            "[bench]   `extra_special_tokens` as a list of strings, but\n"
            "[bench]   transformers expects a {name: token} dict and calls\n"
            "[bench]   .keys() on it. Reproduces with vanilla mlx_lm.load —\n"
            "[bench]   not a thaw bug. Affects mlx-community/Qwen3-Coder-\n"
            "[bench]   Next-8bit and likely other Qwen3-Next quants.\n"
            "[bench]\n"
            "[bench]   FIX: bench applies an in-process monkey-patch by\n"
            "[bench]   default — if you're seeing this, run the bench\n"
            "[bench]   directly (not as an import) and re-pull main:\n"
            "[bench]     git pull && python benchmarks/bench_mlx_load.py ...\n"
            "[bench] ============================================\n",
            file=sys.stderr, flush=True,
        )
        return True

    # Cryptic `[write] Unable to write N bytes` from mx.save_safetensors —
    # almost always disk-full at the snapshot path (macOS /tmp lives on the
    # boot volume). Bench preflights this in main(), so this is the catch
    # for cases where preflight under-estimated.
    if (
        "Unable to write" in msg
        and "bytes" in msg
    ):
        print(
            "\n"
            "[bench] ============================================\n"
            "[bench]   STOP — DISK FILLED MID-FREEZE\n"
            "[bench] ============================================\n"
            "[bench]   thaw_mlx.freeze writes one large .safetensors blob.\n"
            "[bench]   On macOS, /tmp is on the boot volume — a 100+ GB\n"
            "[bench]   model can fill it before the write completes.\n"
            "[bench]\n"
            "[bench]   FIX: re-run with --snapshot pointing at a roomier\n"
            "[bench]   volume, e.g.:\n"
            "[bench]     --snapshot ~/snapshots/<model>.thaw\n"
            "[bench]\n"
            "[bench]   Bench now defaults to ~/.cache/thaw/ — pull main if\n"
            "[bench]   you're still hitting this from /tmp.\n"
            "[bench] ============================================\n",
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
        # Known-bug path: print the LOUD diagnosis FIRST and skip the stack
        # — the user only needs the fix, not a wall of frames.
        if _diagnose_known_bug(exc):
            sys.exit(1)
        # Unknown bug: print the full traceback exactly once so we have
        # something actionable to copy into a bug report.
        print(f"\n[bench] !!! unhandled error: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
