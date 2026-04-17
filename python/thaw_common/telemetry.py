"""
thaw_common.telemetry — centralized fallback logging and strict-mode.

thaw's entire value prop is FAST cold starts. A silent fallback to a
slower path is the worst possible bug for this product. A production
user hitting a pinned-memory exhaustion, an O_DIRECT permission denial,
or a Rust extension load failure should NEVER see "restored in 12s" with
no explanation — they should see a WARNING, and in strict mode they
should see an exception.

Use this module wherever there is a performance-critical fallback:

    from thaw_common.telemetry import fallback_warning, strict_mode

    try:
        stats = rust_pipelined(...)
    except Exception as e:
        fallback_warning("restore_model_pipelined", e, dst="python")
        if strict_mode():
            raise
        stats = python_fallback(...)

Environment:
    THAW_STRICT=1    — fallback paths re-raise instead of degrading
    THAW_QUIET=1     — suppress fallback warnings (not recommended)
"""

import logging
import os
import traceback


logger = logging.getLogger("thaw")
if not logger.handlers:
    # Attach a default handler so users see warnings even without
    # explicit logging configuration. WARNING-level by default.
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.WARNING)
    logger.propagate = False


def strict_mode() -> bool:
    """True if THAW_STRICT=1 — fallbacks re-raise instead of degrading."""
    return os.environ.get("THAW_STRICT", "").lower() in ("1", "true", "yes", "on")


def quiet_mode() -> bool:
    """True if THAW_QUIET=1 — suppress fallback warnings."""
    return os.environ.get("THAW_QUIET", "").lower() in ("1", "true", "yes", "on")


def fallback_warning(label: str, exc: BaseException, *, dst: str = "") -> None:
    """Log a performance-path fallback with the original exception.

    Full traceback is logged at DEBUG level so operators can bump the
    log level when they need to diagnose a slowdown without changing
    code.
    """
    if quiet_mode():
        return
    suffix = f" -> {dst}" if dst else ""
    logger.warning(
        "FALLBACK in %s%s (%s: %s). This path is significantly slower. "
        "Set THAW_STRICT=1 to raise instead, or bump log level to DEBUG "
        "for the full traceback.",
        label, suffix, type(exc).__name__, exc,
    )
    logger.debug("Traceback for %s fallback:\n%s", label, traceback.format_exc())


def check_pinned(tensor, name: str = "buffer") -> None:
    """Verify a tensor is actually pinned.

    torch.empty(pin_memory=True) can silently return pageable memory
    under pressure (locked-memory limit reached, pool exhausted). When
    that happens, cudaMemcpyAsync(..., non_blocking=True) downgrades to
    a synchronous transfer and throughput drops 2-5x with NO error
    signal. This check surfaces that condition immediately.
    """
    # is_pinned() only exists on torch.Tensor. For anything else, skip the
    # check rather than mask a genuine error.
    if not hasattr(tensor, "is_pinned"):
        return
    is_pinned = tensor.is_pinned()
    if is_pinned:
        return

    msg = (
        f"{name}: pin_memory=True requested but tensor is NOT pinned. "
        f"cudaMemcpyAsync will fall back to synchronous transfer "
        f"(2-5x slowdown). Usually caused by exhausted pinned-memory "
        f"pool or low locked-memory ulimit. Check: `ulimit -l`, "
        f"nvidia-smi, and host RAM pressure."
    )
    if strict_mode():
        raise RuntimeError(msg)
    logger.warning(msg)
