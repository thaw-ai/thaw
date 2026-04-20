"""
Receipt builder for thaw demos.

Produces sanitized JSON receipts that can be posted publicly without
leaking pod-specific paths, tokens, or credentials. Intended for
site/receipts/.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version as _pkg_version, PackageNotFoundError  # type: ignore


RECEIPT_VERSION = 1


_PATH_PATTERNS = [
    # Strip absolute workspace / tmp paths that might leak.
    re.compile(r"/tmp/thaw_fork_[A-Za-z0-9_]+/?"),
    re.compile(r"/workspace/[^\s'\"]*"),
    re.compile(r"/root/[^\s'\"]*"),
    re.compile(r"/home/[^/]+/[^\s'\"]*"),
    re.compile(r"/Users/[^/]+/[^\s'\"]*"),
]

_SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"pypi-[A-Za-z0-9_-]{20,}"),
]


def sanitize_text(s: str) -> str:
    """Strip absolute paths and known secret patterns."""
    if not isinstance(s, str):
        return s
    out = s
    for pat in _PATH_PATTERNS:
        out = pat.sub("<PATH>", out)
    for pat in _SECRET_PATTERNS:
        out = pat.sub("<REDACTED>", out)
    return out


def truncate_preview(s: str, max_chars: int = 200) -> str:
    s = sanitize_text((s or "").strip().replace("\n", " "))
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + "…"


def _pkg(name: str) -> str | None:
    try:
        return _pkg_version(name)
    except Exception:
        return None


def software_info() -> dict[str, str | None]:
    import sys
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "vllm": _pkg("vllm"),
        "torch": _pkg("torch"),
        "thaw_native": _pkg("thaw-native"),
        "thaw_vllm": _pkg("thaw-vllm"),
    }


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5,
        )
        return out.decode().strip().splitlines()[0].strip()
    except Exception:
        return None


def pod_info() -> dict[str, Any]:
    gpu_name = _nvidia_smi("name")
    mem_total = _nvidia_smi("memory.total")
    driver = _nvidia_smi("driver_version")
    cuda = None
    try:
        import torch
        cuda = torch.version.cuda
    except Exception:
        pass
    return {
        "gpu_name": gpu_name,
        "gpu_memory_mib": int(mem_total) if (mem_total or "").isdigit() else None,
        "driver_version": driver,
        "cuda_version": cuda,
    }


def git_info() -> dict[str, str | None]:
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, timeout=5,
            )
            return out.decode().strip() or None
        except Exception:
            return None
    commit = _git("rev-parse", "HEAD")
    short = _git("rev-parse", "--short", "HEAD")
    dirty = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        dirty = bool(status)
    except Exception:
        pass
    return {"commit": commit, "short": short, "dirty": dirty}


def build_receipt(
    *,
    demo: str,
    model: str,
    trunk_tokens: int | None,
    timings: dict[str, float],
    modes: dict[str, dict[str, Any]] | None = None,
    checks: dict[str, Any] | None = None,
    samples: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a sanitized receipt dict ready to write as JSON."""
    return {
        "receipt_version": RECEIPT_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "demo": demo,
        "model": model,
        "trunk_tokens": trunk_tokens,
        "timings_s": {k: round(float(v), 3) for k, v in timings.items()},
        "modes": modes or {},
        "checks": checks or {},
        "samples": [
            {k: (truncate_preview(v, 200) if isinstance(v, str) else v) for k, v in s.items()}
            for s in (samples or [])
        ],
        "pod": pod_info(),
        "software": software_info(),
        "git": git_info(),
        "extra": extra or {},
    }


def write_receipt(path: str | os.PathLike, receipt: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=2, sort_keys=False) + "\n")
