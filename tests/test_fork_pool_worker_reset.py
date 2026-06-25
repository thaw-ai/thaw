"""
Tests for _fork_pool_worker._reset_prefix_cache — issue #77.

CPU-only. The worker module redirects fd 1 at import time (the IPC
handshake dance), so importing it in-process would wreck pytest's
stdout. Instead each test runs a small driver subprocess that imports
the real worker module by path and calls _reset_prefix_cache against a
fake llm object.

The fake llm has no engine internals, so mechanism 1 (the direct
block-pool hash-map clear) always fails — with vLLM installed it fails
on attribute access, without vLLM it fails on import. Either way the
tests pin the contract added for #77:

  - one mechanism succeeds  → no raise, loud "degraded" warning on
    stderr, counter increments
  - every mechanism fails   → RuntimeError (no more silent pass over
    stale prefix-cache state)
"""

from __future__ import annotations

import os
import subprocess
import sys

WORKER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "python",
    "thaw_vllm",
    "_fork_pool_worker.py",
)

DRIVER = r'''
import importlib.util, sys

worker_path, case = sys.argv[1], sys.argv[2]

spec = importlib.util.spec_from_file_location("fpw", worker_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class GoodLLM:
    def reset_prefix_cache(self):
        pass


class RaisingLLM:
    def reset_prefix_cache(self):
        raise RuntimeError("boom")


class MissingLLM:
    pass


llm = {"good": GoodLLM, "raising": RaisingLLM, "missing": MissingLLM}[case]()

calls = 2 if case == "good" else 1
for _ in range(calls):
    try:
        mod._reset_prefix_cache(llm)
    except RuntimeError as e:
        sys.stderr.write("RAISED: " + str(e) + "\n")
        sys.exit(7)

sys.stderr.write("OK degraded=%d\n" % mod._RESET_DEGRADED_COUNT)
sys.exit(0)
'''


def _run_case(tmp_path, case: str) -> subprocess.CompletedProcess:
    driver = tmp_path / "reset_driver.py"
    driver.write_text(DRIVER)
    return subprocess.run(
        [sys.executable, str(driver), WORKER_PATH, case],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_partial_failure_warns_and_continues(tmp_path):
    """Mechanism 1 fails (fake llm), mechanism 2 succeeds: no raise,
    loud degraded warning, counter increments per call."""
    proc = _run_case(tmp_path, "good")
    assert proc.returncode == 0, proc.stderr
    assert "prefix-cache reset degraded" in proc.stderr
    # Two calls in this case — the counter must accumulate.
    assert "OK degraded=2" in proc.stderr


def test_total_failure_raises(tmp_path):
    """Both mechanisms fail (public reset raises): RuntimeError, not
    a silent pass."""
    proc = _run_case(tmp_path, "raising")
    assert proc.returncode == 7, proc.stderr
    assert "RAISED:" in proc.stderr
    assert "failed on every mechanism" in proc.stderr
    # The error must carry both underlying failures for debuggability.
    assert "hash-map clear:" in proc.stderr
    assert "reset_prefix_cache(): RuntimeError: boom" in proc.stderr


def test_missing_public_reset_raises(tmp_path):
    """No reset_prefix_cache on the llm and internals unreachable:
    RuntimeError naming the absent API."""
    proc = _run_case(tmp_path, "missing")
    assert proc.returncode == 7, proc.stderr
    assert "not exposed by this vLLM version" in proc.stderr
