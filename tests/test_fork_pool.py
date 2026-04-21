"""
Tests for thaw_vllm.fork_pool — ForkPool IPC + lifecycle.

CPU-only. Each test spawns a small Python script that speaks the same
JSON-line protocol as the real worker but does not import vLLM. GPU
behavior is exercised end-to-end by tests/gpu/test_fork_pool_gpu.py on
an H100 pod.

Coverage
--------
  - boot handshake + ready line
  - error during boot → ForkPoolError
  - boot timeout → WorkerBootTimeout
  - simple generate round-trip, single worker, single prompt
  - round-robin prompt → worker assignment, multiple workers
  - worker error during op is propagated as ForkPoolError
  - close() is idempotent and cleans up processes
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time

import pytest


# ---------------------------------------------------------------------------
# Fake worker scripts — write to temp files, pass as the worker_cmd.
# Each script reads one JSON line, writes one JSON line back. No vLLM.
# ---------------------------------------------------------------------------


FAKE_WORKER_HAPPY = r'''
import json, os, sys, time

def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def _recv():
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line)

# Boot
cfg = _recv()
assert cfg["op"] == "boot", cfg
_send({"status": "ready", "boot_s": 0.01, "pid": os.getpid()})

# Command loop
while True:
    req = _recv()
    if req is None:
        sys.exit(0)
    op = req.get("op")
    if op == "shutdown":
        _send({"status": "ok"})
        sys.exit(0)
    if op == "hydrate":
        _send({"status": "ok", "weights_s": 0.1, "kv_s": 0.05,
               "weights": {}, "kv": {}})
    elif op == "generate":
        prompts = req["prompts"]
        outs = [
            {"text": f"echo:{p}", "token_ids": [1, 2, 3],
             "finish_reason": "stop"}
            for p in prompts
        ]
        _send({"status": "ok", "gen_s": 0.02, "outputs": outs})
    elif op == "reset":
        _send({"status": "ok"})
    else:
        _send({"status": "error", "type": "ProtocolError",
               "message": f"unknown op {op}", "traceback": ""})
'''


FAKE_WORKER_BOOT_FAIL = r'''
import json, sys
cfg = json.loads(sys.stdin.readline())
sys.stdout.write(json.dumps(
    {"status": "error", "type": "RuntimeError",
     "message": "boot failed on purpose", "traceback": ""}
) + "\n")
sys.stdout.flush()
sys.exit(3)
'''


FAKE_WORKER_BOOT_HANG = r'''
import sys, time
# Read the boot message, then hang forever without emitting ready.
sys.stdin.readline()
while True:
    time.sleep(10)
'''


FAKE_WORKER_GEN_FAIL = r'''
import json, os, sys

def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n"); sys.stdout.flush()

def _recv():
    line = sys.stdin.readline()
    return json.loads(line) if line else None

_recv()
_send({"status": "ready", "boot_s": 0.0, "pid": os.getpid()})

while True:
    req = _recv()
    if req is None: sys.exit(0)
    op = req["op"]
    if op == "shutdown":
        _send({"status": "ok"}); sys.exit(0)
    if op == "generate":
        _send({"status": "error", "type": "CUDAError",
               "message": "illegal memory access", "traceback": "(fake)"})
    else:
        _send({"status": "ok"})
'''


@pytest.fixture
def script_path(tmp_path):
    def _write(body: str) -> str:
        p = tmp_path / f"fake_worker_{abs(hash(body)) % 10**8}.py"
        p.write_text(body)
        return str(p)
    return _write


def _make_pool(monkeypatch):
    """Import ForkPool with real module, but the boot-time _fork function
    is only used by fork_completions, not by the IPC tests. Tests that
    need fork() are patched locally."""
    from thaw_vllm.fork_pool import ForkPool
    return ForkPool


# ---------------------------------------------------------------------------
# Boot handshake
# ---------------------------------------------------------------------------


def test_init_pool_boots_workers_and_reports_ready(script_path, monkeypatch):
    ForkPool = _make_pool(monkeypatch)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=2, boot_timeout_s=10)
        status = pool.status()
        assert len(status["workers"]) == 2
        for w in status["workers"]:
            assert not w["dead"]
            assert w["pid"] > 0
    finally:
        pool.close()


def test_init_pool_raises_when_worker_boot_errors(script_path):
    from thaw_vllm.fork_pool import ForkPool, ForkPoolError
    script = script_path(FAKE_WORKER_BOOT_FAIL)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    with pytest.raises(ForkPoolError):
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=5)
    pool.close()


def test_init_pool_raises_on_boot_timeout(script_path):
    from thaw_vllm.fork_pool import ForkPool, WorkerBootTimeout
    script = script_path(FAKE_WORKER_BOOT_HANG)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    with pytest.raises(WorkerBootTimeout):
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=0.5)
    pool.close()


def test_init_pool_rejects_second_call(script_path):
    from thaw_vllm.fork_pool import ForkPool, ForkPoolError
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="m", workers=1, boot_timeout_s=5)
        with pytest.raises(ForkPoolError):
            pool.init_pool(model="m", workers=1, boot_timeout_s=5)
    finally:
        pool.close()


def test_close_is_idempotent(script_path):
    from thaw_vllm.fork_pool import ForkPool
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    pool.init_pool(model="m", workers=1, boot_timeout_s=5)
    pool.close()
    pool.close()  # must not raise


# ---------------------------------------------------------------------------
# Dispatch — fork_completions against the pool
# ---------------------------------------------------------------------------


def _install_fake_fork(monkeypatch, tmp_path):
    """Patch thaw_vllm.fork_pool._fork with a stub that writes empty
    weights + kv files and returns a handle pointing at them. Avoids
    pulling in the real vLLM dependency chain.
    """
    import thaw_vllm.fork_pool as fpmod
    from thaw_vllm.fork import ForkHandle

    def _fake_fork(llm, include_weights=False, state_dir=None):
        import tempfile
        sd = state_dir or tempfile.mkdtemp(prefix="fakefork_")
        os.makedirs(sd, exist_ok=True)
        wp = os.path.join(sd, "weights.thaw")
        kp = os.path.join(sd, "kv.thawkv")
        with open(wp, "wb") as f:
            f.write(b"fake-weights")
        with open(kp, "wb") as f:
            f.write(b"fake-kv")
        h = ForkHandle(
            model_id="fake/model",
            state_dir=sd,
            kv_path=kp,
            weights_path=wp,
            prefix_tokens=16,
            block_shape=[2, 16, 8, 128],
            num_layers=32,
            max_block_id=0,
            num_kv_blocks=1,
        )
        h._owns_state_dir = state_dir is None
        return h

    monkeypatch.setattr(fpmod, "_fork", _fake_fork)
    # fork_pool also uses _tp_size; give it a version that always says 1.
    monkeypatch.setattr(fpmod, "_tp_size", lambda llm: 1)


class _FakeLLM:
    """Stand-in for vllm.LLM — fork_pool only reads _tp_size, which is patched."""


def test_fork_completions_single_worker_single_prompt(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm.fork_pool import ForkPool
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=5)
        results = pool.fork_completions(
            _FakeLLM(), prompts=["hi"], sampling_params=_FakeSP(),
        )
        assert len(results) == 1
        assert results[0].prompt == "hi"
        assert results[0].text == "echo:hi"
        assert results[0].token_ids == [1, 2, 3]
        assert results[0].mode == "subprocess"
    finally:
        pool.close()


def test_fork_completions_round_robin_over_workers(
    script_path, monkeypatch, tmp_path,
):
    """4 prompts across 2 workers → each sees 2 prompts, results ordered."""
    from thaw_vllm.fork_pool import ForkPool
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=2, boot_timeout_s=5)
        prompts = ["a", "b", "c", "d"]
        results = pool.fork_completions(
            _FakeLLM(), prompts=prompts, sampling_params=_FakeSP(),
        )
        assert [r.prompt for r in results] == prompts
        assert [r.text for r in results] == [f"echo:{p}" for p in prompts]
        # Round-robin: 0,2 → worker 0; 1,3 → worker 1.
        assert results[0].worker_index == results[2].worker_index
        assert results[1].worker_index == results[3].worker_index
        assert results[0].worker_index != results[1].worker_index
    finally:
        pool.close()


def test_fork_completions_with_fewer_prompts_than_workers(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm.fork_pool import ForkPool
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=4, boot_timeout_s=5)
        results = pool.fork_completions(
            _FakeLLM(), prompts=["only-one"], sampling_params=_FakeSP(),
        )
        assert len(results) == 1
        assert results[0].text == "echo:only-one"
    finally:
        pool.close()


def test_fork_completions_empty_prompts_returns_empty(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm.fork_pool import ForkPool
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=2, boot_timeout_s=5)
        assert pool.fork_completions(
            _FakeLLM(), prompts=[], sampling_params=_FakeSP()
        ) == []
    finally:
        pool.close()


def test_worker_error_during_generate_is_surfaced(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm.fork_pool import ForkPool, ForkPoolError
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_GEN_FAIL)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=5)
        with pytest.raises(ForkPoolError) as exc_info:
            pool.fork_completions(
                _FakeLLM(), prompts=["x"], sampling_params=_FakeSP(),
            )
        assert "CUDAError" in str(exc_info.value)
    finally:
        pool.close()


def test_public_fork_completions_rejects_both_workers_and_pool(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm import fork_completions
    from thaw_vllm.fork_pool import ForkPool
    from thaw_vllm.fork import ForkError
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=5)
        with pytest.raises(ForkError):
            fork_completions(
                _FakeLLM(), ["x"], _FakeSP(),
                workers=1, pool=pool,
            )
    finally:
        pool.close()


def test_fork_completions_via_public_api_routes_through_pool(
    script_path, monkeypatch, tmp_path,
):
    from thaw_vllm import fork_completions
    from thaw_vllm.fork_pool import ForkPool
    _install_fake_fork(monkeypatch, tmp_path)
    script = script_path(FAKE_WORKER_HAPPY)
    pool = ForkPool(worker_cmd=[sys.executable, script])
    try:
        pool.init_pool(model="fake/model", workers=1, boot_timeout_s=5)
        results = fork_completions(
            _FakeLLM(), ["hi"], _FakeSP(), pool=pool,
        )
        assert results[0].text == "echo:hi"
    finally:
        pool.close()


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------


class _FakeSP:
    """Stand-in for vllm.SamplingParams — _sampling_params_to_dict
    iterates known attribute names; we expose a couple to prove that
    they round-trip through the protocol."""

    n = 1
    temperature = 0.0
    max_tokens = 16
