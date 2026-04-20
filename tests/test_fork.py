"""
Tests for thaw_vllm.fork — primitive + convenience layer.

CPU-only, mocked. Real GPU behavior is exercised by
tests/gpu/test_fork_gpu.py (opt-in, run on H100).

Coverage
--------
  - ForkHandle dataclass: fields, save/load round-trip, context manager
    cleanup, close() idempotence.
  - fork(): happy path writes the manifest + delegates to the right
    primitives; parent-quiescence and prefix-caching validators fire.
  - hydrate(): validators fire BEFORE any GPU mutation
    (ModelMismatchError, BlockShapeMismatchError, BlockPoolTooSmallError,
    PrefixCachingDisabledError).
  - fork_completions: workers=None delegates to llm.generate; workers>0
    spawns N subprocess workers with the right env.
"""

import json
import os
import sys
import tempfile
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fake vLLM objects — narrow replicas of the ones test_pool.py uses, tuned
# to exercise fork.py's paths (block_pool + kv_caches + scheduler config).
# ---------------------------------------------------------------------------


class _FakeBlockPool:
    def __init__(self, num_blocks=256):
        self.blocks = [MagicMock(block_id=i) for i in range(num_blocks)]
        self.cached_block_hash_to_block = MagicMock()


class _FakeKVManager:
    def __init__(self, num_blocks=256, block_size=16):
        self.block_pool = _FakeBlockPool(num_blocks)
        self.block_size = block_size


class _FakeScheduler:
    def __init__(self, num_blocks=256, block_size=16):
        self.kv_cache_manager = _FakeKVManager(num_blocks, block_size)
        self.block_size = block_size


class _FakeModelRunner:
    def __init__(self, num_layers=32, block_shape=(2, 16, 8, 128)):
        # Each "kv_cache" is a mock tensor whose [:, 0] returns something
        # with a .shape matching block_shape[1:] preceded by 2 at dim 0.
        # fork.py reads kv_caches[0][:, 0].shape — so we mock that indexing.
        sample_block = MagicMock()
        sample_block.shape = tuple(block_shape)

        class _Layer:
            def __getitem__(self, idx):
                return sample_block

        self.kv_caches = [_Layer() for _ in range(num_layers)]


class _FakeDriverWorker:
    def __init__(self, num_layers=32, block_shape=(2, 16, 8, 128)):
        self.model_runner = _FakeModelRunner(num_layers, block_shape)


class _FakeExecutor:
    def __init__(self, num_layers=32, block_shape=(2, 16, 8, 128)):
        self.driver_worker = _FakeDriverWorker(num_layers, block_shape)


class _FakeEngineCore:
    def __init__(self, num_blocks=256, num_layers=32, block_size=16,
                 block_shape=(2, 16, 8, 128)):
        self.scheduler = _FakeScheduler(num_blocks, block_size)
        self.model_executor = _FakeExecutor(num_layers, block_shape)


class _FakeLLMEngine:
    def __init__(self, *, model_id="meta-llama/Meta-Llama-3-8B", tp_size=1,
                 has_unfinished=False, num_blocks=256, num_layers=32,
                 block_size=16, block_shape=(2, 16, 8, 128)):
        self.engine_core = _FakeEngineCore(
            num_blocks=num_blocks, num_layers=num_layers,
            block_size=block_size, block_shape=block_shape,
        )
        self.vllm_config = MagicMock()
        self.vllm_config.parallel_config.tensor_parallel_size = tp_size
        self.vllm_config.model_config.model = model_id
        self._has_unfinished = has_unfinished

    def has_unfinished_requests(self):
        return self._has_unfinished


class _FakeLLM:
    def __init__(self, **kwargs):
        self.llm_engine = _FakeLLMEngine(**kwargs)
        self._generate_calls = []

    def generate(self, prompts, sampling_params):
        self._generate_calls.append((list(prompts), sampling_params))
        outputs = []
        for p in prompts:
            first = MagicMock()
            first.text = f"completion-of:{p[:20]}"
            first.token_ids = [1, 2, 3]
            out = MagicMock()
            out.outputs = [first]
            outputs.append(out)
        return outputs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm():
    return _FakeLLM()


@pytest.fixture(autouse=True)
def mock_kv_and_weight_ops(monkeypatch, tmp_path):
    """Stub the real freeze/restore primitives; write realistic sidecars.

    Each call to freeze_kv_cache writes a tiny stub KV file and a valid
    .meta sidecar so fork() can read it back. freeze_model_tp writes an
    empty weights stub. restore_* just return dicts.
    """
    import thaw_vllm.fork as fork_module  # noqa: F401 — forces import

    def fake_freeze_kv_cache(llm, path):
        # Match the sidecar shape produced by the real freeze_kv_cache.
        meta = {
            "num_blocks": 4,
            "block_ids": [0, 1, 2, 3],
            "block_hashes": [None, None, None, None],
            "num_layers": 32,
            "block_shape": [2, 16, 8, 128],
            "block_size": 16,
            "block_bytes": 1024,
            "dtype": "torch.float16",
        }
        with open(path, "wb") as f:
            f.write(b"stub-kv-payload")
        from thaw_vllm.kv_snapshot import _write_meta_sidecar
        _write_meta_sidecar(path, meta)
        return {"num_blocks": 4, "total_bytes": len(b"stub-kv-payload"),
                "elapsed_s": 0.0, "block_ids": meta["block_ids"]}

    def fake_freeze_kv_cache_tp(llm, base_path):
        return fake_freeze_kv_cache(llm, base_path)

    def fake_freeze_model_tp(llm, base_path, vllm_commit=None):
        with open(base_path, "wb") as f:
            f.write(b"stub-weights-payload")
        return {"num_regions": 1, "total_bytes": 20, "elapsed_s": 0.0,
                "throughput_gb_s": 0.0, "tensor_parallel_size": 1,
                "backend": "stub", "per_rank": []}

    def fake_restore_kv_cache(llm, path):
        return {"num_blocks": 4, "total_bytes": 16, "elapsed_s": 0.0}

    def fake_restore_kv_cache_tp(llm, path):
        return fake_restore_kv_cache(llm, path)

    def fake_restore_model_tp(llm, path, chunk_size_mb=64):
        return {"num_regions": 1, "total_bytes": 20, "elapsed_s": 0.0,
                "throughput_gb_s": 0.0, "tensor_parallel_size": 1,
                "backend": "stub", "per_rank": []}

    # fork.py imports these lazily inside the function bodies, so patch
    # at the source modules.
    import thaw_vllm.kv_snapshot as kvs
    import thaw_vllm.snapshot as snap
    monkeypatch.setattr(kvs, "freeze_kv_cache", fake_freeze_kv_cache)
    monkeypatch.setattr(kvs, "freeze_kv_cache_tp", fake_freeze_kv_cache_tp)
    monkeypatch.setattr(kvs, "restore_kv_cache", fake_restore_kv_cache)
    monkeypatch.setattr(kvs, "restore_kv_cache_tp", fake_restore_kv_cache_tp)
    monkeypatch.setattr(snap, "freeze_model_tp", fake_freeze_model_tp)
    monkeypatch.setattr(snap, "restore_model_tp", fake_restore_model_tp)


# ---------------------------------------------------------------------------
# ForkHandle basics
# ---------------------------------------------------------------------------


def test_fork_returns_handle_with_expected_fields(fake_llm):
    from thaw_vllm import fork
    handle = fork(fake_llm)
    try:
        assert handle.model_id == "meta-llama/Meta-Llama-3-8B"
        assert handle.num_kv_blocks == 4
        assert handle.num_layers == 32
        assert handle.block_shape == [2, 16, 8, 128]
        assert handle.max_block_id == 3
        assert handle.prefix_tokens == 4 * 16
        assert handle.tensor_parallel_size == 1
        assert os.path.exists(handle.kv_path)
        # Weights are off by default.
        assert handle.weights_path is None
    finally:
        handle.close()


def test_fork_include_weights_writes_weights_file(fake_llm):
    from thaw_vllm import fork
    with fork(fake_llm, include_weights=True) as handle:
        assert handle.weights_path is not None
        assert os.path.exists(handle.weights_path)


def test_fork_context_manager_cleans_state_dir(fake_llm):
    from thaw_vllm import fork
    with fork(fake_llm) as handle:
        state_dir = handle.state_dir
        assert os.path.isdir(state_dir)
    assert not os.path.exists(state_dir)


def test_fork_with_explicit_state_dir_does_not_delete(fake_llm, tmp_path):
    """Caller-provided state_dir is NOT owned by the handle; close() leaves it."""
    from thaw_vllm import fork
    custom_dir = str(tmp_path / "forkdir")
    with fork(fake_llm, state_dir=custom_dir) as handle:
        assert handle.state_dir == os.path.abspath(custom_dir)
    assert os.path.isdir(custom_dir)


def test_fork_save_roundtrip(fake_llm, tmp_path):
    from thaw_vllm import fork, ForkHandle
    with fork(fake_llm, include_weights=True) as handle:
        target = str(tmp_path / "persisted")
        copied = handle.save(target)
        loaded = ForkHandle.load(target)
    # loaded handle outlives the context manager — save() did copy.
    assert loaded.model_id == copied.model_id
    assert loaded.num_kv_blocks == copied.num_kv_blocks
    assert loaded.block_shape == copied.block_shape
    assert os.path.exists(loaded.kv_path)
    assert os.path.exists(loaded.weights_path)


def test_close_is_idempotent(fake_llm):
    from thaw_vllm import fork
    handle = fork(fake_llm)
    handle.close()
    handle.close()  # must not raise


def test_handle_closed_error_on_save(fake_llm, tmp_path):
    from thaw_vllm import fork, HandleClosedError
    handle = fork(fake_llm)
    handle.close()
    with pytest.raises(HandleClosedError):
        handle.save(str(tmp_path / "target"))


# ---------------------------------------------------------------------------
# Parent-side validators
# ---------------------------------------------------------------------------


def test_fork_raises_when_parent_has_unfinished_requests():
    from thaw_vllm import fork, UnfinishedRequestsError
    llm = _FakeLLM(has_unfinished=True)
    with pytest.raises(UnfinishedRequestsError):
        fork(llm)


def test_fork_raises_when_prefix_caching_disabled(monkeypatch):
    from thaw_vllm import fork, PrefixCachingDisabledError
    llm = _FakeLLM()
    # Strip the cached_block_hash_to_block attr on the parent's block pool
    # to simulate a engine loaded without enable_prefix_caching.
    bp = llm.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool
    # MagicMock auto-creates attrs; replace with a minimal class.
    class _BareBP:
        def __init__(self, blocks): self.blocks = blocks
    llm.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool = (
        _BareBP(bp.blocks)
    )
    with pytest.raises(PrefixCachingDisabledError):
        fork(llm)


# ---------------------------------------------------------------------------
# Child-side validators (hydrate)
# ---------------------------------------------------------------------------


def test_hydrate_raises_on_model_mismatch(fake_llm):
    from thaw_vllm import fork, ModelMismatchError
    with fork(fake_llm) as handle:
        child = _FakeLLM(model_id="meta-llama/Meta-Llama-3-70B")
        with pytest.raises(ModelMismatchError):
            handle.hydrate(child)


def test_hydrate_raises_on_block_pool_too_small(fake_llm):
    from thaw_vllm import fork, BlockPoolTooSmallError
    with fork(fake_llm) as handle:
        # handle.max_block_id == 3 → child needs at least 4 blocks.
        child = _FakeLLM(num_blocks=3)
        with pytest.raises(BlockPoolTooSmallError):
            handle.hydrate(child)


def test_hydrate_raises_on_block_shape_mismatch(fake_llm):
    from thaw_vllm import fork, BlockShapeMismatchError
    with fork(fake_llm) as handle:
        # Different num_kv_heads → block_shape differs.
        child = _FakeLLM(block_shape=(2, 16, 16, 128))
        with pytest.raises(BlockShapeMismatchError):
            handle.hydrate(child)


def test_hydrate_raises_on_layer_count_mismatch(fake_llm):
    from thaw_vllm import fork, BlockShapeMismatchError
    with fork(fake_llm) as handle:
        child = _FakeLLM(num_layers=24)
        with pytest.raises(BlockShapeMismatchError):
            handle.hydrate(child)


def test_hydrate_raises_on_child_without_prefix_caching(fake_llm):
    from thaw_vllm import fork, PrefixCachingDisabledError
    with fork(fake_llm) as handle:
        child = _FakeLLM()
        class _BareBP:
            def __init__(self, blocks): self.blocks = blocks
        child_bp = child.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool
        child.llm_engine.engine_core.scheduler.kv_cache_manager.block_pool = (
            _BareBP(child_bp.blocks)
        )
        with pytest.raises(PrefixCachingDisabledError):
            handle.hydrate(child)


def test_hydrate_happy_path_calls_restore(fake_llm):
    from thaw_vllm import fork
    with fork(fake_llm) as handle:
        child = _FakeLLM()
        stats = handle.hydrate(child)
    assert stats["weights_restored"] is False
    assert stats["kv"]["num_blocks"] == 4


def test_hydrate_restores_weights_when_included(fake_llm):
    from thaw_vllm import fork
    with fork(fake_llm, include_weights=True) as handle:
        child = _FakeLLM()
        stats = handle.hydrate(child)
    assert stats["weights_restored"] is True


# ---------------------------------------------------------------------------
# fork_completions
# ---------------------------------------------------------------------------


def test_fork_completions_same_process_delegates_to_generate(fake_llm):
    from thaw_vllm import fork_completions
    sp = MagicMock()
    prompts = ["prompt-a", "prompt-b", "prompt-c"]
    results = fork_completions(fake_llm, prompts, sp, workers=None)
    assert len(results) == 3
    # Same-process should call generate exactly once with all prompts.
    assert len(fake_llm._generate_calls) == 1
    assert fake_llm._generate_calls[0][0] == prompts
    for r in results:
        assert r.mode == "same_process"


def test_fork_completions_subprocess_rejects_tp_gt_1():
    from thaw_vllm import fork_completions, ForkError
    sp = MagicMock()
    llm = _FakeLLM(tp_size=2)
    with pytest.raises(ForkError):
        fork_completions(llm, ["p"], sp, workers=2)


def test_fork_completions_subprocess_spawns_n_workers(fake_llm, monkeypatch):
    from thaw_vllm import fork_completions

    # Capture Popen invocations and fake successful exits.
    spawn_log = []

    class _FakePopen:
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            spawn_log.append((list(cmd), dict(env)))
            # Pull the payload path out of the cmd and write a fake result.
            payload_path = cmd[-1]
            with open(payload_path) as f:
                payload = json.load(f)
            result = {
                "worker_index": payload["worker_index"],
                "load_s": 0.01,
                "gen_s": 0.02,
                "outputs": [
                    {"prompt_index": pi, "text": f"worker{payload['worker_index']}-out{pi}",
                     "token_ids": [9, 9, 9]}
                    for pi in payload["prompt_indices"]
                ],
            }
            with open(payload["result_path"], "w") as rf:
                json.dump(result, rf)

        def wait(self):
            return 0

    fork_module = sys.modules["thaw_vllm.fork"]
    monkeypatch.setattr(fork_module.subprocess, "Popen", _FakePopen)

    sp = MagicMock()
    sp.n = 1
    sp.temperature = 0.7
    sp.top_p = 0.9
    sp.max_tokens = 64
    # stop attribute must be None so serialization skips it.
    sp.stop = None
    sp.seed = None
    sp.best_of = None
    sp.top_k = None
    sp.min_p = None
    sp.repetition_penalty = None
    sp.presence_penalty = None
    sp.frequency_penalty = None
    sp.min_tokens = None
    sp.logprobs = None
    sp.ignore_eos = None

    prompts = ["pa", "pb", "pc", "pd"]
    results = fork_completions(fake_llm, prompts, sp, workers=2)

    assert len(spawn_log) == 2  # two workers
    for cmd, env in spawn_log:
        assert env["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"
        assert env["VLLM_ALLOW_INSECURE_SERIALIZATION"] == "1"
    assert len(results) == 4
    for r in results:
        assert r.mode == "subprocess"
        assert r.text.startswith("worker")


def test_fork_completions_subprocess_surfaces_worker_failure(fake_llm, monkeypatch):
    from thaw_vllm import fork_completions, ForkError

    class _FailingPopen:
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            pass
        def wait(self):
            return 1

    fork_module = sys.modules["thaw_vllm.fork"]
    monkeypatch.setattr(fork_module.subprocess, "Popen", _FailingPopen)

    sp = MagicMock()
    for name in ("n", "temperature", "top_p", "max_tokens", "stop", "seed",
                 "best_of", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "min_tokens",
                 "logprobs", "ignore_eos"):
        setattr(sp, name, None)

    with pytest.raises(ForkError):
        fork_completions(fake_llm, ["p"], sp, workers=1)


def test_fork_completions_rejects_zero_workers(fake_llm):
    from thaw_vllm import fork_completions, ForkError
    sp = MagicMock()
    with pytest.raises(ForkError):
        fork_completions(fake_llm, ["p"], sp, workers=0)


def test_fork_completions_reuses_passed_handle(fake_llm, monkeypatch):
    from thaw_vllm import fork, fork_completions

    spawn_count = {"n": 0}

    class _FakePopen:
        def __init__(self, cmd, env=None, stdout=None, stderr=None):
            spawn_count["n"] += 1
            payload_path = cmd[-1]
            with open(payload_path) as f:
                payload = json.load(f)
            with open(payload["result_path"], "w") as rf:
                json.dump({
                    "worker_index": payload["worker_index"],
                    "load_s": 0, "gen_s": 0,
                    "outputs": [
                        {"prompt_index": pi, "text": "ok", "token_ids": []}
                        for pi in payload["prompt_indices"]
                    ],
                }, rf)
        def wait(self): return 0

    fork_module = sys.modules["thaw_vllm.fork"]
    monkeypatch.setattr(fork_module.subprocess, "Popen", _FakePopen)

    sp = MagicMock()
    for name in ("n", "temperature", "top_p", "max_tokens", "stop", "seed",
                 "best_of", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "min_tokens",
                 "logprobs", "ignore_eos"):
        setattr(sp, name, None)

    with fork(fake_llm, include_weights=True) as handle:
        state_dir = handle.state_dir
        results = fork_completions(
            fake_llm, ["p1", "p2"], sp, workers=2, handle=handle,
        )
        assert len(results) == 2
        # Passed handle should NOT be closed by fork_completions.
        assert os.path.isdir(state_dir)


def test_fork_completions_rejects_handle_without_weights(fake_llm):
    from thaw_vllm import fork, fork_completions, ForkError
    sp = MagicMock()
    for name in ("n", "temperature", "top_p", "max_tokens", "stop", "seed",
                 "best_of", "top_k", "min_p", "repetition_penalty",
                 "presence_penalty", "frequency_penalty", "min_tokens",
                 "logprobs", "ignore_eos"):
        setattr(sp, name, None)
    with fork(fake_llm, include_weights=False) as handle:
        with pytest.raises(ForkError):
            fork_completions(
                fake_llm, ["p"], sp, workers=1, handle=handle,
            )


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_error_types_inherit_from_fork_error():
    from thaw_vllm import (
        ForkError, ModelMismatchError, BlockShapeMismatchError,
        BlockPoolTooSmallError, PrefixCachingDisabledError,
        UnfinishedRequestsError, HandleClosedError,
    )
    for cls in (
        ModelMismatchError, BlockShapeMismatchError,
        BlockPoolTooSmallError, PrefixCachingDisabledError,
        UnfinishedRequestsError, HandleClosedError,
    ):
        assert issubclass(cls, ForkError)


def test_sampling_params_to_dict_skips_none_and_non_json():
    from thaw_vllm.fork import _sampling_params_to_dict
    sp = MagicMock()
    sp.n = 1
    sp.temperature = 0.7
    sp.top_p = None
    sp.max_tokens = 64
    sp.stop = None
    sp.seed = None
    # Object without a JSON encoding — must be dropped silently.
    class _Opaque: pass
    sp.best_of = _Opaque()
    for name in ("top_k", "min_p", "repetition_penalty", "presence_penalty",
                 "frequency_penalty", "min_tokens", "logprobs", "ignore_eos"):
        setattr(sp, name, None)
    out = _sampling_params_to_dict(sp)
    assert out["n"] == 1
    assert out["temperature"] == 0.7
    assert out["max_tokens"] == 64
    assert "top_p" not in out
    assert "best_of" not in out
