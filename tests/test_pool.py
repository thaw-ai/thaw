"""
Tests for thaw_vllm.pool — engine pool + FastAPI app.

All tests run locally without GPU by mocking vLLM's LLM class and
thaw's restore functions. Tests cover:
  - Pool init and slot management
  - Model registration and snapshot validation
  - Acquire with model affinity (prefer already-loaded slot)
  - Acquire triggers DMA swap when model not loaded
  - Acquire blocks when all slots busy, unblocks on release
  - Preload (synchronous model loading)
  - Pool status reporting
  - FastAPI endpoints: /v1/completions, /v1/chat/completions,
    /v1/models, /health, /admin/*
  - Streaming responses
  - Error cases (unregistered model, missing snapshot, etc.)
"""

import asyncio
import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest
import httpx

# ---------------------------------------------------------------------------
# Mock vLLM objects
# ---------------------------------------------------------------------------


class FakeOutput:
    def __init__(self, text="Hello world", token_ids=None, finish_reason="stop"):
        self.text = text
        self.token_ids = token_ids or [1, 2, 3]
        self.finish_reason = finish_reason


class FakeRequestOutput:
    def __init__(self, text="Hello world", token_ids=None):
        self.prompt_token_ids = [10, 20, 30]
        self.outputs = [FakeOutput(text, token_ids)]


class FakeLLM:
    """Mock vLLM LLM that returns canned responses."""

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.generate_calls = []

    def generate(self, prompts: List[str], sampling_params=None):
        self.generate_calls.append((prompts, sampling_params))
        return [FakeRequestOutput()]


class FakeModelRunner:
    def __init__(self):
        self.model = MagicMock()


class FakeDriverWorker:
    def __init__(self):
        self.model_runner = FakeModelRunner()


class FakeModelExecutor:
    def __init__(self):
        self.driver_worker = FakeDriverWorker()


class FakeEngineCore:
    def __init__(self):
        self.model_executor = FakeModelExecutor()
        self.vllm_config = MagicMock()
        self.vllm_config.parallel_config.tensor_parallel_size = 1


class FakeLLMEngine:
    def __init__(self):
        self.engine_core = FakeEngineCore()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_files():
    """Create temporary fake snapshot files."""
    files = {}
    tmpdir = tempfile.mkdtemp(prefix="thaw_test_")
    for name in ["model_a.thaw", "model_b.thaw", "model_c.thaw"]:
        path = os.path.join(tmpdir, name)
        with open(path, "wb") as f:
            f.write(b"THAW" + b"\x00" * 100)  # fake thaw file
        files[name.replace(".thaw", "")] = path
    yield files
    for path in files.values():
        if os.path.exists(path):
            os.unlink(path)
    os.rmdir(tmpdir)


@pytest.fixture
def pool_with_slots(snapshot_files):
    """Create an EnginePool with 2 fake slots, model_a registered."""
    from thaw_vllm.pool import EnginePool, EngineSlot

    pool = EnginePool()
    pool.base_model = "test-model"
    pool.tp_size = 1

    for i in range(2):
        llm = FakeLLM()
        llm.llm_engine = FakeLLMEngine()
        slot = EngineSlot(id=i, llm=llm)
        pool.slots.append(slot)

    pool.register("model_a", snapshot_files["model_a"])
    pool.register("model_b", snapshot_files["model_b"])

    return pool


_FAKE_STATS = {
    "num_regions": 10,
    "total_bytes": 1_000_000_000,
    "elapsed_s": 0.5,
    "throughput_gb_s": 2.0,
}


def _patch_restore():
    """Patch restore_model_from_ram to be a no-op that returns stats."""
    return patch(
        "thaw_vllm.snapshot.restore_model_from_ram",
        return_value=dict(_FAKE_STATS),
    )


def _patch_restore_tp():
    """Patch restore_model_tp for TP>1 tests."""
    return patch(
        "thaw_vllm.snapshot.restore_model_tp",
        return_value=dict(_FAKE_STATS),
    )


# ---------------------------------------------------------------------------
# Pool unit tests
# ---------------------------------------------------------------------------


class TestEnginePool:

    def test_register_valid(self, snapshot_files):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        pool.register("test", snapshot_files["model_a"])
        assert "test" in pool.snapshots
        assert pool.snapshots["test"] == snapshot_files["model_a"]

    def test_register_missing_file(self):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        with pytest.raises(FileNotFoundError):
            pool.register("bad", "/nonexistent/path.thaw")

    def test_unregister(self, snapshot_files):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        pool.register("test", snapshot_files["model_a"])
        pool.unregister("test")
        assert "test" not in pool.snapshots

    def test_unregister_nonexistent(self):
        from thaw_vllm.pool import EnginePool

        pool = EnginePool()
        pool.unregister("nope")  # should not raise

    def test_status(self, pool_with_slots):
        status = pool_with_slots.status()
        assert status["pool_size"] == 2
        assert status["base_model"] == "test-model"
        assert "model_a" in status["registered_models"]
        assert "model_b" in status["registered_models"]
        assert len(status["slots"]) == 2
        assert status["slots"][0]["model"] is None
        assert status["slots"][0]["busy"] is False

    def test_acquire_swaps_model(self, pool_with_slots):
        """Acquiring a model on an empty slot triggers DMA swap."""
        pool = pool_with_slots

        async def run():
            with _patch_restore() as mock_restore:
                slot = await pool.acquire("model_a")
                assert slot.model_name == "model_a"
                assert mock_restore.called
                pool.release(slot)

        asyncio.run(run())

    def test_acquire_prefers_loaded_slot(self, pool_with_slots):
        """If a slot already has the model, it should be preferred."""
        pool = pool_with_slots
        pool.slots[1].model_name = "model_a"

        async def run():
            with _patch_restore() as mock_restore:
                slot = await pool.acquire("model_a")
                assert slot.id == 1
                assert not mock_restore.called
                pool.release(slot)

        asyncio.run(run())

    def test_acquire_skips_busy_slots(self, pool_with_slots):
        """Busy slots should be skipped."""
        pool = pool_with_slots

        async def run():
            with _patch_restore():
                slot0 = await pool.acquire("model_a")
                assert slot0.id == 0

                slot1 = await pool.acquire("model_b")
                assert slot1.id == 1

                pool.release(slot0)
                pool.release(slot1)

        asyncio.run(run())

    def test_acquire_waits_when_all_busy(self, pool_with_slots):
        """When all slots are busy, acquire blocks until one is released."""
        pool = pool_with_slots

        async def run():
            with _patch_restore():
                slot0 = await pool.acquire("model_a")
                slot1 = await pool.acquire("model_b")

                acquired = []

                async def delayed_acquire():
                    s = await pool.acquire("model_a")
                    acquired.append(s)

                async def release_after_delay():
                    await asyncio.sleep(0.05)
                    pool.release(slot0)

                await asyncio.gather(delayed_acquire(), release_after_delay())

                assert len(acquired) == 1
                assert acquired[0].id == 0
                pool.release(acquired[0])
                pool.release(slot1)

        asyncio.run(run())

    def test_acquire_unregistered_model(self, pool_with_slots):
        """Acquiring an unregistered model raises ValueError."""
        pool = pool_with_slots

        async def run():
            with pytest.raises(ValueError, match="not registered"):
                await pool.acquire("nonexistent")

        asyncio.run(run())

    def test_preload(self, pool_with_slots):
        """Preload synchronously loads a model into a slot."""
        pool = pool_with_slots
        with _patch_restore() as mock_restore:
            stats = pool.preload("model_a", slot_id=0)
            assert pool.slots[0].model_name == "model_a"
            assert mock_restore.called
            assert "throughput_gb_s" in stats

    def test_preload_specific_slot(self, pool_with_slots):
        pool = pool_with_slots
        with _patch_restore():
            pool.preload("model_b", slot_id=1)
            assert pool.slots[1].model_name == "model_b"
            assert pool.slots[0].model_name is None

    def test_preload_unregistered(self, pool_with_slots):
        with pytest.raises(ValueError, match="not registered"):
            pool_with_slots.preload("nonexistent")

    def test_preload_invalid_slot(self, pool_with_slots):
        with pytest.raises(ValueError, match="does not exist"):
            pool_with_slots.preload("model_a", slot_id=99)

    def test_swap_model_tp(self, pool_with_slots, snapshot_files):
        """TP>1 uses restore_model_tp instead of restore_model_from_ram."""
        pool = pool_with_slots
        pool.tp_size = 2

        with _patch_restore_tp() as mock_tp_restore:
            pool.preload("model_a", slot_id=0)
            assert mock_tp_restore.called
            # First arg is the llm instance, second is the path
            call_args = mock_tp_restore.call_args
            assert call_args[0][0] is pool.slots[0].llm
            assert call_args[0][1] == snapshot_files["model_a"]


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(pool_with_slots):
    """HTTPX test client backed by a pool with model_a pre-loaded on slot 0."""
    from thaw_vllm.pool import create_pool_app

    pool = pool_with_slots

    with _patch_restore():
        pool.preload("model_a", slot_id=0)

    # Patch restore for runtime swaps during requests
    with _patch_restore():
        app = create_pool_app(pool)

        # We need to patch vllm.SamplingParams inside the endpoint module
        with patch("thaw_vllm.pool.SamplingParams") as mock_sp:
            mock_sp.return_value = MagicMock()

            # Also need to make the import work inside create_pool_app
            # The SamplingParams is imported at function scope, so we patch
            # it at the vllm module level
            import sys
            mock_vllm = MagicMock()
            mock_vllm.SamplingParams = MagicMock(return_value=MagicMock())
            old_vllm = sys.modules.get("vllm")
            sys.modules["vllm"] = mock_vllm

            from starlette.testclient import TestClient
            client = TestClient(app)
            yield client

            if old_vllm is not None:
                sys.modules["vllm"] = old_vllm
            else:
                sys.modules.pop("vllm", None)


@pytest.fixture
def simple_client(pool_with_slots):
    """Simpler test client that patches at the right level."""
    from thaw_vllm.pool import create_pool_app

    pool = pool_with_slots

    with _patch_restore():
        pool.preload("model_a", slot_id=0)

    app = create_pool_app(pool)

    # Use starlette TestClient for sync testing of async FastAPI app
    from starlette.testclient import TestClient
    with TestClient(app) as client:
        yield client, pool


class TestHealthEndpoint:
    def test_health(self, simple_client):
        client, pool = simple_client
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["pool_size"] == 2
        assert data["idle_slots"] == 2
        assert data["registered_models"] == 2


class TestModelsEndpoint:
    def test_list_models(self, simple_client):
        client, pool = simple_client
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        names = [m["id"] for m in data["data"]]
        assert "model_a" in names
        assert "model_b" in names


class TestAdminEndpoints:
    def test_pool_status(self, simple_client):
        client, pool = simple_client
        resp = client.get("/admin/pool")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pool_size"] == 2
        assert data["slots"][0]["model"] == "model_a"
        assert data["slots"][1]["model"] is None

    def test_list_snapshots(self, simple_client):
        client, pool = simple_client
        resp = client.get("/admin/snapshots")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_a" in data["snapshots"]

    def test_register_snapshot(self, simple_client, snapshot_files):
        client, pool = simple_client
        resp = client.post(
            "/admin/snapshots",
            json={"name": "model_c", "snapshot": snapshot_files["model_c"]},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "registered"
        assert "model_c" in pool.snapshots

    def test_register_missing_file(self, simple_client):
        client, pool = simple_client
        resp = client.post(
            "/admin/snapshots",
            json={"name": "bad", "snapshot": "/nonexistent.thaw"},
        )
        assert resp.status_code == 404

    def test_register_missing_fields(self, simple_client):
        client, pool = simple_client
        resp = client.post("/admin/snapshots", json={})
        assert resp.status_code == 400

    def test_unregister_snapshot(self, simple_client):
        client, pool = simple_client
        resp = client.delete("/admin/snapshots/model_b")
        assert resp.status_code == 200
        assert "model_b" not in pool.snapshots

    def test_preload(self, simple_client):
        client, pool = simple_client
        with _patch_restore():
            resp = client.post(
                "/admin/preload",
                json={"model": "model_b"},
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "preloaded"

    def test_preload_unregistered(self, simple_client):
        client, pool = simple_client
        resp = client.post(
            "/admin/preload",
            json={"model": "nonexistent"},
        )
        assert resp.status_code == 400


class TestCompletionsEndpoint:
    def test_completions(self, simple_client):
        client, pool = simple_client
        with _patch_restore():
            resp = client.post(
                "/v1/completions",
                json={
                    "model": "model_a",
                    "prompt": "Hello",
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert data["model"] == "model_a"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]
        assert "usage" in data
        assert "thaw_metadata" in data
        assert "slot_id" in data["thaw_metadata"]

    def test_completions_default_model(self, simple_client):
        """When only one model registered, model field can be omitted."""
        client, pool = simple_client
        pool.unregister("model_b")  # leave only model_a

        with _patch_restore():
            resp = client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )
        assert resp.status_code == 200
        assert resp.json()["model"] == "model_a"

    def test_completions_unregistered_model(self, simple_client):
        client, pool = simple_client
        resp = client.post(
            "/v1/completions",
            json={"model": "nonexistent", "prompt": "Hello"},
        )
        assert resp.status_code == 404

    def test_completions_no_model_multi(self, simple_client):
        """When multiple models registered, model field is required."""
        client, pool = simple_client
        resp = client.post(
            "/v1/completions",
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 400

    def test_completions_stream(self, simple_client):
        client, pool = simple_client
        with _patch_restore():
            resp = client.post(
                "/v1/completions",
                json={
                    "model": "model_a",
                    "prompt": "Hello",
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = resp.text.strip().split("\n\n")
        assert lines[-1] == "data: [DONE]"
        # Parse first data chunk
        first = json.loads(lines[0].replace("data: ", ""))
        assert first["object"] == "text_completion"


class TestChatCompletionsEndpoint:
    def test_chat_completions(self, simple_client):
        client, pool = simple_client
        with _patch_restore():
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "model_a",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_stream(self, simple_client):
        client, pool = simple_client
        with _patch_restore():
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "model_a",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        lines = [l for l in resp.text.strip().split("\n\n") if l.startswith("data:")]
        assert lines[-1] == "data: [DONE]"

        # First chunk should have role
        first = json.loads(lines[0].replace("data: ", ""))
        assert first["choices"][0]["delta"]["role"] == "assistant"

        # Last data chunk (before [DONE]) should have finish_reason
        last_data = json.loads(lines[-2].replace("data: ", ""))
        assert last_data["choices"][0]["finish_reason"] is not None


class TestModelSwapDuringRequest:
    def test_request_triggers_swap(self, simple_client):
        """Requesting model_b when only model_a is loaded triggers a swap."""
        client, pool = simple_client
        assert pool.slots[0].model_name == "model_a"
        assert pool.slots[1].model_name is None

        with _patch_restore() as mock_restore:
            resp = client.post(
                "/v1/completions",
                json={"model": "model_b", "prompt": "Hi"},
            )
        assert resp.status_code == 200
        # model_b should now be loaded on slot 1 (slot 0 has model_a)
        assert any(s.model_name == "model_b" for s in pool.slots)
        assert mock_restore.called
