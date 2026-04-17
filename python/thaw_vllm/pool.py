"""
thaw_vllm.pool — Pre-warmed vLLM engine pool with hot model swapping.

The engine pool eliminates cold start overhead by keeping vLLM engines
pre-initialized with dummy weights. When a request arrives for model X:
  1. If a slot already has model X loaded -> serve immediately (0s)
  2. If an idle slot exists -> DMA weights from snapshot (~1s) -> serve
  3. If all slots busy -> wait for one to free up

Think PgBouncer for GPU inference.

Usage:
    pool = EnginePool()
    pool.init_pool("meta-llama/Llama-3.1-8B-Instruct", pool_size=2,
                   dtype="float16", enforce_eager=True)
    pool.register("llama-8b", "/snapshots/llama-8b.thaw")
    pool.register("llama-8b-ft", "/snapshots/finetune-v2.thaw")

    app = create_pool_app(pool)
    uvicorn.run(app, host="0.0.0.0", port=8000)

API:
    POST /v1/chat/completions   — OpenAI-compatible (model field selects snapshot)
    POST /v1/completions        — OpenAI-compatible
    GET  /v1/models             — list registered models
    GET  /health                — health check + pool status
    POST /admin/snapshots       — register a model snapshot
    GET  /admin/snapshots       — list registered snapshots
    DELETE /admin/snapshots/{n} — unregister a snapshot
    GET  /admin/pool            — pool slot status
    POST /admin/preload         — pre-load model into idle slot
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("thaw.pool")


@dataclass
class EngineSlot:
    """A single pre-warmed vLLM engine slot.

    Slot-persistent pinned-mmap state lives inside the vLLM worker
    process (see thaw_vllm._pool_worker), not on this dataclass. Under
    V1 MP, cudaHostRegister state cannot cross the IPC boundary, so all
    DMA work dispatches through llm.collective_rpc.
    """
    id: int
    llm: object  # vllm.LLM instance
    model_name: Optional[str] = None
    snapshot_path: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class EnginePool:
    """Pool of pre-warmed vLLM engines with model hot-swapping.

    All engines share the same base model architecture (hidden_dim,
    num_layers, etc). Model swapping replaces weight data via DMA —
    the engine structure stays identical. This means all registered
    snapshots must be for the same architecture.
    """

    def __init__(self):
        self.slots: list[EngineSlot] = []
        self.snapshots: dict[str, str] = {}  # model_name -> snapshot_path
        self.base_model: str = ""
        self.tp_size: int = 1
        self._slot_available = asyncio.Event()
        self._slot_available.set()

    def init_pool(
        self,
        model: str,
        pool_size: int = 1,
        tensor_parallel_size: int = 1,
        **llm_kwargs,
    ):
        """Initialize pool with pre-warmed dummy-weight engines.

        This is the expensive one-time cost (9-20s per engine). After
        this, all model loads are DMA-only (~1s).
        """
        from vllm import LLM

        self.base_model = model
        self.tp_size = tensor_parallel_size

        llm_kwargs.setdefault("enforce_eager", True)
        llm_kwargs.setdefault("dtype", "float16")

        for i in range(pool_size):
            logger.info("Initializing engine slot %d/%d...", i + 1, pool_size)
            t0 = time.perf_counter()
            llm = LLM(
                model=model,
                load_format="dummy",
                tensor_parallel_size=tensor_parallel_size,
                **llm_kwargs,
            )
            elapsed = time.perf_counter() - t0
            logger.info("Slot %d ready in %.1fs", i, elapsed)
            self.slots.append(EngineSlot(id=i, llm=llm))

    def register(self, model_name: str, snapshot_path: str):
        """Register a model snapshot for hot-loading.

        Accepts local paths and remote URIs (s3://...). Remote URIs are
        not verified at registration time; the download happens lazily
        inside _swap_model via resolve_snapshot_path, so S3 errors
        surface at load time with a typed exception.
        """
        from thaw_common.cloud import is_remote

        if not is_remote(snapshot_path) and not os.path.exists(snapshot_path):
            raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")
        self.snapshots[model_name] = snapshot_path
        logger.info("Registered '%s' -> %s", model_name, snapshot_path)

    def unregister(self, model_name: str):
        """Remove a model from the registry."""
        self.snapshots.pop(model_name, None)
        logger.info("Unregistered '%s'", model_name)

    async def acquire(self, model_name: str) -> EngineSlot:
        """Acquire a slot with the requested model loaded.

        Priority:
          1. Idle slot already loaded with this model (zero swap cost)
          2. Any idle slot (swap cost: ~1s DMA)
          3. Wait for a slot to free up
        """
        if model_name not in self.snapshots:
            raise ValueError(
                f"Model '{model_name}' not registered. "
                f"Available: {list(self.snapshots.keys())}"
            )

        while True:
            # Prefer slot that already has the right model
            for slot in self.slots:
                if slot.model_name == model_name and not slot.lock.locked():
                    await slot.lock.acquire()
                    return slot

            # Fall back to any idle slot
            for slot in self.slots:
                if not slot.lock.locked():
                    await slot.lock.acquire()
                    if slot.model_name != model_name:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, self._swap_model, slot, model_name
                        )
                    return slot

            # All slots busy — wait for a release signal
            self._slot_available.clear()
            await self._slot_available.wait()

    def release(self, slot: EngineSlot):
        """Return a slot to the pool."""
        if slot.lock.locked():
            slot.lock.release()
        self._slot_available.set()

    def _swap_model(self, slot: EngineSlot, model_name: str):
        """DMA weights from snapshot into engine slot (blocking).

        Dispatches to each TP worker via llm.collective_rpc. The worker
        maintains the slot-persistent pinned mmap (fast path) and
        falls back through RAM restore → pipelined DMA → pure Python.
        Works transparently under V0, V1 inproc, and V1 MP.
        """
        base_path = self.snapshots[model_name]
        logger.info(
            "Slot %d: swapping '%s' -> '%s'",
            slot.id, slot.model_name or "(empty)", model_name,
        )
        t0 = time.perf_counter()

        from thaw_common.cloud import resolve_snapshot_path
        from thaw_vllm._pool_worker import swap_model as _worker_swap

        # Resolve the top-level path (S3 → local cache) in the parent;
        # each worker applies rank_snapshot_path on top for TP > 1.
        resolved_path = resolve_snapshot_path(base_path)

        results = slot.llm.collective_rpc(
            _worker_swap, args=(slot.id, resolved_path),
        )

        # TP=1 → single-entry list; TP>1 → per-rank stats.
        if len(results) == 1:
            stats = results[0]
        else:
            total_bytes = sum(r['total_bytes'] for r in results)
            total_elapsed = max(r['elapsed_s'] for r in results)
            stats = {
                'num_regions': sum(r['num_regions'] for r in results),
                'total_bytes': total_bytes,
                'elapsed_s': total_elapsed,
                'throughput_gb_s': (
                    (total_bytes / 1e9) / total_elapsed
                    if total_elapsed > 0 else 0
                ),
                'tensor_parallel_size': self.tp_size,
                'per_rank': results,
                'backend': results[0].get('backend', 'unknown'),
            }

        elapsed = time.perf_counter() - t0
        slot.model_name = model_name
        slot.snapshot_path = resolved_path

        logger.info(
            "Slot %d: loaded '%s' in %.2fs (%.1f GB/s, %.2f GB, backend=%s)",
            slot.id, model_name, elapsed,
            stats.get("throughput_gb_s", 0),
            stats.get("total_bytes", 0) / 1e9,
            stats.get("backend", "unknown"),
        )
        return stats

    def preload(self, model_name: str, slot_id: Optional[int] = None):
        """Synchronously pre-load a model into a slot.

        If slot_id is None, picks the first idle slot.
        """
        if model_name not in self.snapshots:
            raise ValueError(f"Model '{model_name}' not registered")

        if slot_id is not None:
            if slot_id >= len(self.slots):
                raise ValueError(f"Slot {slot_id} does not exist")
            slot = self.slots[slot_id]
        else:
            slot = next(
                (s for s in self.slots if not s.lock.locked()),
                None,
            )
            if slot is None:
                raise RuntimeError("No idle slots available for preload")

        return self._swap_model(slot, model_name)

    def status(self) -> dict:
        return {
            "base_model": self.base_model,
            "tensor_parallel_size": self.tp_size,
            "pool_size": len(self.slots),
            "registered_models": {
                name: path for name, path in self.snapshots.items()
            },
            "slots": [
                {
                    "id": s.id,
                    "model": s.model_name,
                    "busy": s.lock.locked(),
                }
                for s in self.slots
            ],
        }


# ── FastAPI application ─────────────────────────────────────────────


def create_pool_app(pool: EnginePool) -> "FastAPI":
    """Create an OpenAI-compatible FastAPI app backed by an EnginePool.

    The `model` field in each request selects which snapshot to serve.
    The pool handles slot acquisition, model swapping, and release.
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from vllm import SamplingParams

    app = FastAPI(title="thaw serve")

    # ── Special token filtering ─────────────────────────────────

    def _is_special_token(tokenizer, token_id: int) -> bool:
        """Check if a token is a special/control token that should be filtered."""
        if tokenizer is None:
            return False
        special_ids = getattr(tokenizer, 'all_special_ids', [])
        if token_id in special_ids:
            return True
        return False

    # ── Tokenizer cache ──────────────────────────────────────────

    _tokenizer_cache = {}

    def _get_tokenizer(model_name: str):
        if model_name not in _tokenizer_cache:
            try:
                from transformers import AutoTokenizer
                _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(
                    pool.base_model
                )
            except Exception as e:
                from thaw_common.telemetry import fallback_warning
                fallback_warning(
                    f"tokenizer_load({pool.base_model})", e,
                    dst="manual_prompt_formatting",
                )
                _tokenizer_cache[model_name] = None
        return _tokenizer_cache[model_name]

    def _build_chat_prompt(messages, model_name):
        tokenizer = _get_tokenizer(model_name)
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except (ValueError, KeyError):
                pass  # base model without chat template — fall through
        return (
            "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            + "\nassistant:"
        )

    # ── Helpers ──────────────────────────────────────────────────

    def _resolve_model(request: dict) -> str:
        model = request.get("model")
        if not model:
            # If only one model registered, use it as default
            if len(pool.snapshots) == 1:
                return next(iter(pool.snapshots))
            raise HTTPException(
                400,
                "model field required (registered: "
                f"{list(pool.snapshots.keys())})",
            )
        if model not in pool.snapshots:
            raise HTTPException(
                404,
                f"Model '{model}' not registered. "
                f"Available: {list(pool.snapshots.keys())}",
            )
        return model

    # ── OpenAI endpoints ─────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        data = [
            {"id": name, "object": "model", "owned_by": "thaw"}
            for name in pool.snapshots
        ]
        return JSONResponse({"object": "list", "data": data})

    @app.post("/v1/completions")
    async def completions(request: dict):
        model_name = _resolve_model(request)
        prompt = request.get("prompt", "")
        stream = request.get("stream", False)

        sampling = SamplingParams(
            temperature=request.get("temperature", 0.0),
            max_tokens=request.get("max_tokens", 100),
            top_p=request.get("top_p", 1.0),
        )

        request_id = f"cmpl-thaw-{int(time.time() * 1000)}"
        created = int(time.time())

        try:
            slot = await pool.acquire(model_name)
        except ValueError as e:
            raise HTTPException(404, str(e))

        try:
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()
            outputs = await loop.run_in_executor(
                None, slot.llm.generate, [prompt], sampling
            )
            latency = time.perf_counter() - t0

            output = outputs[0]

            if stream:
                token_ids = output.outputs[0].token_ids
                tokenizer = _get_tokenizer(model_name)

                def gen():
                    for i, tid in enumerate(token_ids):
                        if _is_special_token(tokenizer, tid):
                            continue
                        text = (
                            tokenizer.decode([tid])
                            if tokenizer
                            else output.outputs[0].text[i : i + 1]
                        )
                        is_last = i == len(token_ids) - 1
                        finish = (
                            output.outputs[0].finish_reason if is_last else None
                        )
                        chunk = {
                            "id": request_id,
                            "object": "text_completion",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "text": text,
                                    "index": 0,
                                    "finish_reason": finish,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                # Release slot before streaming tokens back
                pool.release(slot)
                slot = None
                return StreamingResponse(gen(), media_type="text/event-stream")

            text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            finish = output.outputs[0].finish_reason

            return JSONResponse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "text": text,
                            "index": 0,
                            "finish_reason": (
                                "stop" if finish == "stop" else "length"
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "thaw_metadata": {
                        "latency_s": round(latency, 3),
                        "slot_id": slot.id,
                    },
                }
            )
        finally:
            if slot is not None:
                pool.release(slot)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        model_name = _resolve_model(request)
        messages = request.get("messages", [])
        stream = request.get("stream", False)

        prompt = _build_chat_prompt(messages, model_name)

        sampling = SamplingParams(
            temperature=request.get("temperature", 0.0),
            max_tokens=request.get("max_tokens", 100),
            top_p=request.get("top_p", 1.0),
        )

        request_id = f"chatcmpl-thaw-{int(time.time() * 1000)}"
        created = int(time.time())

        try:
            slot = await pool.acquire(model_name)
        except ValueError as e:
            raise HTTPException(404, str(e))

        try:
            loop = asyncio.get_event_loop()
            t0 = time.perf_counter()
            outputs = await loop.run_in_executor(
                None, slot.llm.generate, [prompt], sampling
            )
            latency = time.perf_counter() - t0

            output = outputs[0]

            if stream:
                token_ids = output.outputs[0].token_ids
                tokenizer = _get_tokenizer(model_name)

                def gen():
                    # First chunk: role
                    yield "data: " + json.dumps(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    ) + "\n\n"

                    # Content chunks
                    for i, tid in enumerate(token_ids):
                        if _is_special_token(tokenizer, tid):
                            continue
                        text = (
                            tokenizer.decode([tid])
                            if tokenizer
                            else output.outputs[0].text[i : i + 1]
                        )
                        yield "data: " + json.dumps(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        ) + "\n\n"

                    # Final chunk
                    finish = output.outputs[0].finish_reason or "stop"
                    yield "data: " + json.dumps(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish,
                                }
                            ],
                        }
                    ) + "\n\n"
                    yield "data: [DONE]\n\n"

                pool.release(slot)
                slot = None
                return StreamingResponse(gen(), media_type="text/event-stream")

            text = output.outputs[0].text
            prompt_tokens = len(output.prompt_token_ids)
            completion_tokens = len(output.outputs[0].token_ids)
            finish = output.outputs[0].finish_reason

            return JSONResponse(
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": text,
                            },
                            "finish_reason": (
                                "stop" if finish == "stop" else "length"
                            ),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                    "thaw_metadata": {
                        "latency_s": round(latency, 3),
                        "slot_id": slot.id,
                    },
                }
            )
        finally:
            if slot is not None:
                pool.release(slot)

    # ── Admin endpoints ──────────────────────────────────────────

    @app.get("/admin/pool")
    async def pool_status():
        return JSONResponse(pool.status())

    @app.post("/admin/snapshots")
    async def register_snapshot(request: dict):
        name = request.get("name") or request.get("model")
        path = request.get("snapshot") or request.get("path")
        if not name or not path:
            raise HTTPException(400, "name and snapshot fields required")
        try:
            pool.register(name, path)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return JSONResponse({"status": "registered", "model": name, "snapshot": path})

    @app.get("/admin/snapshots")
    async def list_snapshots():
        return JSONResponse(
            {"snapshots": {k: v for k, v in pool.snapshots.items()}}
        )

    @app.delete("/admin/snapshots/{name}")
    async def unregister_snapshot(name: str):
        pool.unregister(name)
        return JSONResponse({"status": "unregistered", "model": name})

    @app.post("/admin/preload")
    async def preload_model(request: dict):
        model_name = request.get("model")
        slot_id = request.get("slot_id")
        if not model_name:
            raise HTTPException(400, "model field required")
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None, pool.preload, model_name, slot_id
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(400, str(e))
        return JSONResponse(
            {
                "status": "preloaded",
                "model": model_name,
                "throughput_gb_s": round(stats.get("throughput_gb_s", 0), 2),
                "elapsed_s": round(stats.get("elapsed_s", 0), 2),
            }
        )

    @app.get("/health")
    async def health():
        status = pool.status()
        idle = sum(1 for s in status["slots"] if not s["busy"])
        return JSONResponse(
            {
                "status": "ok",
                "pool_size": status["pool_size"],
                "idle_slots": idle,
                "registered_models": len(status["registered_models"]),
            }
        )

    return app
