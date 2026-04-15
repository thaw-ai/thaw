"""
thaw_vllm.server — OpenAI-compatible API server backed by a vLLM LLM instance.

Usage:
    from thaw_vllm.server import create_app
    app = create_app(llm, "meta-llama/Meta-Llama-3-8B")
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

import json
import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from vllm import SamplingParams


def create_app(llm, model_name: str) -> FastAPI:
    """Create an OpenAI-compatible FastAPI app around a vLLM LLM instance."""

    app = FastAPI(title="thaw")

    # Special token filtering
    def _is_special_token(tokenizer, token_id: int) -> bool:
        if tokenizer is None or tokenizer is False:
            return False
        special_ids = getattr(tokenizer, 'all_special_ids', [])
        return token_id in special_ids

    # Cache tokenizer for chat template rendering
    _tokenizer = [None]

    def _get_tokenizer():
        if _tokenizer[0] is None:
            try:
                from transformers import AutoTokenizer
                _tokenizer[0] = AutoTokenizer.from_pretrained(model_name)
            except Exception:
                _tokenizer[0] = False
        return _tokenizer[0]

    def _build_chat_prompt(messages):
        tokenizer = _get_tokenizer()
        if tokenizer and hasattr(tokenizer, 'apply_chat_template'):
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        ) + "\nassistant:"

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "owned_by": "thaw",
            }]
        })

    @app.post("/v1/completions")
    async def completions(request: dict):
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.0)
        stream = request.get("stream", False)

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=request.get("top_p", 1.0),
        )

        request_id = f"cmpl-thaw-{int(time.time() * 1000)}"
        created = int(time.time())

        if stream:
            def generate_stream():
                outputs = llm.generate([prompt], sampling)
                output = outputs[0]
                token_ids = output.outputs[0].token_ids
                tokenizer = _get_tokenizer()

                for i, token_id in enumerate(token_ids):
                    if _is_special_token(tokenizer, token_id):
                        continue
                    if tokenizer:
                        text = tokenizer.decode([token_id])
                    else:
                        text = output.outputs[0].text[i:i+1]

                    is_last = (i == len(token_ids) - 1)
                    finish = output.outputs[0].finish_reason if is_last else None

                    chunk = {
                        "id": request_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "text": text,
                            "index": 0,
                            "finish_reason": finish,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )

        # Non-streaming
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        latency = time.perf_counter() - t0

        text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        return JSONResponse({
            "id": request_id,
            "object": "text_completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "text": text,
                "index": 0,
                "finish_reason": "stop" if outputs[0].outputs[0].finish_reason == "stop" else "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "thaw_metadata": {
                "latency_s": round(latency, 3),
            }
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict):
        messages = request.get("messages", [])
        max_tokens = request.get("max_tokens", 100)
        temperature = request.get("temperature", 0.0)
        stream = request.get("stream", False)

        prompt = _build_chat_prompt(messages)

        sampling = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=request.get("top_p", 1.0),
        )

        request_id = f"chatcmpl-thaw-{int(time.time() * 1000)}"
        created = int(time.time())

        if stream:
            def generate_stream():
                outputs = llm.generate([prompt], sampling)
                output = outputs[0]
                token_ids = output.outputs[0].token_ids
                tokenizer = _get_tokenizer()

                # First chunk: role
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

                # Content chunks
                for i, token_id in enumerate(token_ids):
                    if _is_special_token(tokenizer, token_id):
                        continue
                    if tokenizer:
                        text = tokenizer.decode([token_id])
                    else:
                        text = output.outputs[0].text[i:i+1]

                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk: finish reason
                finish = output.outputs[0].finish_reason or "stop"
                chunk = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": finish,
                    }],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
            )

        # Non-streaming
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling)
        latency = time.perf_counter() - t0

        text = outputs[0].outputs[0].text
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        return JSONResponse({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop" if outputs[0].outputs[0].finish_reason == "stop" else "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "thaw_metadata": {
                "latency_s": round(latency, 3),
            }
        })

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    return app
