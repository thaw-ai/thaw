"""ChatThaw — LangChain BaseChatModel backed by thaw's fork primitive.

Usage with LangGraph::

    from thaw_vllm.langgraph import ChatThaw, fork_fanout
    llm = ChatThaw(model="meta-llama/Llama-3.1-8B-Instruct")
    # Drop into a StateGraph; concurrent Send() calls are auto-batched
    # via vLLM continuous batching. For guaranteed fork dispatch, call
    # fork_fanout(llm, parent_messages, suffix_message_lists) — that is
    # the supported primitive and what the fork_pool_rl receipt measures.

Requires ``langchain-core>=0.3``. Install with ``pip install thaw-vllm[langgraph]``.
"""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, Field, PrivateAttr

from thaw_vllm.langgraph._message_utils import approximate_token_count
from thaw_vllm.langgraph.coalescer import ForkCoalescer


_ROLE_MAP = {
    "SystemMessage": "system",
    "HumanMessage": "user",
    "AIMessage": "assistant",
    "ToolMessage": "tool",
    "FunctionMessage": "function",
    "ChatMessage": None,  # use message.role
}


def _message_to_hf(msg: BaseMessage) -> dict:
    """Convert a LangChain BaseMessage to the dict shape HF chat templates expect."""
    type_name = type(msg).__name__
    role = _ROLE_MAP.get(type_name)
    if role is None:
        role = getattr(msg, "role", "user")
    content = msg.content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)
    return {"role": role, "content": content}


def _prefix_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8", errors="ignore")).hexdigest()


class ChatThaw(BaseChatModel):
    """LangChain chat model backed by a vLLM parent + thaw ForkPool.

    Single ``ainvoke`` calls route through the parent LLM. Concurrent calls
    (e.g. from a LangGraph ``Send`` fan-out) are buffered for ``fork_window_ms``
    and dispatched as one batched ``LLM.generate`` call (continuous batching).

    To route qualifying batches through ``ForkPool.fork_completions`` for
    prefill-skip amortization, set ``enable_auto_fork=True``. Auto-routing is
    off by default because the interaction between repeated identical prefixes
    and vLLM's V1 prefix cache still has a rough edge that corrupts output on
    round 2+ of a fan-out. Users who want guaranteed fork dispatch for a known
    parent/suffix split should call ``fork_fanout()`` directly — that entry
    point is the primitive and is what the fork_pool_rl receipt measures.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model: str = Field(description="HF repo id or local path of the model to load.")

    fork_window_ms: float = 2.0
    fork_min_prefix_tokens: int = 500
    enable_auto_fork: bool = False
    workers: int = 4

    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    gpu_memory_utilization: float = 0.25
    worker_gpu_memory_utilization: float = 0.55

    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 1.0

    extra_llm_kwargs: dict = Field(default_factory=dict)
    extra_pool_kwargs: dict = Field(default_factory=dict)

    _llm: Any = PrivateAttr(default=None)
    _pool: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _coalescer: Optional[ForkCoalescer] = PrivateAttr(default=None)
    _init_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)
    _warmed_prefixes: set = PrivateAttr(default_factory=set)
    _warmed_lock: Optional[asyncio.Lock] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "chat-thaw"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "fork_window_ms": self.fork_window_ms,
            "fork_min_prefix_tokens": self.fork_min_prefix_tokens,
            "enable_auto_fork": self.enable_auto_fork,
            "workers": self.workers,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

    # ---- lazy init ---------------------------------------------------------

    def _load_sync(self) -> None:
        if self._coalescer is not None:
            return
        # The fork path (thaw_vllm.fork_completions → _assert_prefix_caching_enabled)
        # needs in-proc scheduler access to snapshot the block pool, which V1 MP's
        # subprocess EngineCoreClient does not expose. setdefault, so a caller who
        # explicitly wants V1 MP on can still override (they just can't fork).
        import os
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        from vllm import LLM
        from thaw_vllm.fork_pool import ForkPool

        llm_kwargs = {
            "enable_prefix_caching": True,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "enforce_eager": self.enforce_eager,
            **self.extra_llm_kwargs,
        }
        # Build parent + pool into locals first; publish them atomically only
        # after both succeed, so a worker-init failure doesn't leave a partially
        # loaded ChatThaw whose `self._llm is not None` would fool the guard.
        llm = LLM(model=self.model, **llm_kwargs)
        try:
            pool = ForkPool()
            pool_kwargs = {
                "workers": self.workers,
                "preload_weights": True,
                "gpu_memory_utilization": self.worker_gpu_memory_utilization,
                "tensor_parallel_size": self.tensor_parallel_size,
                **self.extra_pool_kwargs,
            }
            pool.init_pool(model=self.model, **pool_kwargs)
        except Exception:
            del llm
            raise
        self._llm = llm
        self._tokenizer = llm.get_tokenizer()
        self._pool = pool

    async def _load_async(self) -> None:
        if self._coalescer is not None:
            return
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._coalescer is not None:
                return
            await asyncio.to_thread(self._load_sync)
            self._warmed_lock = asyncio.Lock()
            # When auto-fork is off, pin min_prefix_tokens to a value the
            # token_counter can never produce, so the coalescer always falls
            # through to the batched-singles path.
            effective_min = (
                self.fork_min_prefix_tokens if self.enable_auto_fork else 10**12
            )
            self._coalescer = ForkCoalescer(
                fork_callable=self._do_fork,
                single_callable=self._do_single,
                batch_single_callable=self._do_singles,
                window_ms=self.fork_window_ms,
                min_prefix_tokens=effective_min,
                token_counter=self._count_tokens,
            )

    # ---- prompt + sampling helpers ----------------------------------------

    def _messages_to_prompt(
        self,
        messages: Sequence[BaseMessage],
        *,
        add_generation_prompt: bool = True,
    ) -> str:
        hf = [_message_to_hf(m) for m in messages]
        return self._tokenizer.apply_chat_template(
            hf,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    def _count_tokens(self, messages: Sequence[BaseMessage]) -> int:
        if self._tokenizer is None:
            return approximate_token_count(messages)
        prompt = self._messages_to_prompt(messages, add_generation_prompt=False)
        return len(self._tokenizer.encode(prompt))

    def _make_sampling_params(
        self, stop: Optional[List[str]] = None, **overrides: Any
    ):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=overrides.get("temperature", self.temperature),
            top_p=overrides.get("top_p", self.top_p),
            max_tokens=overrides.get("max_tokens", self.max_tokens),
            stop=list(stop) if stop else None,
        )

    # ---- coalescer callbacks ---------------------------------------------

    async def _do_fork(
        self,
        prefix_messages: Sequence[BaseMessage],
        suffix_message_lists: Sequence[Sequence[BaseMessage]],
        sampling_params: Any,
    ) -> List[str]:
        from thaw_vllm.fork import fork_completions

        prefix_prompt = self._messages_to_prompt(
            prefix_messages, add_generation_prompt=False
        )
        full_prompts = [
            self._messages_to_prompt(list(prefix_messages) + list(suffix))
            for suffix in suffix_message_lists
        ]
        await self._ensure_prefix_warm(prefix_prompt)
        results = await asyncio.to_thread(
            fork_completions,
            self._llm,
            full_prompts,
            sampling_params,
            pool=self._pool,
        )
        return [r.text for r in results]

    async def _do_single(
        self, messages: Sequence[BaseMessage], sampling_params: Any
    ) -> str:
        prompt = self._messages_to_prompt(messages)
        outputs = await asyncio.to_thread(
            self._llm.generate, prompt, sampling_params
        )
        return outputs[0].outputs[0].text

    async def _do_singles(
        self,
        messages_list: Sequence[Sequence[BaseMessage]],
        sampling_params: Any,
    ) -> List[str]:
        # vLLM V1 `LLM.generate` is not thread-safe: N concurrent threads each
        # calling it deadlock the EngineCore request loop. Pass all N prompts
        # in one call so the scheduler batches them via continuous batching.
        prompts = [self._messages_to_prompt(m) for m in messages_list]
        outputs = await asyncio.to_thread(
            self._llm.generate, prompts, sampling_params
        )
        return [o.outputs[0].text for o in outputs]

    async def _ensure_prefix_warm(self, prefix_prompt: str) -> None:
        # fork_completions snapshots the parent's cached prefix blocks, so
        # the parent must have prefilled the prefix at least once before
        # the first fork. Dedup by prefix hash so repeated fan-outs on the
        # same parent do not re-run prefill.
        from vllm import SamplingParams

        prefix_hash = _prefix_hash(prefix_prompt)
        async with self._warmed_lock:
            if prefix_hash in self._warmed_prefixes:
                return
        warm_sp = SamplingParams(temperature=0.0, max_tokens=1)
        await asyncio.to_thread(self._llm.generate, prefix_prompt, warm_sp)
        async with self._warmed_lock:
            self._warmed_prefixes.add(prefix_hash)

    # ---- BaseChatModel implementation ------------------------------------

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        await self._load_async()
        sp = self._make_sampling_params(stop=stop, **kwargs)
        text = await self._coalescer.submit(messages, sp)
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=text))]
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "ChatThaw._generate was called from inside a running event loop. "
                "Use ainvoke/abatch/agenerate instead."
            )
        except RuntimeError as exc:
            if "running event loop" in str(exc):
                raise
        return asyncio.run(
            self._agenerate(
                messages, stop=stop, run_manager=None, **kwargs
            )
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        # Streaming isn't implemented yet; fall back to full generate.
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in result.generations:
            yield gen

    # ---- lifecycle --------------------------------------------------------

    def close(self) -> None:
        """Shut down the pool. Safe to call multiple times."""
        pool = self._pool
        if pool is not None:
            try:
                pool.close()
            except Exception:
                pass
            self._pool = None


async def fork_fanout(
    llm: ChatThaw,
    parent_messages: Sequence[BaseMessage],
    suffix_message_lists: Sequence[Sequence[BaseMessage]],
    *,
    stop: Optional[List[str]] = None,
    **sampling_overrides: Any,
) -> List[str]:
    """Explicit fork fan-out — guaranteed fork dispatch, bypasses the coalescer.

    This is the supported primitive: the caller tells us the parent/suffix
    split, we snapshot the parent's cached prefix once and fan out to the
    ForkPool. It is what the fork_pool_rl receipt measures and what YC-demo
    workloads should route through.
    """
    await llm._load_async()
    sp = llm._make_sampling_params(stop=stop, **sampling_overrides)
    return await llm._do_fork(parent_messages, suffix_message_lists, sp)


__all__ = ["ChatThaw", "fork_fanout"]
