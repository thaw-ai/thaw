"""Async coalescer for LangGraph fan-out → ForkPool routing.

LangGraph's ``Send()`` primitive dispatches N independent asyncio tasks, each
calling the LLM's ``ainvoke`` separately. There is no runtime signal that those
N calls share a parent message prefix. This coalescer reconstructs the signal:
it buffers concurrent submit() calls for a short window, finds the longest
common message prefix across the batch, and routes groups with enough shared
context through a fork callable. Singletons and low-overlap groups fall back
to a single callable.

The coalescer is intentionally decoupled from langchain-core and vllm: it only
sees message-like objects (anything with a ``content`` attribute) and opaque
``sampling_params``. Wiring to real ForkPool + ChatThaw happens one layer up.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from thaw_vllm.langgraph._message_utils import (
    approximate_token_count,
    common_prefix_length,
)


ForkCallable = Callable[
    [Sequence[Any], Sequence[Sequence[Any]], Any],
    Awaitable[list[str]],
]
"""(prefix_messages, suffix_messages_list, sampling_params) → one generation per suffix."""

SingleCallable = Callable[[Sequence[Any], Any], Awaitable[str]]
"""(messages, sampling_params) → one generation."""

TokenCounter = Callable[[Sequence[Any]], int]


@dataclass
class _PendingCall:
    messages: Sequence[Any]
    sampling_params: Any
    future: asyncio.Future = field(repr=False)


class ForkCoalescer:
    """Buffer concurrent LLM calls, route prefix-sharing groups through a fork path.

    Parameters
    ----------
    fork_callable
        Invoked with (prefix_messages, suffix_messages_list, sampling_params) when
        ≥2 pending calls share a prefix longer than ``min_prefix_tokens``. Must
        return one generated string per suffix, in order.
    single_callable
        Invoked with (messages, sampling_params) for any call that does not
        participate in a fork group. Used for singletons and low-overlap groups.
    window_ms
        Time to buffer pending calls before dispatching. Larger values capture
        more concurrent LangGraph Sends at the cost of per-call latency on
        non-fork workloads. Default 2ms.
    min_prefix_tokens
        Minimum shared-prefix token count required to route through
        ``fork_callable``. Below this, fork setup cost typically exceeds savings.
        Default 500.
    token_counter
        Maps a message sequence to an approximate token count. Defaults to a
        4-chars-per-token heuristic; override with a real tokenizer for
        precision around the threshold boundary.
    """

    def __init__(
        self,
        *,
        fork_callable: ForkCallable,
        single_callable: SingleCallable,
        window_ms: float = 2.0,
        min_prefix_tokens: int = 500,
        token_counter: Optional[TokenCounter] = None,
    ) -> None:
        self._fork_callable = fork_callable
        self._single_callable = single_callable
        self._window_s = max(window_ms, 0.0) / 1000.0
        self._min_prefix_tokens = min_prefix_tokens
        self._token_counter = token_counter or approximate_token_count
        self._pending: list[_PendingCall] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def submit(self, messages: Sequence[Any], sampling_params: Any) -> str:
        """Queue a call. Returns the generated string once the batch flushes."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        async with self._lock:
            self._pending.append(_PendingCall(messages, sampling_params, fut))
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_after_delay())
        return await fut

    async def _flush_after_delay(self) -> None:
        if self._window_s > 0:
            await asyncio.sleep(self._window_s)
        async with self._lock:
            pending = self._pending
            self._pending = []
            self._flush_task = None
        if pending:
            await self._dispatch(pending)

    async def _dispatch(self, pending: list[_PendingCall]) -> None:
        if len(pending) == 1:
            await self._run_single(pending[0])
            return

        all_msgs = [p.messages for p in pending]
        lcp_len = common_prefix_length(all_msgs)
        if lcp_len == 0:
            await asyncio.gather(
                *(self._run_single(p) for p in pending),
                return_exceptions=True,
            )
            return

        prefix = list(all_msgs[0][:lcp_len])
        if self._token_counter(prefix) < self._min_prefix_tokens:
            await asyncio.gather(
                *(self._run_single(p) for p in pending),
                return_exceptions=True,
            )
            return

        suffix_lists = [list(p.messages[lcp_len:]) for p in pending]
        sampling_params = pending[0].sampling_params
        try:
            results = await self._fork_callable(prefix, suffix_lists, sampling_params)
        except Exception as exc:
            for call in pending:
                if not call.future.done():
                    call.future.set_exception(exc)
            return

        if len(results) != len(pending):
            exc = RuntimeError(
                f"fork_callable returned {len(results)} results for {len(pending)} calls"
            )
            for call in pending:
                if not call.future.done():
                    call.future.set_exception(exc)
            return

        for call, result in zip(pending, results):
            if not call.future.done():
                call.future.set_result(result)

    async def _run_single(self, call: _PendingCall) -> None:
        if call.future.done():
            return
        try:
            result = await self._single_callable(call.messages, call.sampling_params)
        except Exception as exc:
            if not call.future.done():
                call.future.set_exception(exc)
            return
        if not call.future.done():
            call.future.set_result(result)


__all__ = ["ForkCoalescer", "ForkCallable", "SingleCallable", "TokenCounter"]
