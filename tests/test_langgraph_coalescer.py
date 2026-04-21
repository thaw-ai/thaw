"""Tests for thaw_vllm.langgraph.coalescer — ForkCoalescer + prefix detection.

CPU-only. The coalescer has no vllm/langchain dependency, so we pass plain
dicts as stand-in messages and async mock callables as the fork/single paths.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest.mock import AsyncMock

import pytest

from thaw_vllm.langgraph._message_utils import (
    approximate_token_count,
    common_prefix_length,
    message_key,
)
from thaw_vllm.langgraph.coalescer import ForkCoalescer


# ---------------------------------------------------------------------------
# Test fixtures: lightweight "message" objects + callable recorders
# ---------------------------------------------------------------------------


@dataclass
class FakeMessage:
    """Duck-typed stand-in for a LangChain BaseMessage."""

    content: str
    role: str = "user"


def msg(content: str, cls: type = FakeMessage) -> FakeMessage:
    return cls(content=content)


@dataclass
class CallLog:
    fork_calls: list = field(default_factory=list)
    single_calls: list = field(default_factory=list)


def make_callables(log: CallLog, fork_return=None, fork_error=None, single_error=None):
    async def fork_callable(prefix, suffix_lists, sampling_params):
        log.fork_calls.append((list(prefix), [list(s) for s in suffix_lists], sampling_params))
        if fork_error is not None:
            raise fork_error
        if fork_return is not None:
            return fork_return
        return [f"fork-result-{i}" for i in range(len(suffix_lists))]

    async def single_callable(messages, sampling_params):
        log.single_calls.append((list(messages), sampling_params))
        if single_error is not None:
            raise single_error
        return f"single-{messages[-1].content}"

    return fork_callable, single_callable


def make_batch_single(log: CallLog, error=None, wrong_count=False):
    async def batch_single(messages_list, sampling_params):
        log.single_calls.append(([list(m) for m in messages_list], sampling_params))
        if error is not None:
            raise error
        if wrong_count:
            return ["only-one"]
        return [f"batch-{m[-1].content}" for m in messages_list]

    return batch_single


# ---------------------------------------------------------------------------
# _message_utils tests
# ---------------------------------------------------------------------------


def test_message_key_same_type_same_content():
    a = msg("hello")
    b = msg("hello")
    assert message_key(a) == message_key(b)


def test_message_key_different_content():
    assert message_key(msg("a")) != message_key(msg("b"))


def test_message_key_different_type():
    @dataclass
    class OtherMessage:
        content: str

    assert message_key(msg("hi")) != message_key(OtherMessage(content="hi"))


def test_message_key_multipart_content():
    @dataclass
    class MultipartMsg:
        content: Any

    a = MultipartMsg(content=[{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": "x"}])
    b = MultipartMsg(content=[{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": "x"}])
    assert message_key(a) == message_key(b)


def test_common_prefix_length_full_match():
    msgs = [[msg("a"), msg("b"), msg("c")]] * 3
    assert common_prefix_length(msgs) == 3


def test_common_prefix_length_partial():
    msgs = [
        [msg("a"), msg("b"), msg("c1")],
        [msg("a"), msg("b"), msg("c2")],
        [msg("a"), msg("b"), msg("c3")],
    ]
    assert common_prefix_length(msgs) == 2


def test_common_prefix_length_no_match():
    msgs = [[msg("a")], [msg("b")], [msg("c")]]
    assert common_prefix_length(msgs) == 0


def test_common_prefix_length_empty():
    assert common_prefix_length([]) == 0


def test_common_prefix_length_single_list():
    msgs = [[msg("a"), msg("b"), msg("c")]]
    assert common_prefix_length(msgs) == 3


def test_common_prefix_length_varying_lengths():
    msgs = [
        [msg("a"), msg("b"), msg("c"), msg("d")],
        [msg("a"), msg("b")],
        [msg("a"), msg("b"), msg("c")],
    ]
    assert common_prefix_length(msgs) == 2


def test_approximate_token_count_strings():
    m = [msg("a" * 400), msg("b" * 400)]
    # 800 chars / 4 = 200 tokens
    assert approximate_token_count(m) == 200


def test_approximate_token_count_empty():
    assert approximate_token_count([]) == 0


# ---------------------------------------------------------------------------
# Coalescer: single-call fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_call_uses_single_callable():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=1.0,
        min_prefix_tokens=100,
    )
    result = await coalescer.submit([msg("hello")], sampling_params={"t": 0.7})
    assert result == "single-hello"
    assert len(log.single_calls) == 1
    assert len(log.fork_calls) == 0


# ---------------------------------------------------------------------------
# Coalescer: N-way fan-out with shared prefix triggers fork
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_four_way_fanout_triggers_fork():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=50,
    )

    # Shared prefix: 400 chars → ~100 tokens. Above threshold of 50.
    prefix = [msg("a" * 400)]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"branch-{i}")], sampling_params="sp"))
        for i in range(4)
    ]
    results = await asyncio.gather(*tasks)

    assert results == [f"fork-result-{i}" for i in range(4)]
    assert len(log.fork_calls) == 1
    assert len(log.single_calls) == 0

    prefix_got, suffix_lists_got, sp_got = log.fork_calls[0]
    assert len(prefix_got) == 1
    assert prefix_got[0].content == "a" * 400
    assert len(suffix_lists_got) == 4
    assert [s[0].content for s in suffix_lists_got] == [f"branch-{i}" for i in range(4)]
    assert sp_got == "sp"


# ---------------------------------------------------------------------------
# Coalescer: N calls with prefix below threshold → all fall through to single
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_overlap_falls_back_to_single():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=500,
    )

    # Prefix of only 40 chars → ~10 tokens. Well below 500.
    prefix = [msg("short prefix only 40 characters long xxx")]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"b{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert set(results) == {"single-b0", "single-b1", "single-b2"}
    assert len(log.fork_calls) == 0
    assert len(log.single_calls) == 3


# ---------------------------------------------------------------------------
# Coalescer: N calls with fully-disjoint messages → all fall through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_common_prefix_falls_back_to_single():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=10,
    )

    tasks = [
        asyncio.create_task(coalescer.submit([msg(f"totally-different-{i}" * 100)], sampling_params=None))
        for i in range(3)
    ]
    await asyncio.gather(*tasks)

    assert len(log.fork_calls) == 0
    assert len(log.single_calls) == 3


# ---------------------------------------------------------------------------
# Coalescer: fork callable error propagates per-branch, doesn't kill others
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_error_propagates_to_all_pending():
    log = CallLog()
    fork, single = make_callables(log, fork_error=RuntimeError("boom"))
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=50,
    )

    prefix = [msg("a" * 400)]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"branch-{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(r, RuntimeError) and "boom" in str(r) for r in results)
    assert len(log.fork_calls) == 1
    # single_callable should NOT be used as a fallback when the whole fork errors;
    # each call gets the fork exception. (We don't retry per-branch on fork failure.)
    assert len(log.single_calls) == 0


# ---------------------------------------------------------------------------
# Coalescer: single callable error isolates to its own call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_error_isolates_to_one_call():
    log = CallLog()

    async def selective_single(messages, sampling_params):
        log.single_calls.append((list(messages), sampling_params))
        if messages[-1].content == "bad":
            raise ValueError("one bad branch")
        return f"ok-{messages[-1].content}"

    async def fork(prefix, suffix_lists, sp):  # unused in this test
        log.fork_calls.append(True)
        return ["x"] * len(suffix_lists)

    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=selective_single,
        window_ms=5.0,
        min_prefix_tokens=10000,  # force fallback
    )

    t_good = asyncio.create_task(coalescer.submit([msg("good")], sampling_params=None))
    t_bad = asyncio.create_task(coalescer.submit([msg("bad")], sampling_params=None))
    t_also_good = asyncio.create_task(coalescer.submit([msg("also-good")], sampling_params=None))

    results = await asyncio.gather(t_good, t_bad, t_also_good, return_exceptions=True)

    assert results[0] == "ok-good"
    assert isinstance(results[1], ValueError)
    assert results[2] == "ok-also-good"


# ---------------------------------------------------------------------------
# Coalescer: window triggers a second batch for calls arriving after flush
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_two_sequential_batches_dispatch_independently():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=2.0,
        min_prefix_tokens=50,
    )

    prefix = [msg("a" * 400)]

    # First batch
    t1 = asyncio.create_task(coalescer.submit(prefix + [msg("x1")], sampling_params=None))
    t2 = asyncio.create_task(coalescer.submit(prefix + [msg("x2")], sampling_params=None))
    await asyncio.gather(t1, t2)

    # Wait for flush to finalize
    await asyncio.sleep(0.01)

    # Second batch, arriving well after the first flush window
    t3 = asyncio.create_task(coalescer.submit(prefix + [msg("y1")], sampling_params=None))
    t4 = asyncio.create_task(coalescer.submit(prefix + [msg("y2")], sampling_params=None))
    await asyncio.gather(t3, t4)

    assert len(log.fork_calls) == 2
    assert [m.content for m in log.fork_calls[0][0]] == ["a" * 400]
    assert [m.content for m in log.fork_calls[1][0]] == ["a" * 400]
    # First batch suffixes
    assert [s[0].content for s in log.fork_calls[0][1]] == ["x1", "x2"]
    # Second batch suffixes
    assert [s[0].content for s in log.fork_calls[1][1]] == ["y1", "y2"]


# ---------------------------------------------------------------------------
# Coalescer: custom token_counter affects threshold decisions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_token_counter_used_for_threshold():
    log = CallLog()
    fork, single = make_callables(log)

    # Custom counter that always returns 1000 regardless of prefix size
    def fake_counter(messages):
        return 1000

    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=2.0,
        min_prefix_tokens=500,
        token_counter=fake_counter,
    )

    # Even with a tiny shared prefix, fake_counter says it's 1000 tokens → fork.
    prefix = [msg("hi")]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"b{i}")], sampling_params=None))
        for i in range(2)
    ]
    await asyncio.gather(*tasks)

    assert len(log.fork_calls) == 1
    assert len(log.single_calls) == 0


# ---------------------------------------------------------------------------
# Coalescer: zero window flushes on next event loop tick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_window_still_works():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=0.0,
        min_prefix_tokens=50,
    )

    prefix = [msg("a" * 400)]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"b{i}")], sampling_params=None))
        for i in range(2)
    ]
    results = await asyncio.gather(*tasks)
    # With window=0 the flush fires after first awaited sleep(0); both submits
    # should land in the same pending list because they run sequentially in
    # create_task order before the loop yields to the flush task.
    assert results == ["fork-result-0", "fork-result-1"]
    assert len(log.fork_calls) == 1


# ---------------------------------------------------------------------------
# Coalescer: sampling params from first pending call are used for the group
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sampling_params_from_first_call():
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=50,
    )

    prefix = [msg("a" * 400)]
    t1 = asyncio.create_task(coalescer.submit(prefix + [msg("one")], sampling_params="first-sp"))
    t2 = asyncio.create_task(coalescer.submit(prefix + [msg("two")], sampling_params="second-sp"))
    await asyncio.gather(t1, t2)

    assert log.fork_calls[0][2] == "first-sp"


# ---------------------------------------------------------------------------
# Coalescer: mismatched result count → exception to all pending
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coalescer: batched single path — N pending, prefix below threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_batch_single_callable_used_when_prefix_below_threshold():
    log = CallLog()
    fork, single = make_callables(log)
    batch_single = make_batch_single(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        batch_single_callable=batch_single,
        window_ms=2.0,
        min_prefix_tokens=10000,  # unreachable → always fall through
    )

    prefix = [msg("a" * 40)]  # tiny — well below threshold
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"b{i}")], sampling_params="sp"))
        for i in range(4)
    ]
    results = await asyncio.gather(*tasks)

    assert results == [f"batch-b{i}" for i in range(4)]
    # batch_single gets recorded in log.single_calls; fork must not be touched
    assert len(log.fork_calls) == 0
    # Exactly one batch invocation, not four
    assert len(log.single_calls) == 1
    msgs_list, sp = log.single_calls[0]
    assert len(msgs_list) == 4
    assert sp == "sp"


@pytest.mark.asyncio
async def test_batch_single_callable_used_when_lcp_zero():
    log = CallLog()
    fork, single = make_callables(log)
    batch_single = make_batch_single(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        batch_single_callable=batch_single,
        window_ms=2.0,
        min_prefix_tokens=10,
    )

    tasks = [
        asyncio.create_task(coalescer.submit([msg(f"totally-different-{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks)

    assert set(results) == {"batch-totally-different-0", "batch-totally-different-1", "batch-totally-different-2"}
    assert len(log.single_calls) == 1


@pytest.mark.asyncio
async def test_batch_single_error_propagates_to_all():
    log = CallLog()
    fork, single = make_callables(log)
    batch_single = make_batch_single(log, error=RuntimeError("batch boom"))
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        batch_single_callable=batch_single,
        window_ms=2.0,
        min_prefix_tokens=10000,
    )

    tasks = [
        asyncio.create_task(coalescer.submit([msg(f"b{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) and "batch boom" in str(r) for r in results)


@pytest.mark.asyncio
async def test_batch_single_result_count_mismatch_raises():
    log = CallLog()
    fork, single = make_callables(log)
    batch_single = make_batch_single(log, wrong_count=True)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        batch_single_callable=batch_single,
        window_ms=2.0,
        min_prefix_tokens=10000,
    )

    tasks = [
        asyncio.create_task(coalescer.submit([msg(f"b{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(r, RuntimeError) for r in results)
    assert all("returned 1 results for 3 calls" in str(r) for r in results)


@pytest.mark.asyncio
async def test_no_batch_callable_falls_back_to_gather_of_singles():
    """Back-compat: if batch_single_callable isn't supplied, gather singles."""
    log = CallLog()
    fork, single = make_callables(log)
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        # no batch_single_callable
        window_ms=2.0,
        min_prefix_tokens=10000,
    )

    tasks = [
        asyncio.create_task(coalescer.submit([msg(f"b{i}")], sampling_params=None))
        for i in range(3)
    ]
    await asyncio.gather(*tasks)

    assert len(log.single_calls) == 3  # one per pending, via gather
    assert len(log.fork_calls) == 0


# ---------------------------------------------------------------------------
# Coalescer: mismatched result count → exception to all pending
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mismatched_fork_result_count_raises():
    log = CallLog()
    fork, single = make_callables(log, fork_return=["only-one-result"])  # but 3 pending
    coalescer = ForkCoalescer(
        fork_callable=fork,
        single_callable=single,
        window_ms=5.0,
        min_prefix_tokens=50,
    )

    prefix = [msg("a" * 400)]
    tasks = [
        asyncio.create_task(coalescer.submit(prefix + [msg(f"b{i}")], sampling_params=None))
        for i in range(3)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    assert all(isinstance(r, RuntimeError) for r in results)
    assert all("returned 1 results for 3 calls" in str(r) for r in results)
