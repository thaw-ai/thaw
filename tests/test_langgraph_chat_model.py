"""Tests for thaw_vllm.langgraph.chat_model — ChatThaw + message conversion.

CPU-only. Real langchain-core is required (subclass of BaseChatModel must be
instantiable); vllm is mocked via conftest.py. We stub the vLLM LLM, ForkPool,
and fork_completions on a ChatThaw instance so the logic can be exercised
without GPU.
"""
from __future__ import annotations

import asyncio
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# `thaw_vllm.__init__` does `from thaw_vllm.fork import fork, fork_completions`,
# which overrides the auto-submodule attribute — so `thaw_vllm.fork` resolves
# to the `fork` function, not the submodule. Both `from thaw_vllm import fork`
# and `import thaw_vllm.fork as Z` end up bound to the function on CPython.
# Grab the real submodule out of sys.modules and patch.object against it.
import importlib
_thaw_fork_module = importlib.import_module("thaw_vllm.fork")
from thaw_vllm.langgraph.chat_model import (
    ChatThaw,
    _message_to_hf,
    _prefix_hash,
    fork_fanout,
)


# ---------------------------------------------------------------------------
# Message conversion
# ---------------------------------------------------------------------------


def test_message_to_hf_system():
    assert _message_to_hf(SystemMessage(content="sys")) == {
        "role": "system",
        "content": "sys",
    }


def test_message_to_hf_human():
    assert _message_to_hf(HumanMessage(content="hi")) == {
        "role": "user",
        "content": "hi",
    }


def test_message_to_hf_ai():
    assert _message_to_hf(AIMessage(content="ok")) == {
        "role": "assistant",
        "content": "ok",
    }


def test_message_to_hf_multipart_text():
    msg = HumanMessage(content=[{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}])
    assert _message_to_hf(msg) == {"role": "user", "content": "hello world"}


def test_prefix_hash_deterministic():
    h1 = _prefix_hash("some prefix prompt")
    h2 = _prefix_hash("some prefix prompt")
    assert h1 == h2
    assert _prefix_hash("different") != h1


# ---------------------------------------------------------------------------
# ChatThaw: pydantic init + field defaults
# ---------------------------------------------------------------------------


def test_chat_thaw_defaults():
    llm = ChatThaw(model="test-model")
    assert llm.model == "test-model"
    assert llm.fork_window_ms == 2.0
    assert llm.fork_min_prefix_tokens == 500
    # Auto-fork is OFF by default: concurrent submits dispatch via batched
    # singles. Users opt in with enable_auto_fork=True or call fork_fanout()
    # directly for guaranteed fork dispatch.
    assert llm.enable_auto_fork is False
    assert llm.workers == 4
    assert llm.tensor_parallel_size == 1
    assert llm._llm is None
    assert llm._pool is None
    assert llm._coalescer is None


def test_chat_thaw_llm_type():
    llm = ChatThaw(model="test-model")
    assert llm._llm_type == "chat-thaw"


def test_chat_thaw_identifying_params_includes_model():
    llm = ChatThaw(model="foo/bar", workers=8, fork_window_ms=5.0)
    params = llm._identifying_params
    assert params["model"] == "foo/bar"
    assert params["workers"] == 8
    assert params["fork_window_ms"] == 5.0


# ---------------------------------------------------------------------------
# Helpers: install fake vLLM plumbing on a ChatThaw without real GPU/model
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Stand-in for HF AutoTokenizer. apply_chat_template renders role:content lines."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        rendered = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            rendered += "\nassistant:"
        return rendered

    def encode(self, text):
        # Rough: one token per word
        return text.split()


class FakeCompletionOutput:
    def __init__(self, text):
        self.text = text


class FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [FakeCompletionOutput(text)]


def install_fake_plumbing(llm: ChatThaw, gen_response: str = "fake-response"):
    """Populate the private state that _load_async would set up."""
    fake_llm = MagicMock()
    fake_llm.generate = MagicMock(return_value=[FakeRequestOutput(gen_response)])
    llm._llm = fake_llm
    llm._tokenizer = FakeTokenizer()
    llm._pool = MagicMock()
    llm._warmed_lock = asyncio.Lock()
    # _coalescer is the short-circuit guard in _load_async; install a sentinel
    # so tests that invoke _load_async (e.g. fork_fanout) don't try to boot a
    # real vLLM + ForkPool.
    llm._coalescer = MagicMock()
    return fake_llm


# ---------------------------------------------------------------------------
# Prompt rendering via chat template
# ---------------------------------------------------------------------------


def test_messages_to_prompt_renders_with_generation_prompt():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)
    prompt = llm._messages_to_prompt(
        [SystemMessage(content="s"), HumanMessage(content="h")]
    )
    assert "system: s" in prompt
    assert "user: h" in prompt
    assert prompt.endswith("assistant:")


def test_messages_to_prompt_without_generation_prompt():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)
    prompt = llm._messages_to_prompt(
        [HumanMessage(content="hi")], add_generation_prompt=False
    )
    assert prompt == "user: hi"


def test_count_tokens_uses_tokenizer():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)
    # "system: s\nuser: hi hello world" → 6 words
    count = llm._count_tokens(
        [SystemMessage(content="s"), HumanMessage(content="hi hello world")]
    )
    assert count >= 5


# ---------------------------------------------------------------------------
# _do_single path — single generation via parent LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_do_single_routes_to_parent_llm():
    llm = ChatThaw(model="m")
    fake_llm = install_fake_plumbing(llm, gen_response="single-output")
    result = await llm._do_single([HumanMessage(content="hi")], sampling_params="sp")
    assert result == "single-output"
    fake_llm.generate.assert_called_once()


@pytest.mark.asyncio
async def test_do_singles_batches_into_one_generate_call():
    """_do_singles must pass all N prompts to generate() in a single call so
    vLLM's continuous batcher schedules them together; N threaded generate()
    calls deadlock V1 EngineCore."""
    llm = ChatThaw(model="m")
    fake_llm = install_fake_plumbing(llm)
    # Stage distinct outputs, one per prompt
    fake_llm.generate = MagicMock(
        return_value=[
            FakeRequestOutput("a-out"),
            FakeRequestOutput("b-out"),
            FakeRequestOutput("c-out"),
        ]
    )
    results = await llm._do_singles(
        [
            [HumanMessage(content="a")],
            [HumanMessage(content="b")],
            [HumanMessage(content="c")],
        ],
        sampling_params="sp",
    )
    assert results == ["a-out", "b-out", "c-out"]
    fake_llm.generate.assert_called_once()
    # First positional arg is the list of rendered prompts
    prompts_arg = fake_llm.generate.call_args[0][0]
    assert isinstance(prompts_arg, list)
    assert len(prompts_arg) == 3


# ---------------------------------------------------------------------------
# _do_fork path — routes to fork_completions with prefix-warm
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_do_fork_warms_prefix_and_calls_fork_completions():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)

    fake_results = [
        MagicMock(text="fork-a"),
        MagicMock(text="fork-b"),
        MagicMock(text="fork-c"),
    ]
    with patch.object(
        _thaw_fork_module, "fork_completions", return_value=fake_results
    ) as mock_fork:
        prefix = [SystemMessage(content="shared system prompt")]
        suffixes = [
            [HumanMessage(content="q1")],
            [HumanMessage(content="q2")],
            [HumanMessage(content="q3")],
        ]
        results = await llm._do_fork(prefix, suffixes, sampling_params="sp")

    assert results == ["fork-a", "fork-b", "fork-c"]
    mock_fork.assert_called_once()
    # Called as fork_completions(llm, prompts, sp, pool=pool)
    _args, kwargs = mock_fork.call_args
    assert kwargs["pool"] is llm._pool


@pytest.mark.asyncio
async def test_do_fork_skips_rewarm_on_same_prefix():
    """Warming a fully-cached prompt on vLLM V1 forces ≥1 token of prefill
    by uncaching the last block — the re-cached block then has a different
    hash than the one snapshotted to the pool workers, producing garbage
    output on round 2+. Dedup by prefix hash keeps the warm as a one-time
    setup cost."""
    llm = ChatThaw(model="m")
    fake_llm = install_fake_plumbing(llm)

    with patch.object(
        _thaw_fork_module, "fork_completions",
        return_value=[MagicMock(text="x"), MagicMock(text="y")],
    ):
        prefix = [SystemMessage(content="same prefix")]
        suffixes_a = [[HumanMessage(content="q1")], [HumanMessage(content="q2")]]
        await llm._do_fork(prefix, suffixes_a, sampling_params="sp")

        warm_calls_after_first = fake_llm.generate.call_count

        suffixes_b = [[HumanMessage(content="q3")], [HumanMessage(content="q4")]]
        await llm._do_fork(prefix, suffixes_b, sampling_params="sp")

        # Second round with the same prefix should NOT re-warm.
        assert fake_llm.generate.call_count == warm_calls_after_first


# ---------------------------------------------------------------------------
# Sampling params builder
# ---------------------------------------------------------------------------


def test_make_sampling_params_uses_defaults():
    llm = ChatThaw(model="m", temperature=0.5, max_tokens=123, top_p=0.9)
    sp = llm._make_sampling_params()
    # vllm.SamplingParams is mocked in conftest, but the call args get recorded.
    # We verify that the method ran without error; detailed SP shape is checked
    # on pod under GPU validation.
    assert sp is not None


def test_make_sampling_params_overrides_via_kwargs():
    llm = ChatThaw(model="m", temperature=0.5)
    sp = llm._make_sampling_params(temperature=0.9, max_tokens=50)
    assert sp is not None


# ---------------------------------------------------------------------------
# enable_auto_fork wiring — coalescer threshold gate
# ---------------------------------------------------------------------------


def _fake_load_sync(llm: ChatThaw) -> None:
    """Stand-in for the real _load_sync so we can exercise _load_async without
    spinning up vLLM. Sets exactly the attributes _load_sync would set."""
    llm._llm = MagicMock()
    llm._tokenizer = FakeTokenizer()
    llm._pool = MagicMock()


@pytest.mark.asyncio
async def test_load_async_pins_fork_threshold_when_auto_fork_off():
    # Default path: auto-fork off → coalescer's effective threshold is
    # unreachable by the token counter, so every flush falls through to the
    # batched-singles path regardless of how long the shared prefix is.
    llm = ChatThaw(model="m")
    with patch.object(ChatThaw, "_load_sync", _fake_load_sync):
        await llm._load_async()
    assert llm._coalescer is not None
    assert llm._coalescer._min_prefix_tokens >= 10**12


@pytest.mark.asyncio
async def test_load_async_uses_fork_min_when_auto_fork_on():
    llm = ChatThaw(model="m", enable_auto_fork=True, fork_min_prefix_tokens=321)
    with patch.object(ChatThaw, "_load_sync", _fake_load_sync):
        await llm._load_async()
    assert llm._coalescer._min_prefix_tokens == 321


# ---------------------------------------------------------------------------
# fork_fanout helper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fork_fanout_bypasses_coalescer():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)

    fake_results = [MagicMock(text="a"), MagicMock(text="b")]
    with patch.object(
        _thaw_fork_module, "fork_completions", return_value=fake_results
    ) as mock_fork:
        results = await fork_fanout(
            llm,
            [SystemMessage(content="prefix")],
            [[HumanMessage(content="x")], [HumanMessage(content="y")]],
        )

    assert results == ["a", "b"]
    mock_fork.assert_called_once()


# ---------------------------------------------------------------------------
# close() is idempotent
# ---------------------------------------------------------------------------


def test_close_is_idempotent():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)
    pool = llm._pool
    llm.close()
    pool.close.assert_called_once()
    llm.close()  # should not raise
    pool.close.assert_called_once()  # still only one call


def test_close_swallows_pool_errors():
    llm = ChatThaw(model="m")
    install_fake_plumbing(llm)
    llm._pool.close.side_effect = RuntimeError("pool gone")
    llm.close()  # should not raise
