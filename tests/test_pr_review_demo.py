"""Integration smoke-test for demos/pr_review_langgraph.py.

Runs the real LangGraph StateGraph against a ChatThaw whose fork and single
callables are stubbed. Verifies that:

  1. The graph compiles and a fan-out produces the expected 4 Send branches.
  2. With the shared-system-prompt structure, the coalescer detects a common
     message prefix and routes the 4 concurrent calls through the fork path.
  3. Each specialist receives its own perspective instruction.

GPU is not required; ForkPool.fork_completions is patched.
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pytest

# Put the repo root on sys.path so `demos.` and `tests.` resolve as namespace
# packages. `testpaths=tests` + `pythonpath=python` in pyproject doesn't cover
# these, and the prior `sys.path.insert(0, "demos")` shadowed the intended
# `demos.pr_review_langgraph` dotted import.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


import importlib  # noqa: E402

# `thaw_vllm.__init__` re-exports the `fork` function, shadowing the submodule
# attribute. Grab the real submodule from sys.modules for patch.object.
_thaw_fork_module = importlib.import_module("thaw_vllm.fork")  # noqa: E402

from thaw_vllm.langgraph.chat_model import ChatThaw  # noqa: E402


def _install_fake_plumbing(llm: ChatThaw):
    """Fill in the private state the real _load_sync would populate."""
    from tests.test_langgraph_chat_model import FakeRequestOutput, FakeTokenizer

    fake_llm = MagicMock()
    fake_llm.generate = MagicMock(return_value=[FakeRequestOutput("warm-ok")])
    llm._llm = fake_llm
    llm._tokenizer = FakeTokenizer()
    llm._pool = MagicMock()
    llm._warmed_lock = asyncio.Lock()
    return fake_llm


async def test_pr_review_graph_triggers_fork_path():
    """The LangGraph fan-out should coalesce into one fork call with 4 suffixes."""
    from demos.pr_review_langgraph import (
        _DEFAULT_CONTEXT,
        _DEFAULT_DIFF,
        _build_graph,
    )
    from thaw_vllm.langgraph.coalescer import ForkCoalescer

    llm = ChatThaw(
        model="test-model",
        fork_window_ms=10.0,
        fork_min_prefix_tokens=50,
    )
    _install_fake_plumbing(llm)
    # Pre-install the coalescer (normally done in _load_async, which we skip).
    llm._coalescer = ForkCoalescer(
        fork_callable=llm._do_fork,
        single_callable=llm._do_single,
        window_ms=llm.fork_window_ms,
        min_prefix_tokens=llm.fork_min_prefix_tokens,
        token_counter=llm._count_tokens,
    )
    # Skip the expensive _load_async by pre-asserting already loaded.
    async def _noop_load():
        return None
    llm._load_async = _noop_load

    fork_call_log = []

    def fake_fork_completions(parent_llm, prompts, sp, pool=None):
        fork_call_log.append({"n_prompts": len(prompts), "prompts": prompts, "pool": pool})
        return [MagicMock(text=f"fork-review-{i}") for i in range(len(prompts))]

    graph = _build_graph()

    with patch.object(
        _thaw_fork_module, "fork_completions", side_effect=fake_fork_completions
    ):
        result = await graph.ainvoke(
            {
                "diff": _DEFAULT_DIFF,
                "context": _DEFAULT_CONTEXT,
                "reviews": [],
            },
            config={"configurable": {"llm": llm}},
        )

    # All 4 specialists produced reviews
    assert len(result["reviews"]) == 4
    perspectives = {r["perspective"] for r in result["reviews"]}
    assert perspectives == {"security", "performance", "style", "correctness"}

    # The coalescer should have batched them into one fork call with 4 prompts
    assert len(fork_call_log) == 1
    assert fork_call_log[0]["n_prompts"] == 4

    # Each prompt should contain a different perspective instruction (verifies
    # that the suffix messages actually differ per branch)
    prompts = fork_call_log[0]["prompts"]
    assert any("security" in p for p in prompts)
    assert any("performance" in p for p in prompts)
    assert any("style" in p for p in prompts)
    assert any("correctness" in p for p in prompts)


async def test_pr_review_graph_baseline_mode_uses_single_path():
    """With fork_min_prefix_tokens unreachable, all calls fall through to single."""
    from demos.pr_review_langgraph import (
        _DEFAULT_CONTEXT,
        _DEFAULT_DIFF,
        _build_graph,
    )
    from thaw_vllm.langgraph.coalescer import ForkCoalescer

    llm = ChatThaw(
        model="test-model",
        fork_window_ms=10.0,
        fork_min_prefix_tokens=10 ** 9,  # baseline mode
    )
    fake_parent = _install_fake_plumbing(llm)
    fake_parent.generate.return_value = [
        MagicMock(outputs=[MagicMock(text="baseline-review")])
    ]

    llm._coalescer = ForkCoalescer(
        fork_callable=llm._do_fork,
        single_callable=llm._do_single,
        window_ms=llm.fork_window_ms,
        min_prefix_tokens=llm.fork_min_prefix_tokens,
        token_counter=llm._count_tokens,
    )
    async def _noop_load():
        return None
    llm._load_async = _noop_load

    fork_call_log = []

    def fake_fork_completions(*args, **kwargs):
        fork_call_log.append(True)
        return []

    graph = _build_graph()

    with patch.object(
        _thaw_fork_module, "fork_completions", side_effect=fake_fork_completions
    ):
        result = await graph.ainvoke(
            {"diff": _DEFAULT_DIFF, "context": _DEFAULT_CONTEXT, "reviews": []},
            config={"configurable": {"llm": llm}},
        )

    assert len(result["reviews"]) == 4
    # fork_completions should NOT have been called (threshold unreachable)
    assert fork_call_log == []
    # Each branch goes through parent.generate via _do_single
    # (4 branches; the first call also did a warm for the prefix hash cache)
    assert fake_parent.generate.call_count >= 4
