"""Integration smoke-test for demos/pr_review_langgraph.py.

Runs the real LangGraph StateGraph against a ChatThaw whose fork and single
callables are stubbed. Verifies that:

  1. ``--mode thaw`` — the graph compiles to a single reviewer node that
     calls ``fork_fanout`` once with four divergent suffixes, and each
     prompt carries the right perspective instruction.
  2. ``--mode baseline`` — the Send-based fan-out produces four specialist
     branches, none of which hit the fork path.

GPU is not required; fork_completions + parent.generate are patched.
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
from thaw_vllm.langgraph.coalescer import ForkCoalescer  # noqa: E402


def _install_fake_plumbing(llm: ChatThaw):
    """Fill in the private state the real _load_sync would populate."""
    from tests.test_langgraph_chat_model import FakeRequestOutput, FakeTokenizer

    fake_llm = MagicMock()
    fake_llm.generate = MagicMock(return_value=[FakeRequestOutput("warm-ok")])
    llm._llm = fake_llm
    llm._tokenizer = FakeTokenizer()
    llm._pool = MagicMock()
    llm._warmed_lock = asyncio.Lock()
    # Install a ready coalescer so ChatThaw short-circuits _load_async.
    llm._coalescer = ForkCoalescer(
        fork_callable=llm._do_fork,
        single_callable=llm._do_single,
        batch_single_callable=llm._do_singles,
        window_ms=llm.fork_window_ms,
        min_prefix_tokens=10**12,  # auto-fork off; matches production default
        token_counter=llm._count_tokens,
    )

    async def _noop_load():
        return None

    llm._load_async = _noop_load
    return fake_llm


@pytest.mark.asyncio
async def test_pr_review_graph_thaw_mode_uses_fork_fanout():
    """thaw mode dispatches a single fork_fanout with 4 divergent suffixes."""
    from demos.pr_review_langgraph import (
        _DEFAULT_CONTEXT,
        _DEFAULT_DIFF,
        _build_graph,
    )

    llm = ChatThaw(model="test-model")
    _install_fake_plumbing(llm)

    fork_call_log = []

    def fake_fork_completions(parent_llm, prompts, sp, pool=None):
        fork_call_log.append({"n_prompts": len(prompts), "prompts": prompts, "pool": pool})
        return [MagicMock(text=f"fork-review-{i}") for i in range(len(prompts))]

    graph = _build_graph("thaw")

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

    # Exactly one fork_fanout → one fork_completions call with all 4 suffixes
    assert len(fork_call_log) == 1
    assert fork_call_log[0]["n_prompts"] == 4

    prompts = fork_call_log[0]["prompts"]
    assert any("security" in p for p in prompts)
    assert any("performance" in p for p in prompts)
    assert any("style" in p for p in prompts)
    assert any("correctness" in p for p in prompts)


@pytest.mark.asyncio
async def test_pr_review_graph_baseline_mode_uses_send_fanout():
    """baseline mode dispatches 4 per-specialist ainvokes — no fork_completions."""
    from demos.pr_review_langgraph import (
        _DEFAULT_CONTEXT,
        _DEFAULT_DIFF,
        _build_graph,
    )
    from tests.test_langgraph_chat_model import FakeRequestOutput

    llm = ChatThaw(model="test-model")
    fake_parent = _install_fake_plumbing(llm)

    # _do_singles passes all N prompts in one generate() call and expects N
    # outputs back. _do_single passes one and expects one. Adapt the fake
    # to the shape of the first positional arg.
    def _variable_length_generate(prompt_or_prompts, *args, **kwargs):
        if isinstance(prompt_or_prompts, list):
            return [FakeRequestOutput(f"baseline-{i}") for i in range(len(prompt_or_prompts))]
        return [FakeRequestOutput("baseline")]

    fake_parent.generate.side_effect = _variable_length_generate

    fork_call_log = []

    def fake_fork_completions(*args, **kwargs):
        fork_call_log.append(True)
        return []

    graph = _build_graph("baseline")

    with patch.object(
        _thaw_fork_module, "fork_completions", side_effect=fake_fork_completions
    ):
        result = await graph.ainvoke(
            {"diff": _DEFAULT_DIFF, "context": _DEFAULT_CONTEXT, "reviews": []},
            config={"configurable": {"llm": llm}},
        )

    assert len(result["reviews"]) == 4
    # fork_completions must not run in baseline mode
    assert fork_call_log == []
    # Every specialist branch drives parent.generate via _do_single or _do_singles
    assert fake_parent.generate.call_count >= 1
