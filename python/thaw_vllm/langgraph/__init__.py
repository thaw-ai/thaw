"""LangGraph / LangChain integration for thaw's fork primitive.

Exposes two entry points:

  * :class:`ChatThaw` — a LangChain ``BaseChatModel`` backed by a vLLM
    parent + thaw ``ForkPool``. Behaves like any other chat model for
    single and concurrent ``ainvoke`` calls.

  * :func:`fork_fanout` — the explicit fork primitive. Given a parent
    message list and a list of divergent suffix message lists, snapshots
    the parent's cached prefix once and fans out to the pool. This is
    what the fork_pool_rl receipt measures and the recommended way to
    get sub-second amortized fork latency inside a LangGraph node.

Requires langchain-core and langgraph (``pip install thaw-vllm[langgraph]``).
The coalescer module itself has no langchain dependency and can be reused
directly if you are wiring thaw into a non-LangGraph framework.
"""
from __future__ import annotations

from thaw_vllm.langgraph.coalescer import (
    ForkCallable,
    ForkCoalescer,
    SingleCallable,
    TokenCounter,
)

__all__ = [
    "ForkCoalescer",
    "ForkCallable",
    "SingleCallable",
    "TokenCounter",
]

# ChatThaw import is lazy: it requires langchain-core, which is an optional dep.
# Users who install thaw-vllm[langgraph] get `from thaw_vllm.langgraph import ChatThaw`;
# users who install bare thaw-vllm get a clear ImportError at use time.
try:
    from thaw_vllm.langgraph.chat_model import ChatThaw, fork_fanout  # noqa: F401

    __all__ += ["ChatThaw", "fork_fanout"]
except ImportError:
    pass
