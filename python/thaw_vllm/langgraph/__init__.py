"""LangGraph / LangChain integration for thaw's fork primitive.

Exposes :class:`ChatThaw`, a drop-in replacement for chat models like
``ChatOpenAI`` that transparently routes LangGraph ``Send`` fan-out through
``ForkPool.fork_completions``. The "change one import line, N× faster" story
for agent workflows with shared parent context.

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
